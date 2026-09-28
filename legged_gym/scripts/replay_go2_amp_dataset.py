"""Kinematically replay Go2 AMP expert motions in the project Isaac Gym task.

The script deliberately does not call ``env.step``.  Every rendered pose is
written directly from the expert frame, and the corresponding project AMP
observation is checked against the 30 values selected by ``AMP_COLUMNS``.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from isaacgym import gymapi, gymtorch  # Isaac Gym must be imported before torch.
from isaacgym.torch_utils import quat_rotate

import cv2
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers go2_amp_stage1.
from legged_gym.utils import get_args, task_registry
from rl.MGDP.amp.motions import AMP_COLUMNS


TASK_NAME = "go2_amp_stage1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTION_ROOT = PROJECT_ROOT / "datasets" / "go2_motion"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "go2_amp_replay.mp4"

REPLAY_PARAMETERS = [
    {"name": "--clip", "type": str, "default": None,
     "help": "Expert clip filename from datasets/go2_motion."},
    {"name": "--all", "action": "store_true", "default": False,
     "help": "Replay every expert clip in filename order."},
    {"name": "--loop", "action": "store_true", "default": False,
     "help": "Loop until the interactive viewer is closed."},
    {"name": "--record", "action": "store_true", "default": False,
     "help": "Record an Isaac Gym camera sensor to MP4."},
    {"name": "--output", "type": str, "default": str(DEFAULT_OUTPUT),
     "help": "MP4 output path; relative paths are resolved from the project root."},
    {"name": "--overwrite", "action": "store_true", "default": False,
     "help": "Allow --record to replace an existing MP4."},
    {"name": "--tolerance", "type": float, "default": 1e-5,
     "help": "Maximum allowed absolute error for the 30-D AMP state."},
    {"name": "--width", "type": int, "default": 1280,
     "help": "Recorded video width in pixels."},
    {"name": "--height", "type": int, "default": 720,
     "help": "Recorded video height in pixels."},
    {"name": "--camera_distance", "type": float, "default": 2.2,
     "help": "Distance of the following viewer/video camera in metres."},
]


@dataclass(frozen=True)
class MotionClip:
    path: Path
    frame_dt: float
    frames: np.ndarray


def load_clip(path: Path) -> MotionClip:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        frame_dt = float(raw["FrameDuration"])
        frames = np.asarray(raw["Frames"], dtype=np.float32)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read an AMP motion clip") from exc

    if frames.ndim != 2 or frames.shape[1] != 61 or len(frames) < 2:
        raise ValueError(f"{path}: expected at least two 61-D frames")
    if not np.isfinite(frames).all():
        raise ValueError(f"{path}: contains non-finite frame values")
    if not math.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError(f"{path}: FrameDuration must be positive")

    quaternion_norms = np.linalg.norm(frames[:, 3:7], axis=1)
    worst_quaternion_error = float(np.max(np.abs(quaternion_norms - 1.0)))
    if worst_quaternion_error > 1e-4:
        raise ValueError(
            f"{path}: root quaternion norm error {worst_quaternion_error:.3e} exceeds 1e-4"
        )
    return MotionClip(path=path, frame_dt=frame_dt, frames=frames)


def select_clips(clip_name: Optional[str], play_all: bool) -> List[MotionClip]:
    if bool(clip_name) == bool(play_all):
        raise ValueError("select exactly one of --clip FILE or --all")
    if not MOTION_ROOT.is_dir():
        raise ValueError(f"motion directory does not exist: {MOTION_ROOT}")

    if play_all:
        paths = sorted(MOTION_ROOT.glob("*.txt"))
        if not paths:
            raise ValueError(f"no .txt motion clips found in {MOTION_ROOT}")
    else:
        name = Path(clip_name)
        if name.name != clip_name:
            raise ValueError("--clip must be a filename from datasets/go2_motion")
        paths = [MOTION_ROOT / name]
        if not paths[0].is_file():
            raise ValueError(f"motion clip does not exist: {paths[0]}")

    return [load_clip(path) for path in paths]


def amp_field_name(index: int, dof_names: Sequence[str]) -> str:
    if index < 12:
        return f"joint_pos.{dof_names[index]}"
    if index < 15:
        return f"base_lin_vel.{('x', 'y', 'z')[index - 12]}"
    if index < 18:
        return f"base_ang_vel.{('x', 'y', 'z')[index - 15]}"
    return f"joint_vel.{dof_names[index - 18]}"


def configure_replay_environment(env_cfg) -> Path:
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False

    env_cfg.camera.use_camera = False
    env_cfg.camera.use_lidar = False
    env_cfg.camera.camera_res = None
    env_cfg.camera.world_model = False
    env_cfg.camera.use_memory = False
    env_cfg.camera.use_map_decoder = False
    env_cfg.camera.update_wm = False
    env_cfg.camera.load_world_model_policy = False

    for name in (
        "push_robots", "randomize_friction", "randomize_gains",
        "randomize_base_mass", "randomize_limb_mass", "randomize_action_latency",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)

    asset_names = list(env_cfg.asset.asset_name)
    if asset_names != ["go2"]:
        raise ValueError(f"{TASK_NAME} must resolve exactly one Go2 asset, got {asset_names}")
    asset_root = Path(env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    urdf_path = asset_root / "go2" / "urdf" / "go2.urdf"
    if not urdf_path.is_file():
        raise ValueError(f"project Go2 URDF does not exist: {urdf_path}")
    return urdf_path.resolve()


class VideoRecorder:
    def __init__(self, env, width: int, height: int, output: Path,
                 fps: float, camera_distance: float):
        properties = gymapi.CameraProperties()
        properties.width = width
        properties.height = height
        properties.enable_tensors = False
        self.env = env
        self.width = width
        self.height = height
        self.camera_distance = camera_distance
        self.camera_handle = env.gym.create_camera_sensor(env.envs[0], properties)
        if self.camera_handle < 0:
            raise RuntimeError("Isaac Gym failed to create the replay camera sensor")

        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV failed to open MP4 output: {output}")
        self.output = output

    def update_camera(self, root_position: np.ndarray) -> None:
        target = root_position + np.array([0.25, 0.0, 0.10], dtype=np.float64)
        offset = self.camera_distance * np.array([-1.0, 0.65, 0.45], dtype=np.float64)
        position = target + offset
        self.env.gym.set_camera_location(
            self.camera_handle,
            self.env.envs[0],
            gymapi.Vec3(*position),
            gymapi.Vec3(*target),
        )

    def write_frame(self) -> None:
        raw = self.env.gym.get_camera_image(
            self.env.sim,
            self.env.envs[0],
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        rgba = np.asarray(raw)
        if rgba.shape == (self.height, self.width) and rgba.dtype.itemsize == 4:
            rgba = rgba.view(np.uint8)
        try:
            rgba = rgba.reshape(self.height, self.width, 4)
        except ValueError as exc:
            raise RuntimeError(
                f"unexpected Isaac Gym camera image shape {rgba.shape}; "
                f"expected ({self.height}, {self.width}, 4)"
            ) from exc
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        self.writer.write(bgr)

    def close(self) -> None:
        self.writer.release()


class KinematicReplayer:
    def __init__(self, env, tolerance: float, camera_distance: float,
                 recorder: Optional[VideoRecorder]):
        self.env = env
        self.tolerance = tolerance
        self.camera_distance = camera_distance
        self.recorder = recorder
        self.amp_indices = torch.as_tensor(
            AMP_COLUMNS, dtype=torch.long, device=env.device
        )
        actor_index = env.gym.get_actor_index(
            env.envs[0], env.actor_handles[0], gymapi.DOMAIN_SIM
        )
        self.actor_ids = torch.tensor(
            [actor_index], dtype=torch.int32, device=env.device
        )
        self.maximum_error = 0.0
        self.worst_case: Optional[Tuple[str, int, int, float, float]] = None

    def apply_frame(self, frame: np.ndarray) -> None:
        state = torch.as_tensor(frame, dtype=self.env.root_states.dtype,
                                device=self.env.device)
        root_quaternion = state[3:7].unsqueeze(0)

        self.env.root_states[0, 0:3] = state[0:3]
        self.env.root_states[0, 3:7] = state[3:7]
        self.env.root_states[0, 7:10] = quat_rotate(
            root_quaternion, state[31:34].unsqueeze(0)
        )[0]
        self.env.root_states[0, 10:13] = quat_rotate(
            root_quaternion, state[34:37].unsqueeze(0)
        )[0]
        self.env.dof_pos[0] = state[7:19]
        self.env.dof_vel[0] = state[37:49]

        root_updated = self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.root_states),
            gymtorch.unwrap_tensor(self.actor_ids),
            len(self.actor_ids),
        )
        dof_updated = self.env.gym.set_dof_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.dof_state),
            gymtorch.unwrap_tensor(self.actor_ids),
            len(self.actor_ids),
        )
        if root_updated is False or dof_updated is False:
            raise RuntimeError("Isaac Gym rejected an expert root or DOF state")

        # The replay forces the CPU tensor pipeline, where setters take effect
        # immediately. Read the state back from PhysX before AMP validation.
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)

    def validate_frame(self, clip: MotionClip, frame_index: int) -> None:
        expected = torch.as_tensor(
            clip.frames[frame_index], dtype=self.env.root_states.dtype,
            device=self.env.device,
        )[self.amp_indices]
        actual = self.env.get_amp_observations()[0]
        errors = torch.abs(actual - expected)
        error, dimension_tensor = torch.max(errors, dim=0)
        error_value = float(error.item())
        dimension = int(dimension_tensor.item())

        if error_value > self.maximum_error:
            self.maximum_error = error_value
            self.worst_case = (
                clip.path.name,
                frame_index,
                dimension,
                float(expected[dimension].item()),
                float(actual[dimension].item()),
            )
        if error_value > self.tolerance:
            field = amp_field_name(dimension, self.env.dof_names)
            raise RuntimeError(
                "AMP VALIDATION FAILED\n"
                f"  clip: {clip.path.name}\n"
                f"  frame: {frame_index} / {len(clip.frames) - 1}\n"
                f"  time: {frame_index * clip.frame_dt:.6f} s\n"
                f"  dimension: {dimension}\n"
                f"  field: {field}\n"
                f"  expected: {expected[dimension].item():.9g}\n"
                f"  actual: {actual[dimension].item():.9g}\n"
                f"  absolute error: {error_value:.3e}\n"
                f"  tolerance: {self.tolerance:.3e}"
            )

    def update_cameras(self, root_position: np.ndarray) -> None:
        target = root_position + np.array([0.25, 0.0, 0.10], dtype=np.float64)
        offset = self.camera_distance * np.array([-1.0, 0.65, 0.45], dtype=np.float64)
        if self.env.viewer is not None:
            self.env.set_camera(target + offset, target)
        if self.recorder is not None:
            self.recorder.update_camera(root_position)

    def render(self) -> bool:
        if self.env.viewer is not None:
            if self.env.gym.query_viewer_has_closed(self.env.viewer):
                return False
            for event in self.env.gym.query_viewer_action_events(self.env.viewer):
                if event.action == "QUIT" and event.value > 0:
                    return False

        self.env.gym.step_graphics(self.env.sim)
        if self.recorder is not None:
            self.env.gym.render_all_camera_sensors(self.env.sim)
            self.recorder.write_frame()
        if self.env.viewer is not None:
            self.env.gym.draw_viewer(self.env.viewer, self.env.sim, True)
        return True

    def play_clip(self, clip: MotionClip) -> Tuple[bool, int]:
        print(
            f"Playing {clip.path.name}: {len(clip.frames)} frames, "
            f"{(len(clip.frames) - 1) * clip.frame_dt:.2f} s at "
            f"{1.0 / clip.frame_dt:.2f} FPS"
        )
        next_frame_time = time.perf_counter()
        for frame_index, frame in enumerate(clip.frames):
            self.apply_frame(frame)
            self.validate_frame(clip, frame_index)
            root_position = frame[0:3].astype(np.float64, copy=False)
            self.update_cameras(root_position)
            if not self.render():
                return False, frame_index + 1

            if self.env.viewer is not None:
                next_frame_time += clip.frame_dt
                remaining = next_frame_time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        return True, len(clip.frames)


def validate_arguments(args, clips: Sequence[MotionClip]) -> Path:
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("--tolerance must be finite and non-negative")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be positive even integers")
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        raise ValueError("--camera_distance must be finite and positive")
    if args.headless and not args.record:
        raise ValueError("--headless requires --record because no viewer would be visible")
    if args.loop and args.headless:
        raise ValueError("--loop requires the interactive viewer")

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    if args.record and output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    if args.record and output.exists() and not args.overwrite:
        raise ValueError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    if args.record:
        first_dt = clips[0].frame_dt
        for clip in clips[1:]:
            if not math.isclose(clip.frame_dt, first_dt, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("all clips in one MP4 must have the same FrameDuration")
    return output


def main() -> None:
    args = get_args(
        extra_custom_parameters=REPLAY_PARAMETERS,
        description="Replay Go2 AMP expert motions on the project Isaac Gym URDF",
    )
    try:
        clips = select_clips(args.clip, args.all)
        output = validate_arguments(args, clips)
    except ValueError as exc:
        raise SystemExit(str(exc))

    args.task = TASK_NAME
    args.num_envs = 1
    # GPU tensor setters are deferred until gym.simulate(), which would advance
    # the expert state under physics.  The CPU tensor pipeline applies setters
    # immediately while PhysX itself may still run on the selected sim device.
    args.use_gpu_pipeline = False
    if hasattr(args, "pipeline"):
        args.pipeline = "cpu"
    env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME)
    try:
        urdf_path = configure_replay_environment(env_cfg)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"Task: {TASK_NAME}")
    print(f"URDF: {urdf_path}")
    print(f"Motion directory: {MOTION_ROOT}")
    print(f"Clips: {len(clips)}")
    print(f"AMP tolerance: {args.tolerance:.3e}")
    print("Tensor pipeline: CPU (required for exact kinematic state application)")

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    print("DOFs: " + ", ".join(env.dof_names))

    recorder = None
    if args.record:
        recorder = VideoRecorder(
            env=env,
            width=args.width,
            height=args.height,
            output=output,
            fps=1.0 / clips[0].frame_dt,
            camera_distance=args.camera_distance,
        )
        print(f"Recording MP4: {output}")

    # Prime PhysX and any replay camera sensor once. Every expert pose is
    # applied after this step, so physics never advances a displayed frame.
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    replayer = KinematicReplayer(
        env=env,
        tolerance=args.tolerance,
        camera_distance=args.camera_distance,
        recorder=recorder,
    )
    completed_frames = 0
    keep_running = True
    try:
        while keep_running:
            for clip in clips:
                keep_running, frames_played = replayer.play_clip(clip)
                completed_frames += frames_played
                if not keep_running:
                    break
            if not args.loop:
                break
    finally:
        if recorder is not None:
            recorder.close()

    if keep_running:
        print("AMP validation passed")
    else:
        print("Viewer closed; all displayed frames passed AMP validation")
    print(f"Completed frames: {completed_frames}")
    print(f"Maximum absolute error: {replayer.maximum_error:.3e}")
    if replayer.worst_case is not None:
        clip_name, frame_index, dimension, expected, actual = replayer.worst_case
        print(
            "Worst case: "
            f"{clip_name} frame={frame_index} dimension={dimension} "
            f"field={amp_field_name(dimension, env.dof_names)} "
            f"expected={expected:.9g} actual={actual:.9g}"
        )
    if recorder is not None:
        print(f"Video written: {recorder.output}")


if __name__ == "__main__":
    main()
