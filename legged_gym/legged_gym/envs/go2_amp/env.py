"""Go2 AMP state and pre-reset terminal capture."""

from legged_gym.envs.random_dog.random_dog import Randomdog
from isaacgym.torch_utils import quat_rotate_inverse
from rl.MGDP.amp.state import pack_amp_state, validate_go2_dofs


class Go2AmpRandomDog(Randomdog):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        validate_go2_dofs(self.dof_names)

    def get_amp_observations(self):
        quat = self.root_states[:, 3:7]
        lin = quat_rotate_inverse(quat, self.root_states[:, 7:10])
        ang = quat_rotate_inverse(quat, self.root_states[:, 10:13])
        return pack_amp_state(self.dof_pos, lin, ang, self.dof_vel)

    def reset_idx(self, env_ids):
        packet = None
        if env_ids.numel() > 0:
            packet = (env_ids.clone(), self.get_amp_observations()[env_ids].clone())
        super().reset_idx(env_ids)
        if packet is not None:
            self.extras["terminal_amp_states"] = packet

    def step(self, actions):
        self.extras.pop("terminal_amp_states", None)
        return super().step(actions)
