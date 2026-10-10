"""Go2 AMP state and pre-reset terminal capture."""

import torch

from legged_gym.envs.random_dog.random_dog import Randomdog
from isaacgym.torch_utils import quat_rotate_inverse
from rl.MGDP.amp.state import pack_amp_state, validate_go2_dofs
from .terrain import assign_amp_columns


class Go2AmpRandomDog(Randomdog):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        validate_go2_dofs(self.dof_names)

    def get_amp_observations(self):
        quat = self.root_states[:, 3:7]
        lin = quat_rotate_inverse(quat, self.root_states[:, 7:10])
        ang = quat_rotate_inverse(quat, self.root_states[:, 10:13])
        return pack_amp_state(self.dof_pos, lin, ang, self.dof_vel)

    def _uses_explicit_amp_terrain(self):
        return getattr(self.cfg.terrain, "explicit_terrain_columns", None) is not None

    def set_amp_training_iteration(self, iteration):
        if self._uses_explicit_amp_terrain():
            self.amp_training_iteration = int(iteration)

    def _get_env_origins(self):
        if not self._uses_explicit_amp_terrain():
            return super()._get_env_origins()

        columns, anchors = assign_amp_columns(
            self.num_envs,
            getattr(self.cfg.terrain, "amp_anchor_fraction", 0.15),
        )
        self.custom_origins = self.cfg.terrain.custom_origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.terrain_types = torch.as_tensor(columns, dtype=torch.long, device=self.device)
        self.is_amp_anchor = torch.as_tensor(anchors, dtype=torch.bool, device=self.device)
        self.amp_training_iteration = 0
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(
            device=self.device, dtype=torch.float
        )
        self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(
            device=self.device, dtype=torch.float
        )
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

    def _update_terrain_curriculum(self, env_ids):
        if not self._uses_explicit_amp_terrain():
            return super()._update_terrain_curriculum(env_ids)
        if env_ids.numel() == 0:
            return

        anchor_ids = env_ids[self.is_amp_anchor[env_ids]]
        course_ids = env_ids[~self.is_amp_anchor[env_ids]]
        if anchor_ids.numel() > 0:
            self.terrain_levels[anchor_ids] = 0
            self.terrain_types[anchor_ids] = 0
            self.env_origins[anchor_ids] = self.terrain_origins[0, 0]
            self.env_class[anchor_ids] = self.terrain_class[0, 0]

        unlock_iteration = getattr(
            self.cfg.terrain, "amp_curriculum_unlock_iteration", 500
        )
        if course_ids.numel() == 0:
            return
        if self.amp_training_iteration < unlock_iteration:
            self.terrain_levels[course_ids] = 0
            self.env_origins[course_ids] = self.terrain_origins[0, self.terrain_types[course_ids]]
            self.env_class[course_ids] = self.terrain_class[0, self.terrain_types[course_ids]]
            return
        super()._update_terrain_curriculum(course_ids)

    def _reward_feet_air_time(self):
        air_time_cap = getattr(self.cfg.rewards, "amp_air_time_cap", None)
        if air_time_cap is None:
            return super()._reward_feet_air_time()

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_air_time = torch.sum(
            (torch.clamp(self.feet_air_time, max=air_time_cap) - 0.5) * first_contact,
            dim=1,
        )
        rew_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_air_time

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
