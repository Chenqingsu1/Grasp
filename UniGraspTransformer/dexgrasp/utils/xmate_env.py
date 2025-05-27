# xmate_env.py – final stable version (accepts xmate_actor / xmate_asset aliases)

import os
import torch
from isaacgym import gymapi, gymtorch, gymutil

# -----------------------------------------------------------------------------
# Quaternion / RPY helpers ------------------------------------------------------
# -----------------------------------------------------------------------------

def rpy_to_quat(rpy: torch.Tensor):
    cr, sr = torch.cos(rpy[0] / 2), torch.sin(rpy[0] / 2)
    cp, sp = torch.cos(rpy[1] / 2), torch.sin(rpy[1] / 2)
    cy, sy = torch.cos(rpy[2] / 2), torch.sin(rpy[2] / 2)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])

# -----------------------------------------------------------------------------
# Jacobian‑DLS IK Controller ----------------------------------------------------
# -----------------------------------------------------------------------------

class XMateIKController:
    def __init__(self, gym, sim, env, *, actor=None, asset=None,
                 xmate_actor=None, xmate_asset=None,
                 ee_name="xMateSR4C_link6", device="cuda"):

        # ---- 兼容两个参数名 --------------------------------------------------
        self.actor = actor or xmate_actor
        self.asset = asset or xmate_asset
        if self.actor is None or self.asset is None:
            raise ValueError("need actor / asset (or xmate_actor / xmate_asset)")

        self.gym, self.sim, self.env = gym, sim, env
        self.ee_name, self.device = ee_name, device

        # ---- acquire PhysX 原生张量 -----------------------------------------
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.rb_state_tensor  = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.jac_tensor       = self.gym.acquire_jacobian_tensor(
            self.sim, self.gym.get_actor_name(self.env, self.actor))

        # ---- wrap 为 torch ---------------------------------------------------
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        rb_raw         = gymtorch.wrap_tensor(self.rb_state_tensor)
        jac_raw        = gymtorch.wrap_tensor(self.jac_tensor)

        self.num_envs = self.gym.get_env_count(self.sim)
        self.num_dofs = self.gym.get_asset_dof_count(self.asset)

        # ---- 由两份张量分别推刚体数 -----------------------------------------
        rb_bodies  = rb_raw.numel()  // (self.num_envs * 13)
        jac_bodies = jac_raw.shape[0] // (self.num_envs * 6)
        self.num_bodies = min(rb_bodies, jac_bodies)   # 取较小者安全

        if rb_bodies != jac_bodies:
            print(f"[XMateIK]  rb_state bodies={rb_bodies}, Jacobian bodies={jac_bodies}."
                  " 继续运行，将按 min(bodies) 使用。")

        # ---- reshape ---------------------------------------------------------
        self.rb_state = rb_raw.view(self.num_envs, rb_bodies, 13)
        self.jacobian = jac_raw.view(self.num_envs, jac_bodies, 6, self.num_dofs)
        self.dof_pos  = self.dof_state[:, 0].view(self.num_envs, self.num_dofs)

        # ---- 末端刚体索引 -----------------------------------------------------
        self.ee_index = self.gym.find_actor_rigid_body_index(
            self.env, self.actor, self.ee_name, gymapi.DOMAIN_SIM)


    # ---------------------------------------------------------------------
    def step_ik(self, target_pos: torch.Tensor, damping: float = 0.05):
        """One IK step toward *target_pos* (shape (1,3))."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        ee_pos = self.rb_state[0, self.ee_index, 0:3].to(self.device)
        delta_x = target_pos.to(self.device).squeeze(0) - ee_pos

        J = self.jacobian[0, self.ee_index, 0:3, :]
        JT = J.T
        inv = torch.inverse(J @ JT + damping ** 2 * torch.eye(3, device=self.device))
        delta_q = JT @ inv @ delta_x

        new_q = self.dof_pos[0] + delta_q
        self.gym.set_actor_dof_position_targets(self.env, self.actor, new_q.cpu().numpy())

# -----------------------------------------------------------------------------
# Simple demo – run this file directly ----------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams(); sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.8); sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    gym.add_ground(sim, gymapi.PlaneParams())

    asset_root = os.path.abspath("./assets"); asset_file = "xmate/xMateSR4C.urdf"
    opts = gymapi.AssetOptions(); opts.fix_base_link = True; opts.disable_gravity = True
    opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset = gym.load_asset(sim, asset_root, asset_file, opts)
    actor = gym.create_actor(env, asset, gymapi.Transform(), "xmate", 0, 0)

    ctrl = XMateIKController(gym, sim, env, actor=actor, asset=asset)
    tgt  = torch.tensor([[0.4, 0.2, 0.5]], device="cuda")
    gym.prepare_sim(sim)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    while not gymutil.is_headless():
        ctrl.step_ik(tgt)
        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
