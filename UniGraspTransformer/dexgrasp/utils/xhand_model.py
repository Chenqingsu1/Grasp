import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch


class XHandModel:
    def __init__(self, gym, sim, urdf_path, asset_root, device='cuda:0'):
        """
        XHandModel: 使用 Isaac Gym 加载 XHand 并获取其 kinematics 相关信息
        ----------
        gym: isaacgym.gymapi.Gym
            Isaac Gym API 实例
        sim: isaacgym.gymapi.Sim
            当前物理仿真环境
        urdf_path: str
            XHand 的 URDF 文件路径
        asset_root: str
            资产文件夹（包含 `urdf_path`）
        device: str
            计算设备（默认 `cuda:0`）
        """
        self.gym = gym
        self.sim = sim
        self.device = device

        # 载入 XHand URDF
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        

        print(f"✅ Loading XHand URDF: {urdf_path}")
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, urdf_path, asset_options)

        # 获取 DOF（关节）信息
        self.num_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        self.dof_names = [self.gym.get_asset_dof_name(self.hand_asset, i) for i in range(self.num_dofs)]
        self.dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        # 关节限制
        self.dof_lower_limits = torch.tensor(self.dof_props['lower'], device=self.device)
        self.dof_upper_limits = torch.tensor(self.dof_props['upper'], device=self.device)


        # 获取刚体信息
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        
        
        self.body_names = [self.gym.get_asset_rigid_body_name(self.hand_asset, i) for i in range(self.num_bodies)]

    
    def create_actor(self, env, position=(0, 0.5, 0), orientation=(0, 0, 0, 1), name="xhand"):
        """
        在指定环境中创建 XHand 角色
        ----------
        env: isaacgym.gymapi.Env
            目标环境
        position: tuple
            机械手初始位置
        orientation: tuple
            机械手初始姿态（四元数）
        name: str
            角色名称
        """
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(*position)
        initial_pose.r = gymapi.Quat(*orientation)

        
        actor_handle = self.gym.create_actor(env, self.hand_asset, initial_pose, name, 0, 1)

        return actor_handle
