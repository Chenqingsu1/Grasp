import isaacgym
from isaacgym import gymapi
import numpy as np

# 初始化 gym
gym = gymapi.acquire_gym()

# 创建模拟环境
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 创建地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 加载手模型资产
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = False
asset_options.armature = 0.01

# 加载 xhand_right.urdf
hand_asset = gym.load_asset(sim, "/PROJECT", "Assets/XAssets/xhand_right.urdf", asset_options)

# 获取手掌(palm)的链接索引
num_bodies = gym.get_asset_rigid_body_count(hand_asset)
palm_index = -1
for i in range(num_bodies):
    body_name = gym.get_asset_rigid_body_name(hand_asset, i)
    print(f"Body name: {body_name}")
    if "right_hand_link" in body_name.lower():
        palm_index = i
        break

if palm_index == -1:
    print("找不到right_hand_link链接，请检查您的模型链接名称")
    exit()
else:
    print(f"找到palm链接，索引为 {palm_index}")

# 设置环境
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# 创建手的actor
hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)  # 手的位置
hand_pose.r = gymapi.Quat(0, 0, 0, 1)  # 手的旋转
hand_handle = gym.create_actor(env, hand_asset, hand_pose, "hand", 0, 1)

# 创建标记球资产
marker_options = gymapi.AssetOptions()
marker_radius = 0.01  # 1厘米半径的小球
marker_asset = gym.create_sphere(sim, marker_radius, marker_options)

# 获取palm的全局位置
hand_transforms = gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_POS)
palm_transform = hand_transforms[palm_index]
palm_position = palm_transform['pose']['p']

# 创建标记球在palm位置上
marker_pose = gymapi.Transform()
marker_pose.p = gymapi.Vec3(palm_position[0], palm_position[1], palm_position[2])
marker_handle = gym.create_actor(env, marker_asset, marker_pose, "marker", 0, 1)

# 设置标记球颜色（红色）
gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))

# 创建查看器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("Failed to create viewer")
    exit()

# 设置相机位置
cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 模拟循环
while not gym.query_viewer_has_closed(viewer):
    # 模拟物理
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 获取手掌当前位置
    hand_transforms = gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_POS)
    palm_transform = hand_transforms[palm_index]
    palm_position = palm_transform['pose']['p']
    
    # 创建偏移向量（基于原始代码的逻辑）
    offset_z = np.array([0, 0, -0.05])  # 沿Z轴向下偏移0.2
    offset_y = np.array([0, 0.02, 0])  # 沿Y轴向后偏移0.02
    
    # 应用偏移到标记球位置
    marker_state = gym.get_actor_rigid_body_states(env, marker_handle, gymapi.STATE_POS)
    
    # 单独设置每个坐标分量
    marker_state['pose']['p'][0][0] = float(palm_position[0] + offset_z[0] + offset_y[0])
    marker_state['pose']['p'][0][1] = float(palm_position[1] + offset_z[1] + offset_y[1])
    marker_state['pose']['p'][0][2] = float(palm_position[2] + offset_z[2] + offset_y[2])
    
    gym.set_actor_rigid_body_states(env, marker_handle, marker_state, gymapi.STATE_POS)
    
    # 更新视图
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)