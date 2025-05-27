from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
import time

# 1. 初始化仿真环境
my_world = World(stage_units_in_meters=1.0)

# 2. 机器人配置参数
ROBOT_CONFIG = {
    "usd_path": "/home/hans/文档/xwechat_files/wxid_bqwg9dysy4jn12_e34b/msg/file/2025-03/charge42/SR5Bot/xMateSR4C_gen.usd",
    "urdf_path": "/home/hans/文档/xwechat_files/wxid_bqwg9dysy4jn12_e34b/msg/file/2025-02/xMateSR4C_description/urdf/xMateSR4C_gen.urdf",
    "robot_description_path": "/home/hans/文档/xwechat_files/wxid_bqwg9dysy4jn12_e34b/msg/file/2025-03/charge42/robot_description.yaml",
    "prim_path": "/World/xMateSR4C",
    "ee_frame_name": "xMateSR4C_link6",
    "joint_names": [
        "xmate_joint_1", "xmate_joint_2", "xmate_joint_3",
        "xmate_joint_4", "xmate_joint_5", "xmate_joint_6"
    ]
}

# 3. 加载机器人USD
add_reference_to_stage(ROBOT_CONFIG["usd_path"], ROBOT_CONFIG["prim_path"])

# 4. 创建机器人实例（修正基座方向）
robot = SingleArticulation(
    prim_path=ROBOT_CONFIG["prim_path"],
    name="xMateSR4C",
    translation=np.array([0, 0, 0]),
    orientation=np.array([1, 0, 0, 0]), #R.from_euler('x', 180, degrees=True).as_quat()  # 显式绕X轴旋转180度
)
my_world.scene.add(robot)
my_world.scene.add_default_ground_plane()
my_world.reset()

# 5. 初始化Lula运动学求解器
try:
    lula_kinematics = LulaKinematicsSolver(
        robot_description_path=ROBOT_CONFIG["robot_description_path"],
        urdf_path=ROBOT_CONFIG["urdf_path"]
    )
    print("成功加载Lula运动学求解器")
except Exception as e:
    print(f"加载失败: {str(e)}")
    simulation_app.close()
    exit()

# 6. 验证运动学参数
print("\n=== 运动学参数验证 ===")
print("有效关节:", lula_kinematics.get_joint_names())
print("末端执行器帧:", lula_kinematics.get_all_frame_names())
lower_limits, upper_limits = lula_kinematics.get_cspace_position_limits()
print("关节位置限制（度）:")
print(f"下限: {np.rad2deg(lower_limits)}")
print(f"上限: {np.rad2deg(upper_limits)}")

# 7. 创建关节运动学包装器
articulation_solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    kinematics_solver=lula_kinematics,
    end_effector_frame_name=ROBOT_CONFIG["ee_frame_name"]
)

# 8. 设置初始猜测（多组种子提高成功率）
lula_kinematics.set_default_cspace_seeds([
    np.deg2rad([0, 0, 0, 0, 0, 0]),      # 零位
    np.deg2rad([30, 45, -30, 60, 0, 0]),  # 猜测1
    np.deg2rad([-30, 30, 45, -45, 0, 0]) # 猜测2
])

# 9. 设置可达测试目标（基于修正后的坐标系）
target_position = np.array([0.521, 0.136, 0.7])  # Z轴保持高位
target_orientation_first = np.array([1, 0, 0, 0]) #wxyz
'''# 获取基座的实际旋转（必须在my_world.reset()之后调用）
base_pos, base_quat = robot.get_world_pose()
base_rotation = R.from_quat(base_quat)

# 定义世界坐标系中的目标姿态
world_target_position = np.array([0.7, 0.14, 0.5])  # 世界坐标系位置
world_target_orientation = R.from_euler('xyz', [0, 0, 0], degrees=True)  # 世界坐标系方向

# 转换到基座坐标系
target_position = base_rotation.inv().apply(world_target_position - base_pos)  # 位置转换
target_orientation = (base_rotation.inv() * world_target_orientation).as_quat()  # 方向转换
# ========== 新增代码结束 ========== '''

# 打印验证
print(f"\n转换后的目标位置（基座坐标系）: {target_position}")
print(f"转换后的目标方向（基元四元数）: {target_orientation_first}")
# 10. 配置求解器参数
lula_kinematics.set_default_position_tolerance(0.005)  # 5mm精度
lula_kinematics.set_default_orientation_tolerance(np.deg2rad(2))  # 2度精度
lula_kinematics.bfgs_max_iterations = 1000
lula_kinematics.ccd_max_iterations = 200
lula_kinematics.bfgs_orientation_weight = 2.0
lula_kinematics.ccd_orientation_weight = 2.0

# 11. 验证基座坐标系
base_pos, base_quat = robot.get_world_pose()
print(f"\n基座坐标系原点（世界坐标系）: {base_pos}")
print(f"\n基座方向: {base_quat}")
# print(f"基座方向（欧拉角-度）: {R.from_quat(base_quat).as_euler('xyz', degrees=True)}")

# 12. 零位验证
zero_joints = np.zeros(6)
fk_pos, fk_rot = articulation_solver.compute_end_effector_pose(zero_joints)
print(f"\n零位末端位置（基座坐标系）: {fk_pos}")

# 13. 仿真循环
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    
    
    # 在每一帧之后延时 0.05 秒（可以根据需求调整时间）
    time.sleep(0.08)

    if my_world.is_stopped():
        reset_needed = True
        
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            
        # 获取当前状态
        current_joints = robot.get_joint_positions()
        current_pos, current_rot = articulation_solver.compute_end_effector_pose()
        
        # 计算误差
        target_orientation_last = np.concatenate((target_orientation_first[1:], target_orientation_first[:1])) #xyzw
        target_rot_mat = R.from_quat(target_orientation_last).as_matrix()
        current_rot_mat = current_rot if current_rot.shape == (3,3) else R.from_quat(current_rot).as_matrix()
        
        # 位置误差计算
        pos_error = np.linalg.norm(target_position - current_pos)
        
        # 角度误差计算
        orientation_error = np.arccos(
            np.clip((np.trace(target_rot_mat.T @ current_rot_mat) - 1) / 2, -1.0, 1.0))
        
        print(f"\n当前位置: {np.round(current_pos, 3)} | 目标位置: {target_position}")
        print(f"位置误差: {pos_error:.4f}m | 角度误差: {np.rad2deg(orientation_error):.2f}°")
        
        # 求解逆运动学
        ik_action, success = articulation_solver.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation_first
        )
        
        if success:
            print("\n=== IK解算成功 ===")
            print("目标关节角度（度）:", np.rad2deg(ik_action.joint_positions))
            
            # 关节限位检查
            has_joint_limit_issue = False
            for i in range(len(ik_action.joint_positions)):
                joint_name = ROBOT_CONFIG["joint_names"][i]
                pos = ik_action.joint_positions[i]
                lower = lower_limits[i]
                upper = upper_limits[i]
                
                if pos < lower or pos > upper:
                    has_joint_limit_issue = True
                    carb.log_warn(f"{joint_name} 越界: {np.rad2deg(pos):.2f}°")
                    carb.log_warn(f"允许范围: {np.rad2deg(lower):.2f}° ~ {np.rad2deg(upper):.2f}°")
            
            '''# 使用解算得到的关节角度重新计算FK (放在循环外，避免重复计算)
            fk_pos, fk_rot = articulation_solver.compute_end_effector_pose(ik_action.joint_positions)
            
            # 位置验证
            pos_error = np.linalg.norm(fk_pos - target_position)
            print(f"正向运动学验证误差: {pos_error:.4f}m (应接近0)")
            
            # 方向验证（使用旋转向量误差）
            target_rot = R.from_quat(target_orientation)
            current_rot = R.from_matrix(fk_rot) if fk_rot.shape == (3,3) else R.from_quat(fk_rot)
            orientation_error = np.linalg.norm((target_rot.inv() * current_rot).as_rotvec())
            print(f"方向误差（弧度）: {orientation_error:.4f} ≈ {np.rad2deg(orientation_error):.2f}°")'''

            current_rot_mat = current_rot if current_rot.shape == (3,3) else R.from_quat(current_rot).as_matrix()
            print("实际旋转矩阵:\n", np.round(current_rot_mat, 2))
            print("目标旋转矩阵:\n", R.from_quat(target_orientation_last).as_matrix())
           
           
                    
            # 应用控制指令
            robot.get_articulation_controller().apply_action(ik_action)
        else:
            carb.log_warn("IK求解失败，建议：")
            carb.log_warn("1. 检查目标是否在红色坐标系显示范围内")
            carb.log_warn("2. 尝试调整初始猜测种子")
            carb.log_warn("3. 检查URDF/YAML关节限制")

simulation_app.close()