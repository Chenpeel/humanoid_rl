"""
行走任务奖励函数

基于开源实现（Legged Gym, Isaac Lab）的行走奖励函数。
使用 JAX 实现纯函数式设计，支持 JIT 编译。

主要参考:
- Legged Gym (ETH Zurich): https://github.com/leggedrobotics/legged_gym
- Isaac Lab Locomotion: omni.isaac.lab_tasks/locomotion
- ANYmal/Go1 机器人实现
"""

from typing import Dict, Optional, Tuple, List
import jax
import jax.numpy as jp


# ============================================================================================
# ======================================= 默认奖励权重 =========================================
# ============================================================================================
DEFAULT_WALKING_REWARD_WEIGHTS = {
    "forward_velocity": 1.5,
    "gait_symmetry": 0.3,
    "foot_clearance": 0.2,
    "trunk_height": 0.5,
    "orientation": -0.3,
    "trunk_lin_vel_z": -0.5,
    "drag": -1.0,
    "action_rate": -0.01,
    "torques": -0.0001,
    "alive": 0.3,
}
# ============================================================================================
# ===================================== END: 默认奖励权重 ======================================
# ============================================================================================


# ============================================================================================
# ================================ 三阶段课程学习权重配置 ========================================
# ============================================================================================

# -------------------------------- 阶段 1: 站立平衡 --------------------------------
STAGE1_WEIGHTS = {
    "termination": -200.0,
    "alive": 5.0,
    "upright_bonus": 2.0,
    "trunk_height": 2.0,
    "orientation": -1.5,
    "trunk_lin_vel_z": -1.0,
    "action_rate": -0.02,
    "torques": -0.0002,
    "joint_limits": -0.2,
    "forward_velocity": 0.0,
    "gait_symmetry": 0.0,
    "foot_clearance": 0.0,
    "drag": 0.0,
}

# -------------------------------- 阶段 2: 低速行走 --------------------------------
STAGE2_WEIGHTS = {
    "termination": -200.0,
    "alive": 2.0,
    "gait_symmetry": 1.5,
    "foot_clearance": 0.8,
    "feet_contact_forces": 0.3,
    "feet_air_time": 0.5,
    "forward_velocity": 1.0,
    "upright_bonus": 0.5,
    "trunk_height": 1.0,
    "orientation": -0.8,
    "trunk_lin_vel_z": -0.5,
    "drag": -0.8,
    "feet_slide": -0.8,
    "action_rate": -0.01,
    "torques": -0.0001,
    "joint_symmetry": 0.2,
    "stumbling": -0.5,
}

# -------------------------------- 阶段 3: 全速行走 --------------------------------
STAGE3_WEIGHTS = {
    "termination": -200.0,
    "alive": 0.3,
    "velocity_tracking": 2.0,
    "gait_symmetry": 0.3,
    "foot_clearance": 0.2,
    "feet_air_time": 0.25,
    "feet_contact_forces": 0.1,
    "trunk_height": 0.8,
    "orientation": -0.5,
    "upright_bonus": 0.0,
    "trunk_lin_vel_z": -0.5,
    "drag": -1.0,
    "feet_slide": -0.5,
    "stumbling": -0.8,
    "action_rate": -0.01,
    "torques": -0.001,
    "energy_efficiency": 0.001,
    "joint_symmetry": 0.15,
    "joint_limits": -0.1,
    "landing_impact": 0.2,
    "stability": 0.1,
}
# ============================================================================================
# ============================== END: 三阶段课程学习权重配置 =====================================
# ============================================================================================


# =============================================================================================
# ========================================= 数学工具函数 ========================================
# =============================================================================================

def quat_to_euler(quat: jax.Array) -> jax.Array:
    """四元数转欧拉角 [roll, pitch, yaw]"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp) * jp.pi / 2, jp.arcsin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)
    return jp.stack([roll, pitch, yaw], axis=-1)

# ---------------------------------------------------------------------------------------------

def normalize_quaternion(quat: jax.Array) -> jax.Array:
    """归一化四元数"""
    norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    return jp.where(norm > 1e-8, quat / norm, jp.array([1.0, 0.0, 0.0, 0.0]))

# ---------------------------------------------------------------------------------------------

def wrap_to_pi(angles: jax.Array) -> jax.Array:
    """角度包装到 [-π, π]"""
    return jp.arctan2(jp.sin(angles), jp.cos(angles))

# =============================================================================================
# ======================================= END: 数学工具函数  ===================================
# =============================================================================================


# ============================================================================================
# ======================================= 接触检测辅助函数 ======================================
# ============================================================================================

def get_feet_contacts(contact_sensors: jax.Array, threshold: float = 1.0) -> jax.Array:
    """提取脚部接触状态 [right_foot, left_foot]"""
    contact_sensors = jp.asarray(contact_sensors)
    if contact_sensors.shape[-1] == 4:
        right_contact = jp.maximum(contact_sensors[..., 0], contact_sensors[..., 1])
        left_contact = jp.maximum(contact_sensors[..., 2], contact_sensors[..., 3])
        return jp.stack([right_contact > threshold, left_contact > threshold], axis=-1)
    return contact_sensors > threshold

# ============================================================================================
# ===================================== END: 接触检测辅助函数 =====================================
# ============================================================================================


# ==============================================================================================
# ======================================= 核心运动奖励 ===========================================
# ==============================================================================================

def compute_forward_velocity_reward(
    base_linvel: jax.Array, target_velocity: float, tolerance: float = 0.5
) -> jax.Array:
    """前向速度跟踪奖励"""
    forward_vel = base_linvel[..., 0]
    vel_error = jp.abs(forward_vel - target_velocity)
    return jp.exp(-vel_error / tolerance)

# ---------------------------------------------------------------------------------------------

def compute_velocity_tracking_reward(
    actual_velocity: jax.Array,
    command: jax.Array,
    tracking_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    tolerance: float = 0.1,
) -> jax.Array:
    """多轴速度跟踪奖励 [vx, vy, vyaw]"""
    error = actual_velocity - command
    reward_x = jp.exp(-jp.abs(error[..., 0]) / tolerance)
    reward_y = jp.exp(-jp.abs(error[..., 1]) / tolerance)
    reward_yaw = jp.exp(-jp.abs(error[..., 2]) / tolerance)
    wx, wy, wyaw = tracking_weights
    return (wx * reward_x + wy * reward_y + wyaw * reward_yaw) / (wx + wy + wyaw)

# ==============================================================================================
# ===================================== END: 核心运动奖励 =========================================
# ==============================================================================================


# =============================================================================================
# ======================================= 步态质量奖励 ==========================================
# =============================================================================================

def compute_gait_symmetry_reward(contacts: jax.Array) -> jax.Array:
    """步态对称性奖励 (XOR 模式)"""
    right_contact = contacts[..., 0].astype(jp.float32)
    left_contact = contacts[..., 1].astype(jp.float32)
    return jp.abs(right_contact - left_contact)

# ---------------------------------------------------------------------------------------------

def compute_foot_clearance_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    target_clearance: float = 0.05,
    tolerance: float = 0.02,
) -> jax.Array:
    """脚部抬高奖励 (摆动相)"""
    feet_heights = feet_positions[..., 2]
    swing_phase = 1.0 - contacts.astype(jp.float32)
    height_error = jp.abs(feet_heights - target_clearance)
    clearance_reward = jp.exp(-height_error / tolerance) * swing_phase
    return jp.sum(clearance_reward, axis=-1)

# ---------------------------------------------------------------------------------------------

def compute_feet_air_time_reward(
    contact_history: jax.Array, target_duty_cycle: float = 0.5
) -> jax.Array:
    """脚部空中时间奖励"""
    actual_duty_cycle = jp.mean(contact_history, axis=0)
    duty_error = jp.mean(jp.abs(actual_duty_cycle - target_duty_cycle))
    return jp.exp(-duty_error / 0.2)

# =============================================================================================
# ===================================== END: 步态质量奖励 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 稳定性与姿态 ==========================================
# =============================================================================================

def compute_trunk_height_reward(
    torso_z: jax.Array, target_height: float = 0.35, tolerance: float = 0.08
) -> jax.Array:
    """躯干高度保持奖励"""
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)

# ---------------------------------------------------------------------------------------------

def compute_upright_bonus(base_quat: jax.Array, threshold: float = 0.93) -> jax.Array:
    """直立姿态奖励"""
    qw, qx, qy, qz = base_quat[..., 0], base_quat[..., 1], base_quat[..., 2], base_quat[..., 3]
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return jp.where(z_z > threshold, 1.0, 0.0)

# ---------------------------------------------------------------------------------------------

def compute_stability_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    target_height: float = 0.35,
    height_tolerance: float = 0.05,
) -> jax.Array:
    """综合稳定性奖励"""
    height_reward = jp.exp(-jp.abs(torso_z - target_height) / height_tolerance)
    ang_vel_reward = jp.exp(-jp.linalg.norm(base_angvel, axis=-1))
    return (height_reward + ang_vel_reward) / 2.0

# =============================================================================================
# ===================================== END: 稳定性与姿态 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 约束与惩罚项 ==========================================
# =============================================================================================

def compute_trunk_orientation_penalty(quat: jax.Array, max_tilt: float = 0.3) -> jax.Array:
    """躯干过度倾斜惩罚"""
    euler = quat_to_euler(normalize_quaternion(quat))
    roll_penalty = jp.clip(jp.abs(euler[..., 0]) - max_tilt, min=0.0)
    pitch_penalty = jp.clip(jp.abs(euler[..., 1]) - max_tilt, min=0.0)
    return roll_penalty + pitch_penalty

# ---------------------------------------------------------------------------------------------

def compute_drag_penalty(
    feet_positions: jax.Array, contacts: jax.Array, threshold: float = 0.02
) -> jax.Array:
    """拖地惩罚"""
    is_dragging = (feet_positions[..., 2] < threshold) * (1.0 - contacts.astype(jp.float32))
    return jp.sum(is_dragging, axis=-1)

# ---------------------------------------------------------------------------------------------

def compute_feet_slide_penalty(
    feet_linvel: jax.Array, contacts: jax.Array, threshold: float = 0.1
) -> jax.Array:
    """脚部滑动惩罚"""
    slide_vel = jp.linalg.norm(feet_linvel[..., :2], axis=-1)
    is_sliding = (slide_vel > threshold) * (contacts > 0.1)
    return jp.sum(is_sliding.astype(jp.float32) * slide_vel) / jp.maximum(feet_linvel.shape[-2], 1)

# =============================================================================================
# ===================================== END: 约束与惩罚项 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 完整奖励函数 ==========================================
# =============================================================================================

def compute_walking_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    contact_sensors: jax.Array,
    feet_positions: Optional[jax.Array] = None,
    action: jax.Array = None,
    last_action: jax.Array = None,
    torques: jax.Array = None,
    feet_velocities: Optional[jax.Array] = None,
    joint_pos: Optional[jax.Array] = None,
    joint_vel: Optional[jax.Array] = None,
    joint_limits: Optional[tuple] = None,
    contact_history: Optional[jax.Array] = None,
    command: jax.Array = None,
    actual_velocity: jax.Array = None,
    target_velocity: float = 0.5,
    target_height: float = 0.35,
    reward_weights: Dict[str, float] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的行走任务奖励"""
    
    # 初始化
    base_quat = normalize_quaternion(base_quat)
    contacts = get_feet_contacts(contact_sensors)
    reward = jp.array(0.0)
    reward_info = {}
    weights = reward_weights or DEFAULT_WALKING_REWARD_WEIGHTS

    # 1. 速度跟踪
    if "velocity_tracking" in weights and command is not None:
        val = weights["velocity_tracking"] * compute_velocity_tracking_reward(actual_velocity, command)
        reward += val
        reward_info["reward/velocity_tracking"] = val
    elif "forward_velocity" in weights:
        val = weights["forward_velocity"] * compute_forward_velocity_reward(base_linvel, target_velocity)
        reward += val
        reward_info["reward/forward_velocity"] = val

    # 2. 姿态与稳定性
    if "trunk_height" in weights:
        val = weights["trunk_height"] * compute_trunk_height_reward(torso_z, target_height)
        reward += val
        reward_info["reward/trunk_height"] = val
    
    if "orientation" in weights:
        val = weights["orientation"] * compute_trunk_orientation_penalty(base_quat)
        reward += val
        reward_info["reward/orientation"] = val

    if "upright_bonus" in weights:
        val = weights["upright_bonus"] * compute_upright_bonus(base_quat)
        reward += val
        reward_info["reward/upright_bonus"] = val

    # 3. 步态质量
    if "gait_symmetry" in weights:
        val = weights["gait_symmetry"] * compute_gait_symmetry_reward(contacts)
        reward += val
        reward_info["reward/gait_symmetry"] = val

    if "feet_air_time" in weights and contact_history is not None:
        val = weights["feet_air_time"] * compute_feet_air_time_reward(contact_history)
        reward += val
        reward_info["reward/feet_air_time"] = val

    # 4. 物理惩罚项
    if "drag" in weights and feet_positions is not None:
        val = weights["drag"] * compute_drag_penalty(feet_positions, contacts)
        reward += val
        reward_info["reward/drag"] = val

    if "torques" in weights and torques is not None:
        val = weights["torques"] * jp.sum(jp.square(torques), axis=-1)
        reward += val
        reward_info["reward/torques"] = val

    if "alive" in weights:
        val = weights["alive"] * 1.0
        reward += val
        reward_info["reward/alive"] = val

    # 最终保护
    reward = jp.clip(reward, -10.0, 10.0)
    reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

    return reward, reward_info

# =============================================================================================
# ===================================== END: 完整奖励函数 =======================================
# =============================================================================================
