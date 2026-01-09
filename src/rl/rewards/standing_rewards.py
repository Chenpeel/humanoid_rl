"""
站立平衡奖励函数

设计用于 Jiyuan 机器人的基础平衡训练阶段。
侧重于高度保持、垂直姿态和最小化关节动作。
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jp

# =============================================================================================
# ======================================= 默认奖励权重 =========================================
# =============================================================================================

DEFAULT_STANDING_REWARD_WEIGHTS = {
    "height": 2.0,
    "orientation": 1.0,
    "lin_vel": -0.5,
    "ang_vel": -0.5,
    "alive": 1.0,
    "action_rate": -0.01,
    "torques": -0.001,
}

# =============================================================================================
# ======================================= 终止条件检查 ==========================================
# =============================================================================================


def check_standing_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.25,
    upright_threshold: float = 0.5,
) -> jax.Array:
    """
    检查站立是否终止 (摔倒检测)

    Args:
        torso_z: 躯干高度
        base_quat: 基座四元数
        height_threshold: 最小高度阈值 (m)
        upright_threshold: 最小直立度阈值 (z-axis projection)

    Returns:
        bool array, True 表示终止
    """
    # 1. 高度过低
    is_fallen_height = torso_z < height_threshold

    # 2. 姿态倾斜过大
    # z-component of z-axis in world frame: 1 - 2(x^2 + y^2)
    qx, qy = base_quat[..., 1], base_quat[..., 2]
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    is_fallen_orientation = z_z < upright_threshold

    return is_fallen_height | is_fallen_orientation


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


# =============================================================================================
# ======================================= END: 数学工具函数  ===================================
# =============================================================================================


# =============================================================================================
# ======================================= 基础奖励分量 ===========================================
# =============================================================================================


def compute_action_rate_penalty(action: jax.Array, last_action: jax.Array) -> jax.Array:
    """计算动作变化率惩罚"""
    return jp.sum(jp.square(action - last_action), axis=-1)


# ---------------------------------------------------------------------------------------------


def compute_torque_penalty(torques: jax.Array) -> jax.Array:
    """计算扭矩惩罚"""
    return jp.sum(jp.square(torques), axis=-1)


# ---------------------------------------------------------------------------------------------


def compute_height_reward(
    torso_z: jax.Array, target_height: float, tolerance: float = 0.05
) -> jax.Array:
    """计算高度保持奖励"""
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_orientation_reward(
    base_quat: jax.Array, tolerance: float = 0.1
) -> jax.Array:
    """计算姿态稳定奖励（最小化 Roll 和 Pitch）"""
    euler = quat_to_euler(normalize_quaternion(base_quat))
    roll, pitch = euler[..., 0], euler[..., 1]
    error = jp.sqrt(jp.square(roll) + jp.square(pitch))
    return jp.exp(-error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_velocity_penalties(
    base_linvel: jax.Array, base_angvel: jax.Array
) -> Dict[str, jax.Array]:
    """计算速度惩罚（鼓励静止）"""
    return {
        "lin_vel_penalty": jp.sum(jp.square(base_linvel), axis=-1),
        "ang_vel_penalty": jp.sum(jp.square(base_angvel), axis=-1),
    }


# ---------------------------------------------------------------------------------------------


def compute_velocity_penalty(
    base_linvel: jax.Array, base_angvel: jax.Array
) -> jax.Array:
    """计算总速度惩罚 (Wrapper for compatibility)"""
    penalties = compute_velocity_penalties(base_linvel, base_angvel)
    return penalties["lin_vel_penalty"] + penalties["ang_vel_penalty"]


# ---------------------------------------------------------------------------------------------


def compute_joint_deviation_penalty(
    joint_pos: jax.Array,
    joint_pos_default: jax.Array,
    indices: Optional[jax.Array] = None,
) -> jax.Array:
    """关节姿态偏离惩罚 (保持接近默认/home pose)"""
    joint_pos = jp.asarray(joint_pos)
    joint_pos_default = jp.asarray(joint_pos_default)
    if indices is not None:
        indices = jp.asarray(indices)
        joint_pos = joint_pos[..., indices]
        joint_pos_default = joint_pos_default[..., indices]
    diff = joint_pos - joint_pos_default
    return jp.mean(jp.square(diff), axis=-1)


# =============================================================================================
# ===================================== END: 基础奖励分量 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 完整奖励函数 ==========================================
# =============================================================================================


def compute_standing_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    torques: jax.Array,
    target_height: float,
    reward_weights: Dict[str, float],
    joint_pos: Optional[jax.Array] = None,
    joint_pos_default: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的站立平衡奖励

    采用动态奖励组合模式：只计算reward_weights中配置的奖励项，
    提供良好的扩展性和课程学习支持。
    """

    # 初始化
    reward = jp.array(0.0)
    reward_info = {}

    # 1. 高度奖励
    if "height" in reward_weights:
        reward_height = compute_height_reward(torso_z, target_height)
        weighted = reward_weights["height"] * reward_height
        reward += weighted
        reward_info["reward/height"] = weighted

    # 2. 姿态奖励
    if "orientation" in reward_weights:
        reward_orientation = compute_orientation_reward(base_quat)
        weighted = reward_weights["orientation"] * reward_orientation
        reward += weighted
        reward_info["reward/orientation"] = weighted

    # 3. 线速度惩罚
    if "lin_vel" in reward_weights:
        vel_penalties = compute_velocity_penalties(base_linvel, base_angvel)
        weighted = reward_weights["lin_vel"] * vel_penalties["lin_vel_penalty"]
        reward += weighted
        reward_info["reward/lin_vel"] = weighted

    # 4. 角速度惩罚
    if "ang_vel" in reward_weights:
        if "lin_vel" not in reward_weights:  # 避免重复计算
            vel_penalties = compute_velocity_penalties(base_linvel, base_angvel)
        weighted = reward_weights["ang_vel"] * vel_penalties["ang_vel_penalty"]
        reward += weighted
        reward_info["reward/ang_vel"] = weighted

    # 5. 存活奖励
    if "alive" in reward_weights:
        weighted = reward_weights["alive"] * 1.0
        reward += weighted
        reward_info["reward/alive"] = weighted

    # 6. 动作平滑
    if "action_rate" in reward_weights:
        action_rate_penalty = jp.sum(jp.square(action - last_action), axis=-1)
        weighted = reward_weights["action_rate"] * action_rate_penalty
        reward += weighted
        reward_info["reward/action_rate"] = weighted

    # 7. 扭矩惩罚
    if "torques" in reward_weights:
        torque_penalty = jp.sum(jp.square(torques), axis=-1)
        weighted = reward_weights["torques"] * torque_penalty
        reward += weighted
        reward_info["reward/torques"] = weighted

    # 8. 站立姿态正则（防止髋部长期偏置/歪斜）
    if "joint_deviation" in reward_weights:
        if joint_pos is not None and joint_pos_default is not None:
            weighted = reward_weights["joint_deviation"] * compute_joint_deviation_penalty(
                joint_pos, joint_pos_default
            )
        else:
            weighted = jp.array(0.0)
        reward += weighted
        reward_info["reward/joint_deviation"] = weighted

    if "hip_deviation" in reward_weights:
        if joint_pos is not None and joint_pos_default is not None:
            hip_indices = jp.array([0, 1, 2, 8, 9, 10])
            weighted = reward_weights["hip_deviation"] * compute_joint_deviation_penalty(
                joint_pos, joint_pos_default, indices=hip_indices
            )
        else:
            weighted = jp.array(0.0)
        reward += weighted
        reward_info["reward/hip_deviation"] = weighted

    return reward, reward_info


# =============================================================================================
# ===================================== END: 完整奖励函数 =======================================
# =============================================================================================
