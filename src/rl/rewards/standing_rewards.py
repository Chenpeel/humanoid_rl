"""
站立平衡奖励函数

设计用于 Jiyuan 机器人的基础平衡训练阶段。
侧重于高度保持、垂直姿态和最小化关节动作。
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jp

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
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的站立平衡奖励"""

    # 计算分量
    reward_height = compute_height_reward(torso_z, target_height)
    reward_orientation = compute_orientation_reward(base_quat)
    vel_penalties = compute_velocity_penalties(base_linvel, base_angvel)

    # 惩罚项
    action_rate_penalty = jp.sum(jp.square(action - last_action), axis=-1)
    torque_penalty = jp.sum(jp.square(torques), axis=-1)

    # 初始化
    reward = jp.array(0.0)
    reward_info = {}

    # 1. 高度奖励
    weighted = reward_weights["height"] * reward_height
    reward += weighted
    reward_info["reward/height"] = weighted

    # 2. 姿态奖励
    weighted = reward_weights["orientation"] * reward_orientation
    reward += weighted
    reward_info["reward/orientation"] = weighted

    # 3. 线速度惩罚
    weighted = reward_weights["lin_vel"] * vel_penalties["lin_vel_penalty"]
    reward += weighted
    reward_info["reward/lin_vel"] = weighted

    # 4. 角速度惩罚
    weighted = reward_weights["ang_vel"] * vel_penalties["ang_vel_penalty"]
    reward += weighted
    reward_info["reward/ang_vel"] = weighted

    # 5. 存活奖励
    weighted = reward_weights["alive"] * 1.0
    reward += weighted
    reward_info["reward/alive"] = weighted

    # 6. 动作平滑
    weighted = reward_weights["action_rate"] * action_rate_penalty
    reward += weighted
    reward_info["reward/action_rate"] = weighted

    # 7. 扭矩惩罚
    weighted = reward_weights["torques"] * torque_penalty
    reward += weighted
    reward_info["reward/torques"] = weighted

    return reward, reward_info


# =============================================================================================
# ===================================== END: 完整奖励函数 =======================================
# =============================================================================================
