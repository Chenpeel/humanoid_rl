"""
站立任务奖励函数

参考 configs/standing_config.yaml 中的奖励配置
"""

from typing import Dict

import jax
import jax.numpy as jp


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """四元数转欧拉角

    Args:
        quat: 四元数 [w, x, y, z]

    Returns:
        欧拉角 [roll, pitch, yaw]
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(
        jp.abs(sinp) >= 1,
        jp.sign(sinp) * jp.pi / 2,
        jp.arcsin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    return jp.array([roll, pitch, yaw])


def compute_height_reward(
    torso_z: jax.Array,
    target_height: float,
    tolerance: float = 0.08
) -> jax.Array:
    """计算高度保持奖励

    使用指数形式的奖励，高度越接近目标值奖励越高。

    Args:
        torso_z: 躯干高度 (m)
        target_height: 目标高度 (m)
        tolerance: 容差参数，控制奖励衰减速度

    Returns:
        高度奖励值 [0, 1]
    """
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)


def compute_orientation_reward(
    quat: jax.Array,
    tolerance: float = 0.1
) -> jax.Array:
    """计算姿态稳定奖励

    pitch和roll角度应该接近0（保持直立）。

    Args:
        quat: 躯干四元数 [w, x, y, z]
        tolerance: 容差参数

    Returns:
        姿态奖励值 [0, 1]
    """
    euler = quat_to_euler(quat)
    roll, pitch = euler[0], euler[1]
    orientation_error = jp.square(roll) + jp.square(pitch)
    return jp.exp(-orientation_error / tolerance)


def compute_velocity_penalty(
    base_linvel: jax.Array,
    base_angvel: jax.Array
) -> Dict[str, jax.Array]:
    """计算速度惩罚

    站立任务中，机器人不应该移动或旋转。

    Args:
        base_linvel: 基座线速度 [vx, vy, vz]
        base_angvel: 基座角速度 [wx, wy, wz]

    Returns:
        包含线速度和角速度惩罚的字典
    """
    lin_vel_penalty = jp.sum(jp.square(base_linvel))
    ang_vel_penalty = jp.sum(jp.square(base_angvel))
    return {
        "lin_vel_penalty": lin_vel_penalty,
        "ang_vel_penalty": ang_vel_penalty,
    }


def compute_action_rate_penalty(
    action: jax.Array,
    last_action: jax.Array
) -> jax.Array:
    """计算动作变化率惩罚

    鼓励平滑的动作变化。

    Args:
        action: 当前动作
        last_action: 上一步动作

    Returns:
        动作变化率惩罚值
    """
    return jp.sum(jp.square(action - last_action))


def compute_torque_penalty(torques: jax.Array) -> jax.Array:
    """计算扭矩惩罚

    鼓励能量效率。

    Args:
        torques: 执行器扭矩

    Returns:
        扭矩惩罚值
    """
    return jp.sum(jp.square(torques))


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
) -> jax.Array:
    """计算完整的站立任务奖励

    奖励分量：
    1. height: 高度保持奖励
    2. orientation: 姿态稳定奖励
    3. lin_vel: 线速度惩罚
    4. ang_vel: 角速度惩罚
    5. alive: 存活奖励
    6. action_rate: 动作平滑惩罚
    7. torques: 能量效率惩罚

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        action: 当前动作
        last_action: 上一步动作
        torques: 执行器扭矩
        target_height: 目标高度
        reward_weights: 奖励权重字典

    Returns:
        总奖励值
    """
    # 归一化四元数
    base_quat = base_quat / jp.linalg.norm(base_quat)

    # 计算各项奖励
    reward_height = compute_height_reward(torso_z, target_height)
    reward_orientation = compute_orientation_reward(base_quat)
    velocity_penalties = compute_velocity_penalty(base_linvel, base_angvel)
    action_rate_penalty = compute_action_rate_penalty(action, last_action)
    torque_penalty = compute_torque_penalty(torques)

    # 组合奖励
    reward = (
        reward_weights["height"] * reward_height
        + reward_weights["orientation"] * reward_orientation
        + reward_weights["lin_vel"] * velocity_penalties["lin_vel_penalty"]
        + reward_weights["ang_vel"] * velocity_penalties["ang_vel_penalty"]
        + reward_weights["alive"] * 1.0
        + reward_weights["action_rate"] * action_rate_penalty
        + reward_weights["torques"] * torque_penalty
    )

    return reward


def check_standing_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.1,
    angle_threshold: float = 0.8,
) -> jax.Array:
    """检查站立任务是否终止

    终止条件：
    1. 高度太低（摔倒）
    2. 倾斜角度太大

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        height_threshold: 高度阈值 (m)，低于此值认为摔倒
        angle_threshold: 角度阈值 (rad)，超过此值认为摔倒

    Returns:
        是否终止的布尔值
    """
    # 归一化四元数
    base_quat = base_quat / jp.linalg.norm(base_quat)
    euler = quat_to_euler(base_quat)
    roll, pitch = euler[0], euler[1]

    # 检查终止条件
    height_fail = torso_z < height_threshold
    orientation_fail = (jp.abs(roll) > angle_threshold) | (jp.abs(pitch) > angle_threshold)

    return height_fail | orientation_fail


# 默认奖励权重（参考standing_config.yaml）
DEFAULT_STANDING_REWARD_WEIGHTS = {
    "height": 1.0,           # 高度保持奖励
    "orientation": 1.0,      # 姿态稳定奖励
    "lin_vel": -0.5,         # 线速度惩罚
    "ang_vel": -0.3,         # 角速度惩罚
    "alive": 0.2,            # 存活奖励
    "action_rate": -0.01,    # 动作平滑惩罚
    "torques": -0.0001,      # 能量效率惩罚
}
