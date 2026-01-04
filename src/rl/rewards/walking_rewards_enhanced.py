"""
增强行走奖励函数

包含高级行走特性奖励：
1. 步态周期性 (Gait Periodicity)
2. 摆动轨迹 (Swing Trajectory)
3. 着地冲击控制 (Landing Impact)
4. 能量效率优化 (Energy Efficiency)
5. 综合稳定性 (Stability)
6. 增强速度跟踪 (Enhanced Velocity Tracking)
"""

import jax
import jax.numpy as jp
from typing import Tuple


def compute_gait_periodicity_reward(
    contacts: jax.Array,
    phase: jax.Array,
    stance_duration: float = 0.6,
    swing_duration: float = 0.4,
    tolerance: float = 0.1
) -> jax.Array:
    """计算步态周期性奖励

    奖励脚部接触状态与预定义的步态相位匹配。

    Args:
        contacts: 接触状态 [..., num_feet] (bool 或 float)
        phase: 步态相位 [..., num_feet] (0~1)
        stance_duration: 支撑相持续比例 (0~1)
        swing_duration: 摆动相持续比例 (0~1)
        tolerance: 容差

    Returns:
        周期性奖励 [0, 1]
    """
    # 期望的接触状态：相位 < 支撑时长 = 接触(1)，否则 = 摆动(0)
    # 注意：这取决于相位的定义，这里假设 0~stance_duration 为支撑相
    expected_contact = (phase < stance_duration).astype(jp.float32)

    # 实际接触状态
    actual_contact = contacts.astype(jp.float32)

    # 计算误差
    error = jp.abs(expected_contact - actual_contact)

    # 奖励匹配程度
    return jp.mean(jp.exp(-error / tolerance), axis=-1)


def compute_swing_trajectory_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    phase: jax.Array,
    target_height: float = 0.08,
    swing_start_phase: float = 0.6
) -> jax.Array:
    """计算摆动轨迹奖励

    鼓励在摆动相中期达到目标高度，并在着地前降低。

    Args:
        feet_positions: 脚部位置 [..., num_feet, 3]
        contacts: 接触状态
        phase: 步态相位
        target_height: 目标最大抬高高度
        swing_start_phase: 摆动相开始的相位值

    Returns:
        轨迹奖励
    """
    feet_z = feet_positions[..., 2]

    # 计算摆动进程 (0~1)
    # 假设 phase > swing_start_phase 为摆动相
    swing_progress = (phase - swing_start_phase) / (1.0 - swing_start_phase)
    swing_progress = jp.clip(swing_progress, 0.0, 1.0)

    # 期望高度曲线：正弦波形
    # 在摆动相中间(0.5)达到最高，两端为0
    expected_height = target_height * jp.sin(swing_progress * jp.pi)

    # 只在摆动相计算奖励
    is_swing = (phase > swing_start_phase).astype(jp.float32)

    error = jp.abs(feet_z - expected_height)
    reward = jp.exp(-error / 0.02)

    return jp.sum(reward * is_swing, axis=-1)


def compute_landing_impact_reward(
    contact_forces: jax.Array,
    landing_events: jax.Array,
    max_impact_force: float = 500.0,
    tolerance: float = 100.0
) -> jax.Array:
    """计算着地冲击奖励（惩罚大冲击）

    Args:
        contact_forces: 接触力 [..., num_feet]
        landing_events: 着地事件掩码 [..., num_feet] (刚刚着地为1)
        max_impact_force: 最大允许冲击力
        tolerance: 容差

    Returns:
        冲击奖励 [0, 1] (冲击越小越接近1)
    """
    # 只在着地瞬间计算
    impact = contact_forces * landing_events

    # 惩罚超过阈值的冲击
    excess_force = jp.clip(impact - max_impact_force, min=0.0)

    return jp.mean(jp.exp(-excess_force / tolerance), axis=-1)


def compute_energy_efficiency_reward(
    torques: jax.Array,
    joint_velocities: jax.Array,
    target_efficiency: float = 0.8,
    penalty_weight: float = 0.001
) -> jax.Array:
    """计算能量效率奖励

    最小化机械功 (Power = Torque * Velocity) 和热损耗 (Torque^2)。

    Args:
        torques: 关节力矩
        joint_velocities: 关节速度
        target_efficiency: 目标效率指标 (未使用，保留接口)
        penalty_weight: 惩罚权重

    Returns:
        效率奖励 (负值，越接近0越好)
    """
    # 机械功 (绝对值)
    mechanical_power = jp.sum(jp.abs(torques * joint_velocities), axis=-1)

    # 热损耗 (焦耳热类似项)
    thermal_loss = jp.sum(jp.square(torques), axis=-1)

    # 总能量消耗
    total_energy = mechanical_power + thermal_loss

    return -penalty_weight * total_energy


def compute_stability_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    target_height: float = 0.35,
    height_tolerance: float = 0.05,
    angular_velocity_penalty: float = 1.0
) -> jax.Array:
    """计算综合稳定性奖励

    结合高度稳定、姿态稳定和角速度稳定。

    Args:
        torso_z: 躯干高度
        base_quat: 基座四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        target_height: 目标高度
        height_tolerance: 高度容差
        angular_velocity_penalty: 角速度惩罚系数

    Returns:
        稳定性奖励 [0, 1]
    """
    # 1. 高度稳定
    height_error = jp.abs(torso_z - target_height)
    height_reward = jp.exp(-height_error / height_tolerance)

    # 2. 角速度稳定 (希望角速度小，除了指令要求的偏航角速度)
    # 这里简单惩罚所有角速度，更精细的应该减去指令角速度
    ang_vel_mag = jp.linalg.norm(base_angvel, axis=-1)
    ang_vel_reward = jp.exp(-ang_vel_mag * angular_velocity_penalty)

    # 3. 姿态稳定 (Z轴朝上)
    # 假设 quaternion [w, x, y, z]
    # Z轴向量变换后应接近 [0, 0, 1]
    # 简化：惩罚 x, y 分量
    w, x, y, z = base_quat[..., 0], base_quat[..., 1], base_quat[..., 2], base_quat[..., 3]
    # 重力向量在机体坐标系下的投影 (projected gravity)
    projected_gravity_x = 2 * (x * z - w * y)
    projected_gravity_y = 2 * (y * z + w * x)
    orientation_error = jp.square(projected_gravity_x) + jp.square(projected_gravity_y)
    orientation_reward = jp.exp(-orientation_error * 5.0)

    return (height_reward + ang_vel_reward + orientation_reward) / 3.0


def compute_velocity_tracking_reward(
    actual_velocity: jax.Array,
    command: jax.Array,
    tracking_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    tolerance: float = 0.1
) -> jax.Array:
    """计算增强速度跟踪奖励

    分别计算 X, Y, Yaw 的跟踪误差。

    Args:
        actual_velocity: 实际速度 [vx, vy, vyaw]
        command: 指令速度 [vx, vy, vyaw]
        tracking_weights: 各轴权重 (x, y, yaw)
        tolerance: 容差

    Returns:
        跟踪奖励
    """
    # 确保维度匹配
    if actual_velocity.shape[-1] != 3:
        # 假设输入是完整状态，提取前3维
        # 注意：需要调用者确保输入正确
        pass

    error = actual_velocity - command
    error_x = jp.abs(error[..., 0])
    error_y = jp.abs(error[..., 1])
    error_yaw = jp.abs(error[..., 2])

    reward_x = jp.exp(-error_x / tolerance)
    reward_y = jp.exp(-error_y / tolerance)
    reward_yaw = jp.exp(-error_yaw / tolerance)

    wx, wy, wyaw = tracking_weights
    total_weight = wx + wy + wyaw

    return (wx * reward_x + wy * reward_y + wyaw * reward_yaw) / total_weight
