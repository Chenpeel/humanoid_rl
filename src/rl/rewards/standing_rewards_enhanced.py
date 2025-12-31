"""
增强版站立任务奖励函数

基于Unitree等双足机器人的奖励机制研究成果：
1. 添加脚离地间隙奖励（foot clearance）
2. 添加脚悬空时间奖励（swing time）
3. 优化姿态稳定性奖励
4. 参考Unitree G1/H1的奖励设计
"""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jp


def compute_foot_clearance_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    target_clearance: float = 0.08,
    tolerance: float = 0.02,
    min_swing_height: float = 0.02
) -> jax.Array:
    """计算脚部抬高奖励（基于Unitree实现）

    鼓励脚在摆动相抬高到适当高度，参考Unitree G1设计。

    Args:
        feet_positions: 脚部位置，形状 [..., 2, 3] 或 [2, 3]
        contacts: 接触标志，形状 [..., 2] 或 [2]，1=接触，0=离地
        target_clearance: 目标抬高高度 (m)，Unitree G1使用0.08m
        tolerance: 容差参数，控制奖励衰减速度
        min_swing_height: 最小摆动高度 (m)，低于此值不给奖励

    Returns:
        脚部抬高奖励值（0-1范围）
    """
    # 获取脚部高度（Z坐标）
    feet_heights = feet_positions[..., 2]  # shape: [..., 2]

    # 摆动相掩码（脚离地）
    swing_phase = (1.0 - contacts.astype(jp.float32))  # 1=摆动，0=支撑

    # 计算高度误差（只考虑摆动相）
    height_error = jp.abs(jp.maximum(feet_heights - min_swing_height, 0.0) - target_clearance)

    # 指数衰减奖励（在目标高度附近给高奖励）
    clearance_reward = jp.exp(-height_error / tolerance)

    # 只在摆动相激活奖励
    clearance_reward = clearance_reward * swing_phase

    # 对所有脚求平均（避免一只脚异常高获得过高奖励）
    return jp.mean(clearance_reward, axis=-1)


def compute_swing_time_reward(
    contacts_history: jax.Array,
    min_swing_time: float = 0.3,
    target_swing_time: float = 0.4,
    tolerance: float = 0.1
) -> jax.Array:
    """计算脚悬空时间奖励（基于ETH Zurich研究）

    鼓励适当的步态周期，确保有足够的摆动时间。

    Args:
        contacts_history: 历史接触状态，形状 [..., history_length, 2]
        min_swing_time: 最小摆动时间 (s)
        target_swing_time: 目标摆动时间 (s)，Unitree使用0.4s
        tolerance: 容差参数

    Returns:
        摆动时间奖励值（0-1范围）
    """
    # 计算历史长度对应的实际时间（假设50Hz控制频率）
    dt = 0.02  # 50Hz
    history_time = contacts_history.shape[-2] * dt

    if history_time < target_swing_time + tolerance:
        # 历史不够长，返回中性奖励
        return jp.array(0.5)

    # 计算每只脚的摆动时间比例
    swing_ratio = 1.0 - jp.mean(contacts_history.astype(jp.float32), axis=-2)  # [..., 2]

    # 摆动时间奖励（在目标时间附近给高奖励）
    swing_time_error = jp.abs(swing_ratio * history_time - target_swing_time)
    swing_reward = jp.exp(-swing_time_error / tolerance)

    # 惩罚摆动时间过短（防止小碎步）
    too_short_penalty = jp.where(
        swing_ratio * history_time < min_swing_time,
        -0.5 * (min_swing_time - swing_ratio * history_time) / min_swing_time,
        0.0
    )

    # 综合奖励
    total_reward = swing_reward + too_short_penalty

    # 对所有脚求平均
    return jp.mean(jp.clip(total_reward, 0.0, 1.0), axis=-1)


def compute_contact_force_reward(
    contact_forces: jax.Array,
    target_force: float = 150.0,  # N，参考Unitree设置
    tolerance: float = 50.0
) -> jax.Array:
    """计算接触力奖励（减少冲击）

    基于Unitree G1的接触力控制，减少地面反作用力峰值。

    Args:
        contact_forces: 接触力，形状 [..., num_contacts, 3]
        target_force: 目标接触力 (N)
        tolerance: 容差参数

    Returns:
        接触力奖励值
    """
    # 计算总接触力大小
    total_force = jp.linalg.norm(contact_forces, axis=-1)  # [..., num_contacts]

    # 计算力误差
    force_error = jp.abs(total_force - target_force)

    # 指数衰减奖励
    force_reward = jp.exp(-force_error / tolerance)

    # 对所有接触点求平均
    return jp.mean(force_reward, axis=-1)


def compute_stability_reward(
    base_angvel: jax.Array,
    base_linacc: jax.Array,
    angvel_tolerance: float = 0.5,  # rad/s
    acc_tolerance: float = 2.0      # m/s^2
) -> jax.Array:
    """计算稳定性奖励（基于角速度和加速度）

    综合评估机器人稳定性，参考Cassie机器人设计。

    Args:
        base_angvel: 基座角速度 [wx, wy, wz]
        base_linacc: 基座线加速度 [ax, ay, az]
        angvel_tolerance: 角速度容差
        acc_tolerance: 加速度容差

    Returns:
        稳定性奖励值
    """
    # 角速度稳定性
    angvel_magnitude = jp.linalg.norm(base_angvel)
    angvel_stability = jp.exp(-angvel_magnitude / angvel_tolerance)

    # 加速度稳定性
    acc_magnitude = jp.linalg.norm(base_linacc)
    acc_stability = jp.exp(-acc_magnitude / acc_tolerance)

    # 综合稳定性（加权平均）
    return 0.6 * angvel_stability + 0.4 * acc_stability


def compute_enhanced_standing_reward(
    # 基础状态
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    base_linacc: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    torques: jax.Array,
    target_height: float,
    # 增强状态
    feet_positions: Optional[jax.Array] = None,
    contacts: Optional[jax.Array] = None,
    contact_forces: Optional[jax.Array] = None,
    contacts_history: Optional[jax.Array] = None,
    # 奖励权重
    reward_weights: Dict[str, float] = None,
) -> jax.Array:
    """计算增强版站立任务奖励

    基于Unitree研究成果的站立奖励函数，包含：
    1. 基础站立奖励（高度、姿态、速度）
    2. 脚部运动奖励（离地间隙、摆动时间）
    3. 接触力控制奖励
    4. 综合稳定性奖励

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        base_linacc: 基座线加速度
        action: 当前动作
        last_action: 上一步动作
        torques: 执行器扭矩
        target_height: 目标高度
        feet_positions: 脚部位置 [..., 2, 3]
        contacts: 接触状态 [..., 2]
        contact_forces: 接触力 [..., num_contacts, 3]
        contacts_history: 历史接触状态 [..., history_length, 2]
        reward_weights: 奖励权重字典

    Returns:
        总奖励值
    """
    # 默认奖励权重（基于Unitree研究成果）
    if reward_weights is None:
        reward_weights = {
            # 基础站立（来自原有实现）
            "height": 1.0,
            "orientation": 1.0,
            "lin_vel": -0.5,
            "ang_vel": -0.3,
            "alive": 0.2,
            "action_rate": -0.01,
            "torques": -0.0001,
            # 新增奖励项（基于Unitree）
            "foot_clearance": 0.5,      # 脚部抬高
            "swing_time": 0.3,          # 摆动时间
            "contact_force": 0.2,       # 接触力控制
            "stability": 0.8,           # 综合稳定性
        }

    # 基础站立奖励
    base_quat = base_quat / jp.linalg.norm(base_quat)

    reward_height = jp.exp(-jp.square(torso_z - target_height) / 0.05)

    # 姿态奖励（使用四元数直接计算，更稳定）
    gravity_vec = jp.array([0.0, 0.0, -1.0])
    base_rot = jp.quat_rotate(base_quat, gravity_vec)
    orientation_error = jp.square(base_rot[0]) + jp.square(base_rot[1])  # XY平面偏离
    reward_orientation = jp.exp(-orientation_error / 0.1)

    # 速度惩罚
    lin_vel_penalty = jp.sum(jp.square(base_linvel))
    ang_vel_penalty = jp.sum(jp.square(base_angvel))

    # 动作平滑和能量效率
    action_rate_penalty = jp.sum(jp.square(action - last_action))
    torque_penalty = jp.sum(jp.square(torques))

    # 组合基础奖励
    base_reward = (
        reward_weights["height"] * reward_height
        + reward_weights["orientation"] * reward_orientation
        + reward_weights["lin_vel"] * lin_vel_penalty
        + reward_weights["ang_vel"] * ang_vel_penalty
        + reward_weights["alive"] * 1.0
        + reward_weights["action_rate"] * action_rate_penalty
        + reward_weights["torques"] * torque_penalty
    )

    # 增强奖励项（如果提供了必要数据）
    enhanced_reward = base_reward

    # 脚部抬高奖励（站立时的小幅抬脚）
    if feet_positions is not None and contacts is not None:
        clearance_reward = compute_foot_clearance_reward(
            feet_positions, contacts,
            target_clearance=0.02,  # 站立时小幅抬脚2cm
            tolerance=0.01
        )
        enhanced_reward += reward_weights["foot_clearance"] * clearance_reward

    # 摆动时间奖励（站立时的微小摆动）
    if contacts_history is not None:
        swing_reward = compute_swing_time_reward(
            contacts_history,
            min_swing_time=0.1,      # 最小摆动0.1s
            target_swing_time=0.2,   # 目标摆动0.2s
            tolerance=0.05
        )
        enhanced_reward += reward_weights["swing_time"] * swing_reward

    # 接触力奖励
    if contact_forces is not None:
        force_reward = compute_contact_force_reward(
            contact_forces,
            target_force=100.0,  # 站立时目标力100N
            tolerance=30.0
        )
        enhanced_reward += reward_weights["contact_force"] * force_reward

    # 综合稳定性奖励
    stability_reward = compute_stability_reward(
        base_angvel, base_linacc,
        angvel_tolerance=0.3,
        acc_tolerance=1.5
    )
    enhanced_reward += reward_weights["stability"] * stability_reward

    return enhanced_reward


# 增强版默认奖励权重（基于Unitree研究成果优化）
DEFAULT_ENHANCED_STANDING_WEIGHTS = {
    # 基础站立（保持原有权重）
    "height": 1.0,
    "orientation": 1.0,
    "lin_vel": -0.5,
    "ang_vel": -0.3,
    "alive": 0.2,
    "action_rate": -0.01,
    "torques": -0.0001,
    # 新增奖励项（基于Unitree和ETH研究）
    "foot_clearance": 0.3,      # 小幅抬脚奖励
    "swing_time": 0.2,          # 摆动时间奖励
    "contact_force": 0.1,       # 接触力控制
    "stability": 0.5,           # 综合稳定性
}