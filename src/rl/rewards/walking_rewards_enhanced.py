"""
增强版行走任务奖励函数

基于Unitree等双足机器人的行走奖励机制研究：
1. 步态周期性奖励（gait periodicity）
2. 脚摆动轨迹奖励（swing trajectory）
3. 着地冲击控制（landing impact）
4. 能量效率优化（energy efficiency）
"""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jp


def compute_gait_periodicity_reward(
    contacts: jax.Array,
    phase: jax.Array,
    stance_duration: float = 0.6,
    swing_duration: float = 0.4,
    tolerance: float = 0.1
) -> jax.Array:
    """计算步态周期性奖励（基于ETH Zurich研究）

    奖励与期望步态相位匹配的接触状态，参考ANYmal机器人设计。

    Args:
        contacts: 接触状态 [..., 2]，1=接触，0=离地
        phase: 步态相位 [0, 1]，0=开始，1=结束
        stance_duration: 支撑相持续时间比例（0.6）
        swing_duration: 摆动相持续时间比例（0.4）
        tolerance: 容差参数

    Returns:
        步态周期性奖励值
    """
    # 期望接触模式（基于相位）
    # 相位0.0-0.6: 右脚支撑，左脚摆动
    # 相位0.6-1.0: 左脚支撑，右脚摆动
    desired_right_contact = jp.where(phase < stance_duration, 1.0, 0.0)
    desired_left_contact = jp.where(phase >= stance_duration, 1.0, 0.0)

    # 计算接触误差
    right_error = jp.abs(contacts[..., 0] - desired_right_contact)
    left_error = jp.abs(contacts[..., 1] - desired_left_contact)

    # 指数衰减奖励
    contact_reward = jp.exp(-(right_error + left_error) / tolerance)

    return contact_reward


def compute_swing_trajectory_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    phase: jax.Array,
    target_height: float = 0.08,
    swing_start_phase: float = 0.6
) -> jax.Array:
    """计算摆动轨迹奖励（基于Unitree G1实现）

    奖励平滑的摆动轨迹，包括抬升和降落阶段。

    Args:
        feet_positions: 脚部位置 [..., 2, 3]
        contacts: 接触状态 [..., 2]
        phase: 步态相位 [0, 1]
        target_height: 目标抬升高度 (m)
        swing_start_phase: 摆动开始相位

    Returns:
        摆动轨迹奖励值
    """
    # 计算摆动相进度（0=开始摆动，1=结束摆动）
    swing_progress = jp.clip((phase - swing_start_phase) / (1.0 - swing_start_phase), 0.0, 1.0)

    # 期望高度轨迹（抛物线）
    desired_height = 4.0 * target_height * swing_progress * (1.0 - swing_progress)

    # 获取摆动脚的高度
    swing_mask = (1.0 - contacts).astype(jp.float32)  # 1=摆动，0=支撑
    feet_heights = feet_positions[..., 2]  # Z坐标

    # 计算高度误差（仅对摆动脚）
    height_errors = jp.abs(feet_heights - desired_height) * swing_mask

    # 指数衰减奖励
    trajectory_reward = jp.exp(-height_errors / 0.02)  # 2cm容差

    # 对所有脚求平均
    return jp.mean(trajectory_reward, axis=-1)


def compute_landing_impact_reward(
    contact_forces: jax.Array,
    landing_events: jax.Array,
    max_impact_force: float = 500.0,
    tolerance: float = 100.0
) -> jax.Array:
    """计算着地冲击奖励（减少冲击）

    基于伯克利人形机器人研究，控制着地时的冲击力。

    Args:
        contact_forces: 接触力 [..., num_contacts, 3]
        landing_events: 着地事件掩码 [..., num_contacts]
        max_impact_force: 最大允许冲击力 (N)
        tolerance: 容差参数

    Returns:
        着地冲击奖励值
    """
    # 计算冲击力大小
    impact_forces = jp.linalg.norm(contact_forces, axis=-1)  # [..., num_contacts]

    # 只在着地时计算奖励
    impact_penalty = jp.where(
        landing_events > 0.5,
        jp.maximum(0.0, impact_forces - max_impact_force) / max_impact_force,
        0.0
    )

    # 指数衰减惩罚
    impact_reward = jp.exp(-impact_penalty * tolerance / max_impact_force)

    # 对所有接触点求平均
    return jp.mean(impact_reward, axis=-1)


def compute_energy_efficiency_reward(
    torques: jax.Array,
    joint_velocities: jax.Array,
    target_efficiency: float = 0.8,
    penalty_weight: float = 0.001
) -> jax.Array:
    """计算能量效率奖励（基于功率计算）

    更精确的能量消耗评估，考虑扭矩和角速度的乘积。

    Args:
        torques: 关节扭矩 [..., nu]
        joint_velocities: 关节角速度 [..., nu]
        target_efficiency: 目标效率（0-1）
        penalty_weight: 惩罚权重

    Returns:
        能量效率奖励值
    """
    # 计算瞬时功率
    power = jp.abs(torques * joint_velocities)

    # 归一化功率（相对于最大扭矩和最大速度）
    # 假设最大扭矩为100Nm，最大速度为10rad/s
    max_power = 100.0 * 10.0
    normalized_power = power / max_power

    # 计算总功率
    total_power = jp.sum(normalized_power, axis=-1)

    # 效率奖励（功率越低奖励越高）
    efficiency_reward = jp.exp(-total_power * penalty_weight)

    return efficiency_reward


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

    结合高度稳定性、姿态稳定性和角速度稳定性的综合评估。

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        target_height: 目标高度
        height_tolerance: 高度容差
        angular_velocity_penalty: 角速度惩罚权重

    Returns:
        综合稳定性奖励值
    """
    from .walking_rewards import normalize_quaternion, quat_to_euler

    # 1. 高度稳定性
    height_error = jp.abs(torso_z - target_height)
    height_stability = jp.exp(-height_error / height_tolerance)

    # 2. 姿态稳定性（roll和pitch应接近0）
    quat = normalize_quaternion(base_quat)
    euler = quat_to_euler(quat)
    roll, pitch = euler[..., 0], euler[..., 1]
    orientation_error = jp.square(roll) + jp.square(pitch)
    orientation_stability = jp.exp(-orientation_error / 0.1)

    # 3. 角速度稳定性（避免剧烈旋转）
    angular_velocity_magnitude = jp.linalg.norm(base_angvel, axis=-1)
    angular_stability = jp.exp(-angular_velocity_magnitude * angular_velocity_penalty)

    # 综合稳定性（加权平均）
    stability = 0.4 * height_stability + 0.3 * orientation_stability + 0.3 * angular_stability

    return stability


def compute_velocity_tracking_reward(
    actual_velocity: jax.Array,
    command: jax.Array,
    tracking_weights: Tuple[float, float, float] = (2.0, 0.5, 0.3)
) -> jax.Array:
    """计算速度跟踪奖励（增强版）

    更精细的速度跟踪控制，支持不同轴的权重配置。

    Args:
        actual_velocity: 实际速度 [vx, vy, vyaw]
        command: 速度命令 [vx, vy, vyaw]
        tracking_weights: 各轴跟踪权重 (x, y, yaw)

    Returns:
        速度跟踪奖励值
    """
    # 计算跟踪误差
    tracking_error = jp.abs(actual_velocity - command)

    # 应用权重
    weighted_error = (
        tracking_weights[0] * tracking_error[0] +
        tracking_weights[1] * tracking_error[1] +
        tracking_weights[2] * tracking_error[2]
    )

    # 指数衰减奖励
    tracking_reward = jp.exp(-weighted_error / 0.1)

    return tracking_reward


def compute_enhanced_walking_reward(
    # 基础参数
    command: jax.Array,
    actual_velocity: jax.Array,
    feet_positions: jax.Array,
    contacts: jax.Array,
    contact_forces: jax.Array,
    joint_positions: jax.Array,
    joint_velocities: jax.Array,
    torques: jax.Array,
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    # 可选增强参数
    phase: Optional[jax.Array] = None,
    contacts_history: Optional[jax.Array] = None,
    landing_events: Optional[jax.Array] = None,
    # 奖励权重
    reward_weights: Dict[str, float] = None,
    # 目标参数
    target_height: float = 0.35,
    tracking_weights: Tuple[float, float, float] = (2.0, 0.5, 0.3),
) -> jax.Array:
    """计算增强版行走任务奖励

    基于Unitree研究成果的综合行走奖励函数，包含：
    1. 速度跟踪奖励（多轴权重）
    2. 步态周期性奖励（相位匹配）
    3. 摆动轨迹奖励（平滑轨迹）
    4. 着地冲击控制（减少冲击）
    5. 能量效率优化（功率计算）
    6. 基础稳定性奖励（高度、姿态）

    Args:
        command: 速度命令 [vx, vy, vyaw]
        actual_velocity: 实际速度 [vx, vy, vyaw]
        feet_positions: 脚部位置 [..., 2, 3]
        contacts: 接触状态 [..., 2]
        contact_forces: 接触力 [..., num_contacts, 3]
        joint_positions: 关节位置 [..., nu]
        joint_velocities: 关节速度 [..., nu]
        torques: 关节扭矩 [..., nu]
        torso_z: 躯干高度
        base_quat: 躯干四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        action: 当前动作
        last_action: 上一步动作
        phase: 步态相位 [0, 1]
        contacts_history: 历史接触状态
        landing_events: 着地事件掩码
        reward_weights: 奖励权重字典
        target_height: 目标躯干高度
        tracking_weights: 速度跟踪权重

    Returns:
        总奖励值
    """
    # 默认奖励权重（基于Unitree和ETH研究）
    if reward_weights is None:
        reward_weights = {
            # 基础行走（来自原有实现）
            "forward_velocity": 1.5,
            "trunk_height": 0.5,
            "orientation": -0.3,
            "trunk_lin_vel_z": -0.5,
            "drag": -1.0,
            "gait_symmetry": 0.3,
            "foot_clearance": 0.2,
            "action_rate": -0.01,
            "torques": -0.0001,
            "alive": 0.3,
            # 新增增强奖励项
            "gait_periodicity": 0.4,     # 步态周期性
            "swing_trajectory": 0.3,     # 摆动轨迹
            "landing_impact": 0.2,       # 着地冲击
            "energy_efficiency": 0.1,    # 能量效率
            "velocity_tracking": 1.0,    # 增强速度跟踪
        }

    # 基础奖励计算（使用原有函数）
    from .walking_rewards import compute_walking_reward

    # 提取接触状态（如果是传感器数据，需要转换）
    from .walking_rewards import get_feet_contacts
    contact_state = get_feet_contacts(contacts) if contacts.shape[-1] > 2 else contacts

    base_reward = compute_walking_reward(
        torso_z=torso_z,
        base_quat=base_quat,
        base_linvel=base_linvel,
        base_angvel=base_angvel,
        contact_sensors=contacts,
        feet_positions=feet_positions,
        action=action,
        last_action=last_action,
        torques=torques,
        target_velocity=command[0],  # 前向速度命令
        target_height=target_height,
        reward_weights={k: v for k, v in reward_weights.items()
                       if k not in ["gait_periodicity", "swing_trajectory",
                                    "landing_impact", "energy_efficiency",
                                    "velocity_tracking", "stability"]},
    )

    # 增强奖励项
    enhanced_reward = base_reward

    # 1. 步态周期性奖励（需要相位信息）
    if phase is not None:
        periodicity_reward = compute_gait_periodicity_reward(
            contacts, phase,
            stance_duration=0.6,
            swing_duration=0.4,
            tolerance=0.1
        )
        enhanced_reward += reward_weights["gait_periodicity"] * periodicity_reward

    # 2. 摆动轨迹奖励
    if phase is not None:
        trajectory_reward = compute_swing_trajectory_reward(
            feet_positions, contacts, phase,
            target_height=0.08,
            swing_start_phase=0.6
        )
        enhanced_reward += reward_weights["swing_trajectory"] * trajectory_reward

    # 3. 着地冲击控制
    if landing_events is not None:
        impact_reward = compute_landing_impact_reward(
            contact_forces, landing_events,
            max_impact_force=500.0,
            tolerance=100.0
        )
        enhanced_reward += reward_weights["landing_impact"] * impact_reward

    # 4. 能量效率优化
    efficiency_reward = compute_energy_efficiency_reward(
        torques, joint_velocities,
        target_efficiency=0.8,
        penalty_weight=0.001
    )
    enhanced_reward += reward_weights.get("energy_efficiency", 0.0) * efficiency_reward

    # 5. 综合稳定性
    stability_reward = compute_stability_reward(
        torso_z, base_quat, base_linvel, base_angvel,
        target_height=target_height,
        height_tolerance=0.05,
        angular_velocity_penalty=1.0
    )
    enhanced_reward += reward_weights.get("stability", 0.0) * stability_reward

    # 6. 增强速度跟踪（多轴权重）
    if "velocity_tracking" in reward_weights:
        enhanced_tracking = compute_velocity_tracking_reward(
            actual_velocity, command,
            tracking_weights=tracking_weights
        )
        # 替换原有的速度跟踪部分
        base_tracking_weight = reward_weights.get("forward_velocity", 1.5)
        enhanced_reward += (reward_weights["velocity_tracking"] - base_tracking_weight) * enhanced_tracking

    return enhanced_reward


# 增强版默认奖励权重
DEFAULT_ENHANCED_WALKING_WEIGHTS = {
    # 基础行走（保持原有权重）
    "forward_velocity": 1.5,
    "trunk_height": 0.5,
    "orientation": -0.3,
    "trunk_lin_vel_z": -0.5,
    "drag": -1.0,
    "gait_symmetry": 0.3,
    "foot_clearance": 0.2,
    "action_rate": -0.01,
    "torques": -0.0001,
    "alive": 0.3,
    # 新增增强奖励项
    "gait_periodicity": 0.4,     # 步态周期性
    "swing_trajectory": 0.3,     # 摆动轨迹
    "landing_impact": 0.2,       # 着地冲击
    "energy_efficiency": 0.1,    # 能量效率
    "velocity_tracking": 1.0,    # 增强速度跟踪
}