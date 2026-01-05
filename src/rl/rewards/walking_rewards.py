"""
行走任务奖励函数

基于开源实现（Legged Gym, Isaac Lab）的行走奖励函数。
使用 JAX 实现纯函数式设计，支持 JIT 编译。

主要参考:
- Legged Gym (ETH Zurich): https://github.com/leggedrobotics/legged_gym
- Isaac Lab Locomotion: omni.isaac.lab_tasks/locomotion
- ANYmal/Go1 机器人实现

行走任务的核心目标:
1. 前向运动 - 跟踪线速度命令
2. 步态稳定 - 保持对称的步态模式
3. 能量效率 - 最小化能量消耗
4. 姿态稳定 - 保持直立
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jp


# ==================== 数学工具函数 ====================


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """四元数转欧拉角

    Args:
        quat: 四元数 [w, x, y, z]

    Returns:
        欧拉角 [roll, pitch, yaw]
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

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

    return jp.stack([roll, pitch, yaw], axis=-1)


def normalize_quaternion(quat: jax.Array) -> jax.Array:
    """归一化四元数

    Args:
        quat: 四元数 [w, x, y, z]

    Returns:
        归一化后的四元数
    """
    norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    return jp.where(
        norm > 1e-8,
        quat / norm,
        jp.array([1.0, 0.0, 0.0, 0.0])
    )


def wrap_to_pi(angles: jax.Array) -> jax.Array:
    """将角度包装到 [-π, π] 范围

    Args:
        angles: 角度张量（弧度）

    Returns:
        包装后的角度，范围 [-π, π]
    """
    return jp.arctan2(jp.sin(angles), jp.cos(angles))


# ==================== 接触检测辅助函数 ====================


def get_feet_contacts(
    contact_sensors: jax.Array,
    threshold: float = 1.0
) -> jax.Array:
    """从传感器数据中提取脚部接触状态

    Args:
        contact_sensors: 接触传感器数据，形状 (..., num_sensors)
        threshold: 接触力阈值

    Returns:
        接触标志，形状 (..., 2) [right_foot, left_foot]

    注意:
        假设传感器顺序为 [right_foot, right_toe, left_foot, left_toe]
        我们将其聚合为 [right_foot, left_foot]，每脚取最大值
    """
    # 确保输入是数组
    contact_sensors = jp.asarray(contact_sensors)

    # 如果有4个传感器（每脚2个），取最大值聚合
    if contact_sensors.shape[-1] == 4:
        right_contact = jp.maximum(contact_sensors[..., 0], contact_sensors[..., 1])
        left_contact = jp.maximum(contact_sensors[..., 2], contact_sensors[..., 3])
        return jp.stack([right_contact > threshold, left_contact > threshold], axis=-1)
    else:
        # 否则假设只有2个传感器
        return contact_sensors > threshold


# ==================== 前向运动奖励 ====================


def compute_forward_velocity_reward(
    base_linvel: jax.Array,
    target_velocity: float,
    tolerance: float = 0.5
) -> jax.Array:
    """计算前向速度奖励

    Args:
        base_linvel: 基座线速度 [..., 3] 或 [3]
        target_velocity: 目标前向速度 (m/s)
        tolerance: 容差参数

    Returns:
        前向速度奖励值 [0, 1]
    """
    forward_vel = base_linvel[..., 0]  # X方向
    vel_error = jp.abs(forward_vel - target_velocity)
    return jp.exp(-vel_error / tolerance)


# ==================== 步态相关奖励 ====================


def compute_gait_symmetry_reward(
    contacts: jax.Array
) -> jax.Array:
    """计算步态对称性奖励

    奖励左右脚交替接触的步态模式。

    Args:
        contacts: 接触标志，形状 [..., 2] [right, left]

    Returns:
        对称性奖励值 [0, 1]

    注意:
        这是一个简化版本，奖励 XOR 模式（一个接触，另一个离地）。
    """
    right_contact = contacts[..., 0].astype(jp.float32)
    left_contact = contacts[..., 1].astype(jp.float32)

    # XOR 逻辑：一个接触，另一个离地
    symmetry = jp.abs(right_contact - left_contact)

    return symmetry


def compute_foot_clearance_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    target_clearance: float = 0.05,
    tolerance: float = 0.02
) -> jax.Array:
    """计算脚部抬高奖励

    鼓励脚在摆动相抬高到适当高度。

    Args:
        feet_positions: 脚部位置，形状 [..., 2, 3] 或 [2, 3]
        contacts: 接触标志，形状 [..., 2] 或 [2]
        target_clearance: 目标抬高高度 (m)
        tolerance: 容差参数

    Returns:
        抬高奖励值

    注意:
        只在摆动相（脚离地）时计算奖励。
    """
    feet_heights = feet_positions[..., 2]  # Z坐标

    # 只在摆动相（脚离地）时计算
    swing_phase = (1.0 - contacts.astype(jp.float32))

    # 计算高度误差
    height_error = jp.abs(feet_heights - target_clearance)
    clearance_reward = jp.exp(-height_error / tolerance)

    # 只在摆动相有奖励
    clearance_reward = clearance_reward * swing_phase

    # 对所有脚求和
    return jp.sum(clearance_reward, axis=-1)


# ==================== 躯干稳定性奖励 ====================


def compute_trunk_height_reward(
    torso_z: jax.Array,
    target_height: float = 0.35,
    tolerance: float = 0.08
) -> jax.Array:
    """计算躯干高度保持奖励（行走时）

    与站立任务不同，行走时允许小幅度的上下波动。

    Args:
        torso_z: 躯干高度 (m)
        target_height: 目标高度
        tolerance: 容差（更宽松）

    Returns:
        高度奖励值 [0, 1]
    """
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)


def compute_trunk_orientation_penalty(
    quat: jax.Array,
    max_tilt: float = 0.3
) -> jax.Array:
    """计算躯干过度倾斜惩罚

    行走时允许小幅度倾斜，但惩罚过度倾斜。

    Args:
        quat: 躯干四元数 [w, x, y, z]
        max_tilt: 最大允许倾斜（弧度）

    Returns:
        倾斜惩罚值
    """
    quat = normalize_quaternion(quat)
    euler = quat_to_euler(quat)
    roll, pitch = euler[..., 0], euler[..., 1]

    # 只惩罚超过阈值的倾斜
    roll_penalty = jp.clip(jp.abs(roll) - max_tilt, min=0.0)
    pitch_penalty = jp.clip(jp.abs(pitch) - max_tilt, min=0.0)

    return roll_penalty + pitch_penalty


def compute_trunk_lin_vel_z_penalty(
    base_linvel: jax.Array
) -> jax.Array:
    """计算躯干Z方向速度惩罚

    行走时躯干应该平稳移动，避免大幅度上下跳动。

    Args:
        base_linvel: 基座线速度 [..., 3] 或 [3]

    Returns:
        Z速度惩罚值
    """
    return jp.square(base_linvel[..., 2])


# ==================== 行走特定惩罚 ====================


def compute_drag_penalty(
    feet_positions: jax.Array,
    contacts: jax.Array,
    drag_height_threshold: float = 0.02
) -> jax.Array:
    """计算拖地惩罚

    惩罚脚部在摆动相拖地（低高度）。

    Args:
        feet_positions: 脚部位置，形状 [..., 2, 3] 或 [2, 3]
        contacts: 接触标志，形状 [..., 2] 或 [2]
        drag_height_threshold: 拖地检测高度阈值

    Returns:
        拖地惩罚值
    """
    feet_heights = feet_positions[..., 2]
    swing_phase = (1.0 - contacts.astype(jp.float32))

    # 检测拖地：在摆动相但高度太低
    is_dragging = (feet_heights < drag_height_threshold) * swing_phase

    return jp.sum(is_dragging, axis=-1)


# ==================== 能量效率惩罚 ====================


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
    return jp.sum(jp.square(action - last_action), axis=-1)


def compute_torque_penalty(torques: jax.Array) -> jax.Array:
    """计算扭矩惩罚

    鼓励能量效率。

    Args:
        torques: 执行器扭矩

    Returns:
        扭矩惩罚值
    """
    return jp.sum(jp.square(torques), axis=-1)


# ==================== 增强行走奖励分量 ====================


def compute_gait_periodicity_reward(
    contacts: jax.Array,
    phase: jax.Array,
    stance_duration: float = 0.6,
    swing_duration: float = 0.4,
    tolerance: float = 0.1
) -> jax.Array:
    """计算步态周期性奖励"""
    expected_contact = (phase < stance_duration).astype(jp.float32)
    actual_contact = contacts.astype(jp.float32)
    error = jp.abs(expected_contact - actual_contact)
    return jp.mean(jp.exp(-error / tolerance), axis=-1)


def compute_swing_trajectory_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    phase: jax.Array,
    target_height: float = 0.08,
    swing_start_phase: float = 0.6
) -> jax.Array:
    """计算摆动轨迹奖励"""
    feet_z = feet_positions[..., 2]
    swing_progress = (phase - swing_start_phase) / (1.0 - swing_start_phase)
    swing_progress = jp.clip(swing_progress, 0.0, 1.0)
    expected_height = target_height * jp.sin(swing_progress * jp.pi)
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
    """计算着地冲击奖励"""
    impact = contact_forces * landing_events
    excess_force = jp.clip(impact - max_impact_force, min=0.0)
    return jp.mean(jp.exp(-excess_force / tolerance), axis=-1)


def compute_energy_efficiency_reward(
    torques: jax.Array,
    joint_velocities: jax.Array,
    target_efficiency: float = 0.8,
    penalty_weight: float = 0.001
) -> jax.Array:
    """计算能量效率奖励"""
    mechanical_power = jp.sum(jp.abs(torques * joint_velocities), axis=-1)
    thermal_loss = jp.sum(jp.square(torques), axis=-1)
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
    """计算综合稳定性奖励"""
    height_error = jp.abs(torso_z - target_height)
    height_reward = jp.exp(-height_error / height_tolerance)
    ang_vel_mag = jp.linalg.norm(base_angvel, axis=-1)
    ang_vel_reward = jp.exp(-ang_vel_mag * angular_velocity_penalty)
    w, x, y, z = base_quat[..., 0], base_quat[..., 1], base_quat[..., 2], base_quat[..., 3]
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
    """计算增强速度跟踪奖励"""
    error = actual_velocity - command
    error_x = jp.abs(error[..., 0])
    error_y = jp.abs(error[..., 1])
    error_yaw = jp.abs(error[..., 2])
    reward_x = jp.exp(-error_x / tolerance)
    reward_y = jp.exp(-error_y / tolerance)
    reward_yaw = jp.exp(-error_yaw / tolerance)
    wx, wy, wyaw = tracking_weights
    return (wx * reward_x + wy * reward_y + wyaw * reward_yaw) / (wx + wy + wyaw)


# ==================== 完整奖励函数 ====================


def compute_walking_reward(
    # 躯干状态
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    # 接触状态
    contact_sensors: jax.Array,
    # 脚部位置（可选）
    feet_positions: Optional[jax.Array] = None,
    # 动作和力矩
    action: jax.Array = None,
    last_action: jax.Array = None,
    torques: jax.Array = None,
    # 可选的增强参数
    joint_velocities: jax.Array = None,
    phase: jax.Array = None,
    landing_events: jax.Array = None,
    contact_forces: jax.Array = None,
    command: jax.Array = None,
    actual_velocity: jax.Array = None,
    # 目标参数
    target_velocity: float = 0.5,
    target_height: float = 0.35,
    # 奖励权重
    reward_weights: Dict[str, float] = None,
) -> jax.Array:
    """计算完整的行走任务奖励

    奖励分量（基础版）：
    1. forward_velocity: 前向速度奖励
    2. gait_symmetry: 步态对称性奖励
    3. foot_clearance: 脚部抬高奖励（如果提供 feet_positions）
    4. trunk_height: 躯干高度保持
    5. orientation_penalty: 过度倾斜惩罚
    6. trunk_lin_vel_z: Z方向速度惩罚
    7. drag_penalty: 拖地惩罚（如果提供 feet_positions）
    8. alive: 存活奖励
    9. action_rate: 动作平滑惩罚
    10. torques: 能量效率惩罚

    增强奖励分量（需要额外参数）：
    11. gait_periodicity: 步态周期性奖励（需要 phase）
    12. swing_trajectory: 摆动轨迹奖励（需要 phase）
    13. landing_impact: 着地冲击控制（需要 landing_events, contact_forces）
    14. energy_efficiency: 能量效率优化（需要 joint_velocities）
    15. stability: 综合稳定性奖励
    16. velocity_tracking: 增强速度跟踪（需要 command, actual_velocity）

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        contact_sensors: 接触传感器数据
        feet_positions: 脚部位置（可选）
        action: 当前动作
        last_action: 上一步动作
        torques: 执行器扭矩
        joint_velocities: 关节角速度（用于energy_efficiency）
        phase: 步态相位（用于gait_periodicity和swing_trajectory）
        landing_events: 着地事件掩码（用于landing_impact）
        contact_forces: 接触力（用于landing_impact）
        command: 速度命令 [vx, vy, vyaw]（用于velocity_tracking）
        actual_velocity: 实际速度 [vx, vy, vyaw]（用于velocity_tracking）
        target_velocity: 目标前向速度
        target_height: 目标躯干高度
        reward_weights: 奖励权重字典

    Returns:
        总奖励值
    """
    # 归一化四元数
    base_quat = normalize_quaternion(base_quat)

    # 提取接触状态
    contacts = get_feet_contacts(contact_sensors)

    # 计算各项奖励
    reward_forward_vel = compute_forward_velocity_reward(
        base_linvel, target_velocity
    )
    reward_gait_symmetry = compute_gait_symmetry_reward(contacts)
    reward_trunk_height = compute_trunk_height_reward(
        torso_z, target_height
    )
    penalty_orientation = compute_trunk_orientation_penalty(base_quat)
    penalty_lin_vel_z = compute_trunk_lin_vel_z_penalty(base_linvel)

    # 可选的脚部相关奖励/惩罚
    if feet_positions is not None:
        reward_foot_clearance = compute_foot_clearance_reward(
            feet_positions, contacts
        )
        penalty_drag = compute_drag_penalty(feet_positions, contacts)
    else:
        reward_foot_clearance = jp.array(0.0)
        penalty_drag = jp.array(0.0)

    # 能量效率惩罚
    action_rate_penalty = compute_action_rate_penalty(action, last_action)
    torque_penalty = compute_torque_penalty(torques)

    # 如果reward_weights为None，使用默认权重
    if reward_weights is None:
        reward_weights = DEFAULT_WALKING_REWARD_WEIGHTS

    # 组合基础奖励
    reward = (
        reward_weights.get("forward_velocity", 0.0) * reward_forward_vel
        + reward_weights.get("gait_symmetry", 0.0) * reward_gait_symmetry
        + reward_weights.get("foot_clearance", 0.0) * reward_foot_clearance
        + reward_weights.get("trunk_height", 0.0) * reward_trunk_height
        + reward_weights.get("orientation", 0.0) * penalty_orientation
        + reward_weights.get("trunk_lin_vel_z", 0.0) * penalty_lin_vel_z
        + reward_weights.get("drag", 0.0) * penalty_drag
        + reward_weights.get("alive", 0.0) * 1.0
        + reward_weights.get("action_rate", 0.0) * action_rate_penalty
        + reward_weights.get("torques", 0.0) * torque_penalty
    )

    # ==================== 增强奖励项 ====================
    # 检查是否需要计算增强奖励
    has_enhanced_rewards = any(
        key in reward_weights
        for key in ["gait_periodicity", "swing_trajectory", "landing_impact",
                   "energy_efficiency", "stability", "velocity_tracking"]
    )

    if has_enhanced_rewards:
        # 1. 步态周期性奖励（需要相位信息）
        if "gait_periodicity" in reward_weights and phase is not None:
            periodicity_reward = compute_gait_periodicity_reward(
                contacts, phase,
                stance_duration=0.6,
                swing_duration=0.4,
                tolerance=0.1
            )
            reward += reward_weights["gait_periodicity"] * periodicity_reward

        # 2. 摆动轨迹奖励
        if "swing_trajectory" in reward_weights and phase is not None and feet_positions is not None:
            trajectory_reward = compute_swing_trajectory_reward(
                feet_positions, contacts, phase,
                target_height=0.08,
                swing_start_phase=0.6
            )
            reward += reward_weights["swing_trajectory"] * trajectory_reward

        # 3. 着地冲击控制
        if "landing_impact" in reward_weights and landing_events is not None and contact_forces is not None:
            impact_reward = compute_landing_impact_reward(
                contact_forces, landing_events,
                max_impact_force=500.0,
                tolerance=100.0
            )
            reward += reward_weights["landing_impact"] * impact_reward

        # 4. 能量效率优化
        if "energy_efficiency" in reward_weights and joint_velocities is not None and torques is not None:
            efficiency_reward = compute_energy_efficiency_reward(
                torques, joint_velocities,
                target_efficiency=0.8,
                penalty_weight=0.001
            )
            reward += reward_weights["energy_efficiency"] * efficiency_reward

        # 5. 综合稳定性
        if "stability" in reward_weights:
            stability_reward = compute_stability_reward(
                torso_z, base_quat, base_linvel, base_angvel,
                target_height=target_height,
                height_tolerance=0.05,
                angular_velocity_penalty=1.0
            )
            reward += reward_weights["stability"] * stability_reward

        # 6. 增强速度跟踪（多轴权重）
        if "velocity_tracking" in reward_weights and command is not None and actual_velocity is not None:
            enhanced_tracking = compute_velocity_tracking_reward(
                actual_velocity, command,
                tracking_weights=(2.0, 0.5, 0.3)
            )
            # 替换原有的forward_velocity奖励
            base_tracking_weight = reward_weights.get("forward_velocity", 0.0)
            reward += (reward_weights["velocity_tracking"] - base_tracking_weight) * enhanced_tracking

    return reward


# ==================== 终止条件检查 ====================


def check_walking_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.15,
    angle_threshold: float = 0.8,
) -> jax.Array:
    """检查行走任务是否终止

    终止条件：
    1. 高度太低（摔倒）
    2. 倾斜角度太大

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数
        height_threshold: 高度阈值 (m)
        angle_threshold: 角度阈值 (rad)

    Returns:
        是否终止的布尔值
    """
    base_quat = normalize_quaternion(base_quat)
    euler = quat_to_euler(base_quat)
    roll, pitch = euler[..., 0], euler[..., 1]

    height_fail = torso_z < height_threshold
    orientation_fail = (
        (jp.abs(roll) > angle_threshold) | (jp.abs(pitch) > angle_threshold)
    )

    return height_fail | orientation_fail


# ==================== 默认奖励权重 ====================

DEFAULT_WALKING_REWARD_WEIGHTS = {
    # 主要目标：前向运动
    "forward_velocity": 1.5,  # 前向速度奖励
    # 步态质量
    "gait_symmetry": 0.3,  # 步态对称性
    "foot_clearance": 0.2,  # 脚部抬高
    # 躯干稳定
    "trunk_height": 0.5,  # 高度保持
    "orientation": -0.3,  # 姿态惩罚（负权重）
    "trunk_lin_vel_z": -0.5,  # Z方向速度惩罚
    "drag": -1.0,  # 拖地惩罚
    # 能量效率
    "action_rate": -0.01,  # 动作平滑
    "torques": -0.0001,  # 扭矩惩罚
    # 存活
    "alive": 0.3,  # 存活奖励
}
