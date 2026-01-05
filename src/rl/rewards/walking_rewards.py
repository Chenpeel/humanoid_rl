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
    # ==================== 新增参数（课程学习支持）====================
    feet_velocities: Optional[jax.Array] = None,  # 脚部速度（用于滑动和绊脚检测）
    joint_pos: Optional[jax.Array] = None,  # 关节位置（用于对称性和限位检测）
    joint_vel: Optional[jax.Array] = None,  # 关节速度（用于能量效率）
    joint_limits: Optional[tuple] = None,  # 关节限位 (lower, upper)
    contact_history: Optional[jax.Array] = None,  # 接触历史（用于空中时间）
    # 可选的增强参数
    joint_velocities: jax.Array = None,  # 保留兼容性（与joint_vel相同）
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
    """计算完整的行走任务奖励（重构版，支持课程学习）

    基于 Isaac Lab、Brax、dm_control 最佳实践的奖励函数设计。
    支持三阶段课程学习：站立平衡 → 低速行走 → 全速行走。

    基础奖励分量：
    1. forward_velocity: 前向速度奖励
    2. gait_symmetry: 步态对称性奖励
    3. foot_clearance: 脚部抬高奖励
    4. trunk_height: 躯干高度保持
    5. orientation: 过度倾斜惩罚
    6. trunk_lin_vel_z: Z方向速度惩罚
    7. drag: 拖地惩罚
    8. alive: 存活奖励
    9. action_rate: 动作平滑惩罚
    10. torques: 能量效率惩罚

    新增奖励分量（课程学习）：
    11. upright_bonus: 直立姿态奖励（站立阶段关键）
    12. feet_slide: 脚部滑动惩罚
    13. joint_symmetry: 关节对称性奖励
    14. stumbling: 绊脚惩罚
    15. joint_limits: 关节限位惩罚
    16. feet_contact_forces: 接触力平衡奖励
    17. feet_air_time: 脚部空中时间奖励

    增强奖励分量（可选）：
    18. gait_periodicity: 步态周期性奖励
    19. swing_trajectory: 摆动轨迹奖励
    20. landing_impact: 着地冲击控制
    21. energy_efficiency: 能量效率优化
    22. stability: 综合稳定性奖励
    23. velocity_tracking: 增强速度跟踪

    Args:
        torso_z: 躯干高度
        base_quat: 躯干四元数（已校正）
        base_linvel: 基座线速度
        base_angvel: 基座角速度
        contact_sensors: 接触传感器数据
        feet_positions: 脚部位置（可选）
        action: 当前动作
        last_action: 上一步动作
        torques: 执行器扭矩
        feet_velocities: 脚部速度（用于滑动/绊脚检测）
        joint_pos: 关节位置（用于对称性/限位检测）
        joint_vel: 关节速度（用于能量效率）
        joint_limits: 关节限位 (lower, upper)
        contact_history: 接触历史（用于空中时间）
        joint_velocities: 关节角速度（兼容旧版，与joint_vel相同）
        phase: 步态相位
        landing_events: 着地事件掩码
        contact_forces: 接触力
        command: 速度命令 [vx, vy, vyaw]
        actual_velocity: 实际速度 [vx, vy, vyaw]
        target_velocity: 目标前向速度
        target_height: 目标躯干高度
        reward_weights: 奖励权重字典（由课程学习管理器动态更新）

    Returns:
        总奖励值（已裁剪到 [-10, 10] 范围）
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

    # 能量效率惩罚（添加裁剪防止数值爆炸）
    action_rate_penalty = compute_action_rate_penalty(action, last_action)
    action_rate_penalty = jp.clip(action_rate_penalty, 0.0, 100.0)  # 限制最大惩罚

    torque_penalty = compute_torque_penalty(torques)
    torque_penalty = jp.clip(torque_penalty, 0.0, 10000.0)  # 限制最大惩罚

    # 如果reward_weights为None，使用默认权重
    if reward_weights is None:
        reward_weights = DEFAULT_WALKING_REWARD_WEIGHTS

    # 初始化总奖励
    reward = jp.array(0.0)

    # ==================== 核心奖励 ====================

    # 1. 速度跟踪（使用多轴跟踪或单轴）
    if "velocity_tracking" in reward_weights and command is not None and actual_velocity is not None:
        # 多轴速度跟踪（阶段3）
        tracking_reward = compute_velocity_tracking_reward(
            actual_velocity, command, tracking_weights=(2.0, 0.5, 0.3)
        )
        reward += reward_weights["velocity_tracking"] * tracking_reward
    elif "forward_velocity" in reward_weights:
        # 单轴前向速度（阶段1-2）
        reward += reward_weights["forward_velocity"] * reward_forward_vel

    # 2. 高度保持
    if "trunk_height" in reward_weights:
        reward += reward_weights["trunk_height"] * reward_trunk_height

    # 3. 姿态稳定
    if "orientation" in reward_weights:
        reward += reward_weights["orientation"] * penalty_orientation

    # 4. 直立姿态奖励（站立阶段关键，阶段1）
    if "upright_bonus" in reward_weights and reward_weights["upright_bonus"] > 0:
        upright_reward = compute_upright_bonus(base_quat, threshold=0.93)
        reward += reward_weights["upright_bonus"] * upright_reward

    # ==================== 步态质量 ====================

    # 5. 步态对称性
    if "gait_symmetry" in reward_weights:
        reward += reward_weights["gait_symmetry"] * reward_gait_symmetry

    # 6. 脚部抬高
    if "foot_clearance" in reward_weights and feet_positions is not None:
        reward += reward_weights["foot_clearance"] * reward_foot_clearance

    # 7. 脚部空中时间（新增，阶段2-3）
    if "feet_air_time" in reward_weights and contact_history is not None:
        air_time_reward = compute_feet_air_time_reward(contact_history, target_duty_cycle=0.5)
        reward += reward_weights["feet_air_time"] * air_time_reward

    # 8. 接触力平衡（新增，阶段2-3）
    if "feet_contact_forces" in reward_weights:
        contact_reward = compute_contact_force_balance_reward(contact_sensors, target_force=50.0)
        reward += reward_weights["feet_contact_forces"] * contact_reward

    # ==================== 约束和惩罚 ====================

    # 9. Z方向速度惩罚
    if "trunk_lin_vel_z" in reward_weights:
        reward += reward_weights["trunk_lin_vel_z"] * penalty_lin_vel_z

    # 10. 拖地惩罚
    if "drag" in reward_weights and feet_positions is not None:
        reward += reward_weights["drag"] * penalty_drag

    # 11. 脚部滑动惩罚（新增，阶段2-3）
    if "feet_slide" in reward_weights and feet_velocities is not None:
        slide_penalty = compute_feet_slide_penalty(feet_velocities, get_feet_contacts(contact_sensors), slide_threshold=0.1)
        reward += reward_weights["feet_slide"] * slide_penalty

    # 12. 绊脚惩罚（新增，阶段2-3）
    if "stumbling" in reward_weights and feet_positions is not None and feet_velocities is not None:
        feet_heights = feet_positions[..., 2]
        stumble_penalty = compute_stumbling_penalty(feet_heights, feet_velocities, stumble_threshold=0.02)
        reward += reward_weights["stumbling"] * stumble_penalty

    # 13. 关节限位惩罚（新增，阶段1-2）
    if "joint_limits" in reward_weights and joint_limits is not None and joint_pos is not None:
        limits_penalty = compute_joint_limits_penalty(joint_pos, joint_limits[0], joint_limits[1], margin=0.1)
        reward += reward_weights["joint_limits"] * limits_penalty

    # 14. 关节对称性奖励（新增，阶段2-3）
    if "joint_symmetry" in reward_weights and joint_pos is not None:
        symmetry_reward = compute_joint_symmetry_reward(joint_pos)
        reward += reward_weights["joint_symmetry"] * symmetry_reward

    # ==================== 能量效率 ====================

    # 15. 动作平滑（裁剪保护）
    if "action_rate" in reward_weights:
        reward += reward_weights["action_rate"] * action_rate_penalty

    # 16. 扭矩惩罚（裁剪保护）
    if "torques" in reward_weights:
        reward += reward_weights["torques"] * torque_penalty

    # 17. 能量效率（新增，可选，阶段3）
    if "energy_efficiency" in reward_weights:
        # 兼容性：joint_vel 或 joint_velocities
        jvel = joint_vel if joint_vel is not None else joint_velocities
        if jvel is not None and torques is not None:
            efficiency_reward = compute_energy_efficiency_reward(torques, jvel, target_efficiency=0.8, penalty_weight=0.001)
            reward += reward_weights["energy_efficiency"] * efficiency_reward

    # ==================== 存活和其他 ====================

    # 18. 存活奖励
    if "alive" in reward_weights:
        reward += reward_weights["alive"] * 1.0

    # ==================== 增强奖励项（可选）====================

    # 19. 步态周期性奖励（需要相位信息）
    if "gait_periodicity" in reward_weights and phase is not None:
        periodicity_reward = compute_gait_periodicity_reward(
            get_feet_contacts(contact_sensors), phase,
            stance_duration=0.6, swing_duration=0.4, tolerance=0.1
        )
        reward += reward_weights["gait_periodicity"] * periodicity_reward

    # 20. 摆动轨迹奖励
    if "swing_trajectory" in reward_weights and phase is not None and feet_positions is not None:
        trajectory_reward = compute_swing_trajectory_reward(
            feet_positions, get_feet_contacts(contact_sensors), phase,
            target_height=0.08, swing_start_phase=0.6
        )
        reward += reward_weights["swing_trajectory"] * trajectory_reward

    # 21. 着地冲击控制
    if "landing_impact" in reward_weights and landing_events is not None and contact_forces is not None:
        impact_reward = compute_landing_impact_reward(
            contact_forces, landing_events,
            max_impact_force=500.0, tolerance=100.0
        )
        reward += reward_weights["landing_impact"] * impact_reward

    # 22. 综合稳定性
    if "stability" in reward_weights:
        stability_reward = compute_stability_reward(
            torso_z, base_quat, base_linvel, base_angvel,
            target_height=target_height,
            height_tolerance=0.05,
            angular_velocity_penalty=1.0
        )
        reward += reward_weights["stability"] * stability_reward

    # ==================== 最终保护 ====================

    # [CRITICAL FIX] 裁剪总奖励到合理范围（防止训练崩溃）
    # 参考 VelocityTrackingEnv，裁剪到 [-10, 10]
    reward = jp.clip(reward, -10.0, 10.0)

    # NaN/Inf 保护（防止数值问题导致训练失败）
    reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

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


# ==================== 新增奖励函数（基于 Isaac Lab 最佳实践）====================


def compute_termination_penalty() -> jax.Array:
    """终止时施加大负惩罚（固定返回 -200.0）

    注意：此函数不在 compute_walking_reward 中调用，
    而是在环境的 step 函数中检测到 done=True 时应用。

    Returns:
        固定惩罚值 -200.0
    """
    return jp.array(-200.0)


def compute_upright_bonus(
    base_quat: jax.Array,
    threshold: float = 0.93  # cos(20°)
) -> jax.Array:
    """直立姿态奖励（参考 Isaac Lab）

    当机器人接近完美直立时给予奖励。

    Args:
        base_quat: 基座四元数 [qw, qx, qy, qz]（已校正）
        threshold: cos(最大允许倾角)，默认 0.93 对应 20°

    Returns:
        奖励值，范围 [0, 1]
    """
    # 提取 Z 轴在世界坐标系中的方向
    qw, qx, qy, qz = base_quat[..., 0], base_quat[..., 1], base_quat[..., 2], base_quat[..., 3]

    # 计算 Z 轴方向向量 (旋转后的 [0, 0, 1])
    z_x = 2.0 * (qx * qz + qw * qy)
    z_y = 2.0 * (qy * qz - qw * qx)
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)

    # Z 轴与世界 Z 轴的点积（即 z_z 分量）
    upright_projection = z_z

    # 二值奖励：超过阈值给1，否则给0
    reward = jp.where(upright_projection > threshold, 1.0, 0.0)
    return reward


def compute_feet_slide_penalty(
    feet_linvel: jax.Array,  # 脚部线速度 (n_feet, 3)
    contact_sensors: jax.Array,  # 接触传感器 (n_feet,)
    slide_threshold: float = 0.1
) -> jax.Array:
    """脚部滑动惩罚（参考 Isaac Lab H1）

    当脚在接触地面时仍有较大水平速度，则施加惩罚。

    Args:
        feet_linvel: 脚部线速度，形状 (n_feet, 3)
        contact_sensors: 接触力传感器，形状 (n_feet,)
        slide_threshold: 滑动速度阈值 (m/s)

    Returns:
        惩罚值（已归一化）
    """
    # 计算水平速度
    feet_vel_xy = feet_linvel[..., :2]  # 只看 xy 平面
    slide_velocity = jp.linalg.norm(feet_vel_xy, axis=-1)

    # 检测滑动：接触时速度超过阈值
    is_contact = contact_sensors > 0.1  # 接触阈值
    is_sliding = (slide_velocity > slide_threshold) & is_contact

    # 计算惩罚（归一化到每只脚）
    penalty = jp.sum(is_sliding.astype(jp.float32) * slide_velocity) / jp.maximum(len(contact_sensors), 1)
    return penalty


def compute_joint_symmetry_reward(
    joint_pos: jax.Array,  # 关节位置 (16,)
    right_indices: tuple = (0, 1, 2, 3, 4, 5, 6, 7),  # 右腿关节索引
    left_indices: tuple = (8, 9, 10, 11, 12, 13, 14, 15),  # 左腿关节索引
) -> jax.Array:
    """关节镜像对称性奖励

    鼓励左右腿的关节角度保持镜像对称。

    Args:
        joint_pos: 关节位置数组
        right_indices: 右腿关节索引
        left_indices: 左腿关节索引（对应顺序）

    Returns:
        奖励值，范围 [0, 1]
    """
    right_joints = joint_pos[..., jp.array(right_indices)]
    left_joints = joint_pos[..., jp.array(left_indices)]

    # 镜像对称误差（某些关节需要取反，这里简化处理）
    symmetry_error = jp.sum(jp.square(right_joints - left_joints))

    # 指数奖励
    reward = jp.exp(-symmetry_error / 0.2)
    return reward


def compute_stumbling_penalty(
    feet_heights: jax.Array,  # 脚部高度 (n_feet,)
    feet_velocities: jax.Array,  # 脚部速度 (n_feet, 3)
    stumble_threshold: float = 0.02
) -> jax.Array:
    """绊脚检测惩罚

    当脚在很低的高度（接近地面但未完全接触）时有较大速度，
    认为是绊脚动作，施加惩罚。

    Args:
        feet_heights: 脚部离地高度
        feet_velocities: 脚部速度
        stumble_threshold: 绊脚高度阈值

    Returns:
        惩罚值
    """
    # 检测"危险高度"：接近地面但未完全接触
    is_low = (feet_heights < stumble_threshold) & (feet_heights > 0.005)

    # 检测速度较大
    feet_speed = jp.linalg.norm(feet_velocities, axis=-1)
    is_fast = feet_speed > 0.5

    # 绊脚：低高度 + 高速度
    is_stumbling = is_low & is_fast
    penalty = jp.sum(is_stumbling.astype(jp.float32))
    return penalty


def compute_joint_limits_penalty(
    joint_pos: jax.Array,
    joint_limits_lower: jax.Array,  # 从模型中提取
    joint_limits_upper: jax.Array,
    margin: float = 0.1  # 10% 边界容差
) -> jax.Array:
    """关节接近限位惩罚

    当关节角度接近物理限位时施加惩罚。

    Args:
        joint_pos: 当前关节位置
        joint_limits_lower: 关节下限
        joint_limits_upper: 关节上限
        margin: 边界容差（比例）

    Returns:
        惩罚值
    """
    range_size = joint_limits_upper - joint_limits_lower
    lower_margin = joint_limits_lower + margin * range_size
    upper_margin = joint_limits_upper - margin * range_size

    # 检测超出安全边界
    lower_violation = jp.maximum(0.0, lower_margin - joint_pos)
    upper_violation = jp.maximum(0.0, joint_pos - upper_margin)

    penalty = jp.sum(lower_violation + upper_violation)
    return penalty


def compute_contact_force_balance_reward(
    contact_forces: jax.Array,  # 接触力 (n_feet,)
    target_force: float = 50.0  # 目标单脚承重 (N)
) -> jax.Array:
    """接触力平衡奖励

    鼓励左右脚接触力均衡分布。

    Args:
        contact_forces: 各脚接触力大小
        target_force: 理想单脚承重

    Returns:
        奖励值
    """
    # 计算与目标力的偏差
    force_errors = jp.abs(contact_forces - target_force)
    total_error = jp.sum(force_errors)

    # 指数奖励
    reward = jp.exp(-total_error / (target_force * jp.maximum(len(contact_forces), 1)))
    return reward


def compute_feet_air_time_reward(
    contact_history: jax.Array,  # 接触历史 (history_length, n_feet)
    target_duty_cycle: float = 0.5  # 期望接地率 50%
) -> jax.Array:
    """脚部空中时间奖励（参考 Isaac Lab）

    鼓励脚部有适当的摆动相（空中时间）。

    Args:
        contact_history: 最近 N 步的接触状态
        target_duty_cycle: 目标接地率（0-1）

    Returns:
        奖励值
    """
    # 计算实际接地率
    actual_duty_cycle = jp.mean(contact_history, axis=0)

    # 计算与目标的偏差
    duty_error = jp.abs(actual_duty_cycle - target_duty_cycle)
    total_error = jp.mean(duty_error)

    # 指数奖励
    reward = jp.exp(-total_error / 0.2)
    return reward


# ==================== 三阶段课程学习权重配置 ====================


# 站立平衡阶段（0-50k steps）
# 训练目标：学习保持直立不摔倒，建立基本的平衡控制能力，避免剧烈动作
# 优先级：存活 > 姿态 > 能量
STAGE1_WEIGHTS = {
    # 核心目标：不摔倒
    "termination": -200.0,            # 强力终止惩罚
    "alive": 5.0,                     # 高存活奖励

    # 姿态控制
    "upright_bonus": 2.0,             # 直立奖励
    "trunk_height": 2.0,              # 保持目标高度
    "orientation": -1.5,              # 严格姿态约束

    # 抑制移动
    "trunk_lin_vel_z": -1.0,          # 防止跳跃

    # 能量约束
    "action_rate": -0.02,             # 鼓励平滑动作
    "torques": -0.0002,               # 限制扭矩
    "joint_limits": -0.2,             # 防止关节超限

    # 暂时禁用的项（减少干扰）
    "forward_velocity": 0.0,          # 禁用
    "gait_symmetry": 0.0,             # 禁用
    "foot_clearance": 0.0,            # 禁用
    "drag": 0.0,                      # 禁用
}


# 低速行走阶段（50k-150k steps）
# 训练目标：学习基本行走步态，建立双脚交替接触模式，实现低速（0.1-0.3 m/s）稳定前进
# 优先级：步态 > 速度 > 稳定
STAGE2_WEIGHTS = {
    # 终止和存活
    "termination": -200.0,
    "alive": 2.0,                     # 降低：不再是主要目标

    # 步态发展（核心）
    "gait_symmetry": 1.5,             # 激活：步态对称性
    "foot_clearance": 0.8,            # 激活：抬脚奖励
    "feet_contact_forces": 0.3,       # 接触力平衡
    "feet_air_time": 0.5,             # 鼓励摆动相

    # 速度跟踪
    "forward_velocity": 1.0,          # 激活：低权重速度奖励

    # 姿态稳定
    "upright_bonus": 0.5,             # 降低：已学会
    "trunk_height": 1.0,              # 降低
    "orientation": -0.8,              # 降低：允许更多自由度

    # 能量和平滑
    "trunk_lin_vel_z": -0.5,
    "drag": -0.8,                     # 激活：防拖地
    "feet_slide": -0.8,               # 防滑动
    "action_rate": -0.01,
    "torques": -0.0001,
    "joint_symmetry": 0.2,            # 关节对称性
    "stumbling": -0.5,                # 防绊脚
}


# 全速行走阶段（150k+ steps）
# 训练目标：跟踪任意速度命令（-0.2 到 0.8 m/s），优化步态质量和能量效率，提高鲁棒性（应对扰动）
# 优先级：性能 > 质量 > 效率
STAGE3_WEIGHTS = {
    # 终止和存活
    "termination": -200.0,
    "alive": 0.3,                     # 进一步降低

    # 速度跟踪（主任务）
    "velocity_tracking": 2.0,         # 多轴跟踪

    # 步态质量
    "gait_symmetry": 0.3,
    "foot_clearance": 0.2,
    "feet_air_time": 0.25,
    "feet_contact_forces": 0.1,

    # 姿态稳定
    "trunk_height": 0.8,              # 提升：恢复严格要求
    "orientation": -0.5,              # 提升
    "upright_bonus": 0.0,             # 禁用（已内含在orientation中）

    # 约束和惩罚
    "trunk_lin_vel_z": -0.5,
    "drag": -1.0,
    "feet_slide": -0.5,
    "stumbling": -0.8,                # 加强

    # 能量效率
    "action_rate": -0.01,
    "torques": -0.001,                # 提升：更重视能量
    "energy_efficiency": 0.001,       # 激活：机械功率优化
    "joint_symmetry": 0.15,
    "joint_limits": -0.1,

    # 增强项（可选，需要额外传感器数据）
    "landing_impact": 0.2,            # 激活：着地冲击控制
    "stability": 0.1,                 # 激活：综合稳定性
}
