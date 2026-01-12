"""
行走任务奖励函数

基于开源实现（Legged Gym, Isaac Lab）的行走奖励函数。
使用 JAX 实现纯函数式设计，支持 JIT 编译。

主要参考:
- Legged Gym (ETH Zurich): https://github.com/leggedrobotics/legged_gym
- Isaac Lab Locomotion: omni.isaac.lab_tasks/locomotion
- ANYmal/Go1 机器人实现
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jp

from .components import (compute_action_rate_penalty, compute_ang_vel_penalty,
                         compute_joint_deviation_penalty as _compute_joint_deviation_penalty,
                         compute_knee_bend_reward, compute_lin_vel_xy_penalty,
                         compute_toe_only_contact_penalty, compute_torque_penalty,
                         normalize_quaternion, quat_to_euler, wrap_to_pi)

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
    "joint_limits": -0.1,
    "landing_impact": 0.2,
    "stability": 0.1,
}
# ============================================================================================
# ============================== END: 三阶段课程学习权重配置 =====================================
# ============================================================================================


# =============================================================================================
# ========================================= 终止条件检查 ========================================
# =============================================================================================


def check_walking_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.25,
    upright_threshold: float = 0.5,
) -> jax.Array:
    """
    检查行走是否终止 (摔倒检测)

    Args:
        torso_z: 躯干高度
        base_quat: 基座四元数 [w, x, y, z]
        height_threshold: 最小高度阈值 (m)
        upright_threshold: 最小直立度阈值 (z-axis projection)

    Returns:
        bool array, True 表示终止 (episode 结束)
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
# ===================================== END: 终止条件检查 ========================================
# =============================================================================================


# =============================================================================================
# ========================================= 数学工具函数 ========================================
# =============================================================================================

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
        right_contact = jp.maximum(
            contact_sensors[..., 0], contact_sensors[..., 1])
        left_contact = jp.maximum(
            contact_sensors[..., 2], contact_sensors[..., 3])
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
    torso_z: jax.Array, target_height: float = 0.78, tolerance: float = 0.08
) -> jax.Array:
    """躯干高度保持奖励"""
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_upright_bonus(base_quat: jax.Array, threshold: float = 0.93) -> jax.Array:
    """直立姿态奖励"""
    qw, qx, qy, qz = (
        base_quat[..., 0],
        base_quat[..., 1],
        base_quat[..., 2],
        base_quat[..., 3],
    )
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return jp.where(z_z > threshold, 1.0, 0.0)


# ---------------------------------------------------------------------------------------------


def compute_stability_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    target_height: float = 0.78,
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
# ======================================= 能量与扭矩惩罚 =========================================
# =============================================================================================


def compute_normalized_torque_penalty(
    torques: jax.Array,
    joint_names: Optional[List[str]] = None,
) -> jax.Array:
    """
    计算归一化扭矩惩罚

    根据不同关节的实际扭矩限制进行归一化，确保惩罚的公平性。

    扭矩限制 (Nm):
    - Hip pitch: 2.5 Nm (索引 0, 8)
    - Hip yaw: 8 Nm (索引 1, 9)
    - Hip roll: 8 Nm (索引 2, 10)
    - Knee: 8 Nm (索引 3, 11)
    - Ankle 1-3: 2.5 Nm (索引 4, 5, 6, 12, 13, 14)
    - Toe: 8 Nm (索引 7, 15)

    Args:
        torques: 关节扭矩 [batch, 16] 或 [16]
        joint_names: 可选的关节名称列表（用于未来扩展）

    Returns:
        归一化扭矩惩罚 [batch] 或标量
    """
    # 定义每个关节的扭矩限制 (Nm)
    # 假设关节顺序：右腿8个 + 左腿8个
    # 右腿: hip_pitch, hip_yaw, hip_roll, knee, ankle1, ankle2, ankle3, toe
    # 左腿: hip_pitch, hip_yaw, hip_roll, knee, ankle1, ankle2, ankle3, toe
    torque_limits = jp.array(
        [
            # 右腿
            2.5,  # right_hip_pitch
            8.0,  # right_hip_yaw
            8.0,  # right_hip_roll
            8.0,  # right_knee
            2.5,  # right_ankle_1
            2.5,  # right_ankle_2
            2.5,  # right_ankle_3
            8.0,  # right_toe
            # 左腿
            2.5,  # left_hip_pitch
            8.0,  # left_hip_yaw
            8.0,  # left_hip_roll
            8.0,  # left_knee
            2.5,  # left_ankle_1
            2.5,  # left_ankle_2
            2.5,  # left_ankle_3
            8.0,  # left_toe
        ]
    )

    # 归一化扭矩: torque / limit，使得所有关节在 [-1, 1] 范围内
    normalized_torques = torques / torque_limits

    # 计算平方惩罚（归一化后）
    penalty = jp.sum(jp.square(normalized_torques), axis=-1)

    return penalty


# ---------------------------------------------------------------------------------------------


def compute_energy_efficiency_reward(
    torques: jax.Array,
    joint_vel: jax.Array,
) -> jax.Array:
    """
    计算能量效率奖励（负功率惩罚）

    功率 = 扭矩 × 角速度

    Args:
        torques: 关节扭矩 [batch, n_joints]
        joint_vel: 关节速度 [batch, n_joints]

    Returns:
        能量效率惩罚 [batch]
    """
    # Robustness: some robots have extra actuators without corresponding joint_vel entries.
    # Align shapes by truncating both to the common prefix length.
    n = min(torques.shape[-1], joint_vel.shape[-1])
    torques = torques[..., :n]
    joint_vel = joint_vel[..., :n]
    power = jp.abs(torques * joint_vel)
    return jp.sum(power, axis=-1)


# =============================================================================================
# ===================================== END: 能量与扭矩惩罚 ======================================
# =============================================================================================


# =============================================================================================
# ======================================= 约束与惩罚项 ==========================================
# =============================================================================================


def compute_trunk_orientation_penalty(
    quat: jax.Array, max_tilt: float = 0.3
) -> jax.Array:
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
    is_dragging = (feet_positions[..., 2] < threshold) * (
        1.0 - contacts.astype(jp.float32)
    )
    return jp.sum(is_dragging, axis=-1)


# ---------------------------------------------------------------------------------------------


def compute_feet_slide_penalty(
    feet_linvel: jax.Array, contacts: jax.Array, threshold: float = 0.1
) -> jax.Array:
    """脚部滑动惩罚"""
    slide_vel = jp.linalg.norm(feet_linvel[..., :2], axis=-1)
    is_sliding = (slide_vel > threshold) * (contacts > 0.1)
    return jp.sum(is_sliding.astype(jp.float32) * slide_vel) / jp.maximum(
        feet_linvel.shape[-2], 1
    )


# ---------------------------------------------------------------------------------------------


def compute_feet_contact_forces_reward(
    contact_sensors: jax.Array, target_force: float = 50.0, tolerance: float = 20.0
) -> jax.Array:
    """脚部接触力奖励 (鼓励合理接触力)"""
    contact_forces = contact_sensors
    force_error = jp.abs(contact_forces - target_force)
    reward = jp.exp(-force_error / tolerance) * (contact_sensors > 1.0)
    return jp.mean(reward, axis=-1)


# ---------------------------------------------------------------------------------------------


def compute_joint_symmetry_reward(
    joint_pos: jax.Array, tolerance: float = 0.1, mirror_signs: Optional[jax.Array] = None
) -> jax.Array:
    """关节对称性奖励 (左右腿镜像对称)

    注意：Jiyuan 的部分左右关节在 MJCF 中轴向相反（例如 hip_roll/knee/ankle_roll/toe），
    直接比较 right == left 会反向驱动这些关节，容易出现“两条腿同向歪斜/髋部扭曲”的坏解。

    Args:
        joint_pos: 关节位置 [batch, n_joints] 或 [n_joints]
        tolerance: 对称误差容忍度
        mirror_signs: 可选的左右镜像符号 (shape=[n_joints_per_leg])，
            用于把 left_leg 映射到 right_leg 的符号空间：right ≈ left * mirror_signs。
            若为 None 且每条腿 8 关节，则使用 Jiyuan 默认符号：
            [hip_pitch, hip_yaw, hip_roll, knee, ankle_pitch, ankle_roll, ankle_yaw, toe]
            = [ +, +, -, -, +, -, +, - ]。
    """
    joint_pos = jp.asarray(joint_pos)
    if joint_pos.shape[-1] % 2 != 0:
        symmetry_error = jp.mean(jp.abs(joint_pos), axis=-1)
        return jp.exp(-symmetry_error / tolerance)

    num_joints_per_leg = joint_pos.shape[-1] // 2
    right_leg = joint_pos[..., :num_joints_per_leg]
    left_leg = joint_pos[..., num_joints_per_leg:]

    if mirror_signs is None:
        if num_joints_per_leg == 8:
            mirror_signs = jp.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        else:
            mirror_signs = jp.ones((num_joints_per_leg,))
    else:
        mirror_signs = jp.asarray(mirror_signs, dtype=jp.float32)

    broadcast_shape = (1,) * (right_leg.ndim - 1) + (num_joints_per_leg,)
    mirror_signs = mirror_signs.reshape(broadcast_shape)
    left_leg_mirrored = left_leg * mirror_signs

    symmetry_error = jp.mean(jp.abs(right_leg - left_leg_mirrored), axis=-1)
    return jp.exp(-symmetry_error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_joint_deviation_penalty(
    joint_pos: jax.Array,
    joint_pos_default: jax.Array,
    indices: Optional[jax.Array] = None,
) -> jax.Array:
    """关节姿态偏离惩罚 (保持接近默认/home pose)"""
    return _compute_joint_deviation_penalty(
        joint_pos=joint_pos, joint_pos_default=joint_pos_default, indices=indices
    )


# ---------------------------------------------------------------------------------------------


def compute_stumbling_penalty(
    feet_positions: jax.Array,
    feet_velocities: jax.Array,
    contacts: jax.Array,
    threshold: float = 0.05,
) -> jax.Array:
    """绊倒惩罚 (脚部高度低且速度大)"""
    feet_heights = feet_positions[..., 2]
    feet_horizontal_vel = jp.linalg.norm(feet_velocities[..., :2], axis=-1)
    is_stumbling = (
        (feet_heights < threshold)
        * (feet_horizontal_vel > 0.5)
        * (1.0 - contacts.astype(jp.float32))
    )
    return jp.sum(is_stumbling, axis=-1)


# ---------------------------------------------------------------------------------------------


def compute_landing_impact_reward(
    contact_sensors: jax.Array, prev_contact_sensors: jax.Array, threshold: float = 100.0
) -> jax.Array:
    """着陆冲击奖励 (惩罚过大冲击力)"""
    contact_change = contact_sensors - prev_contact_sensors
    new_contacts = contact_change > 1.0
    impact_force = contact_sensors * new_contacts
    return -jp.sum(jp.clip(impact_force - threshold, min=0.0), axis=-1)


# ---------------------------------------------------------------------------------------------


# =============================================================================================
# ===================================== END: 约束与惩罚项 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 完整奖励函数 ==========================================
# =============================================================================================


class _WalkingRewardContext(NamedTuple):
    torso_z: jax.Array
    base_quat: jax.Array
    base_linvel: jax.Array
    base_angvel: jax.Array
    contact_sensors: jax.Array
    contacts: jax.Array
    feet_positions: Optional[jax.Array]
    action: Optional[jax.Array]
    last_action: Optional[jax.Array]
    torques: Optional[jax.Array]
    feet_velocities: Optional[jax.Array]
    joint_pos: Optional[jax.Array]
    joint_vel: Optional[jax.Array]
    joint_pos_default: Optional[jax.Array]
    joint_limits: Optional[tuple]
    contact_history: Optional[jax.Array]
    command: Optional[jax.Array]
    actual_velocity: Optional[jax.Array]
    target_velocity: float
    target_height: float


_HIP_INDICES = jp.array([0, 1, 2, 8, 9, 10])


def _zeros_like_reward(ctx: _WalkingRewardContext) -> jax.Array:
    return jp.zeros_like(ctx.torso_z)


def _ones_like_reward(ctx: _WalkingRewardContext) -> jax.Array:
    return jp.ones_like(ctx.torso_z)


def _walking_trunk_height(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_trunk_height_reward(ctx.torso_z, ctx.target_height)


def _walking_orientation(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_trunk_orientation_penalty(ctx.base_quat)


def _walking_upright_bonus(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_upright_bonus(ctx.base_quat)


def _walking_gait_symmetry(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_gait_symmetry_reward(ctx.contacts)


def _walking_foot_clearance(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.feet_positions is None:
        return _zeros_like_reward(ctx)
    return compute_foot_clearance_reward(ctx.feet_positions, ctx.contacts)


def _walking_feet_air_time(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.contact_history is None:
        return _zeros_like_reward(ctx)
    return compute_feet_air_time_reward(ctx.contact_history)


def _walking_lin_vel(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_lin_vel_xy_penalty(ctx.base_linvel)


def _walking_ang_vel(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_ang_vel_penalty(ctx.base_angvel)


def _walking_drag(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.feet_positions is None:
        return _zeros_like_reward(ctx)
    return compute_drag_penalty(ctx.feet_positions, ctx.contacts)


def _walking_torques(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.torques is None:
        return _zeros_like_reward(ctx)
    return compute_torque_penalty(ctx.torques)


def _walking_alive(ctx: _WalkingRewardContext) -> jax.Array:
    return _ones_like_reward(ctx)


def _walking_action_rate(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.action is None or ctx.last_action is None:
        return _zeros_like_reward(ctx)
    return compute_action_rate_penalty(ctx.action, ctx.last_action)


def _walking_joint_limits(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.joint_pos is None or ctx.joint_limits is None:
        return _zeros_like_reward(ctx)
    lower_limits, upper_limits = ctx.joint_limits
    lower_violation = jp.maximum(0.0, lower_limits - ctx.joint_pos)
    upper_violation = jp.maximum(0.0, ctx.joint_pos - upper_limits)
    return jp.sum(jp.square(lower_violation) + jp.square(upper_violation), axis=-1)


def _walking_trunk_lin_vel_z(ctx: _WalkingRewardContext) -> jax.Array:
    return jp.square(ctx.base_linvel[..., 2])


def _walking_feet_contact_forces(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.contact_sensors is None:
        return _zeros_like_reward(ctx)
    return compute_feet_contact_forces_reward(ctx.contact_sensors)


def _walking_feet_slide(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.feet_velocities is None or ctx.contact_sensors is None:
        return _zeros_like_reward(ctx)
    return compute_feet_slide_penalty(ctx.feet_velocities, ctx.contacts)


def _walking_joint_symmetry(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.joint_pos is None:
        return _zeros_like_reward(ctx)
    return compute_joint_symmetry_reward(ctx.joint_pos)


def _walking_joint_deviation(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.joint_pos is None or ctx.joint_pos_default is None:
        return _zeros_like_reward(ctx)
    return compute_joint_deviation_penalty(ctx.joint_pos, ctx.joint_pos_default)


def _walking_hip_deviation(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.joint_pos is None or ctx.joint_pos_default is None:
        return _zeros_like_reward(ctx)
    return compute_joint_deviation_penalty(
        ctx.joint_pos, ctx.joint_pos_default, indices=_HIP_INDICES
    )


def _walking_stumbling(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.feet_positions is None or ctx.feet_velocities is None:
        return _zeros_like_reward(ctx)
    return compute_stumbling_penalty(
        ctx.feet_positions, ctx.feet_velocities, ctx.contacts
    )


def _walking_landing_impact(ctx: _WalkingRewardContext) -> jax.Array:
    return _zeros_like_reward(ctx)


def _walking_stability(ctx: _WalkingRewardContext) -> jax.Array:
    return compute_stability_reward(
        ctx.torso_z, ctx.base_quat, ctx.base_linvel, ctx.base_angvel, ctx.target_height
    )


def _walking_knee_bend(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.joint_pos is None:
        return _zeros_like_reward(ctx)
    return compute_knee_bend_reward(ctx.joint_pos)


def _walking_double_support(ctx: _WalkingRewardContext) -> jax.Array:
    # Prefer "foot-only" contacts (not toe) when 4 touch sensors are available:
    # [right_foot, right_toe, left_foot, left_toe].
    # Use a short grace window via `contact_history` to avoid penalizing brief recovery steps.
    if ctx.contact_history is not None and getattr(ctx.contact_history, "shape", None) is not None:
        hist = jp.asarray(ctx.contact_history)
        if hist.shape[-1] == 4:
            right_foot = hist[..., 0]
            left_foot = hist[..., 2]
            right_any = (jp.max(right_foot, axis=-2) > 0.5).astype(jp.float32)
            left_any = (jp.max(left_foot, axis=-2) > 0.5).astype(jp.float32)
            return right_any * left_any

    if ctx.contact_sensors is not None:
        sensors = jp.asarray(ctx.contact_sensors)
        if sensors.shape[-1] == 4:
            right = (sensors[..., 0] > 1.0).astype(jp.float32)
            left = (sensors[..., 2] > 1.0).astype(jp.float32)
            return right * left

    right = ctx.contacts[..., 0].astype(jp.float32)
    left = ctx.contacts[..., 1].astype(jp.float32)
    return right * left


def _walking_toe_only(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.contact_history is not None and getattr(ctx.contact_history, "shape", None) is not None:
        hist = jp.asarray(ctx.contact_history)
        if hist.shape[-1] == 4:
            # Penalize toe-only support within the grace window (mean over window).
            toe_only = compute_toe_only_contact_penalty(hist, threshold=0.5)
            return jp.mean(toe_only, axis=-1)

    if ctx.contact_sensors is None:
        return _zeros_like_reward(ctx)
    return compute_toe_only_contact_penalty(ctx.contact_sensors)


def _walking_energy_efficiency(ctx: _WalkingRewardContext) -> jax.Array:
    if ctx.torques is None or ctx.joint_vel is None:
        return _zeros_like_reward(ctx)
    return compute_energy_efficiency_reward(ctx.torques, ctx.joint_vel)


WALKING_REWARD_REGISTRY: Dict[str, Callable[[_WalkingRewardContext], jax.Array]] = {
    # posture / stability
    "trunk_height": _walking_trunk_height,
    "orientation": _walking_orientation,
    "upright_bonus": _walking_upright_bonus,
    "stability": _walking_stability,
    "knee_bend": _walking_knee_bend,
    "double_support": _walking_double_support,
    "toe_only": _walking_toe_only,
    # gait quality
    "gait_symmetry": _walking_gait_symmetry,
    "foot_clearance": _walking_foot_clearance,
    "feet_air_time": _walking_feet_air_time,
    "feet_contact_forces": _walking_feet_contact_forces,
    "feet_slide": _walking_feet_slide,
    "stumbling": _walking_stumbling,
    "landing_impact": _walking_landing_impact,
    # regularization / penalties
    "lin_vel": _walking_lin_vel,
    "ang_vel": _walking_ang_vel,
    "drag": _walking_drag,
    "torques": _walking_torques,
    "action_rate": _walking_action_rate,
    "joint_limits": _walking_joint_limits,
    "trunk_lin_vel_z": _walking_trunk_lin_vel_z,
    "joint_symmetry": _walking_joint_symmetry,
    "joint_deviation": _walking_joint_deviation,
    "hip_deviation": _walking_hip_deviation,
    "energy_efficiency": _walking_energy_efficiency,
    # alive
    "alive": _walking_alive,
}


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
    joint_pos_default: Optional[jax.Array] = None,
    joint_limits: Optional[tuple] = None,
    contact_history: Optional[jax.Array] = None,
    command: jax.Array = None,
    actual_velocity: jax.Array = None,
    target_velocity: float = 0.5,
    target_height: float = 0.78,
    reward_weights: Dict[str, float] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的行走任务奖励"""

    # 初始化
    base_quat = normalize_quaternion(base_quat)
    contacts = get_feet_contacts(contact_sensors)
    weights = reward_weights or DEFAULT_WALKING_REWARD_WEIGHTS
    reward = jp.array(0.0)
    reward_info = {f"reward/{k}": jp.array(0.0) for k in weights.keys()}

    ctx = _WalkingRewardContext(
        torso_z=torso_z,
        base_quat=base_quat,
        base_linvel=base_linvel,
        base_angvel=base_angvel,
        contact_sensors=contact_sensors,
        contacts=contacts,
        feet_positions=feet_positions,
        action=action,
        last_action=last_action,
        torques=torques,
        feet_velocities=feet_velocities,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_pos_default=joint_pos_default,
        joint_limits=joint_limits,
        contact_history=contact_history,
        command=command,
        actual_velocity=actual_velocity,
        target_velocity=target_velocity,
        target_height=target_height,
    )

    # 速度项需要保持“二选一”的旧行为：若启用 velocity_tracking 且 command 可用则优先使用。
    vt_w = weights.get("velocity_tracking", 0.0)
    fv_w = weights.get("forward_velocity", 0.0)
    if vt_w != 0.0 and command is not None:
        if actual_velocity is None:
            val = _zeros_like_reward(ctx)
        else:
            val = compute_velocity_tracking_reward(actual_velocity, command)
        weighted = vt_w * val
        reward += weighted
        reward_info["reward/velocity_tracking"] = weighted
    elif fv_w != 0.0:
        val = compute_forward_velocity_reward(base_linvel, target_velocity)
        weighted = fv_w * val
        reward += weighted
        reward_info["reward/forward_velocity"] = weighted

    for key, weight in weights.items():
        if key in {"velocity_tracking", "forward_velocity"} or weight == 0.0:
            continue
        component = WALKING_REWARD_REGISTRY.get(key, None)
        if component is None:
            weighted = _zeros_like_reward(ctx)
        else:
            weighted = weight * component(ctx)
        reward += weighted
        reward_info[f"reward/{key}"] = weighted

    # 最终保护
    reward = jp.clip(reward, -10.0, 10.0)
    reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

    return reward, reward_info


# =============================================================================================
# ===================================== END: 完整奖励函数 =======================================
# =============================================================================================
