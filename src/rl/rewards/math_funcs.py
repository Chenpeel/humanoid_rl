"""奖励相关的纯数学/纯函数组件（可复用 building blocks）。

说明：
- 本文件仅包含**无副作用**的纯函数：给定输入张量，返回 reward/penalty 标量或向量。
- 站立/行走等任务奖励通过组合这些函数实现，方便复用与课程学习。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jp


def quat_conjugate(quat: jax.Array) -> jax.Array:
    """wxyz 四元数共轭。"""
    quat = jp.asarray(quat)
    return jp.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """wxyz 四元数 Hamilton 乘：`q = q1 * q2`。"""
    q1 = jp.asarray(q1)
    q2 = jp.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jp.stack([w, x, y, z], axis=-1)


def quat_rotate(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """用 wxyz 四元数 `quat` 旋转向量 `vec`（xyz）。"""
    quat = normalize_quaternion(jp.asarray(quat))
    vec = jp.asarray(vec)
    vec_quat = jp.concatenate(
        [jp.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype), vec], axis=-1)
    rotated = quat_mul(quat_mul(quat, vec_quat), quat_conjugate(quat))
    return rotated[..., 1:]


def quat_rotate_inverse(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """用 wxyz 四元数 `quat` 的逆旋转向量 `vec`（xyz）。"""
    quat = normalize_quaternion(jp.asarray(quat))
    vec = jp.asarray(vec)
    vec_quat = jp.concatenate(
        [jp.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype), vec], axis=-1)
    rotated = quat_mul(quat_mul(quat_conjugate(quat), vec_quat), quat)
    return rotated[..., 1:]


def projected_gravity_in_base(base_quat: jax.Array) -> jax.Array:
    """将世界重力向量投影到基座坐标系（wxyz 四元数）。

    Returns:
        `g_base`，shape 为 `(..., 3)`；世界系重力假定为 `[0, 0, -1]`。
    """
    gravity_world = jp.array([0.0, 0.0, -1.0], dtype=jp.float32)
    return quat_rotate_inverse(base_quat, gravity_world)


def quat_angle(quat: jax.Array) -> jax.Array:
    """返回 wxyz 四元数 `quat` 表示的旋转角（rad）。

    与 VideoMimic 常见写法一致：`2 * asin(||q_xyz||)`（带 clamp）。
    """
    quat = normalize_quaternion(jp.asarray(quat))
    vec_norm = jp.linalg.norm(quat[..., 1:], axis=-1)
    vec_norm = jp.clip(vec_norm, a_min=0.0, a_max=1.0)
    return 2.0 * jp.arcsin(vec_norm)


def quat_heading(quat: jax.Array) -> jax.Array:
    """从 wxyz 四元数中提取 heading（yaw）角。"""
    euler = quat_to_euler(normalize_quaternion(jp.asarray(quat)))
    return wrap_to_pi(euler[..., 2])


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """wxyz 四元数转 Euler 角 `[roll, pitch, yaw]`。"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp)
                     * jp.pi / 2, jp.arcsin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)
    return jp.stack([roll, pitch, yaw], axis=-1)


def normalize_quaternion(quat: jax.Array) -> jax.Array:
    """四元数归一化（wxyz）。"""
    quat = jp.asarray(quat)
    norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    default = jp.broadcast_to(
        jp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype), quat.shape)
    mask = norm > 1e-8
    return jp.where(mask, quat / norm, default)


def wrap_to_pi(angles: jax.Array) -> jax.Array:
    """将角度 wrap 到 `[-pi, pi]`。"""
    return jp.arctan2(jp.sin(angles), jp.cos(angles))


def compute_action_rate_penalty(action: jax.Array, last_action: jax.Array) -> jax.Array:
    """动作变化惩罚（相邻步 action 差分平方和）。"""
    return jp.sum(jp.square(action - last_action), axis=-1)


def compute_torque_penalty(torques: jax.Array) -> jax.Array:
    """Torque 幅值惩罚（平方和）。"""
    return jp.sum(jp.square(torques), axis=-1)


def compute_joint_deviation_penalty(
    joint_pos: jax.Array,
    joint_pos_default: jax.Array,
    indices: Optional[jax.Array] = None,
) -> jax.Array:
    """关节偏离默认/站立姿态的惩罚。"""
    joint_pos = jp.asarray(joint_pos)
    joint_pos_default = jp.asarray(joint_pos_default)
    if indices is not None:
        indices = jp.asarray(indices)
        joint_pos = joint_pos[..., indices]
        joint_pos_default = joint_pos_default[..., indices]
    diff = joint_pos - joint_pos_default
    return jp.mean(jp.square(diff), axis=-1)


def compute_lin_vel_penalty(base_linvel: jax.Array) -> jax.Array:
    """线速度惩罚（x,y,z 全轴平方和）。"""
    return jp.sum(jp.square(base_linvel), axis=-1)


def compute_lin_vel_xy_penalty(base_linvel: jax.Array) -> jax.Array:
    """水平线速度惩罚（x,y 平面平方和）。"""
    return jp.sum(jp.square(base_linvel[..., :2]), axis=-1)


def compute_ang_vel_penalty(base_angvel: jax.Array) -> jax.Array:
    """角速度惩罚（平方和）。"""
    return jp.sum(jp.square(base_angvel), axis=-1)


def compute_velocity_penalty(base_linvel: jax.Array, base_angvel: jax.Array) -> jax.Array:
    """线速度与角速度的合成惩罚（x,y,z 全轴）。"""
    return compute_lin_vel_penalty(base_linvel) + compute_ang_vel_penalty(base_angvel)


def compute_feet_air_time_reward(
    contact_history: jax.Array, target_duty_cycle: float = 0.5
) -> jax.Array:
    """脚部空中时间/占空比奖励（窗口近似）。

    兼容 4 触点传感器： [right_foot, right_toe, left_foot, left_toe]，
    会将 toe/foot 合并为每只脚的接触状态，再计算 duty cycle 偏差。
    """
    hist = jp.asarray(contact_history)
    if hist.shape[-1] == 4:
        right = jp.maximum(hist[..., 0], hist[..., 1])
        left = jp.maximum(hist[..., 2], hist[..., 3])
        hist = jp.stack([right, left], axis=-1)
    elif hist.shape[-1] == 2:
        pass
    else:
        return jp.zeros(hist.shape[:-2], dtype=hist.dtype)

    actual_duty_cycle = jp.mean(hist, axis=-2)
    duty_error = jp.mean(
        jp.abs(actual_duty_cycle - target_duty_cycle), axis=-1)
    return jp.exp(-duty_error / 0.2)


def get_feet_contacts(contact_sensors: jax.Array, threshold: float = 1.0) -> jax.Array:
    """提取脚部接触状态 [right_foot, left_foot].

    兼容 4 触点传感器： [right_foot, right_toe, left_foot, left_toe]。
    """
    contact_sensors = jp.asarray(contact_sensors)
    if contact_sensors.shape[-1] == 4:
        right_contact = jp.maximum(
            contact_sensors[..., 0], contact_sensors[..., 1])
        left_contact = jp.maximum(
            contact_sensors[..., 2], contact_sensors[..., 3])
        return jp.stack([right_contact > threshold, left_contact > threshold], axis=-1)
    return contact_sensors > threshold


def compute_height_reward(
    torso_z: jax.Array, target_height: float, tolerance: float = 0.05
) -> jax.Array:
    """高度保持奖励（指数衰减）。"""
    height_error = jp.abs(jp.asarray(torso_z) - target_height)
    return jp.exp(-height_error / tolerance)


def compute_orientation_reward(base_quat: jax.Array, tolerance: float = 0.1) -> jax.Array:
    """姿态稳定奖励（最小化 Roll/Pitch）。"""
    euler = quat_to_euler(normalize_quaternion(jp.asarray(base_quat)))
    roll, pitch = euler[..., 0], euler[..., 1]
    error = jp.sqrt(jp.square(roll) + jp.square(pitch))
    return jp.exp(-error / tolerance)


def compute_forward_velocity_reward(
    base_linvel: jax.Array, target_velocity: float, tolerance: float = 0.5
) -> jax.Array:
    """前向速度跟踪奖励（exp(-|e|/σ)）。"""
    forward_vel = jp.asarray(base_linvel)[..., 0]
    vel_error = jp.abs(forward_vel - target_velocity)
    return jp.exp(-vel_error / tolerance)


def compute_velocity_tracking_reward(
    actual_velocity: jax.Array,
    command: jax.Array,
    tracking_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    tolerance: float = 0.25,
) -> jax.Array:
    """多轴速度跟踪奖励 [vx, vy, vyaw]（exp(-e^2/σ)）。"""
    error = jp.asarray(actual_velocity) - jp.asarray(command)
    reward_x = jp.exp(-jp.square(error[..., 0]) / tolerance)
    reward_y = jp.exp(-jp.square(error[..., 1]) / tolerance)
    reward_yaw = jp.exp(-jp.square(error[..., 2]) / tolerance)
    wx, wy, wyaw = tracking_weights
    return (wx * reward_x + wy * reward_y + wyaw * reward_yaw) / (wx + wy + wyaw)


def compute_gait_symmetry_reward(
    contacts: jax.Array, contact_history: Optional[jax.Array] = None
) -> jax.Array:
    """步态对称性奖励（近似 single-contact）。

    默认使用当前时刻 XOR(contact_left, contact_right)。
    若提供 `contact_history`，则加入短窗口容错：窗口内任意时刻出现过 single-contact 即视为满足。
    """
    right_contact = jp.asarray(contacts)[..., 0].astype(jp.float32)
    left_contact = jp.asarray(contacts)[..., 1].astype(jp.float32)
    xor_now = jp.abs(right_contact - left_contact)

    if contact_history is None:
        return xor_now

    hist = jp.asarray(contact_history)
    if hist.shape[-1] == 4:
        hist_contacts = get_feet_contacts(hist, threshold=0.5)
        right_hist = hist_contacts[..., 0].astype(jp.float32)
        left_hist = hist_contacts[..., 1].astype(jp.float32)
    elif hist.shape[-1] == 2:
        right_hist = (hist[..., 0] > 0.5).astype(jp.float32)
        left_hist = (hist[..., 1] > 0.5).astype(jp.float32)
    else:
        return xor_now

    xor_hist = jp.abs(right_hist - left_hist)
    recent_single = jp.max(xor_hist, axis=-1)
    return jp.maximum(xor_now, recent_single)


def compute_foot_clearance_reward(
    feet_positions: jax.Array,
    contacts: jax.Array,
    target_clearance: float = 0.05,
    tolerance: float = 0.02,
) -> jax.Array:
    """脚部抬高奖励（摆动相）。"""
    feet_positions = jp.asarray(feet_positions)
    contacts = jp.asarray(contacts)
    feet_heights = feet_positions[..., 2]
    swing_phase = 1.0 - contacts.astype(jp.float32)
    height_error = jp.abs(feet_heights - target_clearance)
    clearance_reward = jp.exp(-height_error / tolerance) * swing_phase
    return jp.sum(clearance_reward, axis=-1)


def compute_trunk_height_reward(
    torso_z: jax.Array, target_height: float = 0.78, tolerance: float = 0.08
) -> jax.Array:
    """躯干高度保持奖励。"""
    height_error = jp.abs(jp.asarray(torso_z) - target_height)
    return jp.exp(-height_error / tolerance)


def compute_upright_bonus(base_quat: jax.Array, threshold: float = 0.93) -> jax.Array:
    """直立姿态奖励（z 轴投影阈值）。"""
    base_quat = normalize_quaternion(jp.asarray(base_quat))
    qx, qy = base_quat[..., 1], base_quat[..., 2]
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return jp.where(z_z > threshold, 1.0, 0.0)


def compute_stability_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    target_height: float = 0.78,
    height_tolerance: float = 0.05,
) -> jax.Array:
    """综合稳定性奖励（高度 + 角速度）。"""
    height_reward = jp.exp(-jp.abs(jp.asarray(torso_z) -
                           target_height) / height_tolerance)
    ang_vel_reward = jp.exp(-jp.linalg.norm(jp.asarray(base_angvel), axis=-1))
    return (height_reward + ang_vel_reward) / 2.0


def compute_normalized_torque_penalty(
    torques: jax.Array,
    joint_names: Optional[List[str]] = None,
) -> jax.Array:
    """归一化 Torque 惩罚（Jiyuan 默认 16 DoF 配置）。

    说明：此函数保留原 walking_rewards.py 的行为（使用固定 torque_limits），用于向后兼容。
    """
    torques = jp.asarray(torques)
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
    normalized_torques = torques / torque_limits
    return jp.sum(jp.square(normalized_torques), axis=-1)


def compute_energy_efficiency_reward(torques: jax.Array, joint_vel: jax.Array) -> jax.Array:
    """能量效率惩罚（|Torque * joint_vel| 的和）。"""
    torques = jp.asarray(torques)
    joint_vel = jp.asarray(joint_vel)
    n = min(torques.shape[-1], joint_vel.shape[-1])
    power = jp.abs(torques[..., :n] * joint_vel[..., :n])
    return jp.sum(power, axis=-1)


def compute_trunk_orientation_penalty(quat: jax.Array, max_tilt: float = 0.3) -> jax.Array:
    """躯干过度倾斜惩罚（roll/pitch 超出 max_tilt 的超额部分）。"""
    euler = quat_to_euler(normalize_quaternion(jp.asarray(quat)))
    roll_penalty = jp.clip(jp.abs(euler[..., 0]) - max_tilt, a_min=0.0)
    pitch_penalty = jp.clip(jp.abs(euler[..., 1]) - max_tilt, a_min=0.0)
    return roll_penalty + pitch_penalty


def compute_drag_penalty(
    feet_positions: jax.Array, contacts: jax.Array, threshold: float = 0.02
) -> jax.Array:
    """拖地惩罚（摆动相脚高度过低）。"""
    feet_positions = jp.asarray(feet_positions)
    contacts = jp.asarray(contacts)
    is_dragging = (feet_positions[..., 2] < threshold) * \
        (1.0 - contacts.astype(jp.float32))
    return jp.sum(is_dragging, axis=-1)


def compute_feet_slide_penalty(
    feet_linvel: jax.Array, contacts: jax.Array, threshold: float = 0.1
) -> jax.Array:
    """脚部滑动惩罚。"""
    feet_linvel = jp.asarray(feet_linvel)
    contacts = jp.asarray(contacts)
    slide_vel = jp.linalg.norm(feet_linvel[..., :2], axis=-1)
    is_sliding = (slide_vel > threshold) * (contacts > 0.1)
    return jp.sum(is_sliding.astype(jp.float32) * slide_vel) / jp.maximum(feet_linvel.shape[-2], 1)


def compute_feet_contact_forces_reward(
    contact_sensors: jax.Array, target_force: float = 50.0, tolerance: float = 20.0
) -> jax.Array:
    """脚部接触力奖励（鼓励合理接触力；touch sensor 标量版）。"""
    contact_sensors = jp.asarray(contact_sensors)
    force_error = jp.abs(contact_sensors - target_force)
    reward = jp.exp(-force_error / tolerance) * (contact_sensors > 1.0)
    return jp.mean(reward, axis=-1)


def compute_joint_symmetry_reward(
    joint_pos: jax.Array, tolerance: float = 0.1, mirror_signs: Optional[jax.Array] = None
) -> jax.Array:
    """关节对称性奖励（左右腿镜像对称）。"""
    joint_pos = jp.asarray(joint_pos)
    if joint_pos.shape[-1] % 2 != 0:
        symmetry_error = jp.mean(jp.abs(joint_pos), axis=-1)
        return jp.exp(-symmetry_error / tolerance)

    num_joints_per_leg = joint_pos.shape[-1] // 2
    right_leg = joint_pos[..., :num_joints_per_leg]
    left_leg = joint_pos[..., num_joints_per_leg:]

    if mirror_signs is None:
        if num_joints_per_leg == 8:
            mirror_signs = jp.array(
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        else:
            mirror_signs = jp.ones((num_joints_per_leg,))
    else:
        mirror_signs = jp.asarray(mirror_signs, dtype=jp.float32)

    broadcast_shape = (1,) * (right_leg.ndim - 1) + (num_joints_per_leg,)
    mirror_signs = mirror_signs.reshape(broadcast_shape)
    left_leg_mirrored = left_leg * mirror_signs

    symmetry_error = jp.mean(jp.abs(right_leg - left_leg_mirrored), axis=-1)
    return jp.exp(-symmetry_error / tolerance)


def compute_stumbling_penalty(
    feet_positions: jax.Array,
    feet_velocities: jax.Array,
    contacts: jax.Array,
    threshold: float = 0.05,
) -> jax.Array:
    """绊倒惩罚（脚高度低且水平速度大，且处于摆动相）。"""
    feet_positions = jp.asarray(feet_positions)
    feet_velocities = jp.asarray(feet_velocities)
    contacts = jp.asarray(contacts)
    feet_heights = feet_positions[..., 2]
    feet_horizontal_vel = jp.linalg.norm(feet_velocities[..., :2], axis=-1)
    is_stumbling = (
        (feet_heights < threshold)
        * (feet_horizontal_vel > 0.5)
        * (1.0 - contacts.astype(jp.float32))
    )
    return jp.sum(is_stumbling, axis=-1)


def compute_landing_impact_penalty(
    contact_sensors: jax.Array,
    prev_contacts: jax.Array,
    contact_threshold: float = 1.0,
    impact_force_threshold: float = 100.0,
) -> jax.Array:
    """着陆冲击惩罚（基于接触“状态”变化）。"""
    contact_sensors = jp.asarray(contact_sensors)
    prev_contacts = jp.asarray(prev_contacts)
    current_contacts = contact_sensors > contact_threshold
    prev_contacts = prev_contacts > 0.5
    new_contacts = current_contacts & (~prev_contacts)
    impact_force = contact_sensors * new_contacts.astype(contact_sensors.dtype)
    return -jp.sum(jp.clip(impact_force - impact_force_threshold, a_min=0.0), axis=-1)


def compute_knee_bend_reward(
    joint_pos: jax.Array,
    target_bend: float = 0.4,
    tolerance: float = 0.25,
    knee_indices: Tuple[int, int] = (3, 11),
) -> jax.Array:
    """膝关节弯曲奖励：避免“锁死膝盖”的僵硬站立。

    Notes:
    - 使用 `|knee|` 以兼容左右腿相反的符号约定。
    - `knee_indices` 假设每条腿 8 个驱动关节顺序为：
      `[hip_pitch, hip_yaw, hip_roll, knee, ankle_pitch, ankle_roll, ankle_yaw, toe] * 2`。
    """
    joint_pos = jp.asarray(joint_pos)
    if joint_pos.shape[-1] <= max(knee_indices):
        return jp.zeros(joint_pos.shape[:-1], dtype=joint_pos.dtype)

    knees = joint_pos[..., jp.array(knee_indices)]
    bend = jp.abs(knees)
    bend_error = jp.mean(jp.abs(bend - target_bend), axis=-1)
    return jp.exp(-bend_error / tolerance)


def compute_toe_only_contact_penalty(
    contact_sensors: jax.Array,
    threshold: float = 1.0,
) -> jax.Array:
    """Toe-only 支撑惩罚（只有 toe 接触，foot 未接触）。

    假设 4 个 touch sensors 顺序为：
    `[right_foot, right_toe, left_foot, left_toe]`。
    返回范围约为 `[0, 1]`（对双脚取 mean）。
    """
    contact_sensors = jp.asarray(contact_sensors)
    if contact_sensors.shape[-1] != 4:
        return jp.zeros(contact_sensors.shape[:-1], dtype=contact_sensors.dtype)

    right_foot = contact_sensors[..., 0]
    right_toe = contact_sensors[..., 1]
    left_foot = contact_sensors[..., 2]
    left_toe = contact_sensors[..., 3]

    right_toe_only = (right_toe > threshold) & (right_foot <= threshold)
    left_toe_only = (left_toe > threshold) & (left_foot <= threshold)
    toe_only = jp.stack(
        [right_toe_only.astype(jp.float32), left_toe_only.astype(jp.float32)], axis=-1
    )
    return jp.mean(toe_only, axis=-1)
