"""行走任务奖励函数（类式实现）.

基于开源实现（Legged Gym, Isaac Lab）与 VideoMimic 中常见奖励项组织方式，
使用 `BaseRewards` 的 `_reward_<key>` 方法自动收集机制，按 `reward_weights` 动态组合。

注意：
- 终止（摔倒）判定由 env 的 `_is_done` 决定；本文件仅提供 `check_walking_termination` 供 env 调用。
- reward 项的正负由 `reward_weights` 的符号控制。
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jp

from .base_rewards import BaseRewards
from . import math_funcs

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
# ================================ 三阶段课程学习权重配置 ========================================
# ============================================================================================

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

# =============================================================================================
# ========================================= 终止条件检查 ========================================
# =============================================================================================


def check_walking_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.25,
    upright_threshold: float = 0.5,
) -> jax.Array:
    """检查行走是否终止（摔倒检测）."""
    is_fallen_height = torso_z < height_threshold
    qx, qy = base_quat[..., 1], base_quat[..., 2]
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    is_fallen_orientation = z_z < upright_threshold
    return is_fallen_height | is_fallen_orientation


# =============================================================================================
# ======================================= Reward Context =======================================
# =============================================================================================


class WalkingRewardContext(NamedTuple):
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


# =============================================================================================
# ======================================= Reward Class =========================================
# =============================================================================================


class WalkingRewards(BaseRewards):
    """行走任务奖励项集合。"""

    # ---------- 组合逻辑（保留 velocity_tracking 与 forward_velocity 的“二选一”旧行为） ----------
    def compute(
        self,
        ctx: WalkingRewardContext,
        reward_weights: Dict[str, float],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        weights = reward_weights or DEFAULT_WALKING_REWARD_WEIGHTS

        reward, reward_info = super().compute(
            ctx, weights, exclude_keys=("velocity_tracking", "forward_velocity")
        )

        vt_w = weights.get("velocity_tracking", 0.0)
        fv_w = weights.get("forward_velocity", 0.0)

        if vt_w != 0.0 and ctx.command is not None:
            if ctx.actual_velocity is None:
                val = self.zeros_like(ctx)
            else:
                val = math_funcs.compute_velocity_tracking_reward(ctx.actual_velocity, ctx.command)
            weighted = vt_w * val
            reward += weighted
            reward_info["reward/velocity_tracking"] = weighted
        elif fv_w != 0.0:
            val = math_funcs.compute_forward_velocity_reward(ctx.base_linvel, ctx.target_velocity)
            weighted = fv_w * val
            reward += weighted
            reward_info["reward/forward_velocity"] = weighted

        reward = jp.clip(reward, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        return reward, reward_info

    # ---------- posture / stability ----------
    def _reward_trunk_height(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_trunk_height_reward(ctx.torso_z, ctx.target_height)

    def _reward_orientation(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_trunk_orientation_penalty(ctx.base_quat)

    def _reward_upright_bonus(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_upright_bonus(ctx.base_quat)

    def _reward_stability(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_stability_reward(
            ctx.torso_z, ctx.base_quat, ctx.base_linvel, ctx.base_angvel, ctx.target_height
        )

    def _reward_knee_bend(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.joint_pos is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_knee_bend_reward(ctx.joint_pos)

    def _reward_double_support(self, ctx: WalkingRewardContext) -> jax.Array:
        # Prefer "foot-only" contacts (not toe) when 4 touch sensors are available:
        # [right_foot, right_toe, left_foot, left_toe].
        # Use a short grace window via `contact_history` to avoid penalizing brief recovery steps.
        if ctx.contact_history is not None and getattr(ctx.contact_history, "shape", None) is not None:
            hist = jp.asarray(ctx.contact_history)
            if hist.shape[-1] == 4:
                right_foot = hist[..., 0]
                left_foot = hist[..., 2]
                time_axis = -2 if right_foot.ndim >= 2 else -1
                right_any = (jp.max(right_foot, axis=time_axis) > 0.5).astype(jp.float32)
                left_any = (jp.max(left_foot, axis=time_axis) > 0.5).astype(jp.float32)
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

    def _reward_toe_only(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.contact_history is not None and getattr(ctx.contact_history, "shape", None) is not None:
            hist = jp.asarray(ctx.contact_history)
            if hist.shape[-1] == 4:
                toe_only = math_funcs.compute_toe_only_contact_penalty(hist, threshold=0.5)
                return jp.mean(toe_only, axis=-1)

        if ctx.contact_sensors is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_toe_only_contact_penalty(ctx.contact_sensors)

    # ---------- gait quality ----------
    def _reward_gait_symmetry(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_gait_symmetry_reward(ctx.contacts, ctx.contact_history)

    def _reward_foot_clearance(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.feet_positions is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_foot_clearance_reward(ctx.feet_positions, ctx.contacts)

    def _reward_feet_air_time(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.contact_history is None:
            return self.zeros_like(ctx)
        rew = math_funcs.compute_feet_air_time_reward(ctx.contact_history)
        if ctx.command is None:
            return rew
        speed = jp.linalg.norm(jp.asarray(ctx.command)[..., :2], axis=-1)
        return rew * (speed > 0.1).astype(jp.float32)

    def _reward_feet_contact_forces(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.contact_sensors is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_feet_contact_forces_reward(ctx.contact_sensors)

    def _reward_feet_slide(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.feet_velocities is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_feet_slide_penalty(ctx.feet_velocities, ctx.contacts)

    def _reward_stumbling(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.feet_positions is None or ctx.feet_velocities is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_stumbling_penalty(ctx.feet_positions, ctx.feet_velocities, ctx.contacts)

    def _reward_landing_impact(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.contact_history is None or ctx.contact_sensors is None:
            return self.zeros_like(ctx)
        hist = jp.asarray(ctx.contact_history)
        prev_contacts = hist[..., -1, :]
        return math_funcs.compute_landing_impact_penalty(ctx.contact_sensors, prev_contacts)

    # ---------- regularization / penalties ----------
    def _reward_lin_vel(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_lin_vel_xy_penalty(ctx.base_linvel)

    def _reward_ang_vel(self, ctx: WalkingRewardContext) -> jax.Array:
        return math_funcs.compute_ang_vel_penalty(ctx.base_angvel)

    def _reward_drag(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.feet_positions is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_drag_penalty(ctx.feet_positions, ctx.contacts)

    def _reward_joint_limits(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.joint_pos is None or ctx.joint_limits is None:
            return self.zeros_like(ctx)
        lower_limits, upper_limits = ctx.joint_limits
        lower_violation = jp.maximum(0.0, jp.asarray(lower_limits) - jp.asarray(ctx.joint_pos))
        upper_violation = jp.maximum(0.0, jp.asarray(ctx.joint_pos) - jp.asarray(upper_limits))
        return jp.sum(jp.square(lower_violation) + jp.square(upper_violation), axis=-1)

    def _reward_trunk_lin_vel_z(self, ctx: WalkingRewardContext) -> jax.Array:
        return jp.square(ctx.base_linvel[..., 2])

    def _reward_joint_symmetry(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.joint_pos is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_joint_symmetry_reward(ctx.joint_pos)

    def _reward_joint_deviation(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.joint_pos is None or ctx.joint_pos_default is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_joint_deviation_penalty(ctx.joint_pos, ctx.joint_pos_default)

    def _reward_hip_deviation(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.joint_pos is None or ctx.joint_pos_default is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_joint_deviation_penalty(
            ctx.joint_pos, ctx.joint_pos_default, indices=_HIP_INDICES
        )

    def _reward_energy_efficiency(self, ctx: WalkingRewardContext) -> jax.Array:
        if ctx.torques is None or ctx.joint_vel is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_energy_efficiency_reward(ctx.torques, ctx.joint_vel)


_WALKING_REWARDS = WalkingRewards()


def compute_walking_reward(
    torso_z: jax.Array,
    base_quat: jax.Array,
    base_linvel: jax.Array,
    base_angvel: jax.Array,
    contact_sensors: jax.Array,
    feet_positions: Optional[jax.Array] = None,
    action: Optional[jax.Array] = None,
    last_action: Optional[jax.Array] = None,
    torques: Optional[jax.Array] = None,
    feet_velocities: Optional[jax.Array] = None,
    joint_pos: Optional[jax.Array] = None,
    joint_vel: Optional[jax.Array] = None,
    joint_pos_default: Optional[jax.Array] = None,
    joint_limits: Optional[tuple] = None,
    contact_history: Optional[jax.Array] = None,
    command: Optional[jax.Array] = None,
    actual_velocity: Optional[jax.Array] = None,
    target_velocity: float = 0.5,
    target_height: float = 0.78,
    reward_weights: Optional[Dict[str, float]] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的行走任务奖励（保持旧函数式 API）。"""
    base_quat = math_funcs.normalize_quaternion(base_quat)
    contacts = math_funcs.get_feet_contacts(contact_sensors)
    weights = reward_weights or DEFAULT_WALKING_REWARD_WEIGHTS

    ctx = WalkingRewardContext(
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
        target_velocity=float(target_velocity),
        target_height=float(target_height),
    )

    return _WALKING_REWARDS.compute(ctx, weights)


# =============================================================================================
# ================================ 兼容导出：保持旧的 import 路径 ================================
# =============================================================================================

get_feet_contacts = math_funcs.get_feet_contacts
normalize_quaternion = math_funcs.normalize_quaternion
wrap_to_pi = math_funcs.wrap_to_pi

quat_to_euler = math_funcs.quat_to_euler

compute_forward_velocity_reward = math_funcs.compute_forward_velocity_reward
compute_velocity_tracking_reward = math_funcs.compute_velocity_tracking_reward
compute_gait_symmetry_reward = math_funcs.compute_gait_symmetry_reward
compute_foot_clearance_reward = math_funcs.compute_foot_clearance_reward
compute_feet_air_time_reward = math_funcs.compute_feet_air_time_reward
compute_trunk_height_reward = math_funcs.compute_trunk_height_reward
compute_upright_bonus = math_funcs.compute_upright_bonus
compute_stability_reward = math_funcs.compute_stability_reward
compute_trunk_orientation_penalty = math_funcs.compute_trunk_orientation_penalty
compute_drag_penalty = math_funcs.compute_drag_penalty
compute_feet_slide_penalty = math_funcs.compute_feet_slide_penalty
compute_normalized_torque_penalty = math_funcs.compute_normalized_torque_penalty
compute_energy_efficiency_reward = math_funcs.compute_energy_efficiency_reward
