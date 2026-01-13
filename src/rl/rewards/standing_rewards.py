"""站立平衡奖励函数（类式实现）。

本文件提供：
- `StandingRewards(BaseRewards)`：可复用的站立奖励项集合
- `compute_standing_reward`：对外保持旧的函数式 API，供 env 调用
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jp

from .base_rewards import BaseRewards
from .math_funcs import (
    compute_ang_vel_penalty,
    compute_height_reward,
    compute_joint_deviation_penalty,
    compute_lin_vel_penalty,
    compute_orientation_reward,
)

# =============================================================================================
# ======================================= 默认奖励权重 =========================================
# =============================================================================================

DEFAULT_STANDING_REWARD_WEIGHTS = {
    "height": 2.0,
    "orientation": 1.0,
    "lin_vel": -0.5,
    "ang_vel": -0.5,
    "alive": 1.0,
    "action_rate": -0.01,
    "torques": -0.001,
}

# =============================================================================================
# ======================================= 终止条件检查 ==========================================
# =============================================================================================


def check_standing_termination(
    torso_z: jax.Array,
    base_quat: jax.Array,
    height_threshold: float = 0.25,
    upright_threshold: float = 0.5,
) -> jax.Array:
    """检查站立是否终止（摔倒检测）。"""
    is_fallen_height = torso_z < height_threshold
    qx, qy = base_quat[..., 1], base_quat[..., 2]
    z_z = 1.0 - 2.0 * (qx**2 + qy**2)
    is_fallen_orientation = z_z < upright_threshold
    return is_fallen_height | is_fallen_orientation


# =============================================================================================
# ======================================= 完整奖励函数（组合器）=================================
# =============================================================================================


class StandingRewardContext(NamedTuple):
    torso_z: jax.Array
    base_quat: jax.Array
    base_linvel: jax.Array
    base_angvel: jax.Array
    action: Optional[jax.Array]
    last_action: Optional[jax.Array]
    torques: Optional[jax.Array]
    target_height: float
    joint_pos: Optional[jax.Array]
    joint_pos_default: Optional[jax.Array]


_HIP_INDICES = jp.array([0, 1, 2, 8, 9, 10])


class StandingRewards(BaseRewards):
    """站立任务奖励项集合。"""

    def _reward_height(self, ctx: StandingRewardContext) -> jax.Array:
        return compute_height_reward(ctx.torso_z, ctx.target_height)

    def _reward_orientation(self, ctx: StandingRewardContext) -> jax.Array:
        return compute_orientation_reward(ctx.base_quat)

    def _reward_lin_vel(self, ctx: StandingRewardContext) -> jax.Array:
        return compute_lin_vel_penalty(ctx.base_linvel)

    def _reward_ang_vel(self, ctx: StandingRewardContext) -> jax.Array:
        return compute_ang_vel_penalty(ctx.base_angvel)

    def _reward_joint_deviation(self, ctx: StandingRewardContext) -> jax.Array:
        if ctx.joint_pos is None or ctx.joint_pos_default is None:
            return self.zeros_like(ctx)
        return compute_joint_deviation_penalty(ctx.joint_pos, ctx.joint_pos_default)

    def _reward_hip_deviation(self, ctx: StandingRewardContext) -> jax.Array:
        if ctx.joint_pos is None or ctx.joint_pos_default is None:
            return self.zeros_like(ctx)
        return compute_joint_deviation_penalty(
            ctx.joint_pos, ctx.joint_pos_default, indices=_HIP_INDICES
        )


_STANDING_REWARDS = StandingRewards()


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
    joint_pos: Optional[jax.Array] = None,
    joint_pos_default: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算完整的站立平衡奖励（动态模式）。"""
    ctx = StandingRewardContext(
        torso_z=jp.asarray(torso_z),
        base_quat=base_quat,
        base_linvel=base_linvel,
        base_angvel=base_angvel,
        action=action,
        last_action=last_action,
        torques=torques,
        target_height=jp.asarray(target_height),
        joint_pos=joint_pos,
        joint_pos_default=joint_pos_default,
    )
    return _STANDING_REWARDS.compute(ctx, reward_weights)


# =============================================================================================
# ================================ 兼容导出：保持旧的 import 路径 ================================
# =============================================================================================

from . import math_funcs as _math_funcs

compute_action_rate_penalty = _math_funcs.compute_action_rate_penalty
compute_torque_penalty = _math_funcs.compute_torque_penalty
compute_velocity_penalty = _math_funcs.compute_velocity_penalty

normalize_quaternion = _math_funcs.normalize_quaternion
wrap_to_pi = _math_funcs.wrap_to_pi
quat_to_euler = _math_funcs.quat_to_euler
