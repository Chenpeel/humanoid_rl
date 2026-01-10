"""
站立平衡奖励函数

设计用于 Jiyuan 机器人的基础平衡训练阶段。
侧重于高度保持、垂直姿态和最小化关节动作。
"""

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jp

from .components import (compute_action_rate_penalty, compute_ang_vel_penalty,
                         compute_joint_deviation_penalty as _compute_joint_deviation_penalty,
                         compute_lin_vel_penalty, compute_torque_penalty,
                         normalize_quaternion, quat_to_euler)

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
    """
    检查站立是否终止 (摔倒检测)

    Args:
        torso_z: 躯干高度
        base_quat: 基座四元数
        height_threshold: 最小高度阈值 (m)
        upright_threshold: 最小直立度阈值 (z-axis projection)

    Returns:
        bool array, True 表示终止
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
# ======================================= 基础奖励分量 ===========================================
# =============================================================================================


def compute_height_reward(
    torso_z: jax.Array, target_height: float, tolerance: float = 0.05
) -> jax.Array:
    """计算高度保持奖励"""
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_orientation_reward(
    base_quat: jax.Array, tolerance: float = 0.1
) -> jax.Array:
    """计算姿态稳定奖励（最小化 Roll 和 Pitch）"""
    euler = quat_to_euler(normalize_quaternion(base_quat))
    roll, pitch = euler[..., 0], euler[..., 1]
    error = jp.sqrt(jp.square(roll) + jp.square(pitch))
    return jp.exp(-error / tolerance)


# ---------------------------------------------------------------------------------------------


def compute_velocity_penalties(
    base_linvel: jax.Array, base_angvel: jax.Array
) -> Dict[str, jax.Array]:
    """计算速度惩罚（鼓励静止）"""
    return {
        "lin_vel_penalty": compute_lin_vel_penalty(base_linvel),
        "ang_vel_penalty": compute_ang_vel_penalty(base_angvel),
    }


# ---------------------------------------------------------------------------------------------


def compute_velocity_penalty(
    base_linvel: jax.Array, base_angvel: jax.Array
) -> jax.Array:
    """计算总速度惩罚 (Wrapper for compatibility)"""
    penalties = compute_velocity_penalties(base_linvel, base_angvel)
    return penalties["lin_vel_penalty"] + penalties["ang_vel_penalty"]


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


# =============================================================================================
# ===================================== END: 基础奖励分量 =======================================
# =============================================================================================


# =============================================================================================
# ======================================= 完整奖励函数 ==========================================
# =============================================================================================


class _StandingRewardContext(NamedTuple):
    torso_z: jax.Array
    base_quat: jax.Array
    base_linvel: jax.Array
    base_angvel: jax.Array
    action: jax.Array
    last_action: jax.Array
    torques: jax.Array
    target_height: float
    joint_pos: Optional[jax.Array]
    joint_pos_default: Optional[jax.Array]


_HIP_INDICES = jp.array([0, 1, 2, 8, 9, 10])


def _zeros_like_reward(ctx: _StandingRewardContext) -> jax.Array:
    return jp.zeros_like(ctx.torso_z)


def _ones_like_reward(ctx: _StandingRewardContext) -> jax.Array:
    return jp.ones_like(ctx.torso_z)


def _standing_height(ctx: _StandingRewardContext) -> jax.Array:
    return compute_height_reward(ctx.torso_z, ctx.target_height)


def _standing_orientation(ctx: _StandingRewardContext) -> jax.Array:
    return compute_orientation_reward(ctx.base_quat)


def _standing_lin_vel(ctx: _StandingRewardContext) -> jax.Array:
    return compute_lin_vel_penalty(ctx.base_linvel)


def _standing_ang_vel(ctx: _StandingRewardContext) -> jax.Array:
    return compute_ang_vel_penalty(ctx.base_angvel)


def _standing_alive(ctx: _StandingRewardContext) -> jax.Array:
    return _ones_like_reward(ctx)


def _standing_action_rate(ctx: _StandingRewardContext) -> jax.Array:
    return compute_action_rate_penalty(ctx.action, ctx.last_action)


def _standing_torques(ctx: _StandingRewardContext) -> jax.Array:
    return compute_torque_penalty(ctx.torques)


def _standing_joint_deviation(ctx: _StandingRewardContext) -> jax.Array:
    if ctx.joint_pos is None or ctx.joint_pos_default is None:
        return _zeros_like_reward(ctx)
    return compute_joint_deviation_penalty(ctx.joint_pos, ctx.joint_pos_default)


def _standing_hip_deviation(ctx: _StandingRewardContext) -> jax.Array:
    if ctx.joint_pos is None or ctx.joint_pos_default is None:
        return _zeros_like_reward(ctx)
    return compute_joint_deviation_penalty(
        ctx.joint_pos, ctx.joint_pos_default, indices=_HIP_INDICES
    )


STANDING_REWARD_REGISTRY: Dict[str, Callable[[_StandingRewardContext], jax.Array]] = {
    "height": _standing_height,
    "orientation": _standing_orientation,
    "lin_vel": _standing_lin_vel,
    "ang_vel": _standing_ang_vel,
    "alive": _standing_alive,
    "action_rate": _standing_action_rate,
    "torques": _standing_torques,
    "joint_deviation": _standing_joint_deviation,
    "hip_deviation": _standing_hip_deviation,
}


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
    """计算完整的站立平衡奖励

    采用动态奖励组合模式：只计算reward_weights中配置的奖励项，
    提供良好的扩展性和课程学习支持。
    """

    reward = jp.array(0.0)
    reward_info = {f"reward/{k}": jp.array(0.0) for k in reward_weights.keys()}

    ctx = _StandingRewardContext(
        torso_z=jp.asarray(torso_z),
        base_quat=base_quat,
        base_linvel=base_linvel,
        base_angvel=base_angvel,
        action=action,
        last_action=last_action,
        torques=torques,
        target_height=target_height,
        joint_pos=joint_pos,
        joint_pos_default=joint_pos_default,
    )

    for key, weight in reward_weights.items():
        if weight == 0.0:
            continue
        component = STANDING_REWARD_REGISTRY.get(key, None)
        if component is None:
            weighted = _zeros_like_reward(ctx)
        else:
            weighted = weight * component(ctx)
        reward += weighted
        reward_info[f"reward/{key}"] = weighted

    return reward, reward_info


# =============================================================================================
# ===================================== END: 完整奖励函数 =======================================
# =============================================================================================
