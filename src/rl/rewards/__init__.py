"""
奖励函数模块

包含站立、行走等任务的奖励函数。
"""

from .standing_rewards import (
    DEFAULT_STANDING_REWARD_WEIGHTS,
    check_standing_termination,
    compute_height_reward,
    compute_orientation_reward,
    compute_standing_reward,
)
from .walking_rewards import (
    DEFAULT_WALKING_REWARD_WEIGHTS,
    check_walking_termination,
    compute_walking_reward,
)
from .math_funcs import (
    compute_action_rate_penalty,
    compute_drag_penalty,
    compute_energy_efficiency_reward,
    compute_foot_clearance_reward,
    compute_forward_velocity_reward,
    compute_gait_symmetry_reward,
    compute_normalized_torque_penalty,
    compute_torque_penalty,
    compute_trunk_height_reward,
    compute_trunk_orientation_penalty,
    compute_velocity_penalty,
    get_feet_contacts,
    normalize_quaternion,
    quat_to_euler,
    wrap_to_pi,
)

__all__ = [
    # 站立奖励
    "compute_standing_reward",
    "compute_height_reward",
    "compute_orientation_reward",
    "compute_velocity_penalty",
    "compute_action_rate_penalty",
    "compute_torque_penalty",
    "check_standing_termination",
    "DEFAULT_STANDING_REWARD_WEIGHTS",
    # 行走奖励
    "compute_walking_reward",
    "compute_forward_velocity_reward",
    "compute_gait_symmetry_reward",
    "compute_foot_clearance_reward",
    "compute_trunk_height_reward",
    "compute_trunk_orientation_penalty",
    "compute_drag_penalty",
    "compute_normalized_torque_penalty",
    "compute_energy_efficiency_reward",
    "check_walking_termination",
    "DEFAULT_WALKING_REWARD_WEIGHTS",
    # 工具函数
    "quat_to_euler",
    "normalize_quaternion",
    "wrap_to_pi",
    "get_feet_contacts",
]
