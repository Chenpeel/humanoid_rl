"""兼容层：奖励组件已迁移至 `math_funcs.py`。

说明：
- 历史版本中常用 `rl.rewards.components` 存放可复用的纯函数奖励项/数学工具。
- 现在统一收敛到 `rl.rewards.math_funcs`，本文件仅保留旧 import 路径，避免外部代码断裂。
"""

from __future__ import annotations

from .math_funcs import (
    compute_action_rate_penalty,
    compute_ang_vel_penalty,
    compute_joint_deviation_penalty,
    compute_knee_bend_reward,
    compute_lin_vel_penalty,
    compute_lin_vel_xy_penalty,
    compute_toe_only_contact_penalty,
    compute_torque_penalty,
    normalize_quaternion,
    quat_to_euler,
    wrap_to_pi,
)

__all__ = [
    "quat_to_euler",
    "normalize_quaternion",
    "wrap_to_pi",
    "compute_action_rate_penalty",
    "compute_torque_penalty",
    "compute_joint_deviation_penalty",
    "compute_lin_vel_penalty",
    "compute_lin_vel_xy_penalty",
    "compute_ang_vel_penalty",
    "compute_knee_bend_reward",
    "compute_toe_only_contact_penalty",
]

