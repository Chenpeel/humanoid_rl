"""
Jiyuan 机器人观测函数（ManagerBasedRLEnv）

核心目标：把网络输入统一到 Z-up 语义坐标系。

背景：
- 历史原因，Jiyuan base 在 XML 中存在固定 90° 旋转，导致 body-frame 量（速度、重力投影等）轴向含义错位。
- 训练时如果 reward/termination 做了校正，但 policy 观测没校正，会造成输入/信号不一致，影响收敛与可解释性。
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..utils.math_utils import DEFAULT_BASE_QUAT_CORRECTION_WXYZ, apply_base_quat_correction_to_body_vec


def base_lin_vel_corrected(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """base 线速度（body frame，已校正到 Z-up 语义）。"""
    asset = env.scene[asset_name]
    vel_b = asset.data.root_lin_vel_b
    return apply_base_quat_correction_to_body_vec(vel_b, base_quat_correction)


def base_ang_vel_corrected(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """base 角速度（body frame，已校正到 Z-up 语义）。"""
    asset = env.scene[asset_name]
    ang_b = asset.data.root_ang_vel_b
    return apply_base_quat_correction_to_body_vec(ang_b, base_quat_correction)


def projected_gravity_corrected(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """重力投影（body frame，已校正到 Z-up 语义）。"""
    asset = env.scene[asset_name]
    gravity_b = asset.data.projected_gravity_b
    return apply_base_quat_correction_to_body_vec(gravity_b, base_quat_correction)

