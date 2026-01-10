"""
Jiyuan 机器人奖励函数

从 JAX/MJX 迁移的奖励函数，使用 PyTorch 和 Isaac Lab 接口。

所有奖励函数遵循 Isaac Lab 规范：
- 第一个参数：env (ManagerBasedRLEnv)
- 返回值：torch.Tensor，形状 (num_envs,)
- 使用 env.scene["robot"].data 访问机器人状态

参考:
- 原始实现: src/rl/rewards/standing_rewards.py (jax 分支)
- Isaac Lab MDP: omni.isaac.lab.envs.mdp
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 导入数学工具
from ..utils.math_utils import (
    DEFAULT_BASE_QUAT_CORRECTION_WXYZ as _DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
    apply_base_quat_correction_to_body_vec,
    normalize_quaternion,
    remove_fixed_quat_rotation,
)


# Jiyuan 特有：历史原因 base frame 存在 90° 旋转（wxyz）。
# 在需要把观测/奖励对齐到 Z-up 语义坐标系时，使用 quat ⊗ fixed^{-1} 移除该旋转。
DEFAULT_BASE_QUAT_CORRECTION_WXYZ = _DEFAULT_BASE_QUAT_CORRECTION_WXYZ


try:  # Isaac Lab 运行时可用；纯 Python 环境下保持为 None
    from isaaclab.utils.math import quat_apply_inverse as _quat_apply_inverse
    from isaaclab.utils.math import yaw_quat as _yaw_quat
except Exception:  # pragma: no cover
    _quat_apply_inverse = None
    _yaw_quat = None


def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    asset_cfg,
    threshold: float = 1.0,
) -> Tensor:
    """脚部滑动惩罚（可配置阈值版）。

    参考 Isaac Lab locomotion `feet_slide`，但把接触阈值暴露为参数，便于 YAML 动态调参。

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置（SceneEntityCfg），应匹配脚部 bodies
        asset_cfg: 机器人资产配置（SceneEntityCfg），应匹配脚部 bodies
        threshold: 接触力阈值（N），越大越“严格”
    """
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_hist = getattr(contact_sensor.data, "net_forces_w_history", None)
    if forces_hist is None or sensor_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    contacts = forces_hist[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > float(threshold)

    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


##
# 高度和姿态奖励
##


def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.35,
    tolerance: float = 0.05,
) -> Tensor:
    """高度保持奖励

    机器人应该保持在目标高度附近。使用指数形式的奖励，高度越接近目标值奖励越高。

    Args:
        env: 环境实例
        target_height: 目标高度 (m)，默认 0.35m
        tolerance: 容差参数，控制奖励衰减速度

    Returns:
        高度奖励值，形状 (num_envs,)，范围 [0, 1]
    """
    # 获取机器人base的高度（z坐标）
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]

    # 计算高度误差
    height_error = torch.abs(torso_z - target_height)

    # 指数奖励
    return torch.exp(-height_error / tolerance)


def orientation_reward(
    env: ManagerBasedRLEnv,
    tolerance: float = 0.1,
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """姿态稳定奖励

    机器人应该保持直立（roll和pitch接近0）。

    Args:
        env: 环境实例
        tolerance: 容差参数

    Returns:
        姿态奖励值，形状 (num_envs,)，范围 [0, 1]
    """
    # 使用 projected_gravity_b（Isaac Lab 内部已缓存），避免每步创建 gravity_w 张量。
    asset = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b

    gravity_b = apply_base_quat_correction_to_body_vec(gravity_b, base_quat_correction)

    # 直立时 gravity_b ≈ [0,0,-1]，因此用 XY 分量的平方和作为 tilt 误差（数值稳定、无欧拉角奇异）。
    tilt_error = torch.sum(torch.square(gravity_b[:, :2]), dim=-1)
    return torch.exp(-tilt_error / tolerance)


##
# 速度惩罚
##


def lin_vel_penalty_l2(env: ManagerBasedRLEnv) -> Tensor:
    """线速度L2惩罚

    站立任务中，机器人不应该移动。

    Args:
        env: 环境实例

    Returns:
        线速度惩罚值，形状 (num_envs,)
    """
    # 获取base线速度
    base_lin_vel = env.scene["robot"].data.root_lin_vel_b  # body frame

    # L2范数
    return torch.sum(torch.square(base_lin_vel), dim=-1)


def ang_vel_penalty_l2(env: ManagerBasedRLEnv) -> Tensor:
    """角速度L2惩罚

    站立任务中，机器人不应该旋转。

    Args:
        env: 环境实例

    Returns:
        角速度惩罚值，形状 (num_envs,)
    """
    # 获取base角速度
    base_ang_vel = env.scene["robot"].data.root_ang_vel_b  # body frame

    # L2范数
    return torch.sum(torch.square(base_ang_vel), dim=-1)


def lin_vel_z_l2_corrected(
    env: ManagerBasedRLEnv,
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """Z 方向线速度惩罚（使用校正后的 body frame）。

    说明：Isaac Lab 的 `mdp.lin_vel_z_l2` 直接取 `root_lin_vel_b[:,2]`。
    对于 base 存在固定旋转的机器人，这个 “z” 轴含义会错位，必须先做校正。
    """
    asset = env.scene["robot"]
    vel_b = apply_base_quat_correction_to_body_vec(asset.data.root_lin_vel_b, base_quat_correction)
    return torch.square(vel_b[:, 2])


def ang_vel_xy_l2_corrected(
    env: ManagerBasedRLEnv,
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """XY 平面角速度惩罚（使用校正后的 body frame）。"""
    asset = env.scene["robot"]
    ang_b = apply_base_quat_correction_to_body_vec(asset.data.root_ang_vel_b, base_quat_correction)
    return torch.sum(torch.square(ang_b[:, :2]), dim=-1)


def xy_vel_penalty_l2(env: ManagerBasedRLEnv) -> Tensor:
    """XY平面速度惩罚

    惩罚水平方向的移动，允许少量垂直方向的振动。

    Args:
        env: 环境实例

    Returns:
        XY速度惩罚值，形状 (num_envs,)
    """
    # 使用 world frame 的 XY，避免受 base frame 固定旋转影响
    base_lin_vel_w = env.scene["robot"].data.root_lin_vel_w
    return torch.sum(torch.square(base_lin_vel_w[:, :2]), dim=-1)


##
# 速度跟踪奖励（用于velocity tracking任务）
##


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
) -> Tensor:
    """线速度跟踪奖励（XY平面）

    使用指数形式奖励，速度越接近目标奖励越高。

    Args:
        env: 环境实例
        std: 标准差参数，控制奖励衰减速度
        command_name: 命令名称

    Returns:
        速度跟踪奖励值，形状 (num_envs,)，范围 [0, 1]
    """
    # 获取实际速度
    base_lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]  # XY平面

    # 获取目标速度（从命令管理器）
    command = env.command_manager.get_command(command_name)
    target_vel = command[:, :2]  # 假设命令前2维是目标线速度

    # 计算速度误差
    vel_error = torch.sum(torch.square(base_lin_vel - target_vel), dim=-1)

    # 指数奖励
    return torch.exp(-vel_error / (std**2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
) -> Tensor:
    """角速度跟踪奖励（Yaw方向）

    Args:
        env: 环境实例
        std: 标准差参数
        command_name: 命令名称

    Returns:
        角速度跟踪奖励值，形状 (num_envs,)，范围 [0, 1]
    """
    # 获取实际角速度
    base_ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]  # Yaw方向

    # 获取目标角速度
    command = env.command_manager.get_command(command_name)
    target_ang_vel = command[:, 2]  # 假设命令第3维是目标角速度

    # 计算角速度误差
    ang_vel_error = torch.square(base_ang_vel - target_ang_vel)

    # 指数奖励
    return torch.exp(-ang_vel_error / (std**2))


##
# 动作平滑性奖励
##


def action_rate_l2(env: ManagerBasedRLEnv) -> Tensor:
    """动作变化率L2惩罚

    鼓励平滑的动作变化，避免抖动。

    Args:
        env: 环境实例

    Returns:
        动作变化率惩罚值，形状 (num_envs,)
    """
    # 条件返回安全：部分版本/配置可能没有 prev_action
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action") or not hasattr(env.action_manager, "prev_action"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def action_l2(env: ManagerBasedRLEnv) -> Tensor:
    """动作L2惩罚

    鼓励小幅度的动作。

    Args:
        env: 环境实例

    Returns:
        动作惩罚值，形状 (num_envs,)
    """
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


##
# 能量效率奖励
##


def joint_torques_l2(env: ManagerBasedRLEnv) -> Tensor:
    """关节扭矩L2惩罚

    鼓励能量效率。

    Args:
        env: 环境实例

    Returns:
        扭矩惩罚值，形状 (num_envs,)
    """
    # 获取执行器产生的扭矩
    torques = env.scene["robot"].data.applied_torque

    # L2范数
    return torch.sum(torch.square(torques), dim=-1)


def joint_powers_l1(env: ManagerBasedRLEnv) -> Tensor:
    """关节功率L1惩罚

    功率 = 扭矩 × 速度

    Args:
        env: 环境实例

    Returns:
        功率惩罚值，形状 (num_envs,)
    """
    # 获取扭矩和关节速度
    torques = env.scene["robot"].data.applied_torque
    joint_vel = env.scene["robot"].data.joint_vel

    # 计算功率并求L1范数
    powers = torch.abs(torques * joint_vel)
    return torch.sum(powers, dim=-1)


##
# 关节限制惩罚
##


def joint_pos_limits(env: ManagerBasedRLEnv, margin: float = 0.1) -> Tensor:
    """关节位置限制惩罚

    当关节接近限制时给予惩罚。

    Args:
        env: 环境实例
        margin: 安全边界（归一化）

    Returns:
        限制惩罚值，形状 (num_envs,)
    """
    # 获取关节位置（归一化到[-1, 1]）
    joint_pos = env.scene["robot"].data.joint_pos
    joint_pos_limits = env.scene["robot"].data.soft_joint_pos_limits

    # 计算到限制的距离
    lower_limits = joint_pos_limits[:, :, 0]
    upper_limits = joint_pos_limits[:, :, 1]

    # 归一化位置
    joint_pos_normalized = (joint_pos - lower_limits) / (upper_limits - lower_limits + 1e-8)

    # 当位置超出 [margin, 1-margin] 时惩罚
    lower_violation = torch.clamp(margin - joint_pos_normalized, min=0.0)
    upper_violation = torch.clamp(joint_pos_normalized - (1 - margin), min=0.0)

    # 总惩罚
    return torch.sum(lower_violation + upper_violation, dim=-1)


def joint_vel_limits(env: ManagerBasedRLEnv, margin_factor: float = 0.9) -> Tensor:
    """关节速度限制惩罚

    Args:
        env: 环境实例
        margin_factor: 限制因子（0-1）

    Returns:
        速度限制惩罚值，形状 (num_envs,)
    """
    # 获取关节速度
    joint_vel = env.scene["robot"].data.joint_vel
    joint_vel_limits = env.scene["robot"].data.soft_joint_vel_limits

    # 计算超出限制的部分
    vel_limit_scaled = joint_vel_limits * margin_factor
    violation = torch.clamp(torch.abs(joint_vel) - vel_limit_scaled, min=0.0)

    return torch.sum(violation, dim=-1)


##
# 脚部接触奖励
##


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
    foot_body_regex: str = ".*_foot_link|.*_toe_link",
) -> Tensor:
    """脚部离地时间奖励

    鼓励脚部有一定的离地时间（用于行走任务）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器配置名称
        threshold: 接触力阈值

    Returns:
        离地时间奖励值，形状 (num_envs,)
    """
    if sensor_cfg_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor = env.scene.sensors[sensor_cfg_name]

    # 尝试只选择脚/脚趾 body（避免把躯干/大腿接触也当作“脚接触”）
    body_ids = None
    if hasattr(contact_sensor, "find_bodies"):
        try:
            body_ids, _ = contact_sensor.find_bodies(foot_body_regex)
        except Exception:
            body_ids = None

    contact_forces = contact_sensor.data.net_forces_w_history
    if body_ids is not None:
        contact_forces = contact_forces[:, :, body_ids, :]

    # contact_forces: (num_envs, history, num_bodies, 3)
    is_contact = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > threshold  # (num_envs, num_bodies)
    # 以脚为单位平均，避免脚数量变化导致尺度变化
    return torch.mean((~is_contact).float(), dim=1)


def ankle_workspace_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg,
    max_angle: float = 0.5235987755982988,  # pi/6
    margin: float = 0.1,
) -> Tensor:
    """脚踝工作空间软约束惩罚（基于关节位置）

    目的：为 Sim2Real 准备，限制脚踝关节角度在实体可用范围（±30°）附近。

    注意：使用关节位置而不是动作索引，避免 action 维度/顺序不一致导致误惩罚。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    safe_limit = max(float(max_angle) - float(margin), 0.0)
    violation = torch.clamp(torch.abs(joint_pos) - safe_limit, min=0.0)
    return torch.sum(violation, dim=-1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """线速度跟踪奖励（gravity-aligned yaw frame）

    对齐 Isaac Lab locomotion 的实现：把 world 线速度旋转到“只含 yaw 的机体坐标系”后跟踪命令。
    对于 Jiyuan，如果 base frame 存在固定旋转，需先移除该旋转再提取 yaw。
    """
    if _quat_apply_inverse is None or _yaw_quat is None:  # pragma: no cover
        raise RuntimeError("该奖励函数需要 Isaac Lab 运行时（isaaclab.utils.math）。")

    asset = env.scene["robot"]
    base_quat_w = normalize_quaternion(asset.data.root_quat_w)
    if base_quat_correction is not None:
        base_quat_w = remove_fixed_quat_rotation(base_quat_w, base_quat_correction)

    vel_w = asset.data.root_lin_vel_w[:, :3]
    vel_yaw = _quat_apply_inverse(_yaw_quat(base_quat_w), vel_w)

    command = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(torch.square(command - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
) -> Tensor:
    """角速度跟踪奖励（world frame yaw）"""
    asset = env.scene["robot"]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
