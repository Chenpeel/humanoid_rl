"""
行走任务奖励函数

基于开源实现（Legged Gym, Isaac Lab）的行走奖励函数。

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

from __future__ import annotations

import math
import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 导入数学工具
from ..utils.math_utils import (
    DEFAULT_BASE_QUAT_CORRECTION_WXYZ as _DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
    apply_base_quat_correction_to_body_vec,
)

# 导入 Isaac Lab 管理器工具
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


##
# 步态相关奖励
##


# Jiyuan 特有：历史原因 base frame 存在 90° 旋转（wxyz）。
DEFAULT_BASE_QUAT_CORRECTION_WXYZ = _DEFAULT_BASE_QUAT_CORRECTION_WXYZ


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    threshold: float = 1.0,
) -> Tensor:
    """脚部滑动惩罚

    惩罚脚部在接触地面时的滑动。这有助于确保机器人在支撑相保持稳定的接触，
    而不是拖拽脚部。奖励计算为脚部线速度的范数乘以接触标志。

    参考：Isaac Lab locomotion tasks

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        asset_cfg: 机器人资产配置

    Returns:
        滑动惩罚值，形状 (num_envs,)
    """
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if sensor_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 检测接触（优先历史力，更稳；否则退化到当前力）
    if contact_sensor.data.net_forces_w_history is not None:
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        contacts = forces.norm(dim=-1).max(dim=1)[0] > threshold
    elif contact_sensor.data.net_forces_w is not None:
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        contacts = forces.norm(dim=-1) > threshold
    else:
        return torch.zeros(env.num_envs, device=env.device)

    # 获取资产
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    # 获取脚部线速度（只考虑 XY 平面）
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # 计算滑动：速度范数 * 接触标志
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    return reward


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    threshold: float = 1.0,
) -> Tensor:
    """脚部接触标志（用于其他奖励函数）"""
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, 0, dtype=torch.bool, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor_cfg.body_ids is None:
        return torch.zeros(env.num_envs, 0, dtype=torch.bool, device=env.device)

    # 优先用历史力（更抗抖动），否则退化到当前力
    if contact_sensor.data.net_forces_w_history is not None:
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        force_norm = forces.norm(dim=-1).max(dim=1)[0]
    elif contact_sensor.data.net_forces_w is not None:
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_norm = forces.norm(dim=-1)
    else:
        return torch.zeros(env.num_envs, 0, dtype=torch.bool, device=env.device)

    return force_norm > threshold


def gait_symmetry_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    threshold: float = 1.0,
) -> Tensor:
    """步态对称性奖励

    奖励左右脚交替接触地面的对称步态。

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置（建议只匹配左右脚各一个 body）
        threshold: 接触力阈值

    Returns:
        对称性奖励，形状 (num_envs,)
    """
    contacts = feet_contact_forces(env, sensor_cfg=sensor_cfg, threshold=threshold)
    if contacts.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    # 假设前两个 body 分别对应左右脚（需要在 cfg 中保证匹配顺序/数量）
    left_contact = contacts[:, 0].float()
    right_contact = contacts[:, 1].float()

    # 奖励左右脚的互斥接触（一个接触，另一个离地）
    # XOR 逻辑：left ⊕ right
    symmetry = torch.abs(left_contact - right_contact)

    return symmetry


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    target_air_time: float = 0.5,
    air_time_std: float = 0.2,
) -> Tensor:
    """脚部离地时间奖励

    鼓励脚部在空中停留适当的时间（模仿人类行走）。

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置（需要 track_air_time=True）
        target_air_time: 目标离地时间（秒）

    Returns:
        离地时间奖励，形状 (num_envs,)

    注意:
        依赖 ContactSensor 的 air_time 跟踪（cfg.track_air_time=True）。若不可用则返回 0。
    """
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor_cfg.body_ids is None or contact_sensor.data.last_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    try:
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time_error = torch.abs(last_air_time - float(target_air_time))
    air_time_reward = torch.exp(-air_time_error / float(air_time_std)) * first_contact.float()
    return torch.sum(air_time_reward, dim=1)


def stance_duration_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    target_stance_time: float = 0.3,
    stance_time_std: float = 0.2,
) -> Tensor:
    """支撑相持续时间奖励

    鼓励脚部在地面停留适当的时间。

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置（需要 track_air_time=True）
        target_stance_time: 目标支撑时间（秒）

    Returns:
        支撑时间奖励，形状 (num_envs,)
    """
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor_cfg.body_ids is None or contact_sensor.data.last_contact_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 当脚从接触转为空中时触发（first_air），用 last_contact_time 评价支撑时长
    try:
        first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)

    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    stance_time_error = torch.abs(last_contact_time - float(target_stance_time))
    stance_time_reward = torch.exp(-stance_time_error / float(stance_time_std)) * first_air.float()
    return torch.sum(stance_time_reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    threshold: float = 1.0,
    target_clearance: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    clearance_std: float = 0.02,
) -> Tensor:
    """脚部抬高奖励

    鼓励脚在摆动相抬高到适当高度（避免拖地）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值
        target_clearance: 目标抬高高度（m）

    Returns:
        抬高奖励，形状 (num_envs,)

    注意:
        使用 Articulation 的 body_pos_w 获取脚部高度；摆动相通过 ContactSensor 的 in_air 判定。
    """
    if sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 判定是否在空中（需要 track_air_time=True）；否则用接触力近似摆动相
    if contact_sensor.data.current_air_time is not None:
        in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
    else:
        contacts = feet_contact_forces(env, sensor_cfg=sensor_cfg, threshold=threshold)
        in_air = ~contacts

    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.abs(feet_heights - float(target_clearance))
    clearance_reward = torch.exp(-height_error / float(clearance_std)) * in_air.float()
    return torch.sum(clearance_reward, dim=1)


##
# 躯干稳定性（行走专用）
##


def trunk_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.35,
    tolerance: float = 0.05,
) -> Tensor:
    """躯干高度保持奖励（行走时）

    与站立任务不同，行走时允许小幅度的上下波动。

    Args:
        env: 环境实例
        target_height: 目标高度
        tolerance: 容差（更宽松）

    Returns:
        高度奖励，形状 (num_envs,)
    """
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]
    height_error = torch.abs(torso_z - target_height)
    return torch.exp(-height_error / tolerance)


def trunk_orientation_penalty(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.3,
    base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """躯干过度倾斜惩罚

    行走时允许小幅度倾斜，但惩罚过度倾斜。

    Args:
        env: 环境实例
        max_tilt: 最大允许倾斜（弧度）

    Returns:
        倾斜惩罚，形状 (num_envs,)
    """
    asset = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b

    gravity_b = apply_base_quat_correction_to_body_vec(gravity_b, base_quat_correction)

    # 用 sin(tilt) 近似 tilt（小角度下等价），避免 asin/acos 带来的开销
    tilt_sin = torch.norm(gravity_b[:, :2], dim=-1)
    limit_sin = float(math.sin(float(max_tilt)))
    return torch.clamp(tilt_sin - limit_sin, min=0.0)


def trunk_lin_vel_z_penalty(env: ManagerBasedRLEnv) -> Tensor:
    """躯干Z方向速度惩罚

    行走时躯干应该平稳移动，避免大幅度上下跳动。

    Args:
        env: 环境实例

    Returns:
        Z速度惩罚，形状 (num_envs,)
    """
    base_lin_vel = env.scene["robot"].data.root_lin_vel_w
    return torch.square(base_lin_vel[:, 2])


##
# 说明：本模块不提供硬编码的 reward 权重表（权重应从 YAML/训练配置动态驱动）。
##
