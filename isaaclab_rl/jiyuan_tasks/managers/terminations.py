"""
Jiyuan 机器人终止条件函数

从 JAX/MJX 迁移的终止条件，使用 PyTorch 和 Isaac Lab 接口。

所有终止函数遵循 Isaac Lab 规范：
- 第一个参数：env (ManagerBasedRLEnv)
- 返回值：torch.Tensor，形状 (num_envs,)，dtype bool
- 用于检测需要重置的环境（摔倒、超时等）

参考:
- Isaac Lab MDP: omni.isaac.lab.envs.mdp
- 原始实现: src/rl/envs/robot_envs.py (jax 分支)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 导入数学工具
from ..utils.math_utils import quat_to_euler_xyz, normalize_quaternion


##
# 基础终止条件
##


def time_out(env: ManagerBasedRLEnv) -> Tensor:
    """超时终止

    当回合达到最大时间步时终止。

    Args:
        env: 环境实例

    Returns:
        终止标志，形状 (num_envs,)
    """
    return env.episode_length_buf >= env.max_episode_length


##
# 姿态相关终止条件
##


def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    min_height: float = 0.2,
) -> Tensor:
    """高度过低终止

    当机器人base高度低于阈值时终止（通常表示摔倒）。

    Args:
        env: 环境实例
        min_height: 最小高度阈值 (m)，默认 0.2m

    Returns:
        终止标志，形状 (num_envs,)
    """
    # 获取机器人base的高度（z坐标）
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]

    return torso_z < min_height


def base_height_above_threshold(
    env: ManagerBasedRLEnv,
    max_height: float = 1.0,
) -> Tensor:
    """高度过高终止

    当机器人base高度高于阈值时终止（异常跳跃）。

    Args:
        env: 环境实例
        max_height: 最大高度阈值 (m)，默认 1.0m

    Returns:
        终止标志，形状 (num_envs,)
    """
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]
    return torso_z > max_height


def base_orientation_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_roll: float = 0.8,  # ~45度
    max_pitch: float = 0.8,
) -> Tensor:
    """姿态超出范围终止

    当机器人倾斜角度（roll或pitch）超过阈值时终止（摔倒）。

    Args:
        env: 环境实例
        max_roll: 最大roll角（弧度），默认 0.8 rad (~45度)
        max_pitch: 最大pitch角（弧度），默认 0.8 rad

    Returns:
        终止标志，形状 (num_envs,)
    """
    # 获取机器人base的四元数
    base_quat = env.scene["robot"].data.root_quat_w

    # 归一化四元数
    base_quat = normalize_quaternion(base_quat)

    # 转换为欧拉角
    euler = quat_to_euler_xyz(base_quat)
    roll, pitch = euler[:, 0], euler[:, 1]

    # 检查是否超出范围
    roll_out = torch.abs(roll) > max_roll
    pitch_out = torch.abs(pitch) > max_pitch

    return roll_out | pitch_out


##
# 接触相关终止条件
##


def base_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
) -> Tensor:
    """base接触地面终止

    当机器人base与地面接触时终止（摔倒）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器配置名称
        threshold: 接触力阈值 (N)

    Returns:
        终止标志，形状 (num_envs,)

    注意:
        需要在环境配置中定义名为 sensor_cfg_name 的接触传感器，
        并且该传感器监控base或torso body。
    """
    # 获取base的接触力
    # 注意：这需要在环境配置中正确设置接触传感器
    if sensor_cfg_name not in env.scene.sensors:
        # 如果传感器未配置，返回全False（不终止）
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_forces = env.scene.sensors[sensor_cfg_name].data.net_forces_w

    # 计算接触力的范数
    contact_force_norm = torch.norm(contact_forces, dim=-1)

    # 检测是否有接触（力大于阈值）
    return contact_force_norm > threshold


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
    body_ids: list[int] | None = None,
) -> Tensor:
    """不期望的身体部位接触地面终止

    当指定的身体部位（如大腿、小腿等非脚部）接触地面时终止。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器配置名称
        threshold: 接触力阈值 (N)
        body_ids: 需要检测的body ID列表，如果为None则检测所有

    Returns:
        终止标志，形状 (num_envs,)

    注意:
        需要在环境配置中正确设置接触传感器，监控相应的body。
    """
    if sensor_cfg_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_forces = env.scene.sensors[sensor_cfg_name].data.net_forces_w

    # 如果指定了body_ids，只检测这些body
    if body_ids is not None:
        contact_forces = contact_forces[:, body_ids, :]

    # 计算每个body的接触力范数
    contact_force_norm = torch.norm(contact_forces, dim=-1)

    # 检测是否有任何不期望的body接触地面
    # 使用 max 找到每个环境中最大的接触力
    max_contact = torch.max(contact_force_norm, dim=-1)[0]

    return max_contact > threshold


##
# 速度相关终止条件
##


def linear_velocity_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_velocity: float = 10.0,
) -> Tensor:
    """线速度超出范围终止

    当机器人线速度过大时终止（异常快速移动）。

    Args:
        env: 环境实例
        max_velocity: 最大线速度 (m/s)

    Returns:
        终止标志，形状 (num_envs,)
    """
    base_lin_vel = env.scene["robot"].data.root_lin_vel_w

    # 计算速度的模
    vel_norm = torch.norm(base_lin_vel, dim=-1)

    return vel_norm > max_velocity


def angular_velocity_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_velocity: float = 10.0,
) -> Tensor:
    """角速度超出范围终止

    当机器人角速度过大时终止（异常快速旋转）。

    Args:
        env: 环境实例
        max_velocity: 最大角速度 (rad/s)

    Returns:
        终止标志，形状 (num_envs,)
    """
    base_ang_vel = env.scene["robot"].data.root_ang_vel_w

    # 计算角速度的模
    ang_vel_norm = torch.norm(base_ang_vel, dim=-1)

    return ang_vel_norm > max_velocity


##
# 关节相关终止条件
##


def joint_pos_out_of_limits(
    env: ManagerBasedRLEnv,
    margin: float = 0.01,
) -> Tensor:
    """关节位置超出限制终止

    当关节位置超出软限制时终止。

    Args:
        env: 环境实例
        margin: 安全边界 (rad)，距离限制的最小距离

    Returns:
        终止标志，形状 (num_envs,)
    """
    joint_pos = env.scene["robot"].data.joint_pos
    joint_limits = env.scene["robot"].data.soft_joint_pos_limits

    # 检查是否超出限制
    lower_limits = joint_limits[:, :, 0] + margin
    upper_limits = joint_limits[:, :, 1] - margin

    # 检测任何关节是否超出限制
    out_of_lower = joint_pos < lower_limits
    out_of_upper = joint_pos > upper_limits

    # 如果任何关节超限，该环境就终止
    return torch.any(out_of_lower | out_of_upper, dim=-1)


def joint_vel_out_of_limits(
    env: ManagerBasedRLEnv,
    margin_factor: float = 0.9,
) -> Tensor:
    """关节速度超出限制终止

    当关节速度超出软限制时终止。

    Args:
        env: 环境实例
        margin_factor: 限制因子（0-1），使用限制的百分比

    Returns:
        终止标志，形状 (num_envs,)
    """
    joint_vel = env.scene["robot"].data.joint_vel
    joint_vel_limits = env.scene["robot"].data.soft_joint_vel_limits

    # 应用边界因子
    effective_limits = joint_vel_limits * margin_factor

    # 检测任何关节速度是否超限
    out_of_limits = torch.abs(joint_vel) > effective_limits

    return torch.any(out_of_limits, dim=-1)


##
# 组合终止条件（常用预设）
##


def is_fallen(
    env: ManagerBasedRLEnv,
    min_height: float = 0.2,
    max_roll: float = 0.8,
    max_pitch: float = 0.8,
) -> Tensor:
    """机器人摔倒检测（组合条件）

    检测机器人是否摔倒，综合考虑高度和姿态。

    Args:
        env: 环境实例
        min_height: 最小高度阈值 (m)
        max_roll: 最大roll角（弧度）
        max_pitch: 最大pitch角（弧度）

    Returns:
        终止标志，形状 (num_envs,)
    """
    # 高度过低
    height_fail = base_height_below_threshold(env, min_height)

    # 姿态超限
    orientation_fail = base_orientation_out_of_bounds(env, max_roll, max_pitch)

    # 任一条件满足即判定为摔倒
    return height_fail | orientation_fail


##
# 站立任务默认终止条件
##


def standing_termination(env: ManagerBasedRLEnv) -> Tensor:
    """站立任务默认终止条件

    综合多个条件判断站立任务是否应该终止：
    - 超时
    - 摔倒（高度或姿态）

    Args:
        env: 环境实例

    Returns:
        终止标志，形状 (num_envs,)
    """
    # 超时
    timeout = time_out(env)

    # 摔倒
    fallen = is_fallen(
        env,
        min_height=0.2,  # 20cm
        max_roll=0.785,  # 45度
        max_pitch=0.785,
    )

    return timeout | fallen


##
# 速度跟踪任务默认终止条件
##


def velocity_tracking_termination(env: ManagerBasedRLEnv) -> Tensor:
    """速度跟踪任务默认终止条件

    综合多个条件判断速度跟踪任务是否应该终止：
    - 超时
    - 摔倒
    - 速度超限

    Args:
        env: 环境实例

    Returns:
        终止标志，形状 (num_envs,)
    """
    # 超时
    timeout = time_out(env)

    # 摔倒
    fallen = is_fallen(
        env,
        min_height=0.15,  # 速度跟踪任务允许更低（蹲下姿态）
        max_roll=1.0,  # ~57度
        max_pitch=1.0,
    )

    # 速度异常
    vel_fail = linear_velocity_out_of_bounds(env, max_velocity=8.0)

    return timeout | fallen | vel_fail
