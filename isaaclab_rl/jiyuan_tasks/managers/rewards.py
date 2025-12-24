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
from ..utils.math_utils import quat_to_euler_xyz, normalize_quaternion


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
) -> Tensor:
    """姿态稳定奖励

    机器人应该保持直立（roll和pitch接近0）。

    Args:
        env: 环境实例
        tolerance: 容差参数

    Returns:
        姿态奖励值，形状 (num_envs,)，范围 [0, 1]
    """
    # 获取机器人base的四元数
    base_quat = env.scene["robot"].data.root_quat_w

    # 归一化四元数
    base_quat = normalize_quaternion(base_quat)

    # 转换为欧拉角
    euler = quat_to_euler_xyz(base_quat)
    roll, pitch = euler[:, 0], euler[:, 1]

    # 计算姿态误差（roll和pitch的平方和）
    orientation_error = torch.square(roll) + torch.square(pitch)

    # 指数奖励
    return torch.exp(-orientation_error / tolerance)


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


def xy_vel_penalty_l2(env: ManagerBasedRLEnv) -> Tensor:
    """XY平面速度惩罚

    惩罚水平方向的移动，允许少量垂直方向的振动。

    Args:
        env: 环境实例

    Returns:
        XY速度惩罚值，形状 (num_envs,)
    """
    # 获取base线速度
    base_lin_vel = env.scene["robot"].data.root_lin_vel_b

    # 只惩罚XY方向
    return torch.sum(torch.square(base_lin_vel[:, :2]), dim=-1)


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
    # 获取当前动作和上一步动作
    # Isaac Lab 会自动在环境中存储 last_actions
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
    # 获取接触力
    contact_forces = env.scene.sensors[sensor_cfg_name].data.net_forces_w_history

    # 检测接触（力大于阈值）
    contact_detected = torch.max(torch.norm(contact_forces, dim=-1), dim=-1)[0] > threshold

    # 奖励非接触状态
    return (~contact_detected).float()


##
# 默认奖励权重
##


# 站立任务奖励权重
STANDING_REWARD_WEIGHTS = {
    "height": 1.0,  # 高度保持
    "orientation": 1.0,  # 姿态稳定
    "lin_vel": -0.5,  # 线速度惩罚
    "ang_vel": -0.3,  # 角速度惩罚
    "alive": 0.2,  # 存活奖励
    "action_rate": -0.01,  # 动作平滑
    "torques": -0.0001,  # 能量效率
}

# 速度跟踪任务奖励权重
VELOCITY_TRACKING_REWARD_WEIGHTS = {
    "track_lin_vel": 1.0,  # 线速度跟踪
    "track_ang_vel": 0.5,  # 角速度跟踪
    "lin_vel_z": -2.0,  # Z方向速度惩罚
    "ang_vel_xy": -0.05,  # XY方向角速度惩罚
    "orientation": 0.5,  # 姿态稳定
    "action_rate": -0.01,  # 动作平滑
    "joint_accel": -2.5e-7,  # 关节加速度惩罚
    "joint_powers": -2e-5,  # 能量效率
    "alive": 0.1,  # 存活奖励
}
