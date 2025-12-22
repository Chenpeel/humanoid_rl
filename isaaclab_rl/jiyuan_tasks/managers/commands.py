"""
Jiyuan 机器人命令生成器

命令管理器用于生成机器人的目标指令，如期望的线速度、角速度等。
主要用于速度跟踪任务（VelocityTracking）。

注意:
- Isaac Lab 提供了内置的命令生成器（omni.isaac.lab.envs.mdp.commands），
  可以直接在环境配置中使用，无需自定义实现。
- 本文件提供自定义命令生成器的参考实现。

参考:
- Isaac Lab Commands: omni.isaac.lab.envs.mdp.commands
- 官方示例: source/standalone/workflows/rsl_rl/train.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


##
# 自定义命令生成器（参考实现）
##


def uniform_velocity_command(
    env: ManagerBasedRLEnv,
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    command_name: str = "base_velocity",
) -> Tensor:
    """生成均匀分布的速度命令

    为每个环境随机采样线速度（x, y）和角速度（z）命令。

    Args:
        env: 环境实例
        lin_vel_x_range: 前向速度范围 (m/s)，默认 [-1.0, 1.0]
        lin_vel_y_range: 侧向速度范围 (m/s)，默认 [-0.5, 0.5]
        ang_vel_z_range: 转向速度范围 (rad/s)，默认 [-1.0, 1.0]
        command_name: 命令名称（用于存储到命令管理器）

    Returns:
        速度命令张量，形状 (num_envs, 3)，格式 [vel_x, vel_y, ang_vel_z]

    注意:
        这是自定义实现的示例。在实际使用中，推荐使用 Isaac Lab 内置的
        UniformVelocityCommand 命令生成器（在环境配置中配置）。
    """
    num_envs = env.num_envs
    device = env.device

    # 生成随机命令
    lin_vel_x = torch.rand(num_envs, device=device) * (lin_vel_x_range[1] - lin_vel_x_range[0]) + lin_vel_x_range[0]
    lin_vel_y = torch.rand(num_envs, device=device) * (lin_vel_y_range[1] - lin_vel_y_range[0]) + lin_vel_y_range[0]
    ang_vel_z = torch.rand(num_envs, device=device) * (ang_vel_z_range[1] - ang_vel_z_range[0]) + ang_vel_z_range[0]

    # 组合为 (num_envs, 3)
    commands = torch.stack([lin_vel_x, lin_vel_y, ang_vel_z], dim=-1)

    return commands


def curriculum_velocity_command(
    env: ManagerBasedRLEnv,
    initial_lin_vel_range: tuple[float, float] = (0.0, 0.5),
    final_lin_vel_range: tuple[float, float] = (-1.5, 1.5),
    initial_ang_vel_range: tuple[float, float] = (0.0, 0.3),
    final_ang_vel_range: tuple[float, float] = (-1.5, 1.5),
    curriculum_steps: int = 10000,
) -> Tensor:
    """课程学习速度命令生成器

    根据训练进度逐渐增加命令难度。

    Args:
        env: 环境实例
        initial_lin_vel_range: 初始线速度范围 (m/s)
        final_lin_vel_range: 最终线速度范围 (m/s)
        initial_ang_vel_range: 初始角速度范围 (rad/s)
        final_ang_vel_range: 最终角速度范围 (rad/s)
        curriculum_steps: 课程总步数

    Returns:
        速度命令张量，形状 (num_envs, 3)

    注意:
        需要在环境中维护 global_step 计数器，用于计算课程进度。
    """
    # 获取当前训练步数（假设环境有这个属性）
    current_step = getattr(env, "global_step", 0)

    # 计算课程进度 [0, 1]
    progress = min(current_step / curriculum_steps, 1.0)

    # 插值计算当前速度范围
    lin_vel_min = initial_lin_vel_range[0] + progress * (final_lin_vel_range[0] - initial_lin_vel_range[0])
    lin_vel_max = initial_lin_vel_range[1] + progress * (final_lin_vel_range[1] - initial_lin_vel_range[1])

    ang_vel_min = initial_ang_vel_range[0] + progress * (final_ang_vel_range[0] - initial_ang_vel_range[0])
    ang_vel_max = initial_ang_vel_range[1] + progress * (final_ang_vel_range[1] - initial_ang_vel_range[1])

    # 使用插值后的范围生成命令
    return uniform_velocity_command(
        env,
        lin_vel_x_range=(lin_vel_min, lin_vel_max),
        lin_vel_y_range=(lin_vel_min * 0.5, lin_vel_max * 0.5),  # 侧向速度较小
        ang_vel_z_range=(ang_vel_min, ang_vel_max),
    )


def standing_command(env: ManagerBasedRLEnv) -> Tensor:
    """站立任务命令（零速度）

    站立任务不需要速度跟踪，返回零命令。

    Args:
        env: 环境实例

    Returns:
        零速度命令，形状 (num_envs, 3)
    """
    return torch.zeros(env.num_envs, 3, device=env.device)


##
# 命令管理辅助函数
##


def resample_commands(
    env: ManagerBasedRLEnv,
    env_ids: Tensor,
    command_generator_fn: callable,
) -> None:
    """重新采样指定环境的命令

    当环境重置或达到命令重采样时间时调用。

    Args:
        env: 环境实例
        env_ids: 需要重新采样的环境ID
        command_generator_fn: 命令生成函数

    注意:
        这是辅助函数示例。实际使用中，Isaac Lab 的 CommandManager
        会自动处理命令的重采样逻辑。
    """
    # 生成新命令
    new_commands = command_generator_fn(env)

    # 只更新指定环境的命令
    if hasattr(env, "commands"):
        env.commands[env_ids] = new_commands[env_ids]


##
# 推荐使用方式
##

# 在实际项目中，推荐使用 Isaac Lab 内置的命令生成器，
# 而不是自定义实现。配置示例：
#
# from omni.isaac.lab.envs.mdp import UniformVelocityCommandCfg
#
# @configclass
# class VelocityTrackingEnvCfg(ManagerBasedRLEnvCfg):
#     commands: CommandsCfg = CommandsCfg()
#
#     @configclass
#     class CommandsCfg:
#         base_velocity = UniformVelocityCommandCfg(
#             asset_name="robot",
#             resampling_time_range=(10.0, 10.0),
#             rel_standing_envs=0.0,
#             rel_heading_envs=1.0,
#             heading_command=True,
#             heading_control_stiffness=0.5,
#             debug_vis=True,
#             ranges=UniformVelocityCommandCfg.Ranges(
#                 lin_vel_x=(-1.0, 1.0),
#                 lin_vel_y=(-0.5, 0.5),
#                 ang_vel_z=(-1.0, 1.0),
#             ),
#         )
#
# 详见：Isaac Lab 官方文档和示例
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/omni.isaac.lab.envs.mdp.html#commands
