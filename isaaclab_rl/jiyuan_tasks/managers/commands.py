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
    from isaaclab.envs import ManagerBasedRLEnv

from ..utils.math_utils import DEFAULT_BASE_QUAT_CORRECTION_WXYZ, apply_base_quat_correction_to_body_vec, remove_fixed_quat_rotation


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
# from isaaclab.envs.mdp import UniformVelocityCommandCfg
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


##
# Isaac Lab CommandTerm：校正版速度命令（用于 Jiyuan base 固定旋转）
##


try:  # 仅在 Isaac Lab 运行时可用
    import isaaclab.utils.math as isaac_math
    from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg as _UniformVelocityCommandCfg
    from isaaclab.utils import configclass

    @configclass
    class CorrectedUniformVelocityCommandCfg(_UniformVelocityCommandCfg):
        """对齐 Z-up 语义坐标系的速度命令配置。

        说明：
        - Isaac Lab 默认认为机器人 base frame 的 (x,y) 位于水平面、z 为 up。
        - Jiyuan 的 MJCF 把 base 旋转了 90°，导致 base frame 的 y 轴变成 up（错轴）。
        - 本 cfg 通过自定义 CommandTerm 让 command / obs / reward 在同一“校正后 body frame”工作。
        """

        class_type: type = None  # 在下方绑定，避免定义顺序问题
        base_quat_correction: tuple[float, float, float, float] | None = DEFAULT_BASE_QUAT_CORRECTION_WXYZ


    class CorrectedUniformVelocityCommand(UniformVelocityCommand):
        """均匀速度命令生成器（语义 Z-up 校正版）。"""

        cfg: CorrectedUniformVelocityCommandCfg

        def _current_heading_w_corrected(self) -> torch.Tensor:
            base_quat_w = self.robot.data.root_quat_w
            if self.cfg.base_quat_correction is not None:
                base_quat_w = remove_fixed_quat_rotation(base_quat_w, self.cfg.base_quat_correction)
            forward_w = isaac_math.quat_apply(base_quat_w, self.robot.data.FORWARD_VEC_B)
            return torch.atan2(forward_w[:, 1], forward_w[:, 0])

        def _update_command(self):
            """后处理速度命令（heading 控制 + standing 置零），在校正后的 base frame 上工作。"""
            if self.cfg.heading_command:
                env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
                heading_error = isaac_math.wrap_to_pi(self.heading_target[env_ids] - self._current_heading_w_corrected()[env_ids])
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

            standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[standing_env_ids, :] = 0.0

        def _update_metrics(self):
            max_command_time = self.cfg.resampling_time_range[1]
            max_command_step = max_command_time / self._env.step_dt

            lin_vel_b = apply_base_quat_correction_to_body_vec(self.robot.data.root_lin_vel_b, self.cfg.base_quat_correction)
            ang_vel_b = apply_base_quat_correction_to_body_vec(self.robot.data.root_ang_vel_b, self.cfg.base_quat_correction)

            self.metrics["error_vel_xy"] += (
                torch.norm(self.vel_command_b[:, :2] - lin_vel_b[:, :2], dim=-1) / max_command_step
            )
            self.metrics["error_vel_yaw"] += (torch.abs(self.vel_command_b[:, 2] - ang_vel_b[:, 2]) / max_command_step)

        def _debug_vis_callback(self, event):
            if not self.robot.is_initialized:
                return

            base_pos_w = self.robot.data.root_pos_w.clone()
            base_pos_w[:, 2] += 0.5

            vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
            lin_vel_b = apply_base_quat_correction_to_body_vec(self.robot.data.root_lin_vel_b, self.cfg.base_quat_correction)
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(lin_vel_b[:, :2])

            self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
            self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
            arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
            arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

            heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
            zeros = torch.zeros_like(heading_angle)
            arrow_quat = isaac_math.quat_from_euler_xyz(zeros, zeros, heading_angle)

            base_quat_w = self.robot.data.root_quat_w
            if self.cfg.base_quat_correction is not None:
                base_quat_w = remove_fixed_quat_rotation(base_quat_w, self.cfg.base_quat_correction)
            arrow_quat = isaac_math.quat_mul(base_quat_w, arrow_quat)
            return arrow_scale, arrow_quat


    CorrectedUniformVelocityCommandCfg.class_type = CorrectedUniformVelocityCommand

except Exception:  # pragma: no cover
    # 纯 Python 环境下无需可用；env_cfg 也不会在无 Isaac Lab 环境被导入。
    pass
