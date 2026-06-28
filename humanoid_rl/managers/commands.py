"""CommandsManager —— 训练指令生成。

指令类型：
  - 速度指令: vx (前进), vy (侧向), yaw_rate (转向角速度)
  - 指令在 episode 开始时随机生成，或按 resampling_time 定期更新
  - 站立任务不需要指令
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CommandsManager:
    """指令管理器。

    在每个 episode 开始时或按周期（resampling_time）生成新的速度指令。
    """

    def __init__(self, cfg: object, env: "ManagerBasedRLEnv"):
        self.cfg = cfg
        self.env = env

        # 指令范围
        self.vx_range = getattr(cfg, "vx_range", (-1.0, 1.0))        # m/s
        self.vy_range = getattr(cfg, "vy_range", (-0.5, 0.5))        # m/s
        self.yaw_rate_range = getattr(cfg, "yaw_rate_range", (-1.0, 1.0))  # rad/s
        self.resampling_time = getattr(cfg, "resampling_time", 10.0)  # s

    def compute(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """为指定环境生成新指令。

        Args:
            env_ids: 需要重置指令的环境 ID，None 表示全部

        Returns:
            commands: (num_envs, 3) [vx, vy, yaw_rate]
        """
        num_envs = self.env.num_envs if env_ids is None else len(env_ids)
        device = self.env.device

        # 均匀分布采样
        vx = torch.rand(num_envs, device=device) * (
            self.vx_range[1] - self.vx_range[0]
        ) + self.vx_range[0]
        vy = torch.rand(num_envs, device=device) * (
            self.vy_range[1] - self.vy_range[0]
        ) + self.vy_range[0]
        yaw_rate = torch.rand(num_envs, device=device) * (
            self.yaw_rate_range[1] - self.yaw_rate_range[0]
        ) + self.yaw_rate_range[0]

        commands = torch.stack([vx, vy, yaw_rate], dim=-1)
        return commands

    def should_resample(self, env_ids: torch.Tensor) -> torch.Tensor:
        """判断哪些环境需要重采样指令（基于时间）。"""
        episode_time = self.env.episode_length_buf * self.env.step_dt
        time_to_resample = episode_time % self.resampling_time
        return time_to_resample < self.env.step_dt
