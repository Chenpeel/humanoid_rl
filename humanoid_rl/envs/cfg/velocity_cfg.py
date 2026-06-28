"""VelocityTrackingEnvCfg —— 速度跟踪任务配置。

训练机器人在平面/崎岖地形上跟踪给定的速度指令。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .base_cfg import BaseEnvCfg


@configclass
class VelocityTrackingEnvCfg(BaseEnvCfg):
    """速度跟踪任务。

    核心目标：
    - 精确跟踪 vx / vy / yaw_rate 指令
    - 在崎岖地形上保持稳定
    - 灵活的步态自适应
    """

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "velocity_tracking"

        # ---- 观测 ----
        self.observations.policy.num_obs = 96

        # ---- 奖励 ----
        self.rewards.num_terms = 15

        # ---- 指令 ----
        self.commands.num_commands = 3  # vx, vy, yaw_rate
        self.commands.resampling_time = 10.0  # 每 10s 更新一次指令

        # ---- 课程学习 ----
        self.curriculum.enable = True
