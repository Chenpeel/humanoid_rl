"""WalkingEnvCfg —— 行走任务配置。

训练机器人在平坦地形上稳定行走。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .base_cfg import BaseEnvCfg


@configclass
class WalkingEnvCfg(BaseEnvCfg):
    """行走任务。

    核心目标：
    - 跟踪目标前进速度（vx）
    - 保持躯干稳定
    - 自然步态（对称站立腿、摆动腿周期性）
    - 最小化关节冲击
    """

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "walking"

        # ---- 观测 ----
        self.observations.policy.num_obs = 72

        # ---- 奖励 ----
        self.rewards.num_terms = 12

        # ---- 指令 ----
        self.commands.num_commands = 3  # vx, vy, yaw_rate

        # ---- 课程学习 ----
        self.curriculum.enable = True
