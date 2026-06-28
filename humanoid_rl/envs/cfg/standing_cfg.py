"""StandingEnvCfg —— 站立平衡任务配置。

训练机器人在原地保持站立姿态，最小化躯干倾斜和移动。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .base_cfg import BaseEnvCfg


@configclass
class StandingEnvCfg(BaseEnvCfg):
    """站立平衡任务。

    核心目标：
    - 最小化躯干倾角（roll / pitch → 0）
    - 保持底座高度在目标值
    - 最小化关节力矩
    - 惩罚水平位移
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- 任务元信息 ----
        self.task_name = "standing"

        # ---- 观测配置 ----
        # 站立任务使用精简观测：关节位置/速度 + IMU + 高度
        self.observations.policy.num_obs = 48

        # ---- 奖励配置 ----
        # 各奖励项权重在 YAML 中通过 robot 配置覆盖
        self.rewards.num_terms = 8

        # ---- 终止条件 ----
        # 躯干高度过低 / 倾角过大 → 终止
        self.terminations.time_out = None  # 由 episode_length_s 自动派生

        # ---- 指令 ----
        # 站立任务不需要速度指令
        self.commands.num_commands = 0

        # ---- 课程学习 ----
        # 站立任务不需要课程
        self.curriculum.enable = False
