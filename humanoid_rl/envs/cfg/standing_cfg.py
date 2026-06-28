"""StandingEnvCfg —— 站立平衡任务配置。"""

from __future__ import annotations

try:
    from isaaclab.utils import configclass
except ImportError:
    def configclass(cls):
        return cls

from .base_cfg import BaseEnvCfg


@configclass
class StandingEnvCfg(BaseEnvCfg):
    """站立平衡任务。"""
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "standing"
        self.observations.policy.num_obs = 48
        self.rewards.num_terms = 8
        self.terminations.time_out = None
        self.commands.num_commands = 0
        self.curriculum.enable = False
