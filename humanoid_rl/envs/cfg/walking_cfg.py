"""WalkingEnvCfg —— 行走任务配置。"""

from __future__ import annotations

try:
    from isaaclab.utils import configclass
except ImportError:
    def configclass(cls):
        return cls

from .base_cfg import BaseEnvCfg


@configclass
class WalkingEnvCfg(BaseEnvCfg):
    """行走任务。"""
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "walking"
        self.observations.policy.num_obs = 72
        self.rewards.num_terms = 12
        self.commands.num_commands = 3
        self.curriculum.enable = True
