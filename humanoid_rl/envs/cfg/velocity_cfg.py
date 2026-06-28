"""VelocityTrackingEnvCfg —— 速度跟踪任务配置。"""

from __future__ import annotations

try:
    from isaaclab.utils import configclass
except ImportError:
    def configclass(cls):
        return cls

from .base_cfg import BaseEnvCfg


@configclass
class VelocityTrackingEnvCfg(BaseEnvCfg):
    """速度跟踪任务。"""
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "velocity_tracking"
        self.observations.policy.num_obs = 96
        self.rewards.num_terms = 15
        self.commands.num_commands = 3
        self.commands.resampling_time = 10.0
        self.curriculum.enable = True
