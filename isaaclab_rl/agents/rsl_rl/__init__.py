"""
RSL_RL 算法配置模块

ETH Zurich 的 PPO 实现配置。

模块:
- ppo_cfg.py: PPO 超参数配置
"""

from .ppo_cfg import (
    VelocityTrackingPPORunnerCfg,
    StandingPPORunnerCfg,
    VELOCITY_TRACKING_PPO_CFG,
    STANDING_PPO_CFG,
)

__all__ = [
    "VelocityTrackingPPORunnerCfg",
    "StandingPPORunnerCfg",
    "VELOCITY_TRACKING_PPO_CFG",
    "STANDING_PPO_CFG",
]
