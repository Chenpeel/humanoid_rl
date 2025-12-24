"""
RL 算法配置模块

包含各种 RL 算法的配置类。

支持的算法:
- RSL_RL: ETH Zurich 的 PPO 实现（主要使用）
"""

from . import rsl_rl

__all__ = [
    "rsl_rl",
]
