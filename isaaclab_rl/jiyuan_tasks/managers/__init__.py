"""
管理器模块

包含自定义的 MDP 函数（观测、奖励、命令、终止等）。

模块:
- observations.py: 观测函数（计划中）
- rewards.py: 奖励函数（从 JAX/MJX 迁移）
- walking_rewards.py: 行走任务奖励函数（基于开源实现）
- commands.py: 命令生成器（计划中）
- terminations.py: 终止条件（计划中）

所有函数遵循 Isaac Lab 管理器规范:
- 第一个参数: env: ManagerBasedRLEnv
- 返回值: torch.Tensor，形状: (num_envs,) 或 (num_envs, dim)
"""

from __future__ import annotations

__all__ = [
    "observations",
    "rewards",
    "walking_rewards",
    "terminations",
    "commands",
]


def __getattr__(name: str):
    # 避免在 import jiyuan_tasks.managers 时立即导入 torch/isaaclab 相关重模块。
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
