"""
Isaac Lab RL（Jiyuan）

本包为 Jiyuan 双足机器人提供基于 Isaac Lab + PyTorch + RSL_RL 的训练/评估入口与任务实现。

注意:
- 为了让 `pytest` 等纯 Python 工具在无 Isaac Sim 运行时的环境里也能导入，本模块不在 import 时主动加载
  `jiyuan_tasks` / `agents` 等子模块（这些子模块可能依赖 Isaac Lab）。
"""

__version__ = "0.3.0"

__all__ = [
    "agents",
    "jiyuan_tasks",
]
