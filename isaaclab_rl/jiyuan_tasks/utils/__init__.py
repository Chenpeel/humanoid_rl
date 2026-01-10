"""
工具函数模块

包含数学工具、可视化辅助等实用函数。

模块:
- math_utils.py: 数学工具函数（四元数、欧拉角等）
- sim2real.py: Sim2Real 映射层（并联脚踝映射）
- imitation.py: 模仿学习框架（BVH/FBX 支持）
- config_loader.py: 配置加载工具（支持优先级：CLI > 配置文件 > 预定义）
- visualization.py: 可视化辅助函数（计划中）
"""

from __future__ import annotations

__all__ = [
    "math_utils",
    "sim2real",
    "imitation",
    "config_loader",
]


def __getattr__(name: str):
    # 避免在 import jiyuan_tasks.utils 时立即拉起 torch / isaaclab 等重依赖，保持“纯 Python 可导入”。
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
