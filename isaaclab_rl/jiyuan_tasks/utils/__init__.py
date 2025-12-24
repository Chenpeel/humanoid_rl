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

from . import math_utils
from . import sim2real
from . import imitation
from . import config_loader

__all__ = [
    "math_utils",
    "sim2real",
    "imitation",
    "config_loader",
]
