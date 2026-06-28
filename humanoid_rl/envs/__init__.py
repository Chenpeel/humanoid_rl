"""
机器人环境模块 —— Isaac Lab Task 定义。

按机器人-任务两层配置：
  robot   → 机器人模型资产（URDF/MJCF/USD）
  task    → 奖励、观测、终止、指令
"""

from .base_env import BaseEnv
