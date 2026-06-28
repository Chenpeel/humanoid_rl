"""
任务配置模块 —— Hydra / OmegaConf 驱动的 YAML 配置。

每种任务类型一个文件：
  base_cfg.py       → 基础环境参数（地面、重力、仿真步长）
  standing_cfg.py   → 站立平衡任务
  walking_cfg.py    → 行走任务
  velocity_cfg.py   → 速度跟踪任务
"""
