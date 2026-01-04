"""
环境配置模块

包含所有任务环境的配置类，使用 Isaac Lab 的 @configclass 装饰器。

配置类:
- jiyuan_scene_cfg.py: 场景和机器人配置
- jiyuan_test_env_cfg.py: 测试环境配置（最小可行环境）
- velocity_tracking_env_cfg.py: 速度跟踪任务配置
- standing_env_cfg.py: 站立任务配置
- walking_env_cfg.py: 行走任务配置
- rough_env_cfg.py: 地形训练任务配置（粗糙地形/平面）
"""

# 导入场景配置
from .jiyuan_scene_cfg import JiyuanSceneCfg

# 导入完整任务配置
from .velocity_tracking_env_cfg import VelocityTrackingEnvCfg, VELOCITY_TRACKING_ENV_CFG
from .standing_env_cfg import StandingEnvCfg, STANDING_ENV_CFG

# 导入地形训练配置
from .rough_env_cfg import (
    JiyuanRoughEnvCfg,
    JiyuanFlatEnvCfg,
    JIYUAN_ROUGH_ENV_CFG,
    JIYUAN_FLAT_ENV_CFG,
)

# 导入课程学习配置
from .curriculum_env_cfg import CurriculumEnvCfg, CURRICULUM_ENV_CFG

__all__ = [
    "JiyuanSceneCfg",
    "VelocityTrackingEnvCfg",
    "VELOCITY_TRACKING_ENV_CFG",
    "StandingEnvCfg",
    "STANDING_ENV_CFG",
    "JiyuanRoughEnvCfg",
    "JiyuanFlatEnvCfg",
    "JIYUAN_ROUGH_ENV_CFG",
    "JIYUAN_FLAT_ENV_CFG",
    "CurriculumEnvCfg",
    "CURRICULUM_ENV_CFG",
]
