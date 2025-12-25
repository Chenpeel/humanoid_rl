"""
Jiyuan 机器人任务定义模块

包含 Jiyuan 双足机器人的各种强化学习任务环境。

当前支持的任务:
- TestEnv: 最小可行环境（用于基础验证）
- VelocityTracking: 速度跟踪任务（主要训练任务）
- Standing: 站立平衡任务（预训练/调试）
- Walking: 行走任务（专用步态训练）

环境注册:
所有环境都会在导入时自动注册到 gymnasium，可通过 gym.make() 创建。

示例:
    >>> import gymnasium as gym
    >>> # 测试环境（基础验证）
    >>> test_env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=16)
    >>>
    >>> # 速度跟踪环境（主要训练任务）
    >>> velocity_env = gym.make("Isaac-Jiyuan-Velocity-v0", num_envs=4096)
    >>>
    >>> # 站立环境（预训练）
    >>> standing_env = gym.make("Isaac-Jiyuan-Standing-v0", num_envs=4096)
    >>>
    >>> # 行走环境（步态训练）
    >>> walking_env = gym.make("Isaac-Jiyuan-Walking-v0", num_envs=4096)
"""

import gymnasium as gym

# 导入子模块
from . import envs
from . import managers
from . import utils

##
# 注册环境
##

gym.register(
    id="Isaac-Jiyuan-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "jiyuan_tasks.envs.cfg:JIYUAN_TEST_ENV_CFG",
    },
    disable_env_checker=True,
)

# 速度跟踪环境（主要训练任务）
gym.register(
    id="Isaac-Jiyuan-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "jiyuan_tasks.envs.cfg:VELOCITY_TRACKING_ENV_CFG",
    },
    disable_env_checker=True,
)

# 站立平衡环境（预训练/调试）
gym.register(
    id="Isaac-Jiyuan-Standing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "jiyuan_tasks.envs.cfg:STANDING_ENV_CFG",
    },
    disable_env_checker=True,
)

# 行走环境（专用步态训练）
gym.register(
    id="Isaac-Jiyuan-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "jiyuan_tasks.envs.cfg:WALKING_ENV_CFG",
    },
    disable_env_checker=True,
)

__all__ = [
    "envs",
    "managers",
    "utils",
]
