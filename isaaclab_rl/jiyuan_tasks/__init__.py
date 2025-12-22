"""
Jiyuan 机器人任务定义模块

包含 Jiyuan 双足机器人的各种强化学习任务环境。

当前支持的任务:
- TestEnv: 最小可行环境（用于基础验证）
- VelocityTracking: 速度跟踪任务（计划中）
- Standing: 站立平衡任务（计划中）

环境注册:
所有环境都会在导入时自动注册到 gymnasium，可通过 gym.make() 创建。

示例:
    >>> import gymnasium as gym
    >>> env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=16)
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
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_rl.jiyuan_tasks.envs.cfg:JIYUAN_TEST_ENV_CFG",
    },
    disable_env_checker=True,
)

# TODO: 在实现完整环境后，在这里注册更多环境
# gym.register(
#     id="Isaac-Jiyuan-Velocity-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": "isaaclab_rl.jiyuan_tasks.envs.cfg:VelocityTrackingEnvCfg",
#     },
#     disable_env_checker=True,
# )

__all__ = [
    "envs",
    "managers",
    "utils",
]
