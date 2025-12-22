"""
Jiyuan :hºû¡šI!W

+ Jiyuan Ì³:hº„Í:f`û¡¯ƒ

SM/„û¡:
- VelocityTracking: ¦ß*û¡
- Standing: ÙËsaû¡¡-	

¯ƒèŒ:
@	¯ƒý(üeöê¨èŒ0 gymnasiumïÇ gym.make() ú

:‹:
    >>> import gymnasium as gym
    >>> env = gym.make("Isaac-Jiyuan-Velocity-Tracking-v0", num_envs=4096)
"""

import gymnasium as gym

# üeP!W
from . import envs
from . import managers
from . import utils

# TODO: (ž°¯ƒ(ÙÌèŒ¯ƒ
# from .envs.cfg.velocity_tracking_cfg import VelocityTrackingEnvCfg
#
# gym.register(
#     id="Isaac-Jiyuan-Velocity-Tracking-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": VelocityTrackingEnvCfg,
#     },
#     disable_env_checker=True,
# )

__all__ = [
    "envs",
    "managers",
    "utils",
]
