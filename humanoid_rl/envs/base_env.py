"""BaseEnv —— 通用双足机器人环境基类。

继承 Isaac Lab 的 ManagerBasedRLEnv，组合四个 Manager。
仅在 Linux + GPU 环境下可用，macOS 上导入时优雅降级。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .cfg.base_cfg import BaseEnvCfg

_ISAAC_AVAILABLE = False
try:
    from isaaclab.envs import ManagerBasedRLEnv  # noqa: F811
    _ISAAC_AVAILABLE = True
except ImportError:
    pass


if _ISAAC_AVAILABLE:
    class BaseEnv(ManagerBasedRLEnv):
        """双足机器人基础环境。"""
        cfg: BaseEnvCfg

        def __init__(self, cfg: BaseEnvCfg, render_interval: int = 1):
            super().__init__(cfg, render_interval)

        def _get_observations(self) -> dict[str, torch.Tensor]:
            return self.observations_manager.compute()

        def _get_rewards(self) -> torch.Tensor:
            return self.rewards_manager.compute()

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self.terminations_manager.compute()

        def _reset_idx(self, env_ids: torch.Tensor | None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            super()._reset_idx(env_ids)
else:
    class BaseEnv:
        """占位类 —— Isaac Lab 不可用。"""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "BaseEnv requires Isaac Lab (Linux + NVIDIA GPU). "
                "On macOS, use tests/ for code validation only."
            )
