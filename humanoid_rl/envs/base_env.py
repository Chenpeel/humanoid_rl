"""BaseEnv —— 通用双足机器人环境基类。

继承 Isaac Lab 的 ManagerBasedRLEnv，组合四个 Manager。
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from humanoid_rl.managers import (
    RewardsManager,
    ObservationsManager,
    TerminationsManager,
    CommandsManager,
)

from .cfg.base_cfg import BaseEnvCfg


class BaseEnv(ManagerBasedRLEnv):
    """双足机器人基础环境。

    通过 YAML 配置文件指定机器人和任务后，由 Manager 组合完成环境定义。
    """

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
