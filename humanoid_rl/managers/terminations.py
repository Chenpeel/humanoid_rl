"""TerminationsManager —— 终止条件判断。

终止条件：
  1. 躯干倾角过大      (roll > threshold 或 pitch > threshold)
  2. 躯干高度过低      (torso_height < min_height)
  3. 关节超限          (joint position 超出允许范围)
  4. 超时              (episode_length > max_episode_length)
  5. 基座线速度过大    (可选，用于站立任务)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerminationsManager:
    """终止条件管理器。

    返回两个 mask：
      terminated     - 环境失败（跌倒等）
      time_out       - 正常运行超时
    """

    def __init__(self, cfg: object, env: "ManagerBasedRLEnv"):
        self.cfg = cfg
        self.env = env

        # 阈值（从 YAML 可覆盖）
        self.max_roll = getattr(cfg, "max_roll", 0.8)        # rad
        self.max_pitch = getattr(cfg, "max_pitch", 0.8)      # rad
        self.min_torso_height = getattr(cfg, "min_torso_height", 0.6)  # m
        self.max_episode_length_s = getattr(cfg, "max_episode_length_s", 20.0)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算终止与超时。

        Returns:
            terminated: (num_envs,) 失败终止
            time_out:   (num_envs,) 超时终止
        """
        num_envs = self.env.num_envs
        device = self.env.device

        # ---- 失败终止 ----
        # 躯干倾角过大
        roll, pitch = self._get_torso_rp()
        too_tilted = (roll.abs() > self.max_roll) | (pitch.abs() > self.max_pitch)

        # 躯干高度过低
        torso_height = self.env.root_pos_w[:, 2]
        too_low = torso_height < self.min_torso_height

        terminated = too_tilted | too_low

        # ---- 超时 ----
        episode_length = self.env.episode_length_buf
        max_steps = int(
            self.max_episode_length_s / (self.env.step_dt * self.env.cfg.sim.render_interval)
        )
        time_out = episode_length >= max_steps

        return terminated, time_out

    def _get_torso_rp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """从躯干四元数计算 roll, pitch。"""
        quat = self.env.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.pow(2) + y.pow(2)))
        pitch = torch.asin(2 * (w * y - z * x))
        return roll, pitch
