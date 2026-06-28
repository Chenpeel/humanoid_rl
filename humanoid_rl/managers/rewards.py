"""RewardsManager —— 奖励项计算与加权。

奖励项列表（各任务共享，通过 YAML 权重控制启用/停用）：
  1. 躯干倾角惩罚       (roll + pitch)
  2. 躯干高度惩罚       (deviation from target)
  3. 线速度跟踪奖励     (vx / vy)
  4. 角速度跟踪奖励     (yaw_rate)
  5. 关节加速度惩罚     (smoothness)
  6. 关节力矩惩罚       (efficiency)
  7. 关节位置惩罚       (deviation from default pose)
  8. 足部接触力惩罚     (soft landing)
  9. 生存奖励           (alive bonus)
 10. 摆动腿对称奖励     (gait symmetry)
 11. 躯干角速度惩罚     (gyro stabilization)
 12. 动作变化惩罚       (smooth action)
 13. 足部滑动惩罚       (foot slip)
 14. 步高奖励           (foot clearance)
 15. 工作空间边界惩罚   (joint limits)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardsManager:
    """奖励管理器。

    每个 reward term 是 torch.jit.script 独立的函数，权重从 YAML 加载。
    最终奖励 = sum(weight_i * term_i)。
    """

    def __init__(self, cfg: object, env: "ManagerBasedRLEnv"):
        self.cfg = cfg
        self.env = env
        self._terms: list[tuple[float, callable]] = []
        self._build_terms()

    def _build_terms(self):
        """根据配置构建启用的奖励项。"""
        # 实际实现中，terms 由 hydra 从 YAML 解析注入
        pass

    def compute(self) -> torch.Tensor:
        """计算所有环境的总奖励。

        Returns:
            (num_envs,) 各环境的标量奖励
        """
        total = torch.zeros(self.env.num_envs, device=self.env.device)
        for weight, term_fn in self._terms:
            total += weight * term_fn()
        return total

    def add_term(self, name: str, weight: float, fn: callable):
        """动态添加奖励项。"""
        self._terms.append((weight, fn))

    # ------------------------------------------------------------------
    # 以下为具体奖励函数（静态方法，便于 JIT 序列化）
    # ------------------------------------------------------------------

    @staticmethod
    def torso_orientation_penalty(torso_quat: torch.Tensor) -> torch.Tensor:
        """躯干倾角惩罚：roll² + pitch² 越大惩罚越重。"""
        roll, pitch = _quat_to_rp(torso_quat)
        return roll.pow(2) + pitch.pow(2)

    @staticmethod
    def torso_height_penalty(
        torso_height: torch.Tensor, target_height: float
    ) -> torch.Tensor:
        """躯干高度偏差惩罚。"""
        return (torso_height - target_height).pow(2)

    @staticmethod
    def lin_vel_tracking_reward(
        lin_vel: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """线速度跟踪奖励（指数衰减）。"""
        error = torch.norm(lin_vel[:, :2] - command[:, :2], dim=1)
        return torch.exp(-error / 0.25)

    @staticmethod
    def ang_vel_tracking_reward(
        ang_vel: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """角速度跟踪奖励。"""
        error = (ang_vel[:, 2] - command[:, 2]).abs()
        return torch.exp(-error / 0.25)

    @staticmethod
    def joint_acc_penalty(joint_acc: torch.Tensor) -> torch.Tensor:
        """关节加速度平方和（平滑性）。"""
        return joint_acc.pow(2).sum(dim=1)

    @staticmethod
    def joint_torque_penalty(torques: torch.Tensor) -> torch.Tensor:
        """关节力矩平方和（能效）。"""
        return torques.pow(2).sum(dim=1)

    @staticmethod
    def joint_pos_penalty(
        joint_pos: torch.Tensor, default_pos: torch.Tensor
    ) -> torch.Tensor:
        """偏离默认姿态惩罚。"""
        return (joint_pos - default_pos).pow(2).sum(dim=1)

    @staticmethod
    def alive_reward() -> float:
        """生存奖励（常量）。"""
        return 1.0


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _quat_to_rp(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """四元数 → roll, pitch（简化，适用小角度）。"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.pow(2) + y.pow(2)))
    pitch = torch.asin(2 * (w * y - z * x))
    return roll, pitch
