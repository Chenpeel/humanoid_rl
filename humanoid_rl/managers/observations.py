"""ObservationsManager —— 观测空间定义与数据提取。

观测向量组成（拼接为一个 1D 向量）：
  策略观测 (Policy)        - 输入给 Actor-Critic
  特权观测 (Privileged)    - 输入给 Critic 但不给 Actor（Teacher-Student）
  评价观测 (Critic)        - 完整的 critic 观测

典型观测成分：
  - 关节位置（rad）
  - 关节速度（rad/s）
  - 上一帧动作
  - IMU：躯干角速度（gyro）
  - IMU：重力方向（projected gravity）
  - 速度指令（command）
  - 关节位置误差（相对默认姿态）
  - 躯干线速度（可选）
  - 足部接触状态（可选）
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ObservationsManager:
    """观测管理器。

    负责从仿真状态中提取观测向量，并应用归一化和噪声。
    """

    def __init__(self, cfg: object, env: "ManagerBasedRLEnv"):
        self.cfg = cfg
        self.env = env
        self.num_obs = getattr(cfg, "num_obs", 48)
        self.obs_scales = getattr(cfg, "obs_scales", {})

    def compute(self) -> dict[str, torch.Tensor]:
        """计算策略观测。

        Returns:
            {"policy": (num_envs, num_obs)} 观测字典
        """
        obs_list = []

        # 1. 关节位置（归一化到 [-1, 1]）
        joint_pos = self.env.joint_pos - self.env.default_joint_pos
        joint_pos = joint_pos * self.obs_scales.get("joint_pos", 1.0)
        obs_list.append(joint_pos)

        # 2. 关节速度（归一化）
        joint_vel = self.env.joint_vel * self.obs_scales.get("joint_vel", 1.0)
        obs_list.append(joint_vel)

        # 3. 上一帧动作
        obs_list.append(self.env.action)

        # 4. IMU: 躯干角速度
        ang_vel = self.env.root_ang_vel_b * self.obs_scales.get("ang_vel", 0.25)
        obs_list.append(ang_vel)

        # 5. IMU: 重力方向（projected gravity）
        projected_gravity = self.env.projected_gravity
        obs_list.append(projected_gravity)

        # 6. 速度指令（如果存在）
        if hasattr(self.env, "command") and self.env.command is not None:
            obs_list.append(self.env.command[:, :3] * self.obs_scales.get("command", 1.0))

        # 拼接
        policy_obs = torch.cat(obs_list, dim=-1)

        # 裁剪到指定维度
        policy_obs = policy_obs[:, : self.num_obs]

        return {"policy": policy_obs}
