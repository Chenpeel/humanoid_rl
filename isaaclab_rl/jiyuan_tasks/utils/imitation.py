"""
模仿学习框架（Imitation Learning）

支持从参考动作（BVH/FBX）学习机器人运动。

主要功能:
1. 加载和预处理参考轨迹（BVH/FBX → 关节角度）
2. 轨迹重定向（Retargeting）- 人体动作映射到机器人
3. 模仿奖励函数（Pose Matching, Velocity Matching）
4. 相位变量跟踪

参考:
- AMP (Adversarial Motion Priors): https://arxiv.org/abs/2104.02180
- PHC (Perpetual Humanoid Control): https://arxiv.org/abs/2305.06456
- Isaac Lab Motion Imitation Examples

使用场景:
- 学习人类步态
- 学习复杂动作序列（跳跃、转身等）
- 数据驱动的运动生成
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# 参考轨迹数据结构
##


class ReferenceMotion:
    """参考动作轨迹

    存储和管理参考动作数据（如 BVH/FBX 转换后的关节角度序列）。

    数据格式:
    - joint_positions: (num_frames, num_joints) - 关节位置
    - joint_velocities: (num_frames, num_joints) - 关节速度
    - root_positions: (num_frames, 3) - 根节点位置
    - root_orientations: (num_frames, 4) - 根节点四元数
    - timesteps: (num_frames,) - 时间戳
    """

    def __init__(
        self,
        joint_positions: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None,
        root_positions: Optional[np.ndarray] = None,
        root_orientations: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
        fps: float = 60.0,
    ):
        """初始化参考动作

        Args:
            joint_positions: 关节位置序列，形状 (num_frames, num_joints)
            joint_velocities: 关节速度序列（可选）
            root_positions: 根节点位置序列（可选）
            root_orientations: 根节点朝向序列（可选）
            timesteps: 时间戳序列（可选，如果为 None 则根据 fps 生成）
            fps: 帧率（仅当 timesteps 为 None 时使用）
        """
        self.joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        self.num_frames, self.num_joints = joint_positions.shape

        # 计算或使用提供的速度
        if joint_velocities is not None:
            self.joint_velocities = torch.tensor(joint_velocities, dtype=torch.float32)
        else:
            # 数值微分
            dt = 1.0 / fps
            self.joint_velocities = torch.diff(self.joint_positions, dim=0) / dt
            # 在末尾复制最后一帧
            self.joint_velocities = torch.cat([self.joint_velocities, self.joint_velocities[-1:, :]], dim=0)

        # 根节点数据
        if root_positions is not None:
            self.root_positions = torch.tensor(root_positions, dtype=torch.float32)
        else:
            self.root_positions = None

        if root_orientations is not None:
            self.root_orientations = torch.tensor(root_orientations, dtype=torch.float32)
        else:
            self.root_orientations = None

        # 时间戳
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, dtype=torch.float32)
        else:
            self.timesteps = torch.arange(self.num_frames, dtype=torch.float32) / fps

        self.duration = self.timesteps[-1].item()
        print(f"[Imitation] 参考动作加载完成")
        print(f"  帧数: {self.num_frames}")
        print(f"  关节数: {self.num_joints}")
        print(f"  时长: {self.duration:.2f}s")

    def get_frame(self, time: float, loop: bool = True) -> Dict[str, torch.Tensor]:
        """获取指定时间的参考帧

        Args:
            time: 时间（秒）
            loop: 是否循环播放

        Returns:
            包含 joint_pos, joint_vel, root_pos, root_quat 的字典
        """
        # 循环或限制时间
        if loop:
            time = time % self.duration
        else:
            time = min(time, self.duration)

        # 找到对应的帧（线性插值）
        idx = torch.searchsorted(self.timesteps, torch.tensor(time))
        idx = torch.clamp(idx, 0, self.num_frames - 2)

        # 计算插值系数
        t0 = self.timesteps[idx]
        t1 = self.timesteps[idx + 1]
        alpha = (time - t0) / (t1 - t0 + 1e-8)

        # 插值关节位置和速度
        joint_pos = (1 - alpha) * self.joint_positions[idx] + alpha * self.joint_positions[idx + 1]
        joint_vel = (1 - alpha) * self.joint_velocities[idx] + alpha * self.joint_velocities[idx + 1]

        result = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

        # 插值根节点数据
        if self.root_positions is not None:
            root_pos = (1 - alpha) * self.root_positions[idx] + alpha * self.root_positions[idx + 1]
            result["root_pos"] = root_pos

        if self.root_orientations is not None:
            # 四元数需要球面线性插值（SLERP）
            root_quat = self._slerp(self.root_orientations[idx], self.root_orientations[idx + 1], alpha)
            result["root_quat"] = root_quat

        return result

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """四元数球面线性插值（SLERP）

        Args:
            q0: 起始四元数
            q1: 目标四元数
            t: 插值系数 [0, 1]

        Returns:
            插值后的四元数
        """
        # 计算点积
        dot = torch.sum(q0 * q1)

        # 如果点积为负，反转一个四元数以取短路径
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # 如果四元数非常接近，使用线性插值
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / torch.norm(result)

        # 计算角度
        theta_0 = torch.acos(dot)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        sin_theta_0 = torch.sin(theta_0)

        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q0 + s1 * q1


##
# 模仿奖励函数
##


def pose_matching_reward(
    env: ManagerBasedRLEnv,
    reference_motion: ReferenceMotion,
    phase_variable_name: str = "motion_phase",
    weight_pos: float = 1.0,
    weight_vel: float = 0.1,
) -> Tensor:
    """姿态匹配奖励

    奖励机器人姿态与参考动作的匹配程度。

    Args:
        env: 环境实例
        reference_motion: 参考动作轨迹
        phase_variable_name: 相位变量名称（环境中存储的属性）
        weight_pos: 位置权重
        weight_vel: 速度权重

    Returns:
        姿态匹配奖励，形状 (num_envs,)

    注意:
        需要环境维护 motion_phase 变量（当前播放时间）
    """
    if not hasattr(env, phase_variable_name):
        return torch.zeros(env.num_envs, device=env.device)

    # 获取当前相位
    phases = getattr(env, phase_variable_name)  # 形状 (num_envs,)

    # 获取当前关节状态
    joint_pos = env.scene["robot"].data.joint_pos  # (num_envs, num_joints)
    joint_vel = env.scene["robot"].data.joint_vel

    # 批量获取参考帧
    rewards = torch.zeros(env.num_envs, device=env.device)

    for env_id, phase in enumerate(phases):
        ref_frame = reference_motion.get_frame(phase.item())

        # 计算位置误差
        pos_error = torch.sum(torch.square(joint_pos[env_id] - ref_frame["joint_pos"]))

        # 计算速度误差
        vel_error = torch.sum(torch.square(joint_vel[env_id] - ref_frame["joint_vel"]))

        # 指数奖励
        reward = torch.exp(-(weight_pos * pos_error + weight_vel * vel_error))
        rewards[env_id] = reward

    return rewards


def root_pose_matching_reward(
    env: ManagerBasedRLEnv,
    reference_motion: ReferenceMotion,
    phase_variable_name: str = "motion_phase",
) -> Tensor:
    """根节点姿态匹配奖励

    奖励机器人根节点（base）与参考动作的匹配。

    Args:
        env: 环境实例
        reference_motion: 参考动作轨迹
        phase_variable_name: 相位变量名称

    Returns:
        根节点姿态奖励，形状 (num_envs,)
    """
    if not hasattr(env, phase_variable_name):
        return torch.zeros(env.num_envs, device=env.device)

    if reference_motion.root_positions is None:
        return torch.zeros(env.num_envs, device=env.device)

    phases = getattr(env, phase_variable_name)

    # 获取当前根节点状态
    root_pos = env.scene["robot"].data.root_pos_w  # (num_envs, 3)
    root_quat = env.scene["robot"].data.root_quat_w  # (num_envs, 4)

    rewards = torch.zeros(env.num_envs, device=env.device)

    for env_id, phase in enumerate(phases):
        ref_frame = reference_motion.get_frame(phase.item())

        # 位置误差
        pos_error = torch.sum(torch.square(root_pos[env_id] - ref_frame["root_pos"]))

        # 朝向误差（如果有）
        if "root_quat" in ref_frame:
            # 四元数误差可以用点积衡量（越接近1越相似）
            quat_similarity = torch.abs(torch.sum(root_quat[env_id] * ref_frame["root_quat"]))
            quat_error = 1.0 - quat_similarity
        else:
            quat_error = 0.0

        # 综合奖励
        reward = torch.exp(-(pos_error + quat_error))
        rewards[env_id] = reward

    return rewards


##
# 相位变量管理
##


def update_motion_phase(env: ManagerBasedRLEnv, dt: float, loop: bool = True):
    """更新动作相位变量

    Args:
        env: 环境实例
        dt: 时间步长
        loop: 是否循环播放

    注意:
        这应该在环境的 step() 中调用
    """
    if not hasattr(env, "motion_phase"):
        env.motion_phase = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "motion_duration"):
        # 默认时长（需要从 reference_motion 获取）
        env.motion_duration = 1.0

    # 增加相位
    env.motion_phase += dt

    # 循环或截断
    if loop:
        env.motion_phase = env.motion_phase % env.motion_duration
    else:
        env.motion_phase = torch.clamp(env.motion_phase, 0, env.motion_duration)


##
# BVH/FBX 加载器（占位符）
##


def load_motion_from_bvh(filepath: str) -> ReferenceMotion:
    """从 BVH 文件加载参考动作

    Args:
        filepath: BVH 文件路径

    Returns:
        ReferenceMotion 实例

    TODO:
        实现 BVH 解析和重定向（Retargeting）
        可使用库: python-bvh, ezc3d, pybvh
    """
    raise NotImplementedError(
        "BVH 加载器尚未实现。请使用以下库之一:\n"
        "  - python-bvh: pip install bvh\n"
        "  - pybvh: pip install pybvh\n"
        "然后实现骨骼重定向（人体 → 机器人关节映射）"
    )


def load_motion_from_fbx(filepath: str) -> ReferenceMotion:
    """从 FBX 文件加载参考动作

    Args:
        filepath: FBX 文件路径

    Returns:
        ReferenceMotion 实例

    TODO:
        实现 FBX 解析和重定向
        可使用库: FBX SDK (Python bindings)
    """
    raise NotImplementedError(
        "FBX 加载器尚未实现。请使用 FBX SDK:\n"
        "  - Autodesk FBX SDK: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0\n"
        "然后实现骨骼重定向"
    )


##
# 模仿学习环境配置扩展
##

# 要在环境配置中添加模仿学习，需要:
#
# 1. 在观测空间中添加参考动作（可选）:
#    reference_joint_pos = ObsTerm(
#        func=get_reference_joint_pos,
#        params={"reference_motion": motion, "phase_var": "motion_phase"}
#    )
#
# 2. 在奖励函数中添加模仿奖励:
#    pose_matching = RewTerm(
#        func=pose_matching_reward,
#        weight=1.0,
#        params={"reference_motion": motion}
#    )
#
# 3. 在环境的 step() 中更新相位:
#    def step(self, actions):
#        ...
#        update_motion_phase(self, self.dt)
#        ...
#
# 示例配置见下方注释

"""
示例：使用模仿学习的环境配置

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

# 加载参考动作（需要先实现加载器）
# reference_motion = load_motion_from_bvh("path/to/walk.bvh")

@configclass
class ImitationWalkEnvCfg(VelocityTrackingEnvCfg):
    \"\"\"模仿学习行走环境\"\"\"

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # ... 原有观测 ...

            # 添加参考动作观测（可选）
            reference_joint_pos = ObsTerm(
                func=get_reference_joint_pos,
                params={"reference_motion": reference_motion}
            )

    @configclass
    class RewardsCfg:
        # 模仿奖励（主要）
        pose_matching = RewTerm(
            func=pose_matching_reward,
            weight=2.0,
            params={"reference_motion": reference_motion}
        )

        root_pose_matching = RewTerm(
            func=root_pose_matching_reward,
            weight=1.0,
            params={"reference_motion": reference_motion}
        )

        # ... 其他奖励 ...

    def __post_init__(self):
        super().__post_init__()
        # 初始化相位变量
        self.motion_duration = reference_motion.duration
```
"""
