"""
行走任务奖励函数

基于开源实现（Legged Gym, Isaac Lab）的行走奖励函数。

主要参考:
- Legged Gym (ETH Zurich): https://github.com/leggedrobotics/legged_gym
- Isaac Lab Locomotion: omni.isaac.lab_tasks/locomotion
- ANYmal/Go1 机器人实现

行走任务的核心目标:
1. 前向运动 - 跟踪线速度命令
2. 步态稳定 - 保持对称的步态模式
3. 能量效率 - 最小化能量消耗
4. 姿态稳定 - 保持直立
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 导入数学工具
from ..utils.math_utils import quat_to_euler_xyz, normalize_quaternion

# 导入 Isaac Lab 管理器工具
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


##
# 步态相关奖励
##


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> Tensor:
    """脚部滑动惩罚

    惩罚脚部在接触地面时的滑动。这有助于确保机器人在支撑相保持稳定的接触，
    而不是拖拽脚部。奖励计算为脚部线速度的范数乘以接触标志。

    参考：Isaac Lab locomotion tasks

    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        asset_cfg: 机器人资产配置

    Returns:
        滑动惩罚值，形状 (num_envs,)
    """
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检测接触（使用历史力的最大值）
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # 获取资产
    asset = env.scene[asset_cfg.name]
    # 获取脚部线速度（只考虑 XY 平面）
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # 计算滑动：速度范数 * 接触标志
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    return reward


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
) -> Tensor:
    """获取脚部接触力（用于其他奖励函数）

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值

    Returns:
        接触标志，形状 (num_envs, num_feet)
    """
    if sensor_cfg_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    contact_forces = env.scene.sensors[sensor_cfg_name].data.net_forces_w
    contact_force_norm = torch.norm(contact_forces, dim=-1)

    # 检测接触（力大于阈值）
    return contact_force_norm > threshold


def gait_symmetry_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
) -> Tensor:
    """步态对称性奖励

    奖励左右脚交替接触地面的对称步态。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值

    Returns:
        对称性奖励，形状 (num_envs,)
    """
    contacts = feet_contact_forces(env, sensor_cfg_name, threshold)

    # 假设左脚是第0个，右脚是第1个
    left_contact = contacts[:, 0].float()
    right_contact = contacts[:, 1].float()

    # 奖励左右脚的互斥接触（一个接触，另一个离地）
    # XOR 逻辑：left ⊕ right
    symmetry = torch.abs(left_contact - right_contact)

    return symmetry


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
    target_air_time: float = 0.5,
) -> Tensor:
    """脚部离地时间奖励

    鼓励脚部在空中停留适当的时间（模仿人类行走）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值
        target_air_time: 目标离地时间（秒）

    Returns:
        离地时间奖励，形状 (num_envs,)

    注意:
        需要环境维护 feet_air_time 缓冲区（累计离地时间）
    """
    if not hasattr(env, "feet_air_time"):
        # 如果环境未实现，返回零奖励
        return torch.zeros(env.num_envs, device=env.device)

    contacts = feet_contact_forces(env, sensor_cfg_name, threshold)

    # 获取当前的离地时间（假设环境已累计）
    air_times = env.feet_air_time  # 形状 (num_envs, num_feet)

    # 当脚接触地面时，计算离地时间是否接近目标
    # 使用指数奖励形式
    air_time_error = torch.abs(air_times - target_air_time)
    air_time_reward = torch.exp(-air_time_error / 0.2)

    # 只在接触时计算奖励
    air_time_reward = air_time_reward * contacts.float()

    # 对所有脚求和
    return air_time_reward.sum(dim=-1)


def stance_duration_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
    target_stance_time: float = 0.3,
) -> Tensor:
    """支撑相持续时间奖励

    鼓励脚部在地面停留适当的时间。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值
        target_stance_time: 目标支撑时间（秒）

    Returns:
        支撑时间奖励，形状 (num_envs,)
    """
    if not hasattr(env, "feet_stance_time"):
        return torch.zeros(env.num_envs, device=env.device)

    contacts = feet_contact_forces(env, sensor_cfg_name, threshold)

    # 获取支撑时间
    stance_times = env.feet_stance_time  # 形状 (num_envs, num_feet)

    # 当脚离地时，计算支撑时间是否接近目标
    stance_time_error = torch.abs(stance_times - target_stance_time)
    stance_time_reward = torch.exp(-stance_time_error / 0.2)

    # 只在离地时刻计算奖励
    stance_time_reward = stance_time_reward * (~contacts).float()

    return stance_time_reward.sum(dim=-1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
    target_clearance: float = 0.05,
) -> Tensor:
    """脚部抬高奖励

    鼓励脚在摆动相抬高到适当高度（避免拖地）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值
        target_clearance: 目标抬高高度（m）

    Returns:
        抬高奖励，形状 (num_envs,)

    注意:
        需要环境提供脚部位置信息
    """
    contacts = feet_contact_forces(env, sensor_cfg_name, threshold)

    # 获取脚部高度（需要环境提供）
    if not hasattr(env, "feet_positions"):
        return torch.zeros(env.num_envs, device=env.device)

    feet_positions = env.feet_positions  # 形状 (num_envs, num_feet, 3)
    feet_heights = feet_positions[:, :, 2]  # Z坐标

    # 只在摆动相（脚离地）时计算
    swing_phase = ~contacts

    # 计算高度误差
    height_error = torch.abs(feet_heights - target_clearance)
    clearance_reward = torch.exp(-height_error / 0.02)

    # 只在摆动相有奖励
    clearance_reward = clearance_reward * swing_phase.float()

    return clearance_reward.sum(dim=-1)


##
# 步态频率和周期
##


def stride_frequency_reward(
    env: ManagerBasedRLEnv,
    target_frequency: float = 2.0,
) -> Tensor:
    """步态频率奖励

    鼓励保持目标的步态频率（步/秒）。

    Args:
        env: 环境实例
        target_frequency: 目标步态频率（Hz）

    Returns:
        频率奖励，形状 (num_envs,)

    注意:
        需要环境跟踪步数和时间
    """
    if not hasattr(env, "step_count") or not hasattr(env, "episode_time"):
        return torch.zeros(env.num_envs, device=env.device)

    step_count = env.step_count  # 累计步数
    episode_time = env.episode_time  # episode 时间（秒）

    # 计算当前频率
    current_frequency = step_count / (episode_time + 1e-6)

    # 频率误差
    frequency_error = torch.abs(current_frequency - target_frequency)
    return torch.exp(-frequency_error / 0.5)


##
# 行走特定的惩罚
##


def stumbling_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 1.0,
) -> Tensor:
    """绊倒惩罚

    惩罚脚部在摆动相意外接触地面。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 接触力阈值

    Returns:
        绊倒惩罚值，形状 (num_envs,)
    """
    if not hasattr(env, "feet_in_swing_phase"):
        return torch.zeros(env.num_envs, device=env.device)

    contacts = feet_contact_forces(env, sensor_cfg_name, threshold)
    swing_phase = env.feet_in_swing_phase  # 形状 (num_envs, num_feet)

    # 如果脚应该在摆动相但接触了地面，施加惩罚
    stumbling = contacts.float() * swing_phase.float()

    return stumbling.sum(dim=-1)


def drag_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "contact_forces",
    threshold: float = 0.5,
) -> Tensor:
    """拖地惩罚

    惩罚脚部在摆动相拖地（低高度+小接触力）。

    Args:
        env: 环境实例
        sensor_cfg_name: 接触传感器名称
        threshold: 拖地检测阈值

    Returns:
        拖地惩罚值，形状 (num_envs,)
    """
    if not hasattr(env, "feet_positions"):
        return torch.zeros(env.num_envs, device=env.device)

    # 获取脚部高度和接触力
    feet_heights = env.feet_positions[:, :, 2]
    contact_forces = env.scene.sensors[sensor_cfg_name].data.net_forces_w
    contact_force_norm = torch.norm(contact_forces, dim=-1)

    # 检测拖地：低高度且有小接触力
    is_dragging = (feet_heights < 0.02) & (contact_force_norm > threshold) & (contact_force_norm < 10.0)

    return is_dragging.float().sum(dim=-1)


##
# 躯干稳定性（行走专用）
##


def trunk_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.35,
    tolerance: float = 0.05,
) -> Tensor:
    """躯干高度保持奖励（行走时）

    与站立任务不同，行走时允许小幅度的上下波动。

    Args:
        env: 环境实例
        target_height: 目标高度
        tolerance: 容差（更宽松）

    Returns:
        高度奖励，形状 (num_envs,)
    """
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]
    height_error = torch.abs(torso_z - target_height)
    return torch.exp(-height_error / tolerance)


def trunk_orientation_penalty(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.3,
) -> Tensor:
    """躯干过度倾斜惩罚

    行走时允许小幅度倾斜，但惩罚过度倾斜。

    Args:
        env: 环境实例
        max_tilt: 最大允许倾斜（弧度）

    Returns:
        倾斜惩罚，形状 (num_envs,)
    """
    base_quat = env.scene["robot"].data.root_quat_w
    base_quat = normalize_quaternion(base_quat)
    euler = quat_to_euler_xyz(base_quat)
    roll, pitch = euler[:, 0], euler[:, 1]

    # 只惩罚超过阈值的倾斜
    roll_penalty = torch.clamp(torch.abs(roll) - max_tilt, min=0.0)
    pitch_penalty = torch.clamp(torch.abs(pitch) - max_tilt, min=0.0)

    return roll_penalty + pitch_penalty


def trunk_lin_vel_z_penalty(env: ManagerBasedRLEnv) -> Tensor:
    """躯干Z方向速度惩罚

    行走时躯干应该平稳移动，避免大幅度上下跳动。

    Args:
        env: 环境实例

    Returns:
        Z速度惩罚，形状 (num_envs,)
    """
    base_lin_vel = env.scene["robot"].data.root_lin_vel_w
    return torch.square(base_lin_vel[:, 2])


##
# 前向运动奖励
##


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    target_velocity: float = 0.5,
) -> Tensor:
    """前向速度奖励（简化版，无命令）

    鼓励机器人前向移动。

    Args:
        env: 环境实例
        target_velocity: 目标前向速度（m/s）

    Returns:
        前向速度奖励，形状 (num_envs,)
    """
    base_lin_vel = env.scene["robot"].data.root_lin_vel_w
    forward_vel = base_lin_vel[:, 0]  # X方向

    # 速度误差
    vel_error = torch.abs(forward_vel - target_velocity)
    return torch.exp(-vel_error / 0.5)


##
# 行走任务默认奖励权重
##


WALKING_REWARD_WEIGHTS = {
    # 主要目标：前向运动
    "track_lin_vel_xy": 1.5,  # 速度跟踪
    "track_ang_vel_z": 0.5,  # 转向
    # 步态质量
    "gait_symmetry": 0.5,  # 步态对称性
    "feet_air_time": 0.3,  # 离地时间
    "stance_duration": 0.3,  # 支撑时间
    "foot_clearance": 0.2,  # 脚部抬高
    # 躯干稳定
    "trunk_height": 0.5,  # 高度保持
    "orientation": 0.3,  # 姿态稳定
    "trunk_lin_vel_z": -1.0,  # Z方向速度惩罚
    "trunk_tilt": -0.5,  # 过度倾斜惩罚
    # 步态惩罚
    "stumbling": -2.0,  # 绊倒
    "drag": -1.0,  # 拖地
    # 能量效率
    "action_rate": -0.01,  # 动作平滑
    "joint_powers": -2.0e-5,  # 功率消耗
    # 存活
    "alive": 0.5,
}
