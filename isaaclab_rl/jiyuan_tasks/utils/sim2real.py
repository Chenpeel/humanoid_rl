"""
Sim2Real 映射层

用于将仿真中的动作映射到实体机器人硬件。

主要功能:
1. 调用并联机构求解器（3-parallel-dot ankle）
2. 工作空间限制和安全检查
3. 动作后处理和滤波

集成:
- ROS 包: parallel_3dof_controller
- 运动学求解器: Parallel3DOFKinematicsSolver
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# 并联脚踝映射接口
##


class ParallelAnkleMapper:
    """3-DOF 并联脚踝映射器

    将仿真中的 ankle roll/pitch/yaw 映射到实体机器人的舵机命令。

    工作流程:
    1. 从仿真动作中提取 ankle RPY（假设在动作空间的特定位置）
    2. 调用 ROS 运动学求解器: RPY → theta角 → 舵机位置
    3. 应用工作空间限制（±30°）
    4. 返回映射后的舵机命令

    注意:
        这是 Sim2Real 的关键接口，仅在实体机器人部署时使用。
        仿真训练时不需要此映射（使用简化的串联模型）。
    """

    def __init__(
        self,
        ankle_indices: Dict[str, List[int]],
        l0: float = 0.02,
        l1: float = 0.01,
        l2: float = 0.03,
        enable_filtering: bool = True,
        filter_alpha: float = 0.7,
    ):
        """初始化并联脚踝映射器

        Args:
            ankle_indices: 脚踝关节在动作空间中的索引
                格式: {"left": [idx_roll, idx_pitch, idx_yaw],
                      "right": [idx_roll, idx_pitch, idx_yaw]}
            l0: 平台半径 (m)
            l1: 动平台距离 (m)
            l2: 静平台距离 (m)
            enable_filtering: 是否启用低通滤波（平滑命令）
            filter_alpha: 滤波系数（0-1，越大越平滑）
        """
        self.ankle_indices = ankle_indices
        self.enable_filtering = enable_filtering
        self.filter_alpha = filter_alpha

        # 运动学求解器（延迟导入 ROS 包）
        self.solver = None
        self._init_solver(l0, l1, l2)

        # 滤波缓冲区
        self.last_servo_commands = None

        # 工作空间限制（弧度）
        self.rpy_limits = {
            "roll": (-np.pi / 6, np.pi / 6),  # ±30°
            "pitch": (-np.pi / 6, np.pi / 6),
            "yaw": (-np.pi / 6, np.pi / 6),
        }

        print(f"[Sim2Real] ParallelAnkleMapper 初始化完成")
        print(f"  左脚踝索引: {ankle_indices['left']}")
        print(f"  右脚踝索引: {ankle_indices['right']}")
        print(f"  工作空间: ±30°")
        print(f"  滤波: {'启用' if enable_filtering else '禁用'}")

    def _init_solver(self, l0: float, l1: float, l2: float):
        """初始化运动学求解器（延迟导入）"""
        try:
            # 尝试导入 ROS 包
            import sys

            ros_path = "/home/chenpeel/work/repo/jiyuan/ros/src/parallel_3dof_controller"
            if ros_path not in sys.path:
                sys.path.insert(0, ros_path)

            from parallel_3dof_controller.kinematics_solver import Parallel3DOFKinematicsSolver

            self.solver = Parallel3DOFKinematicsSolver(l0=l0, l1=l1, l2=l2)
            print(f"[Sim2Real] ROS 运动学求解器加载成功")

        except ImportError as e:
            print(f"[Sim2Real] 警告: 无法导入 ROS 运动学求解器")
            print(f"  错误: {e}")
            print(f"  仿真训练时无需此模块，部署时请确保 ROS 环境正确配置")
            self.solver = None

    def clip_rpy(self, roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
        """限制 RPY 在工作空间内

        Args:
            roll, pitch, yaw: 角度（弧度）

        Returns:
            裁剪后的 (roll, pitch, yaw)
        """
        roll = np.clip(roll, *self.rpy_limits["roll"])
        pitch = np.clip(pitch, *self.rpy_limits["pitch"])
        yaw = np.clip(yaw, *self.rpy_limits["yaw"])
        return roll, pitch, yaw

    def map_action(self, action: np.ndarray, ankle_side: str) -> List[Dict]:
        """映射动作到舵机命令

        Args:
            action: 完整的动作向量（包含所有关节）
            ankle_side: 'left' 或 'right'

        Returns:
            舵机命令列表 [{'id': int, 'position': int, 'speed': int}, ...]

        注意:
            如果 solver 未初始化，返回空列表（仅在仿真中使用）
        """
        if self.solver is None:
            return []

        # 提取脚踝 RPY
        indices = self.ankle_indices[ankle_side]
        roll = action[indices[0]]
        pitch = action[indices[1]]
        yaw = action[indices[2]]

        # 限制工作空间
        roll, pitch, yaw = self.clip_rpy(roll, pitch, yaw)

        # 调用运动学求解器
        commands = self.solver.rpy_to_servo_commands(roll, pitch, yaw, ankle_side=ankle_side, speed=100)  # 默认速度

        # 应用滤波
        if self.enable_filtering:
            commands = self._apply_filtering(commands, ankle_side)

        return commands

    def _apply_filtering(self, commands: List[Dict], ankle_side: str) -> List[Dict]:
        """应用低通滤波平滑命令

        Args:
            commands: 当前舵机命令
            ankle_side: 脚踝侧

        Returns:
            滤波后的命令
        """
        if self.last_servo_commands is None:
            self.last_servo_commands = {ankle_side: commands}
            return commands

        if ankle_side not in self.last_servo_commands:
            self.last_servo_commands[ankle_side] = commands
            return commands

        last_commands = self.last_servo_commands[ankle_side]
        filtered_commands = []

        for curr_cmd, last_cmd in zip(commands, last_commands):
            # 低通滤波: position_new = α × position_last + (1-α) × position_curr
            filtered_position = int(
                self.filter_alpha * last_cmd["position"] + (1 - self.filter_alpha) * curr_cmd["position"]
            )

            filtered_cmd = curr_cmd.copy()
            filtered_cmd["position"] = filtered_position
            filtered_commands.append(filtered_cmd)

        self.last_servo_commands[ankle_side] = filtered_commands
        return filtered_commands

    def map_batch_actions(self, actions: torch.Tensor) -> Dict[str, List[Dict]]:
        """批量映射动作（用于多环境）

        Args:
            actions: 批量动作，形状 (num_envs, action_dim)

        Returns:
            字典，格式: {env_id: {'left': [...], 'right': [...]}}

        注意:
            实际部署时通常只有单个环境，此方法用于调试
        """
        if self.solver is None:
            return {}

        results = {}
        actions_np = actions.cpu().numpy()

        for env_id, action in enumerate(actions_np):
            left_commands = self.map_action(action, "left")
            right_commands = self.map_action(action, "right")

            results[env_id] = {
                "left": left_commands,
                "right": right_commands,
            }

        return results


##
# 工作空间约束奖励（Sim2Real 准备）
##


def ankle_workspace_penalty(
    env: ManagerBasedRLEnv,
    ankle_indices: Dict[str, List[int]],
    margin: float = 0.1,
) -> Tensor:
    """脚踝工作空间约束惩罚

    惩罚脚踝 RPY 超出实体机器人工作空间（±30°）的动作。

    Args:
        env: 环境实例
        ankle_indices: 脚踝关节索引
        margin: 安全边界（弧度）

    Returns:
        工作空间惩罚，形状 (num_envs,)

    注意:
        这是 Sim2Real 的软约束，训练时鼓励策略在安全范围内。
    """
    # 获取当前动作
    actions = env.action_manager.action

    # 工作空间限制
    max_angle = np.pi / 6  # ±30°
    safe_limit = max_angle - margin

    penalty = torch.zeros(env.num_envs, device=env.device)

    # 检查左脚踝
    for idx in ankle_indices["left"]:
        angle = actions[:, idx]
        violation = torch.clamp(torch.abs(angle) - safe_limit, min=0.0)
        penalty += violation

    # 检查右脚踝
    for idx in ankle_indices["right"]:
        angle = actions[:, idx]
        violation = torch.clamp(torch.abs(angle) - safe_limit, min=0.0)
        penalty += violation

    return penalty


##
# 使用示例（仅供参考）
##

# 仿真训练时：
# - 使用简化的串联关节模型训练
# - 添加 ankle_workspace_penalty 软约束
# - 动作空间：直接输出关节角度（包括脚踝 RPY）
#
# 实体部署时：
# - 创建 ParallelAnkleMapper 实例
# - 策略输出动作后，调用 map_action() 转换
# - 将舵机命令发送到硬件控制器
#
# 示例代码:
# ```python
# # 初始化映射器
# mapper = ParallelAnkleMapper(
#     ankle_indices={
#         "left": [9, 10, 11],    # 左脚踝 roll/pitch/yaw 在动作空间中的索引
#         "right": [12, 13, 14],  # 右脚踝
#     },
#     l0=0.02, l1=0.01, l2=0.03
# )
#
# # 策略推理
# action = policy.get_action(obs)  # 形状 (action_dim,)
#
# # 映射到舵机命令
# left_servo_cmds = mapper.map_action(action, 'left')
# right_servo_cmds = mapper.map_action(action, 'right')
#
# # 发送到硬件
# for cmd in left_servo_cmds + right_servo_cmds:
#     servo_controller.set_position(cmd['id'], cmd['position'], cmd['speed'])
# ```
