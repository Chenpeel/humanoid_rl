"""
站立平衡环境配置

这是站立平衡环境，训练机器人在原地保持稳定直立。

任务目标:
- 保持在目标高度附近（~0.35m）
- 保持直立姿态（roll 和 pitch 接近 0）
- 尽量不移动（线速度和角速度接近 0）
- 高能量效率

训练参数:
- 并行环境: 4096
- Episode 长度: 30秒（比速度跟踪更长）
- 控制频率: 50Hz / 4 = 12.5Hz
- 训练步数: 10M+

用途:
- 作为速度跟踪任务的预训练
- 验证基础控制能力
- 调试和可视化
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入场景配置
from .jiyuan_scene_cfg import JiyuanSceneCfg

# 导入Isaac Lab内置的MDP函数
import isaaclab.envs.mdp as mdp

# 导入自定义管理器函数
from jiyuan_tasks.managers import rewards, terminations


##
# 站立环境配置
##


@configclass
class StandingEnvCfg(ManagerBasedRLEnvCfg):
    """站立平衡环境配置

    训练机器人在原地保持稳定直立。
    """

    # 场景配置
    # num_envs 在运行时由 train.py 从配置文件或命令行参数设置
    # 默认值仅用于未指定时的后备
    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基础设置
    decimation = 4  # 控制频率：50Hz / 4 = 12.5Hz
    episode_length_s = 30.0  # 每个episode 30秒（更长，鼓励稳定性）

    # 命令配置（固定为 0，为了与行走任务保持维度兼容）
    @configclass
    class CommandsCfg:
        """命令生成器配置"""
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=1.0,  # 100% 站立
            heading_command=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )

    # 观测配置
    @configclass
    class ObservationsCfg:
        """观测空间配置"""

        @configclass
        class PolicyCfg(ObsGroup):
            """策略观测（包含命令，确保维度兼容）"""

            # 基础状态（17维）
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
            projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3

            # 速度命令（3维）- 即使是站立也保留，为了维度兼容
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3

            # 高度信息（相对于目标）
            base_height = ObsTerm(func=mdp.base_pos_z)  # 1

            # 高度扫描（187 维）- 确保课程学习维度兼容
            height_scan = ObsTerm(
                func=mdp.height_scan,
                params={"sensor_cfg": SceneEntityCfg("height_scanner")}
            )

            # 关节状态（32维）
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 16
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 16

            # 上一步动作（16维）
            actions = ObsTerm(func=mdp.last_action)  # 16

            # 总维度: 3 + 3 + 3 + 3 + 1 + 16 + 16 + 16 = 61 维

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # 策略观测
        policy: PolicyCfg = PolicyCfg()

    # 动作配置
    @configclass
    class ActionsCfg:
        """动作空间配置"""

        joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],  # 所有16个关节
            scale=0.25,  # 动作缩放因子
            use_default_offset=True,  # 使用默认关节位置作为偏移
        )

    # 奖励配置（站立任务特化）
    @configclass
    class RewardsCfg:
        """奖励函数配置

        使用自定义奖励函数（从 JAX/MJX 迁移）
        """

        # 高度保持（主要目标）
        height = RewTerm(
            func=rewards.height_reward,
            weight=2.0,  # 更高权重
            params={"target_height": 0.35, "tolerance": 0.05},
        )

        # 姿态稳定（主要目标）
        orientation = RewTerm(
            func=rewards.orientation_reward,
            weight=2.0,  # 更高权重
            params={"tolerance": 0.1},
        )

        # 线速度惩罚（应该尽量不动）
        lin_vel = RewTerm(
            func=rewards.lin_vel_penalty_l2,
            weight=-1.0,
        )

        # 角速度惩罚（应该尽量不转）
        ang_vel = RewTerm(
            func=rewards.ang_vel_penalty_l2,
            weight=-0.5,
        )

        # XY平面速度惩罚（允许少量Z方向振动）
        xy_vel = RewTerm(
            func=rewards.xy_vel_penalty_l2,
            weight=-0.5,
        )

        # 动作平滑性
        action_rate = RewTerm(
            func=rewards.action_rate_l2,
            weight=-0.01,
        )

        # 动作幅度惩罚（鼓励小动作）
        action_l2 = RewTerm(
            func=rewards.action_l2,
            weight=-0.005,
        )

        # 能量效率
        torques = RewTerm(
            func=rewards.joint_torques_l2,
            weight=-0.0002,
        )

        # 存活奖励
        alive = RewTerm(func=mdp.is_alive, weight=0.5)

        # 关节限制惩罚
        joint_pos_limits = RewTerm(
            func=rewards.joint_pos_limits,
            weight=-0.5,
            params={"margin": 0.05},
        )

    # 终止条件（更严格）
    @configclass
    class TerminationsCfg:
        """终止条件配置"""

        # 时间限制
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # 机器人摔倒（更严格的阈值）
        fallen = DoneTerm(
            func=terminations.is_fallen,
            params={
                "min_height": 0.2,  # 站立任务不允许太低
                "max_roll": 0.785,  # 45度
                "max_pitch": 0.785,
            },
        )

        # 高度超限（跳得太高）
        height_too_high = DoneTerm(
            func=terminations.base_height_above_threshold,
            params={"max_height": 0.6},
        )

        # 速度异常
        velocity_out_of_bounds = DoneTerm(
            func=terminations.linear_velocity_out_of_bounds,
            params={"max_velocity": 5.0},  # 站立任务不应快速移动
        )

    # 事件配置（适度随机化）
    @configclass
    class EventsCfg:
        """随机化事件配置"""

        # 重置时随机化机器人姿态（较小范围）
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),  # ±0.1 rad（比速度跟踪小）
                "velocity_range": (0.0, 0.0),  # 从静止开始
            },
        )

        # 重置时随机化 base 高度（小范围）
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (0.3, 0.4),  # 在目标高度附近
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-3.14, 3.14),  # 任意朝向
                },
                "velocity_range": {},  # 零速度
            },
        )

        # 随机化机器人质量（±10%，比速度跟踪小）
        randomize_robot_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )

        # 随机化执行器增益（±5%）
        randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.95, 1.05),
                "damping_distribution_params": (0.95, 1.05),
                "operation": "scale",
            },
        )

        # 定期添加小扰动（测试稳定性）
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),  # 每5-10秒推一次
            params={
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},  # 较小的推力
            },
        )

    # 配置实例
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """后处理配置"""
        # 设置模拟参数
        self.sim.dt = 0.005  # 200Hz 物理模拟
        self.sim.render_interval = self.decimation  # 渲染频率

        # 设置 PhysX GPU 缓冲区（支持多环境并行训练）
        self.sim.physx.gpu_max_rigid_contact_count = 2**26  # 默认 2^23
        self.sim.physx.gpu_max_rigid_patch_count = 2**19  # 默认 5*2^15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24  # 默认 2^21
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24  # 默认 2^21
        self.sim.physx.gpu_collision_stack_size = 2**28  # 默认 2^26
        self.sim.physx.gpu_heap_capacity = 2**28  # 默认 2^26
        self.sim.physx.gpu_temp_buffer_capacity = 2**26  # 默认 2^24

        # 设置查看器参数
        self.viewer.eye = (5.0, 5.0, 3.0)  # 更近的视角，适合观察站立
        self.viewer.lookat = (0.0, 0.0, 0.35)  # 聚焦在目标高度


##
# 环境配置导出
##


# 导出配置类
STANDING_ENV_CFG = StandingEnvCfg()
