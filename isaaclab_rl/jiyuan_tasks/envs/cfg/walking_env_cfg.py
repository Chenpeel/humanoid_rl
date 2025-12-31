"""
Jiyuan 行走任务环境配置

专门的双足机器人行走任务，训练机器人实现稳定、自然的步态。

任务目标:
- 前向运动：保持目标线速度
- 步态质量：对称、有节奏的步态
- 姿态稳定：躯干保持直立
- 能量效率：平滑、低能耗运动

训练参数:
- 并行环境: 4096
- Episode 长度: 20秒
- 控制频率: 50Hz / 4 = 12.5Hz

参考:
- Legged Gym: https://github.com/leggedrobotics/legged_gym
- Isaac Lab locomotion tasks
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
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# 导入场景配置
from .jiyuan_scene_cfg import JiyuanSceneCfg

# 导入Isaac Lab内置的MDP函数
import isaaclab.envs.mdp as mdp

# 导入自定义奖励函数
from jiyuan_tasks.managers import walking_rewards
from jiyuan_tasks.managers import rewards
from jiyuan_tasks.managers import terminations


##
# 行走环境配置
##


@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    """Jiyuan 行走任务环境配置

    训练机器人实现稳定的前向行走。
    """

    # 场景配置
    # num_envs 在运行时由 train.py 从配置文件或命令行参数设置
    # 默认值仅用于未指定时的后备
    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基础设置
    decimation = 4  # 控制频率：50Hz / 4 = 12.5Hz
    episode_length_s = 20.0  # 每个episode 20秒

    # 命令配置（简化版，主要关注前向速度）
    @configclass
    class CommandsCfg:
        """命令生成器配置"""

        # 使用 Isaac Lab 内置的均匀速度命令生成器
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),  # 每10秒重新采样命令
            rel_standing_envs=0.0,  # 0% 环境站立（全部行走）
            rel_heading_envs=0.0,  # 0% 环境只转向
            heading_command=True,  # 包含航向命令
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.5, 1.0),  # 前向速度（行走任务主要关注）
                lin_vel_y=(-0.3, 0.3),  # 侧向速度（较小）
                ang_vel_z=(-0.5, 0.5),  # 转向速度（较小）
                heading=(-3.14, 3.14),
            ),
        )

    # 观测配置
    @configclass
    class ObservationsCfg:
        """观测空间配置"""

        @configclass
        class PolicyCfg(ObsGroup):
            """策略观测"""

            # 基础状态（17维）
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
            projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3

            # 速度命令（3维）
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3

            # 关节状态（32维）
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 16
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 16

            # 上一步动作（16维）
            actions = ObsTerm(func=mdp.last_action)  # 16

            # 总维度: 3 + 3 + 3 + 3 + 16 + 16 + 16 = 60 维

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    # 动作配置
    @configclass
    class ActionsCfg:
        """动作空间配置"""

        joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],  # 所有16个关节
            scale=0.25,
            use_default_offset=True,
        )

    # 奖励配置（行走专用）
    @configclass
    class RewardsCfg:
        """奖励函数配置"""

        # 主要目标：速度跟踪
        track_lin_vel_xy = RewTerm(
            func=walking_rewards.forward_velocity_reward,
            weight=1.5,
            params={"target_velocity": 0.7},  # 默认前向速度
        )

        track_ang_vel_z = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={"std": 0.5, "command_name": "base_velocity"},
        )

        # 步态质量奖励
        gait_symmetry = RewTerm(
            func=walking_rewards.gait_symmetry_reward,
            weight=0.3,
        )

        # 躯干稳定性
        trunk_height = RewTerm(
            func=walking_rewards.trunk_height_reward,
            weight=0.5,
            params={"target_height": 0.80, "tolerance": 0.08},
        )

        orientation = RewTerm(
            func=rewards.orientation_reward,
            weight=0.3,
            params={"tolerance": 0.3},
        )

        # 惩罚项
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        trunk_lin_vel_z = RewTerm(func=walking_rewards.trunk_lin_vel_z_penalty, weight=-0.8)
        trunk_tilt = RewTerm(func=walking_rewards.trunk_orientation_penalty, weight=-0.3)

        # 动作平滑性
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

        # 能量效率
        joint_powers = RewTerm(
            func=rewards.joint_powers_l1,
            weight=-2.0e-5,
        )

        # 存活奖励
        alive = RewTerm(func=mdp.is_alive, weight=0.5)

        # 关节限制
        joint_pos_limits = RewTerm(
            func=rewards.joint_pos_limits,
            weight=-0.8,
            params={"margin": 0.1},
        )

        joint_vel_limits = RewTerm(
            func=rewards.joint_vel_limits,
            weight=-0.8,
            params={"margin_factor": 0.9},
        )

    # 终止条件
    @configclass
    class TerminationsCfg:
        """终止条件配置"""

        # 时间限制
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # 机器人摔倒
        fallen = DoneTerm(
            func=terminations.is_fallen,
            params={
                "min_height": 0.15,
                "max_roll": 1.0,
                "max_pitch": 1.0,
            },
        )

        # 速度异常
        velocity_out_of_bounds = DoneTerm(
            func=terminations.linear_velocity_out_of_bounds,
            params={"max_velocity": 8.0},
        )

        # 关节位置超限
        joint_pos_limits = DoneTerm(
            func=terminations.joint_pos_out_of_limits,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    # 事件配置（领域随机化）
    @configclass
    class EventsCfg:
        """随机化事件配置"""

        # 重置时随机化机器人姿态
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (-0.1, 0.1),
            },
        )

        # 重置时随机化 base 位置
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "yaw": (-0.5, 0.5),  # 较小的航向范围（主要是前向行走）
                },
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "yaw": (-0.3, 0.3),
                },
            },
        )

        # 随机化机器人质量（±20%）
        randomize_robot_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )

        # 随机化执行器增益（±10%）
        randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )

        # 定期添加外部扰动
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),  # 每8-12秒推一次
            params={
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            },
        )

    # 课程学习配置
    @configclass
    class CurriculumCfg:
        """课程学习配置"""
        pass

    # 配置实例
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后处理配置"""
        # 设置模拟参数
        self.sim.dt = 0.005  # 200Hz 物理模拟
        self.sim.render_interval = self.decimation

        # 设置 PhysX GPU 缓冲区（支持多环境并行训练）
        self.sim.physx.gpu_max_rigid_contact_count = 2**26
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26

        # 设置查看器参数
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


##
# 环境配置导出
##


WALKING_ENV_CFG = WalkingEnvCfg()
