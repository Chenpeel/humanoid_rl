"""
Jiyuan 速度跟踪环境配置

这是完整的速度跟踪环境，用于训练机器人跟踪指定的线速度和角速度命令。

任务目标:
- 跟踪随机采样的目标速度（前后、左右、旋转）
- 保持稳定的直立姿态
- 高能量效率和平滑的动作

训练参数:
- 并行环境: 4096
- Episode 长度: 20秒
- 控制频率: 50Hz / 4 = 12.5Hz
- 训练步数: 30M+

参考:
- 原实现: src/rl/envs/robot_envs.py (jax 分支)
- Isaac Lab 示例: source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

# 导入自定义管理器函数
from jiyuan_tasks.managers import rewards, terminations


##
# 速度跟踪环境配置
##


@configclass
class VelocityTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Jiyuan 速度跟踪环境配置

    训练机器人跟踪随机速度命令。
    """

    # 场景配置
    # num_envs 在运行时由 train.py 从配置文件或命令行参数设置
    # 默认值仅用于未指定时的后备
    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基础设置
    decimation = 4  # 控制频率：50Hz / 4 = 12.5Hz
    episode_length_s = 20.0  # 每个episode 20秒

    # 命令配置
    @configclass
    class CommandsCfg:
        """命令生成器配置"""

        # 使用 Isaac Lab 内置的均匀速度命令生成器
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),  # 每10秒重新采样命令
            rel_standing_envs=0.1,  # 10% 的环境站立不动
            rel_heading_envs=0.0,  # 0% 的环境只转向
            heading_command=True,  # 包含航向命令
            heading_control_stiffness=0.5,
            debug_vis=True,  # 显示命令箭头
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.5, 1.5),  # 前后速度 (m/s)
                lin_vel_y=(-0.8, 0.8),  # 左右速度 (m/s)
                ang_vel_z=(-1.5, 1.5),  # 旋转速度 (rad/s)
                heading=(-3.14, 3.14),  # 航向角 (rad)
            ),
        )

    # 观测配置
    @configclass
    class ObservationsCfg:
        """观测空间配置"""

        @configclass
        class PolicyCfg(ObsGroup):
            """策略观测（完整信息）"""

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

    # 奖励配置（完整）
    @configclass
    class RewardsCfg:
        """奖励函数配置

        使用自定义奖励函数（从 JAX/MJX 迁移）
        """

        # 速度跟踪奖励（主要目标）
        track_lin_vel_xy = RewTerm(
            func=rewards.track_lin_vel_xy_exp,
            weight=1.5,
            params={"std": 0.5, "command_name": "base_velocity"},
        )

        track_ang_vel_z = RewTerm(
            func=rewards.track_ang_vel_z_exp,
            weight=0.75,
            params={"std": 0.5, "command_name": "base_velocity"},
        )

        # 姿态稳定性
        orientation = RewTerm(
            func=rewards.orientation_reward,
            weight=0.5,
            params={"tolerance": 0.2},
        )

        # 速度惩罚（Z方向不应有速度）
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

        # 角速度惩罚（XY方向不应旋转）
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

        # 动作平滑性
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

        # 能量效率
        joint_powers = RewTerm(
            func=rewards.joint_powers_l1,
            weight=-2.0e-5,
        )

        # 关节加速度惩罚（平滑运动）
        joint_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 存活奖励
        alive = RewTerm(func=mdp.is_alive, weight=0.5)

        # 关节限制惩罚（软约束）
        joint_pos_limits = RewTerm(
            func=rewards.joint_pos_limits,
            weight=-1.0,
            params={"margin": 0.1},
        )

        joint_vel_limits = RewTerm(
            func=rewards.joint_vel_limits,
            weight=-1.0,
            params={"margin_factor": 0.9},
        )

    # 终止条件（完整）
    @configclass
    class TerminationsCfg:
        """终止条件配置

        使用自定义终止函数（从 JAX/MJX 迁移）
        """

        # 时间限制
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # 机器人摔倒
        fallen = DoneTerm(
            func=terminations.is_fallen,
            params={
                "min_height": 0.15,  # 允许更低（蹲下姿态）
                "max_roll": 1.0,  # ~57度
                "max_pitch": 1.0,
            },
        )

        # 速度异常
        velocity_out_of_bounds = DoneTerm(
            func=terminations.linear_velocity_out_of_bounds,
            params={"max_velocity": 8.0},
        )



    # 事件配置（领域随机化）
    @configclass
    class EventsCfg:
        """随机化事件配置

        实现领域随机化，提高 Sim2Real 鲁棒性。
        """

        # 重置时随机化机器人姿态
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),  # ±0.2 rad
                "velocity_range": (-0.1, 0.1),  # ±0.1 rad/s
            },
        )

        # 重置时随机化 base 位置
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
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

        # 随机化关节摩擦力
        randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.0, 0.01),
                "operation": "abs",
            },
        )

        # 定期添加外部扰动（推力）
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),  # 每10-15秒推一次
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        )

    # 课程学习配置（可选）
    @configclass
    class CurriculumCfg:
        """课程学习配置

        逐步增加任务难度。
        """

        # TODO: 添加课程学习条目
        # 例如：逐步增加命令速度范围、减少站立环境比例等
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
        self.sim.render_interval = self.decimation  # 渲染频率

        # 设置 PhysX GPU 缓冲区（支持多环境并行训练）
        # 默认值对于 64+ 环境不够大，需要增加
        self.sim.physx.gpu_max_rigid_contact_count = 2**26  # 默认 2^23
        self.sim.physx.gpu_max_rigid_patch_count = 2**19  # 默认 5*2^15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24  # 默认 2^21
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24  # 默认 2^21
        self.sim.physx.gpu_collision_stack_size = 2**28  # 默认 2^26
        self.sim.physx.gpu_heap_capacity = 2**28  # 默认 2^26
        self.sim.physx.gpu_temp_buffer_capacity = 2**26  # 默认 2^24

        # 设置查看器参数
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


##
# 环境配置导出
##


# 导出配置类
VELOCITY_TRACKING_ENV_CFG = VelocityTrackingEnvCfg()
