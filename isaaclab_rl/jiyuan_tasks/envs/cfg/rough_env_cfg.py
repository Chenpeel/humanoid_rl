"""
Jiyuan 机器人地形训练环境配置

复用 Isaac Lab 官方的地形训练模板，为 Jiyuan 机器人添加复杂地形行走能力。

地形类型:
- 粗糙地形 (rough): 包含台阶、斜坡、随机障碍
- 自动课程学习: 从简单到复杂逐步提升难度

参考:
- Isaac Lab: isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg
- Anymal C 配置: isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入 Isaac Lab 地形训练基础模板
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

# 导入 Jiyuan 场景配置
from .jiyuan_scene_cfg import JiyuanSceneCfg

# 导入Isaac Lab内置的MDP函数
import isaaclab.envs.mdp as mdp
# 导入地形训练的MDP函数（课程学习）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_locomotion

# 导入自定义奖励函数
from jiyuan_tasks.managers import rewards
from jiyuan_tasks.managers import terminations


##
# 环境配置
##


@configclass
class JiyuanRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Jiyuan 机器人在粗糙地形上的速度跟踪环境

    基于 JiyuanSceneCfg，将 ground 替换为 terrain 生成器。
    """

    # 场景配置
    # num_envs 在运行时由 train.py 从配置文件或命令行参数设置
    # 默认值仅用于未指定时的后备
    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基础设置
    decimation = 4
    episode_length_s = 20.0

    # 命令配置
    @configclass
    class CommandsCfg:
        """命令生成器配置"""

        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.5, 1.5),
                lin_vel_y=(-1.5, 1.5),
                ang_vel_z=(-1.5, 1.5),
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

            # 基础状态
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

            # 速度命令
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

            # 关节状态
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

            # 高度扫描（地形感知）
            height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

            # 上一步动作
            actions = ObsTerm(func=mdp.last_action)

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
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
        )

    # 奖励配置
    @configclass
    class RewardsCfg:
        """奖励函数配置"""

        # 主要目标：速度跟踪
        track_lin_vel_xy = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        track_ang_vel_z = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # 姿态稳定性
        orientation = RewTerm(
            func=rewards.orientation_reward,
            weight=0.5,
            params={"tolerance": 0.3},
        )

        # 惩罚项
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        joint_powers = RewTerm(func=rewards.joint_powers_l1, weight=-2.0e-5)
        joint_accel_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

        # 存活奖励
        alive = RewTerm(func=mdp.is_alive, weight=0.5)

        # 关节限制
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

    # 终止条件
    @configclass
    class TerminationsCfg:
        """终止条件配置"""

        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        fallen = DoneTerm(
            func=terminations.is_fallen,
            params={"min_height": 0.15, "max_roll": 1.0, "max_pitch": 1.0},
        )
        velocity_out_of_bounds = DoneTerm(
            func=terminations.linear_velocity_out_of_bounds,
            params={"max_velocity": 8.0},
        )
        joint_pos_out_of_limits = DoneTerm(
            func=terminations.joint_pos_out_of_limits,
            params={"margin": 0.05},  # 增大到0.05rad (约2.86度),减少误触发
        )

    # 事件配置（领域随机化）
    @configclass
    class EventsCfg:
        """随机化事件配置"""

        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (-0.15, 0.15), "velocity_range": (-0.1, 0.1)},
        )

        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5)},
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
            },
        )

        randomize_robot_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.2),  # 改为0.8-1.2倍（±20%），scale操作要求>0
                "operation": "scale",
            },
        )

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

        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

    # 课程学习配置
    @configclass
    class CurriculumCfg:
        """课程学习配置"""

        terrain_levels = CurrTerm(func=mdp_locomotion.terrain_levels_vel)

    # 配置实例
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后处理配置 - 将 ground 替换为 terrain"""
        # 将 ground 替换为粗糙地形生成器
        self.scene.ground = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # 添加高度扫描传感器（地形感知）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # 启用脚部空中时间跟踪
        if hasattr(self.scene.contact_forces, "track_air_time"):
            self.scene.contact_forces.track_air_time = True

        # 设置模拟参数
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.ground.physics_material

        # 设置 PhysX GPU 缓冲区
        self.sim.physx.gpu_max_rigid_contact_count = 2**26
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26

        # 设置传感器更新周期
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 启用/禁用地形课程学习
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.ground.terrain_generator is not None:
                self.scene.ground.terrain_generator.curriculum = True
        else:
            if self.scene.ground.terrain_generator is not None:
                self.scene.ground.terrain_generator.curriculum = False

        # 设置查看器参数
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class JiyuanFlatEnvCfg(JiyuanRoughEnvCfg):
    """Jiyuan 平坦地形环境（用于初期训练）"""

    def __post_init__(self):
        # 调用父类初始化（设置粗糙地形）
        super().__post_init__()

        # 将地形替换为平面
        self.scene.ground.terrain_type = "plane"
        self.scene.ground.terrain_generator = None

        # 移除高度扫描传感器
        self.scene.height_scanner = None

        # 移除高度扫描观测
        self.observations.policy.height_scan = None

        # 禁用课程学习
        self.curriculum.terrain_levels = None


##
# 导出配置实例
##


JIYUAN_ROUGH_ENV_CFG = JiyuanRoughEnvCfg()
JIYUAN_FLAT_ENV_CFG = JiyuanFlatEnvCfg()
