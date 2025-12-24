"""
Jiyuan 测试环境配置

这是一个最小可行环境（MVE），用于验证：
1. Jiyuan MJCF 模型可正确加载
2. 环境可成功重置和运行
3. 批量并行环境工作正常

此环境使用最简配置：
- 最小观测空间（仅基础状态）
- 最小奖励（仅存活奖励）
- 无命令生成
- 基础终止条件

用于阶段1的基础验证，不用于实际训练。
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unif

# 导入场景配置
from .jiyuan_scene_cfg import JiyuanSceneCfg

# 导入Isaac Lab内置的MDP函数
import isaaclab.envs.mdp as mdp


##
# 预定义配置
##


@configclass
class JiyuanTestEnvCfg(ManagerBasedRLEnvCfg):
    """Jiyuan 测试环境配置（最小可行环境）

    用于验证基础功能，不用于实际训练。
    """

    # 场景配置
    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=16, env_spacing=2.5)

    # 基础设置
    decimation = 4  # 控制频率：50Hz / 4 = 12.5Hz
    episode_length_s = 10.0  # 每个episode 10秒

    # 观测配置（最小）
    @configclass
    class ObservationsCfg:
        """观测空间配置"""

        @configclass
        class PolicyCfg(ObsGroup):
            """策略观测（最小集合）"""

            # 基础状态（17维）
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unif(-0.1, 0.1))  # 3
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unif(-0.2, 0.2))  # 3
            projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unif(-0.05, 0.05))  # 3

            # 关节状态（32维）
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unif(-0.01, 0.01))  # 16
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unif(-1.5, 1.5))  # 16

            # 上一步动作（16维）
            actions = ObsTerm(func=mdp.last_action)  # 16

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

    # 奖励配置（最小）
    @configclass
    class RewardsCfg:
        """奖励函数配置"""

        # 存活奖励
        alive = RewTerm(func=mdp.is_alive, weight=1.0)

        # 惩罚大幅动作
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

        # 惩罚关节速度过大
        joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    # 终止条件
    @configclass
    class TerminationsCfg:
        """终止条件配置"""

        # 时间限制
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # 机器人摔倒（base高度过低）
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
        )

    # 事件配置（最小）
    @configclass
    class EventsCfg:
        """随机化事件配置"""

        # 重置时随机化机器人姿态
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),  # ±0.1 rad
                "velocity_range": (0.0, 0.0),
            },
        )

    # 配置实例
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

        # 设置查看器参数
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


##
# 环境配置导出
##


# 导出配置类
JIYUAN_TEST_ENV_CFG = JiyuanTestEnvCfg()
