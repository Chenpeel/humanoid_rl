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

from isaaclab.utils import configclass

# 导入 Isaac Lab 地形训练基础模板
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
)

# 导入 Jiyuan 机器人配置
from .jiyuan_scene_cfg import JiyuanSceneCfg


##
# Jiyuan 地形训练配置
##


@configclass
class JiyuanRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Jiyuan 机器人在粗糙地形上的速度跟踪环境

    继承 Isaac Lab 的标准地形训练模板，只需替换机器人配置。
    """

    scene: JiyuanSceneCfg = JiyuanSceneCfg(num_envs=8192, env_spacing=2.5)

    def __post_init__(self):
        """后处理配置 - 替换机器人并调整参数"""
        # 调用父类初始化（这会设置地形、传感器等）
        super().__post_init__()

        # 替换为 Jiyuan 机器人配置
        # 注意：使用 JiyuanSceneCfg 中的 robot 配置
        self.scene.robot = JiyuanSceneCfg.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 调整高度扫描传感器路径（Jiyuan 使用 base_link 而不是 base）
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # 调整接触传感器路径（匹配 Jiyuan 的脚部命名）
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*_foot"

        # 调整脚部空中时间奖励的脚部名称匹配
        # Jiyuan 的脚部命名: right_foot_link, left_foot_link
        # 需要在 feet_air_time 奖励中使用正则表达式匹配
        # （这在 Isaac Lab 的 mdp.feet_air_time 中自动处理）

        # 根据 Jiyuan 机器人尺寸调整地形参数
        if self.scene.terrain.terrain_generator is not None:
            # Jiyuan 约 0.5m 高，中型机器人，使用默认地形参数
            # 如需调整（小/大机器人），修改以下参数：
            # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.2)
            # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.08)
            pass


@configclass
class JiyuanRoughEnvCfg_PLAY(JiyuanRoughEnvCfg):
    """Jiyuan 地形训练 - 播放/测试配置

    减小环境数量和地形复杂度，用于本地测试和可视化。
    """

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # 减小环境数量（节省内存）
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # 在整个地形网格中随机放置机器人（而不是按地形层级）
        self.scene.terrain.max_init_terrain_level = None

        # 减少地形数量以节省内存
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 禁用观测噪声（播放模式）
        self.observations.policy.enable_corruption = False

        # 移除随机推力事件
        self.events.push_robot = None
        self.events.base_external_force_torque = None


@configclass
class JiyuanFlatEnvCfg(JiyuanRoughEnvCfg):
    """Jiyuan 平坦地形环境（用于初期训练）

    将粗糙地形替换为平面，简化训练难度。
    """

    def __post_init__(self):
        # 调用父类初始化（这会设置机器人等）
        # 注意：这里我们不调用 super().__post_init__()，
        # 因为那会设置粗糙地形
        LocomotionVelocityRoughEnvCfg.__post_init__(self)

        # 将地形替换为平面
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 移除高度扫描传感器（平面不需要）
        self.scene.height_scanner = None

        # 移除高度扫描观测
        self.observations.policy.height_scan = None

        # 调整奖励权重（平面训练）
        # self.rewards.flat_orientation_l2.weight = -5.0


##
# 导出配置实例
##


JIYUAN_ROUGH_ENV_CFG = JiyuanRoughEnvCfg()
JIYUAN_FLAT_ENV_CFG = JiyuanFlatEnvCfg()
