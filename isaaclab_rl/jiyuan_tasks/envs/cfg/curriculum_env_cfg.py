"""
分阶段课程学习环境配置

本环境旨在通过分阶段课程学习（Curriculum Learning）训练 Jiyuan 机器人，
从最简单的站立任务开始，逐步过渡到在复杂地形上高速行走。

课程阶段设计:
1.  **阶段 1: 站立平衡 (Standing)**
    -   目标: 学会基本站立，保持平衡。
    -   地形: 完全平坦。
    -   速度命令: 0 m/s (纯站立)。

2.  **阶段 2: 小步行走 (Slow Walking)**
    -   目标: 学会原地踏步和慢速行走，避免摔倒。
    -   地形: 完全平坦。
    -   速度命令: 较小的线速度和角速度范围 (±0.2 m/s)。

3.  **阶段 3: 正常行走 (Normal Walking)**
    -   目标: 在平地上稳定跟踪中等速度命令。
    -   地形: 完全平坦。
    -   速度命令: 正常线速度和角速度范围 (±1.0 m/s)。

4.  **阶段 4: 地形适应 (Rough Terrain)**
    -   目标: 在粗糙地形上稳定跟踪高速命令。
    -   地形: 粗糙地形。
    -   速度命令: 扩大的线速度和角速度范围 (±1.5 m/s)。

切换条件:
    -   阶段切换基于平均 Episode 长度。当机器人能够在一个阶段稳定足够长的时间
        （即不容易摔倒），就认为它已经掌握了当前阶段的技能，可以进入下一阶段。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .velocity_tracking_env_cfg import VelocityTrackingEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm


##
# 分阶段课程学习环境配置
##


@configclass
class CurriculumEnvCfg(VelocityTrackingEnvCfg):
    """分阶段课程学习环境配置"""

    def __post_init__(self):
        """后处理配置，定义课程学习阶段。"""
        # 调用父类的后处理
        super().__post_init__()

        # 定义课程学习
        self.curriculum = self.CurriculumCfg(
            # 阶段切换条件：基于平均 episode 长度
            # 当平均 episode 长度超过阈值时，进入下一阶段
            # episode_length_buf 是 RSL_RL runner 内部记录的 episode 长度缓冲区
            net_reward_term=CurrTerm(
                func=lambda runner: runner.episode_length_buf.mean(),
                name="episode_length",
            ),
            # 定义阶段
            # 顺序很重要，从易到难
            # 阶段0 -> 1 的切换阈值
            # 阶段 1 -> 2 的切换阈值 ...
            # 阈值单位是“步数”，不是秒。 1s = 50Hz / decimation = 50 / 4 = 12.5 步
            # 200 步 ~= 16 秒
            # 300 步 ~= 24 秒
            # 400 步 ~= 32 秒 (超过 episode length, 意味着基本不会摔倒)
            thresholds=[200, 300, 400],
            # 定义每个阶段要修改的参数
            # 使用点号分隔的字符串来指定嵌套的配置参数
            # key: 要修改的参数路径
            # value: 包含每个阶段值的列表 (阶段0, 阶段1, 阶段2, 阶段3)
            terms={
                # -----------------
                # 阶段 1: 站立
                # -----------------
                "commands.base_velocity.ranges.lin_vel_x": [(0.0, 0.0), (-0.2, 0.2), (-1.0, 1.0), (-1.5, 1.5)],
                "commands.base_velocity.ranges.lin_vel_y": [(0.0, 0.0), (-0.2, 0.2), (-0.8, 0.8), (-1.0, 1.0)],
                "commands.base_velocity.ranges.ang_vel_z": [(0.0, 0.0), (-0.2, 0.2), (-1.0, 1.0), (-1.5, 1.5)],
                
                # -----------------
                # 阶段 4: 地形适应
                # -----------------
                # 仅在最后一个阶段引入粗糙地形
                "scene.ground.terrain_type": ["plane", "plane", "plane", "rough"],
                "scene.ground.terrain_cfg.sub_terrains.mounts100.max_height": [0.0, 0.0, 0.0, 0.08],
                "scene.ground.terrain_cfg.sub_terrains.pyramid_stairs_inv.step_height_range": [ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.05, 0.15)],
            },
        )


##
# 环境配置导出
##

CURRICULUM_ENV_CFG = CurriculumEnvCfg()
