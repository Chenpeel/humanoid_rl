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

import isaaclab.envs.mdp as mdp

from .velocity_tracking_env_cfg import VelocityTrackingEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm


# -----------------------------------------------------------------------------
# Curriculum helpers
# -----------------------------------------------------------------------------

_CURRICULUM_THRESHOLDS = (200, 300, 400)
_CURRICULUM_LIN_VEL_X = ((0.0, 0.0), (-0.2, 0.2), (-1.0, 1.0), (-1.5, 1.5))
_CURRICULUM_LIN_VEL_Y = ((0.0, 0.0), (-0.2, 0.2), (-0.8, 0.8), (-1.0, 1.0))
_CURRICULUM_ANG_VEL_Z = ((0.0, 0.0), (-0.2, 0.2), (-1.0, 1.0), (-1.5, 1.5))


def _curriculum_stage(env, thresholds):
    mean_len = float(env.episode_length_buf.float().mean().item())
    stage = 0
    for thr in thresholds:
        if mean_len >= thr:
            stage += 1
        else:
            break
    return min(stage, len(thresholds))


def _curriculum_stage_log(env, env_ids, thresholds):
    return float(_curriculum_stage(env, thresholds))


def _curriculum_param_by_stage(env, env_ids, data, thresholds, values):
    stage = _curriculum_stage(env, thresholds)
    if stage >= len(values):
        stage = len(values) - 1
    desired = values[stage]
    try:
        if data == desired:
            return mdp.modify_env_param.NO_CHANGE
    except Exception:
        pass
    return desired


##
# 分阶段课程学习环境配置
##


@configclass
class CurriculumEnvCfg(VelocityTrackingEnvCfg):
    """分阶段课程学习环境配置"""

    @configclass
    class CurriculumCfg:
        """课程学习配置

        根据平均 episode 长度逐步放宽速度命令范围。
        """

        stage = CurrTerm(
            func=_curriculum_stage_log,
            params={"thresholds": _CURRICULUM_THRESHOLDS},
        )
        cmd_lin_vel_x = CurrTerm(
            func=mdp.modify_env_param,
            params={
                "address": "command_manager.cfg.base_velocity.ranges.lin_vel_x",
                "modify_fn": _curriculum_param_by_stage,
                "modify_params": {
                    "thresholds": _CURRICULUM_THRESHOLDS,
                    "values": _CURRICULUM_LIN_VEL_X,
                },
            },
        )
        cmd_lin_vel_y = CurrTerm(
            func=mdp.modify_env_param,
            params={
                "address": "command_manager.cfg.base_velocity.ranges.lin_vel_y",
                "modify_fn": _curriculum_param_by_stage,
                "modify_params": {
                    "thresholds": _CURRICULUM_THRESHOLDS,
                    "values": _CURRICULUM_LIN_VEL_Y,
                },
            },
        )
        cmd_ang_vel_z = CurrTerm(
            func=mdp.modify_env_param,
            params={
                "address": "command_manager.cfg.base_velocity.ranges.ang_vel_z",
                "modify_fn": _curriculum_param_by_stage,
                "modify_params": {
                    "thresholds": _CURRICULUM_THRESHOLDS,
                    "values": _CURRICULUM_ANG_VEL_Z,
                },
            },
        )

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后处理配置，定义课程学习阶段。"""
        # 调用父类的后处理
        super().__post_init__()


##
# 环境配置导出
##

CURRICULUM_ENV_CFG = CurriculumEnvCfg()
