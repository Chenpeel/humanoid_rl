"""
Jiyuan 机器人场景配置

本模块定义 Jiyuan 双足机器人的场景和机器人配置，包括：
- 从 MJCF 加载机器人模型
- 定义初始状态（位置、姿态、关节角度）
- 配置执行器参数（刚度、阻尼等）

## 如何适配其他机器人

### 方法1：复制并修改本文件（当前阶段）
```bash
cp jiyuan_scene_cfg.py unitree_go2_scene_cfg.py
```
然后修改：
1. `MjcfFileCfg.asset_path` → 指向新机器人的MJCF文件
2. `init_state` → 调整初始位置、姿态、关节角度
3. `actuators` → 调整执行器参数（刚度、阻尼、力矩限制）
4. `joint_names_expr` → 匹配新机器人的关节命名

### 方法2：创建机器人配置库（未来阶段）
详见 `.migrate/08-multi-robot-architecture.md` 的长期规划。

参考:
- Isaac Lab 官方文档: https://isaac-sim.github.io/IsaacLab/main/
- MJCF 支持: omni.isaac.lab.sim.spawners.from_files.MjcfFileCfg
- 多机器人架构: ../.migrate/08-multi-robot-architecture.md
"""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import MjcfFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass


##
# 场景配置
##


@configclass
class JiyuanSceneCfg(InteractiveSceneCfg):
    """Jiyuan 双足机器人场景配置

    包含:
    - 机器人资产配置（从 MJCF 加载）
    - 地面/地形配置
    - 传感器配置（可选）
    """

    # 地面平面
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 机器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=MjcfFileCfg(
            # MJCF 文件路径（相对于 isaaclab_rl/assets/）
            asset_path="assets/xmls/models/jiyuan/index.xml",
            make_instanceable=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 初始位置：在地面上方约0.3米（MJCF中是0.99m，但那是整个机器人高度）
            pos=(0.0, 0.0, 0.35),

            # 初始姿态：保持直立（四元数 [w, x, y, z]）
            # MJCF 中的 quat="0.70710678 0.70710678 0 0" 可能是为了坐标系转换
            # Isaac Lab 中直接使用 identity quaternion (w=1)
            rot=(1.0, 0.0, 0.0, 0.0),

            # 关节初始位置（单位：弧度）
            # 从原始MJX代码的_default_pose()推断
            joint_pos={
                # 髋关节pitch：略微前倾
                ".*hip_pitch_engine.*": 0.0,

                # 髋关节yaw：保持中立
                ".*hip_yaw_engine.*": 0.0,

                # 髋关节roll：保持中立
                ".*hip_roll.*": 0.0,

                # 膝关节：弯曲约30度（0.5弧度）以降低重心
                ".*knee.*": 0.5,

                # 踝关节：保持中立（3自由度并联结构）
                ".*ankle_1_3.*": 0.0,
                ".*ankle_2_3.*": 0.0,
                ".*ankle_3_3.*": 0.0,

                # 脚趾关节：略微抬起
                ".*toe.*": 0.0,
            },

            # 初始速度：静止
            joint_vel={".*": 0.0},
        ),

        # 执行器配置
        actuators={
            # 主要关节电机（hip, knee）
            "main_motors": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*hip_pitch_engine.*",
                    ".*hip_yaw_engine.*",
                    ".*hip_roll.*",
                    ".*knee.*",
                ],
                # PD 控制参数（从 MJCF default_classes.xml 映射）
                # MJCF: damping=0.5, armature=0.01
                # Isaac Lab: stiffness和damping需要根据实际调优
                stiffness=80.0,  # 典型值：40-150
                damping=2.0,     # 典型值：1-5

                # 力矩限制（从 MJCF ctrlrange="-1 1" 映射）
                # 实际力矩需要乘以gear ratio，这里假设最大力矩约150Nm
                effort_limit=150.0,

                # 速度限制（弧度/秒）
                velocity_limit=10.0,
            ),

            # 踝关节电机（3自由度并联结构）
            "ankle_motors": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*ankle_1_3.*",
                    ".*ankle_2_3.*",
                    ".*ankle_3_3.*",
                ],
                # 踝关节可能需要不同的参数
                stiffness=60.0,
                damping=1.5,
                effort_limit=100.0,
                velocity_limit=10.0,
            ),

            # 脚趾电机
            "toe_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*toe.*"],
                # MJCF toe_motor: damping=0.2, armature=0.005
                stiffness=40.0,
                damping=1.0,
                effort_limit=50.0,
                velocity_limit=10.0,
            ),
        },
    )

    # 脚部接触传感器（可选，用于奖励计算）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,  # 每步更新
        history_length=3,
        debug_vis=False,
    )

    # 高度扫描传感器（可选，用于地形感知）
    # TODO: 如果需要复杂地形导航，可以启用
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )


##
# 环境事件（领域随机化）
##

# TODO: 阶段2-3会实现更多事件，这里先保留基础框架


##
# 辅助函数
##


def get_joint_names() -> list[str]:
    """获取所有关节名称（按MJCF定义顺序）

    Returns:
        关节名称列表，共16个执行器
    """
    # 右腿（8个）
    right_leg = [
        "right_hip_pitch_engine_joint",
        "right_hip_yaw_engine_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_1_3_joint",
        "right_ankle_2_3_joint",
        "right_ankle_3_3_joint",
        "right_toe_joint",
    ]

    # 左腿（8个）
    left_leg = [
        "left_hip_pitch_engine_joint",
        "left_hip_yaw_engine_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_1_3_joint",
        "left_ankle_2_3_joint",
        "left_ankle_3_3_joint",
        "left_toe_joint",
    ]

    return right_leg + left_leg


def get_actuator_count() -> int:
    """获取执行器数量

    Returns:
        执行器数量（16）
    """
    return 16
