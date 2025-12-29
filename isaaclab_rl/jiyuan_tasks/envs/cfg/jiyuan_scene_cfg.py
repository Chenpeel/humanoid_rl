"""
Jiyuan 机器人场景配置

本模块定义 Jiyuan 双足机器人的场景和机器人配置，包括：
- 从 USD 加载机器人模型（从 MJCF 转换而来）
- 定义初始状态（位置、姿态、关节角度）
- 配置执行器参数（刚度、阻尼等）

## 资产转换

**重要**：本配置使用 USD 格式的机器人模型，需要先从 MJCF 转换：

```bash
# 在服务器上运行转换（只需执行一次）
make convert-usd

# 或者直接使用转换脚本
dep/IsaacLab/isaaclab.sh -p dep/IsaacLab/scripts/tools/convert_mjcf.py \
    assets/xmls/models/jiyuan.xml \
    assets/usd/jiyuan/jiyuan.usd \
    --make-instanceable --import-sites
```

转换后会生成：
- `assets/usd/jiyuan/jiyuan.usd` - 主 USD 文件
- `assets/usd/jiyuan/configuration/` - 可实例化的配置文件

## USD 结构说明

MJCF 导入后的 USD 结构：
```
/jiyuan (根 Prim)
  /worldBody (articulation root，带有 ArticulationRootAPI)
    /base_link (第一个 body，但不是 articulation root)
      /right_hip_pitch_engine_link
      /left_hip_pitch_engine_link
      ...
```

**关键点**：`prim_path` 必须指向 `worldBody`，而不是 `Robot`，因为 MJCF 导入器会创建 `worldBody` 作为 articulation root。

## 如何适配其他机器人

### 方法1：复制并修改本文件（当前阶段）
```bash
cp jiyuan_scene_cfg.py unitree_go2_scene_cfg.py
```
然后修改：
1. `UsdFileCfg.usd_path` → 指向新机器人的 USD 文件
2. `init_state` → 调整初始位置、姿态、关节角度
3. `actuators` → 调整执行器参数（刚度、阻尼、力矩限制）
4. `joint_names_expr` → 匹配新机器人的关节命名

### 方法2：创建机器人配置库（未来阶段）
详见 `.migrate/08-multi-robot-architecture.md` 的长期规划。

参考:
- Isaac Lab 官方文档: https://isaac-sim.github.io/IsaacLab/main/
- USD 支持: omni.isaac.lab.sim.spawners.from_files.UsdFileCfg
- 多机器人架构: ../.migrate/08-multi-robot-architecture.md
"""

from __future__ import annotations

import os
from pathlib import Path
from tkinter.constants import FALSE

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import UsdFileCfg  # 改用 UsdFileCfg 替代 MjcfFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# 项目根目录（相对于当前文件：isaaclab_rl/jiyuan_tasks/envs/cfg/）
# 向上4级：cfg -> envs -> jiyuan_tasks -> isaaclab_rl -> 项目根目录
ISAAC_LAB_RL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# 默认模型名称（可通过环境变量 ROBOT_MODEL 覆盖）
DEFAULT_ROBOT_MODEL = "jiyuan"


def get_usd_path(model_name: str | None = None) -> str:
    """根据模型名称获取 USD 文件路径

    Args:
        model_name: 机器人模型名称
                   - 如果为 None，从环境变量 ROBOT_MODEL 读取
                   - 如果环境变量也不存在，使用默认值 "jiyuan"

    Returns:
        USD 文件的绝对路径字符串

    路径规则：
        assets/usd/{model_name}/{model_name}.usd

    示例：
        >>> get_usd_path("jiyuan")
        '/path/to/project/assets/usd/jiyuan/jiyuan.usd'
        >>> os.environ["ROBOT_MODEL"] = "unitree_go2"
        >>> get_usd_path()
        '/path/to/project/assets/usd/unitree_go2/unitree_go2.usd'
    """
    if model_name is None:
        model_name = os.environ.get("ROBOT_MODEL", DEFAULT_ROBOT_MODEL)
    return str(ISAAC_LAB_RL_ROOT / f"assets/usd/{model_name}/{model_name}.usd")


##
# 场景配置
##


@configclass
class JiyuanSceneCfg(InteractiveSceneCfg):
    """Jiyuan 双足机器人场景配置

    包含:
    - 机器人资产配置（从 USD 加载）
    - 地面/地形配置
    - 传感器配置（可选）

    注意:
    - USD 文件需要先从 MJCF 转换：`make convert-usd`
    - prim_path 指向 `worldBody`（MJCF 导入后的 articulation root）
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
    # USD 文件加载说明：
    # - prim_path 是在场景中创建引用的位置
    # - UsdFileCfg 会在这个位置创建对 jiyuan.usd 的引用
    # - Isaac Lab 会自动查找 articulation root（带有 ArticulationRootAPI 的 prim）
    robot: ArticulationCfg = ArticulationCfg(
        # 明确指向 MJCF 导入器生成的 worldBody，解决发现多个根节点的错误
        prim_path="{ENV_REGEX_NS}/Robot/worldBody",
        spawn=UsdFileCfg(
            # USD 文件路径（自动根据模型名称查找）
            # 从环境变量 ROBOT_MODEL 读取，默认为 "jiyuan"
            usd_path=get_usd_path(),
            # 启用contact sensors以支持contact_forces传感器
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
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
                # 膝关节：弯曲约30度
                # USD中左右膝关节限制相反（坐标系不同）
                # right_knee: [-2.094, 0.000] -> 使用-0.5
                # left_knee: [0.000, 2.094] -> 使用+0.5
                "right_knee.*": -0.5,
                "left_knee.*": 0.5,
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
                    ".*hip_pitch_engine_joint",
                    ".*hip_yaw_engine_joint",
                    ".*hip_roll_joint",
                    ".*knee_joint",
                ],
                # PD 控制参数（从 MJCF default_classes.xml 映射）
                # MJCF: damping=0.5, armature=0.01
                # Isaac Lab: stiffness和damping需要根据实际调优
                stiffness=80.0,  # 典型值：40-150
                damping=2.0,  # 典型值：1-5
                # 力矩限制（从 MJCF ctrlrange="-1 1" 映射）
                # 使用 effort_limit_sim 替代废弃的 effort_limit
                effort_limit_sim=150.0,
                # 速度限制（弧度/秒）
                # 注意: velocity_limit 不会被隐式执行器使用,仅用于文档
                # velocity_limit_sim 尚未实现,保留velocity_limit用于记录
                velocity_limit=10.0,
            ),
            # 踝关节电机（3自由度并联结构）
            "ankle_motors": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*ankle_1_3_joint",
                    ".*ankle_2_3_joint",
                    ".*ankle_3_3_joint",
                ],
                # 踝关节可能需要不同的参数
                stiffness=60.0,
                damping=1.5,
                effort_limit_sim=100.0,
                velocity_limit=10.0,
            ),
            # 脚趾电机
            "toe_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*toe_joint"],
                # MJCF toe_motor: damping=0.2, armature=0.005
                stiffness=40.0,
                damping=1.0,
                effort_limit_sim=50.0,
                velocity_limit=10.0,
            ),
        },
    )

    # 脚部接触传感器（用于奖励计算和步态检测）
    # 使用递归匹配，确保能找到嵌套在任意层级下的脚部链接
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot.*",
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
