# 多机器人架构设计与演进规划

**创建日期**: 2025-01-22
**当前阶段**: 阶段1 - 单机器人（Jiyuan）
**长期目标**: 支持多种机器人（Jiyuan, Unitree Go2, H1等）

---

## 1. 架构演进策略

### 阶段划分

| 阶段 | 时间点 | 机器人数量 | 架构方案 | 复杂度 |
|------|--------|-----------|---------|--------|
| **阶段1** | 当前 | 1个（Jiyuan） | 单机器人专用结构 | ⭐ 简单 |
| **阶段2** | 添加第2-3个机器人时 | 2-3个 | 配置参数化（方案C） | ⭐⭐ 中等 |
| **阶段3** | 3+机器人或需要统一管理时 | 3+个 | 通用任务架构（方案B重构） | ⭐⭐⭐ 复杂 |

---

## 2. 当前架构（阶段1）

### 目录结构

```
isaaclab_rl/
├── jiyuan_tasks/              # Jiyuan 专用任务
│   ├── envs/
│   │   ├── cfg/
│   │   │   ├── __init__.py
│   │   │   └── jiyuan_scene_cfg.py
│   │   ├── jiyuan_base_env.py
│   │   └── velocity_tracking_env.py
│   ├── managers/              # 自定义MDP函数
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── commands.py
│   │   └── terminations.py
│   └── utils/
├── agents/
│   └── rsl_rl/
└── scripts/
```

### 环境注册

```python
gym.register(
    id="Isaac-Jiyuan-Velocity-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": JiyuanVelocityTrackingEnvCfg},
)
```

### 优点
- ✅ 简单直接，专注Jiyuan迁移
- ✅ 避免过早优化
- ✅ 快速迭代验证

### 局限
- ❌ 添加新机器人需要复制整个任务目录
- ❌ MDP函数无法跨机器人复用

---

## 3. 过渡架构（阶段2：配置参数化）

### 触发条件
- 需要支持第2个机器人（如Unitree Go2）
- 发现大量重复代码

### 目录结构演进

```
isaaclab_rl/
├── jiyuan_tasks/              # 保持现有结构
│   └── ...
├── unitree_tasks/             # 新增：第2个机器人
│   └── ...
├── common/                    # 新增：共享代码
│   ├── mdp/                   # 通用MDP函数
│   │   ├── observations.py    # 通用观测（base_lin_vel等）
│   │   ├── rewards.py         # 通用奖励（lin_vel_tracking等）
│   │   └── terminations.py
│   └── utils/
│       ├── math_utils.py
│       └── terrain_utils.py
└── robots/                    # 新增：机器人配置库
    ├── __init__.py
    ├── jiyuan_cfg.py          # Jiyuan机器人配置
    └── unitree_go2_cfg.py     # Unitree Go2配置
```

### 机器人配置库设计

**robots/jiyuan_cfg.py**:
```python
"""Jiyuan 双足机器人配置"""
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sim import MjcfFileCfg

JIYUAN_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=MjcfFileCfg(
        asset_path="assets/xmls/models/jiyuan/index.xml",
        make_instanceable=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*knee.*": 0.5,
            # ...
        },
    ),
    actuators={
        "main_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*"],
            stiffness=80.0,
            damping=2.0,
        ),
    },
)
```

### 使用方式

**方式1：直接导入**
```python
from isaaclab_rl.robots import JIYUAN_CFG, UNITREE_GO2_CFG

@configclass
class VelocityTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg()
    scene.robot = JIYUAN_CFG  # 或 UNITREE_GO2_CFG
```

**方式2：工厂函数**
```python
from isaaclab_rl.common.factories import create_velocity_env

env = create_velocity_env(robot="jiyuan", num_envs=4096)
env = create_velocity_env(robot="unitree_go2", num_envs=4096)
```

### 迁移路径（从阶段1到阶段2）

1. **提取共享代码**
   ```bash
   # 将通用MDP函数移至 common/mdp/
   mv jiyuan_tasks/managers/rewards.py common/mdp/locomotion_rewards.py
   ```

2. **创建机器人配置库**
   ```bash
   # 提取机器人特定配置
   # jiyuan_scene_cfg.py → robots/jiyuan_cfg.py
   ```

3. **更新环境配置**
   ```python
   # 从机器人库导入
   from isaaclab_rl.robots import JIYUAN_CFG
   from isaaclab_rl.common.mdp import locomotion_rewards as rewards
   ```

4. **保持向后兼容**
   ```python
   # jiyuan_tasks/__init__.py 保持不变
   # 旧代码仍然可以用 "Isaac-Jiyuan-Velocity-v0"
   ```

---

## 4. 长期架构（阶段3：通用任务系统）

### 触发条件
- 支持3+个机器人
- 需要统一的任务管理
- 需要支持机器人×任务×地形的组合

### 目录结构（重构）

```
isaaclab_rl/
├── tasks/                     # 通用任务（核心变更）
│   ├── locomotion/
│   │   ├── velocity/
│   │   │   ├── config/
│   │   │   │   ├── base_cfg.py      # 基础配置
│   │   │   │   └── robots/          # 机器人特定覆写
│   │   │   │       ├── jiyuan.py
│   │   │   │       ├── unitree_go2.py
│   │   │   │       └── h1.py
│   │   │   └── velocity_env.py       # 通用环境实现
│   │   ├── standing/
│   │   └── mdp/                      # 任务级MDP函数
│   │       ├── observations.py
│   │       ├── rewards.py
│   │       └── commands.py
│   └── manipulation/                 # 未来：操作任务
│
├── robots/                    # 机器人资产库（扩展）
│   ├── configs/
│   │   ├── jiyuan.py
│   │   ├── unitree_go2.py
│   │   └── h1.py
│   └── utils/
│       └── asset_loader.py
│
├── terrains/                  # 地形库
│   ├── flat.py
│   ├── stairs.py
│   └── rough.py
│
├── common/                    # 全局共享代码
│   ├── mdp/                   # 通用MDP函数
│   └── utils/
│
└── agents/                    # 算法配置（保持不变）
    └── rsl_rl/
```

### 环境注册（统一格式）

```python
# 格式：Isaac-{Task}-{Robot}-{Terrain}-v{Version}
gym.register(
    id="Isaac-Velocity-Jiyuan-Flat-v0",
    id="Isaac-Velocity-UnitreeGo2-Stairs-v0",
    id="Isaac-Standing-H1-Rough-v0",
)

# 简化格式（默认地形）
gym.register(
    id="Isaac-Velocity-Jiyuan-v0",  # 等价于 Flat
)
```

### 使用方式（工厂模式）

```python
from isaaclab_rl.tasks import create_task

# 方式1：字符串参数
env = create_task(
    task="velocity",
    robot="jiyuan",
    terrain="flat",
    num_envs=4096,
)

# 方式2：配置组合
from isaaclab_rl.tasks.locomotion.velocity import VelocityTaskCfg
from isaaclab_rl.robots.configs import JIYUAN_CFG
from isaaclab_rl.terrains import FlatTerrainCfg

cfg = VelocityTaskCfg(
    robot=JIYUAN_CFG,
    terrain=FlatTerrainCfg(),
)
env = gym.make("Isaac-Velocity-v0", cfg=cfg)
```

### 配置覆写机制

**tasks/locomotion/velocity/config/base_cfg.py**:
```python
@configclass
class VelocityTaskCfg(ManagerBasedRLEnvCfg):
    """速度跟踪任务基础配置（机器人无关）"""

    # 观测配置（通用）
    observations = ObservationsCfg()

    # 奖励配置（通用）
    rewards = RewardsCfg(
        lin_vel_tracking=RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0),
        ang_vel_tracking=RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5),
        # ...
    )

    # 机器人配置（由子类覆写）
    robot: ArticulationCfg = None  # 必须由robot-specific配置提供
```

**tasks/locomotion/velocity/config/robots/jiyuan.py**:
```python
@configclass
class JiyuanVelocityTaskCfg(VelocityTaskCfg):
    """Jiyuan机器人的速度跟踪任务配置"""

    # 覆写机器人配置
    robot = JIYUAN_CFG

    # 覆写任务特定参数（如果需要）
    rewards = VelocityTaskCfg.rewards.replace(
        lin_vel_tracking=RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5),
    )
```

### 迁移路径（从阶段2到阶段3）

1. **重构任务目录**
   ```bash
   mkdir -p tasks/locomotion/velocity
   mv jiyuan_tasks/envs/velocity_tracking_env.py tasks/locomotion/velocity/
   ```

2. **提取通用配置**
   ```bash
   # 创建基础配置（去除机器人特定部分）
   # jiyuan_velocity_cfg.py → base_cfg.py
   ```

3. **创建机器人覆写**
   ```bash
   mkdir tasks/locomotion/velocity/config/robots/
   # 机器人特定部分 → robots/jiyuan.py
   ```

4. **更新注册系统**
   ```python
   # 使用新的工厂模式注册
   from isaaclab_rl.tasks import register_all_tasks
   register_all_tasks()
   ```

---

## 5. 命名标准化

### 环境注册ID规范

**阶段1（当前）**:
```python
"Isaac-Jiyuan-Velocity-v0"
"Isaac-Jiyuan-Standing-v0"
```

**阶段2（过渡）**:
```python
# 保持兼容旧格式
"Isaac-Jiyuan-Velocity-v0"
"Isaac-UnitreeGo2-Velocity-v0"

# 可选：新增统一格式
"Isaac-Velocity-Jiyuan-v0"
"Isaac-Velocity-UnitreeGo2-v0"
```

**阶段3（长期）**:
```python
# 标准格式
"Isaac-{Task}-{Robot}-{Terrain}-v{Version}"

# 示例
"Isaac-Velocity-Jiyuan-Flat-v0"
"Isaac-Velocity-Jiyuan-Stairs-v1"
"Isaac-Standing-H1-Rough-v0"

# 简化格式（默认Flat地形）
"Isaac-Velocity-Jiyuan-v0"
```

### 文件命名规范

**阶段1-2**:
```
jiyuan_scene_cfg.py
jiyuan_velocity_cfg.py
unitree_go2_scene_cfg.py
```

**阶段3**:
```
robots/configs/jiyuan.py
robots/configs/unitree_go2.py
tasks/locomotion/velocity/config/base_cfg.py
tasks/locomotion/velocity/config/robots/jiyuan.py
```

---

## 6. 开源库集成策略

### 保持兼容性原则

**无论哪个阶段，都确保**:
```python
# 标准Gym接口
env = gym.make("Isaac-{Task}-{Robot}-v0", num_envs=4096)

# 兼容主流RL库
from rsl_rl.runners import OnPolicyRunner
from stable_baselines3 import PPO
from ray.rllib.algorithms.ppo import PPOConfig
```

### 扩展性保证

**添加自定义机器人示例**:
```python
# 阶段1：复制jiyuan_tasks目录
cp -r jiyuan_tasks/ my_robot_tasks/

# 阶段2：添加机器人配置
# robots/my_robot_cfg.py
MY_ROBOT_CFG = ArticulationCfg(...)

# 阶段3：创建robot配置覆写
# tasks/locomotion/velocity/config/robots/my_robot.py
@configclass
class MyRobotVelocityTaskCfg(VelocityTaskCfg):
    robot = MY_ROBOT_CFG
```

---

## 7. 实施时间表

### 阶段1（当前 - Week 1-8）
- ✅ 实现Jiyuan单机器人
- ✅ 验证训练流程
- ✅ 建立基础架构

### 阶段2（Week 9-16，如需添加第2个机器人）
- 📅 Week 9: 提取共享代码到 `common/`
- 📅 Week 10: 创建 `robots/` 配置库
- 📅 Week 11-12: 添加第2个机器人（如Unitree Go2）
- 📅 Week 13-14: 验证配置复用
- 📅 Week 15-16: 文档和示例更新

### 阶段3（Month 5+，如需支持3+机器人）
- 📅 Month 5: 设计重构方案
- 📅 Month 6: 重构 `tasks/` 目录
- 📅 Month 7: 迁移现有机器人到新架构
- 📅 Month 8: 添加地形系统
- 📅 Month 9: 完整测试和文档

---

## 8. 参考项目架构对比

### legged-loco（参考架构）

```
legged_gym/
  envs/
    base/
      legged_robot.py          # 通用基类
      legged_robot_config.py   # 通用配置
    go2/
      go2_config.py            # 机器人特定配置
    h1/
      h1_config.py
```

**优点**:
- 清晰的继承关系
- 配置参数化

**缺点**:
- 机器人特定代码仍有重复

### bipedal_locomotion_isaaclab（参考案例）

```
bipedal_locomotion/
  tasks/
    tron_env.py                # 单机器人实现
  config/
    tron_config.py
```

**特点**:
- 专注单机器人（与我们阶段1类似）
- 简单直接

### 我们的方案优势

- ✅ 渐进式演进，避免过早优化
- ✅ 每个阶段都可独立使用
- ✅ 向后兼容
- ✅ 明确的迁移路径

---

## 9. 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2025-01-22 | 采用阶段1架构（单机器人） | 专注Jiyuan迁移，避免过早优化 |
| 2025-01-22 | 规划阶段2/3演进路径 | 确保长期可扩展性 |
| 2025-01-22 | 标准化命名规范 | 与Isaac Lab生态保持一致 |

---

## 10. 附录：快速参考

### 当前如何添加新任务（阶段1）

```bash
# 1. 复制velocity配置
cp jiyuan_tasks/envs/cfg/velocity_tracking_cfg.py \
   jiyuan_tasks/envs/cfg/new_task_cfg.py

# 2. 修改配置

# 3. 注册环境
# 在 jiyuan_tasks/__init__.py 添加注册代码
```

### 未来如何添加新机器人（阶段2）

```bash
# 1. 创建机器人配置
vim robots/new_robot_cfg.py

# 2. 复用或创建任务
cp -r jiyuan_tasks/ new_robot_tasks/

# 3. 替换机器人配置
# 在 new_robot_tasks/envs/cfg/*.py 中
# 使用 NEW_ROBOT_CFG 替换 JIYUAN_CFG
```

### 长期如何添加新机器人（阶段3）

```bash
# 1. 创建机器人配置
vim robots/configs/new_robot.py

# 2. 创建任务配置覆写
vim tasks/locomotion/velocity/config/robots/new_robot.py

# 3. 自动注册（工厂模式）
# 无需手动注册，框架自动发现
```

---

**文档版本**: v1.0
**下次审查**: 添加第2个机器人时
**负责人**: Jiyuan Robotics Team
