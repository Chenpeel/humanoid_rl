# 架构设计和目录结构

本文档说明迁移后的 Isaac Lab 项目架构设计和目录组织。

## 整体架构

```
JAX/MJX 架构（原）                 Isaac Lab 架构（目标）
────────────────────                ──────────────────────
┌─────────────────┐                 ┌──────────────────────┐
│  训练脚本        │                 │   训练脚本(RSL_RL)    │
│  train.py       │                 │   scripts/train.py   │
└────────┬────────┘                 └──────────┬───────────┘
         │                                    │
         ↓                                    ↓
┌─────────────────┐                 ┌──────────────────────┐
│ PPO Trainer     │                 │   RSL_RL Runner      │
│ (纯 JAX 实现)   │                 │   (封装训练循环)      │
└────────┬────────┘                 └──────────┬───────────┘
         │                                    │
         ↓                                    ↓
┌─────────────────┐                 ┌──────────────────────┐
│  环境类          │                 │  ManagerBasedRLEnv   │
│  VelocityEnv    │                 │  (配置驱动)          │
└────────┬────────┘                 └──────────┬───────────┘
         │                                    │
         ↓                                    ↓
┌─────────────────┐                 ┌──────────────────────┐
│  MJX Physics    │                 │  Isaac Sim Physics   │
│  (GPU加速)      │                 │  (PhysX GPU)         │
└─────────────────┘                 └──────────────────────┘
```

## 完整目录结构

```
/home/chenpeel/work/repo/jiyuan/rl/
│
├── .migrate/                              # 迁移文档（新增）
│   ├── README.md                          # 迁移概览
│   ├── 01-setup.md                        # 环境搭建指南
│   ├── 02-architecture.md                 # 本文档
│   ├── 03-code-mapping.md                 # 代码映射
│   ├── 04-implementation-phases.md        # 实施计划（待创建）
│   ├── 05-testing-strategy.md             # 测试策略（待创建）
│   ├── 06-optimization-guide.md           # 优化指南（待创建）
│   ├── 07-troubleshooting.md              # 常见问题（待创建）
│   └── checklists/                        # 验收清单
│       ├── phase1-checklist.md            # ✅ 已创建
│       ├── phase2-checklist.md            # 待创建
│       ├── phase3-checklist.md            # 待创建
│       └── phase4-checklist.md            # 待创建
│
├── isaaclab_rl/                           # Isaac Lab 项目（新增）
│   ├── README.md                          # 项目说明
│   ├── setup.py                           # Python 包配置
│   ├── pyproject.toml                     # 现代项目配置
│   │
│   ├── assets/                            # 软链接到 ../assets
│   │   └── jiyuan/mjcf/                   # → ../../assets/xmls/models/jiyuan
│   │
│   ├── jiyuan_tasks/                      # 核心任务定义
│   │   ├── __init__.py                    # 环境注册
│   │   │
│   │   ├── envs/                          # 环境实现
│   │   │   ├── __init__.py
│   │   │   ├── jiyuan_base_env.py         # 基础环境类
│   │   │   ├── velocity_tracking_env.py   # 速度跟踪环境
│   │   │   ├── standing_env.py            # 站立环境
│   │   │   └── cfg/                       # 环境配置
│   │   │       ├── __init__.py
│   │   │       ├── jiyuan_scene_cfg.py    # 场景配置
│   │   │       ├── velocity_tracking_cfg.py # 速度跟踪配置
│   │   │       └── standing_cfg.py        # 站立配置
│   │   │
│   │   ├── managers/                      # 自定义管理器
│   │   │   ├── __init__.py
│   │   │   ├── observations.py            # 观测函数
│   │   │   ├── rewards.py                 # 奖励函数（从 MJX 迁移）
│   │   │   ├── commands.py                # 命令生成器
│   │   │   └── terminations.py            # 终止条件
│   │   │
│   │   └── utils/                         # 工具函数
│   │       ├── __init__.py
│   │       ├── math_utils.py              # 数学工具（quat_to_euler 等）
│   │       └── visualization.py           # 可视化辅助
│   │
│   ├── agents/                            # RL 算法配置
│   │   └── rsl_rl/                        # RSL_RL 配置
│   │       ├── __init__.py
│   │       ├── ppo_cfg.py                 # PPO 超参数
│   │       └── network_cfg.py             # 网络架构
│   │
│   ├── scripts/                           # 训练和评估脚本
│   │   ├── train.py                       # 主训练脚本
│   │   ├── play.py                        # 策略评估
│   │   └── export_policy.py               # 策略导出
│   │
│   ├── tools/                             # 辅助工具
│   │   ├── convert_mjcf_to_usd.py         # MJCF → USD 转换
│   │   ├── analyze_rewards.py             # 奖励分析
│   │   └── visualize_policy.py            # 策略可视化
│   │
│   └── tests/                             # 单元测试
│       ├── test_environments.py
│       ├── test_rewards.py
│       └── test_observations.py
│
├── src/                                   # 保留原 JAX/MJX 代码（jax 分支）
│   └── rl/
│       ├── envs/
│       ├── rewards/                       # 迁移参考
│       ├── models/
│       └── training/
│
├── assets/                                # 保留原模型资源
│   └── xmls/models/jiyuan/
│
└── configs/                               # 保留原配置
    └── train.yaml                         # 超参数参考
```

## 模块设计

### 1. jiyuan_tasks/envs/cfg/

**职责**: 环境配置（声明式）

**关键文件**:
- `jiyuan_scene_cfg.py`: 场景和机器人定义
- `velocity_tracking_cfg.py`: 速度跟踪任务的完整配置
- `standing_cfg.py`: 站立任务的完整配置

**设计原则**:
- 配置驱动，无硬编码
- 可继承和扩展
- 类型安全（使用 `@configclass`）

### 2. jiyuan_tasks/managers/

**职责**: 自定义的观测/奖励/命令/终止函数

**关键文件**:
- `observations.py`: 自定义观测计算函数
- `rewards.py`: 从 MJX 迁移的奖励函数
- `commands.py`: 速度命令生成器
- `terminations.py`: 摔倒检测等终止条件

**设计原则**:
- 纯函数设计
- 第一个参数始终是 `env`
- 返回 `torch.Tensor`

### 3. agents/rsl_rl/

**职责**: RL 算法配置

**关键文件**:
- `ppo_cfg.py`: PPO 超参数配置类

**设计原则**:
- 继承 RSL_RL 的配置基类
- 集中管理所有超参数

## 参考项目

学习这些开源项目的架构设计：

| 项目 | 机器人类型 | 可学习的点 |
|------|-----------|-----------|
| [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab) | TRON1 (双足) | 双足运动的奖励设计、课程学习 |
| [legged-loco](https://github.com/yang-zj1026/legged-loco) | Go2, H1 | 模块化配置、多任务支持 |
| [LeggedRobotsLab](https://github.com/guohua-zhang/LeggedRobotsLab) | Go1 EDU | Sim2Real 流程、真机部署 |
| [IsaacLab-Quadruped-Locomotion](https://github.com/huangfq07/IsaacLab-Quadruped-Locomotion) | 四足 | 高级 PPO 配置、性能优化 |

## 设计决策

### 为什么使用配置驱动设计？

**优势**:
1. 可读性高：配置即文档
2. 易于实验：修改参数无需改代码
3. 类型安全：编译时检查参数类型
4. 可复用：配置可继承和组合

### 为什么保留原 src/ 目录？

**理由**:
1. 作为迁移参考（奖励函数逻辑）
2. 性能基准（对比训练结果）
3. 知识保留（设计思路）
4. 回滚保险（如遇重大问题）

### 为什么使用 RSL_RL 而不是其他库？

**理由**:
1. 专业优化：ETH Zurich 开发，专为足部运动优化
2. GPU 加速：GPU 优化的 PPO 实现
3. 广泛验证：多篇论文验证
4. 官方集成：与 Isaac Lab 无缝集成

## 多机器人支持

### 长期架构规划

当前架构专注于 Jiyuan 单机器人，但已考虑未来扩展性。详细的多机器人架构演进规划见：

📄 **[08-multi-robot-architecture.md](./08-multi-robot-architecture.md)**

**核心策略**:
- **阶段1（当前）**: 单机器人专用结构（Jiyuan）
- **阶段2（2-3个机器人）**: 配置参数化 + 共享代码提取
- **阶段3（3+机器人）**: 通用任务架构，机器人配置库

### 如何添加新机器人

**当前阶段（简单复制）**:
```bash
# 1. 复制jiyuan_tasks目录
cp -r isaaclab_rl/jiyuan_tasks/ isaaclab_rl/new_robot_tasks/

# 2. 修改机器人配置
vim isaaclab_rl/new_robot_tasks/envs/cfg/new_robot_scene_cfg.py

# 3. 更新MJCF路径和执行器参数
```

**未来阶段（配置复用）**:
```python
# 创建机器人配置库
from isaaclab_rl.robots import JIYUAN_CFG, NEW_ROBOT_CFG

@configclass
class VelocityTaskCfg(ManagerBasedRLEnvCfg):
    robot = JIYUAN_CFG  # 或 NEW_ROBOT_CFG
```

详见 [08-multi-robot-architecture.md](./08-multi-robot-architecture.md)。

## 下一步

了解架构设计后：

1. 阅读 [03-code-mapping.md](./03-code-mapping.md) 学习具体的代码转换
2. （可选）阅读 [08-multi-robot-architecture.md](./08-multi-robot-architecture.md) 了解长期规划
3. 开始阶段 1：环境搭建（见 [checklists/phase1-checklist.md](./checklists/phase1-checklist.md)）
