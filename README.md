# Humanoid RL

通用双足机器人强化学习训练框架。

基于 Isaac Lab + RSL-RL + PyTorch，支持多种人形机器人的站立、行走、速度跟踪等任务训练。

## 支持的机器人

| 机器人 | 状态 | 任务 |
|--------|------|------|
| Unitree H1 | ✅ | standing, walking, velocity_tracking |
| Unitree H2 | 🚧 | 规划中 |
| Unitree G1 | 🚧 | 规划中 |

## 快速开始

```bash
# 1. 安装依赖
make install

# 2. 验证环境
make verify

# 3. 训练站立任务（H1）
make train ROBOT=h1 TASK=standing

# 4. 评估策略
make play ROBOT=h1
```

## 项目结构

```
humanoid_rl/
├── pyproject.toml         # 项目配置、依赖
├── Makefile               # 构建和训练命令
├── robots/                # 机器人模型资产
│   └── unitree_h1/
├── configs/               # YAML 配置文件
│   └── robots/            # 每机器人配置
├── humanoid_rl/           # 核心 Python 包
│   ├── envs/              # 环境定义和配置
│   │   └── cfg/           # 任务配置（standing, walking, velocity_tracking）
│   ├── managers/          # Reward / Observation / Termination 管理器
│   ├── agents/            # RL 算法配置（PPO）
│   └── utils/             # 工具函数
├── scripts/               # 入口脚本
│   ├── train.py
│   └── play.py
└── tests/                 # 测试
```

## 技术栈

- **环境管理**: [uv](https://github.com/astral-sh/uv) (Python)
- **仿真**: [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- **RL 算法**: [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (PPO)
- **深度学习**: PyTorch
- **配置**: Hydra / OmegaConf

## 架构

```
Robot Config (YAML)
      │
      ▼
BaseEnv (Isaac Lab Task)
      │
      ├── Observations Manager   ── 关节状态、IMU、速度等
      ├── Commands Manager       ── 速度指令生成
      ├── Rewards Manager        ── 奖励项加权求和
      └── Terminations Manager   ── 终止条件判断
      │
      ▼
PPO Agent (RSL-RL) ── Policy → Action → Reward → …
```
