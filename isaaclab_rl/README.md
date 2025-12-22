# Isaac Lab RL - Jiyuan 双足机器人

这是使用 Isaac Lab + PyTorch + RSL_RL 训练 Jiyuan 双足机器人的完整实现。

## 项目结构

```
isaaclab_rl/
├── jiyuan_tasks/          # 任务环境定义
│   ├── envs/             # 环境实现
│   │   └── cfg/          # 环境配置
│   ├── managers/         # 自定义管理器函数
│   └── utils/            # 工具函数
├── agents/               # RL 算法配置
│   └── rsl_rl/          # RSL_RL PPO 配置
├── scripts/              # 训练和评估脚本
├── tools/                # 辅助工具
└── tests/                # 单元测试
```

## 安装

### 前置依赖

1. **Isaac Sim 2024.1.1+**
   ```bash
   # 通过 Omniverse Launcher 安装
   ```

2. **Isaac Lab**
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ./isaaclab.sh --install
   ./isaaclab.sh --extra rsl_rl
   ```

### 安装本项目

```bash
cd isaaclab_rl
pip install -e .
```

## 使用方法

### 训练

```bash
# 使用 Isaac Lab 启动器
${ISAACLAB_PATH}/isaaclab.sh -p scripts/train.py \
    --task Isaac-Jiyuan-Velocity-Tracking-v0 \
    --num_envs 4096 \
    --headless
```

### 评估

```bash
# 评估训练好的策略
${ISAACLAB_PATH}/isaaclab.sh -p scripts/play.py \
    --task Isaac-Jiyuan-Velocity-Tracking-v0 \
    --checkpoint logs/velocity_tracking/model_10000.pt
```

## 迁移文档

详细的迁移文档位于 `../.migrate/` 目录：
- `README.md` - 迁移概览
- `01-setup.md` - 环境搭建
- `02-architecture.md` - 架构设计
- `03-code-mapping.md` - 代码映射

## 开发状态

- ✅ 项目结构创建
- ✅ 包配置完成
- ⏸️ 环境实现（进行中）
- ⏸️ 奖励函数迁移（计划中）
- ⏸️ 训练脚本（计划中）

## 扩展性和多机器人支持

### 当前架构
- 专注于 Jiyuan 双足机器人
- 清晰的模块化设计，易于理解和维护

### 添加新机器人
**简单方式（当前）**:
```bash
# 复制jiyuan_tasks目录
cp -r jiyuan_tasks/ new_robot_tasks/
# 修改配置文件中的机器人参数
```

**长期规划**:
- 配置参数化：机器人配置库 + 通用任务
- 详见 `../.migrate/08-multi-robot-architecture.md`

### 集成其他RL库
当前架构完全兼容主流RL库：
```python
# RSL_RL
from rsl_rl.runners import OnPolicyRunner

# Stable-Baselines3
from stable_baselines3 import PPO

# Ray RLlib
from ray.rllib.algorithms.ppo import PPOConfig
```

## 许可证

MIT
