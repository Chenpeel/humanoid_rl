# Isaac Lab RL - Jiyuan 双足机器人

使用 Isaac Lab + PyTorch + RSL_RL 训练 Jiyuan 双足机器人的完整实现。

## 🎯 项目状态

**当前阶段**: 阶段1 - 环境搭建和基础验证（进行中）

- ✅ 项目结构创建
- ✅ 包配置完成（setup.py, pyproject.toml）
- ✅ 场景配置（Jiyuan机器人MJCF加载）
- ✅ 测试环境实现（Isaac-Jiyuan-Test-v0）
- ✅ 环境注册系统
- ✅ 测试脚本
- ⏸️ Isaac Lab 安装（需要用户执行）
- ⏸️ 环境验证（待Isaac Lab安装后）
- ⏸️ 完整任务环境（速度跟踪、站立等）

## 📁 项目结构

```
isaaclab_rl/
├── jiyuan_tasks/              # Jiyuan 机器人任务
│   ├── envs/
│   │   └── cfg/
│   │       ├── jiyuan_scene_cfg.py          # 场景和机器人配置
│   │       └── jiyuan_test_env_cfg.py       # 测试环境配置
│   ├── managers/              # 自定义MDP函数（待实现）
│   └── utils/                 # 工具函数（待实现）
├── agents/                    # RL 算法配置（待实现）
│   └── rsl_rl/
├── scripts/                   # 脚本
│   └── test_env.py           # ✅ 环境测试脚本
└── tests/                     # 单元测试（待实现）
```

## 🚀 快速开始

### 前置依赖

#### 1. Isaac Sim 2024.1.1+

通过 Omniverse Launcher 安装：
```bash
# 下载 Omniverse Launcher
# https://www.nvidia.com/en-us/omniverse/download/

# 在 Launcher 中安装 Isaac Sim 2024.1.1+
```

#### 2. Isaac Lab

```bash
# 克隆 Isaac Lab
cd /home/chenpeel/work/repo/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v1.2.0  # 或最新稳定版

# 安装 Isaac Lab
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
./isaaclab.sh --install

# 安装 RSL_RL
./isaaclab.sh --extra rsl_rl
```

详见：`../.migrate/01-setup.md`

### 安装本项目

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
pip install -e .
```

## 🧪 测试环境

### 方法1：直接运行测试脚本

```bash
# 基础测试（16个并行环境，10步）
python scripts/test_env.py

# 更多并行环境
python scripts/test_env.py --num_envs 64

# 更多测试步数
python scripts/test_env.py --num_steps 100

# 无头模式（不显示GUI）
python scripts/test_env.py --headless
```

### 方法2：Python代码测试

```python
import gymnasium as gym
import torch
from isaaclab_rl import jiyuan_tasks  # 触发环境注册

# 创建环境
env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=16)

# 重置环境
obs, info = env.reset()
print(f"观测形状: {obs.shape}")  # (16, 65) - 16个环境，65维观测

# 执行步骤
action = torch.zeros(16, 16)  # 16个环境，16个关节
obs, reward, terminated, truncated, info = env.step(action)

print(f"平均奖励: {reward.mean().item()}")
```

## 📊 测试环境说明

### Isaac-Jiyuan-Test-v0

**用途**: 最小可行环境，用于验证基础功能

**观测空间** (65维):
- 基础线速度 (3)
- 基础角速度 (3)
- 投影重力 (3)
- 关节位置 (16)
- 关节速度 (16)
- 上一步动作 (16)
- 命令（计划中）

**动作空间** (16维):
- 16个关节的位置目标（相对于默认姿态）

**奖励函数**:
- 存活奖励: +1.0
- 动作变化率惩罚: -0.01
- 关节速度惩罚: -0.0001

**终止条件**:
- 时间限制: 10秒
- 机器人摔倒（base接触地面）

## 📚 文档

### 迁移文档

位于 `../.migrate/` 目录：

| 文档 | 内容 | 优先级 |
|------|------|--------|
| [README.md](../.migrate/README.md) | 迁移概览 | ⭐⭐⭐⭐⭐ |
| [01-setup.md](../.migrate/01-setup.md) | 环境搭建详细指南 | ⭐⭐⭐⭐⭐ |
| [02-architecture.md](../.migrate/02-architecture.md) | 架构设计 | ⭐⭐⭐⭐⭐ |
| [03-code-mapping.md](../.migrate/03-code-mapping.md) | MJX → Isaac Lab 代码映射 | ⭐⭐⭐⭐ |
| [08-multi-robot-architecture.md](../.migrate/08-multi-robot-architecture.md) | 多机器人架构规划 | ⭐⭐⭐ |

### 阶段验收清单

- [phase1-checklist.md](../.migrate/checklists/phase1-checklist.md) - 当前阶段

## 🔧 开发指南

### 添加新任务环境

1. **创建环境配置**:
```python
# isaaclab_rl/jiyuan_tasks/envs/cfg/my_task_cfg.py
from .jiyuan_test_env_cfg import JiyuanTestEnvCfg

@configclass
class MyTaskEnvCfg(JiyuanTestEnvCfg):
    # 修改奖励、观测等配置
    pass
```

2. **注册环境**:
```python
# isaaclab_rl/jiyuan_tasks/__init__.py
gym.register(
    id="Isaac-Jiyuan-MyTask-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MyTaskEnvCfg},
)
```

### 添加新机器人

详见：`../.migrate/08-multi-robot-architecture.md`

**简单方式（当前阶段）**:
```bash
# 1. 复制jiyuan_tasks目录
cp -r jiyuan_tasks/ new_robot_tasks/

# 2. 修改场景配置
vim new_robot_tasks/envs/cfg/*_scene_cfg.py
# 更新 MjcfFileCfg.asset_path 和执行器参数

# 3. 注册新环境
vim new_robot_tasks/__init__.py
```

## 🛠️ 故障排查

### 常见问题

**Q: 导入 isaaclab_rl 时报错**
```bash
# 确保安装了项目
pip install -e .

# 确保在正确的conda环境
conda activate isaaclab
```

**Q: 环境创建失败**
```bash
# 检查 Isaac Lab 是否正确安装
python -c "import omni.isaac.lab; print('OK')"

# 检查 RSL_RL 是否安装
python -c "from rsl_rl.algorithms import PPO; print('OK')"
```

**Q: MJCF 模型加载失败**
```bash
# 检查资产路径
ls -la ../assets/xmls/models/jiyuan/index.xml

# 验证 MJCF 语法
python -c "import mujoco; mujoco.MjModel.from_xml_path('path/to/index.xml')"
```

更多问题：`../.migrate/07-troubleshooting.md`

## 🤝 集成其他RL库

### RSL_RL (推荐)

```python
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym

env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=4096)
runner = OnPolicyRunner(env, train_cfg=..., log_dir="logs/")
runner.learn(num_learning_iterations=10000)
```

### Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=1)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1000000)
```

### Ray RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig().environment("Isaac-Jiyuan-Test-v0")
algo = config.build()
algo.train()
```

## 📝 待办事项

### 阶段1（当前）
- [ ] 安装 Isaac Lab（用户执行）
- [ ] 运行测试脚本验证环境
- [ ] 解决MJCF加载问题（如果有）
- [ ] 验证16-4096个并行环境

### 阶段2（Week 3-5）
- [ ] 实现完整的观测管理器
- [ ] 迁移奖励函数（从JAX到PyTorch）
- [ ] 实现命令生成器
- [ ] 集成RSL_RL训练器

### 阶段3（Week 6-7）
- [ ] 完整训练验证（30M steps）
- [ ] 性能基准测试
- [ ] 视频录制

## 📄 许可证

MIT

## 🙏 致谢

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - NVIDIA的强化学习框架
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl) - ETH Zurich的PPO实现
- [legged-loco](https://github.com/yang-zj1026/legged-loco) - 参考架构

---

**文档版本**: v0.1.0
**最后更新**: 2025-01-22
**项目状态**: 🟡 开发中
