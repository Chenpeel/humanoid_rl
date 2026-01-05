# JRL - JAX强化学习训练库

基于JAX + MJX的双足机器人强化学习训练框架，采用自动三阶段课程学习策略。

## 技术栈

- **物理引擎**: MuJoCo (MJX)
- **深度学习**: JAX 0.4.20+
- **算法**: PPO (Proximal Policy Optimization)
- **课程学习**: 自动三阶段渐进式训练
- **硬件**: NVIDIA GPU with CUDA 11.8+

## 核心特性

### 🎓 自动课程学习

从 2026-01-05 起，项目采用自动三阶段课程学习框架：

- **阶段1（0-50k steps）**：站立平衡 - 学习保持直立不摔倒
- **阶段2（50k-150k steps）**：低速行走 - 学习基本步态和低速前进
- **阶段3（150k+ steps）**：全速行走 - 跟踪任意速度命令并优化性能

**优势**：
- ✅ 奖励权重自动切换 - 无需手动配置多个阶段文件
- ✅ 环境参数自动调整 - 速度命令范围、目标高度自动适配
- ✅ 训练日志记录阶段信息 - curriculum_stage、progress等
- ✅ 可复用框架 - 未来可扩展到跳跃、攀爬等任务

## 快速开始

### 安装依赖

```bash
make install
```

### 训练

```bash
# 标准训练（2048 envs，200M steps，约8-12小时）
make train

# 长时间训练（4096 envs，500M steps，约24-48小时）
make train-long

# 快速测试（128 envs，1M steps，约5-10分钟）
make train-test
```

### 监控训练

```bash
make tensorboard
# 访问: http://localhost:6006
```

### 评估模型

```bash
make eval CKPT=logs/ppo_*/checkpoints/best_model
```

## 目录结构

```
jrl/
├── configs/                    # 配置文件
│   ├── train/                  # 标准训练配置
│   ├── train-10h/              # 长时间训练配置
│   └── quick_test/             # 快速测试配置
├── src/rl/                     # 源代码
│   ├── envs/                   # 环境实现
│   ├── models/                 # 神经网络模型
│   ├── rewards/                # 奖励函数（含三阶段权重）
│   ├── curriculum/             # 课程学习模块
│   ├── training/               # 训练逻辑
│   └── utils/                  # 工具函数
├── scripts/                    # 训练脚本
│   ├── train.py                # 主训练脚本（集成课程学习）
│   └── eval.py                 # 评估脚本
├── assets/                     # 机器人资源
├── docs/                       # 项目文档
└── Makefile                    # 命令管理

```

## 配置说明

详细配置说明请参考：[configs/README.md](configs/README.md)

## 常用命令

```bash
# 查看所有可用命令
make help

# 检查环境
make check-env

# 验证配置文件
make validate-config

# 查看项目信息
make info

# 清理缓存和日志
make clean-all
```

## 参考文献

- **Isaac Lab Curriculum**: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- **Gait-Conditioned RL**: https://arxiv.org/abs/2505.20619
