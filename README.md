# JRL - JAX强化学习训练库

基于JAX + MJX的双足机器人强化学习训练框架，采用分阶段课程学习策略。

## 技术栈

- **物理引擎**: MuJoCo (MJX)
- **深度学习**: JAX 0.4.20+
- **算法**: PPO (Proximal Policy Optimization)
- **硬件**: NVIDIA GPU with CUDA 11.8+

## 快速开始

### 安装依赖

```bash
make install
```

### 训练

```bash
make train-all
```

### 监控训练

```bash
make tensorboard
```

### 评估模型

```bash
make eval CKPT=logs/ppo_*/checkpoints/best_model
```

## 目录结构

```
jrl/
├── configs/                    # 配置文件
│   ├── train/                  # 正式训练配置
│   └── test/                   # 测试配置
├── src/rl/                     # 源代码
│   ├── envs/                   # 环境实现
│   ├── models/                 # 神经网络模型
│   ├── rewards/                # 奖励函数
│   ├── training/               # 训练逻辑
│   └── utils/                  # 工具函数
├── scripts/                    # 训练脚本
│   ├── train.py                # 单阶段训练
│   └── train_staged.py         # 流水线训练
├── assets/                     # 机器人资源
├── docs/                       # 项目文档
└── Makefile                    # 命令管理

```
