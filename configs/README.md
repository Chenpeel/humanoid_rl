# 配置文件使用指南

## 核心设计理念 ⭐

**自动课程学习 + 统一配置**

从 2026-01-05 起，项目采用自动三阶段课程学习框架：
- **阶段1（0-50k steps）**：站立平衡 - 学习保持直立不摔倒
- **阶段2（50k-150k steps）**：低速行走 - 学习基本步态和低速前进
- **阶段3（150k+ steps）**：全速行走 - 跟踪任意速度命令并优化性能

**优势**：
✅ **奖励权重自动切换**：无需手动配置多个阶段文件
✅ **环境参数自动调整**：速度命令范围、目标高度自动适配
✅ **训练日志记录阶段信息**：curriculum_stage、curriculum_stage_index、curriculum_progress
✅ **可复用框架**：未来可扩展到跳跃、攀爬等任务

## 目录结构

```
configs/
├── train/                      # 标准训练配置
│   └── train_default.yaml      # 默认配置（2048 envs，200M steps）
│
├── train-10h/                  # 长时间训练配置
│   └── train_long.yaml          # 10小时配置（4096 envs，500M steps）
│
├── quick_test/                 # 快速测试配置
│   └── quick_test.yaml          # 快速验证（128 envs，1M steps）
│
├── train.yaml                  # 通用训练配置（已弃用，使用 train/train_default.yaml）
├── train_walking.yaml          # 高配训练配置（已弃用，使用 train-10h/train_long.yaml）
└── README.md                   # 本文档
```

## 配置类型说明

### 1. 标准训练（train/）

**适用场景**：日常训练，平衡性能和资源消耗

**硬件要求**：
- GPU：RTX 3090 / RTX 4080（16GB+）
- RAM：64GB+

**使用方式**：
```bash
make train CONFIG=configs/train/train_default.yaml
```

**配置特点**：
- `num_envs`: 2048（适中并行度）
- `total_timesteps`: 200M（覆盖完整课程）
- 自动课程学习覆盖三个阶段

### 2. 长时间训练（train-10h/）

**适用场景**：高性能训练，追求最佳效果

**硬件要求**：
- GPU：RTX 4090（24GB）或更高
- RAM：128GB+

**使用方式**：
```bash
make train CONFIG=configs/train-10h/train_long.yaml
```

**配置特点**：
- `num_envs`: 4096（最大并行度）
- `total_timesteps`: 500M（充分训练，确保收敛）
- 适合长时间无人值守训练

### 3. 快速测试（quick_test/）

**适用场景**：代码调试、功能验证

**硬件要求**：
- GPU：GTX 1080Ti（11GB）或更高
- RAM：32GB+

**使用方式**：
```bash
make train CONFIG=configs/quick_test/quick_test.yaml
```

**配置特点**：
- `num_envs`: 128（最小并行度）
- `total_timesteps`: 1M（快速完成）
- `hidden_dims`: [128, 128]（小网络）
- 适合快速迭代和验证

## 课程学习机制

### 自动阶段切换

课程学习模块（`src/rl/curriculum/walking.py`）会根据训练步数自动切换：

| 阶段 | 步数范围 | 速度命令 | 目标高度 | 训练目标 |
|-----|---------|---------|---------|---------|
| 1: 站立平衡 | 0-50k | 0 m/s | 0.45m | 学习保持直立不摔倒 |
| 2: 低速行走 | 50k-150k | 0.1-0.3 m/s | 0.40m | 学习基本步态和低速前进 |
| 3: 全速行走 | 150k+ | -0.2~0.8 m/s | 0.35m | 跟踪任意速度命令 |

### 阶段切换日志

训练时会自动打印阶段切换信息：

```
======================================================================
[课程学习] 阶段切换
  训练步数: 50,000
  站立平衡 → 低速行走
  目标: 学习双脚交替接触、建立基本步态模式、实现低速稳定前进
======================================================================
```

### 日志字段

TensorBoard/WandB 日志中包含以下字段：
- `curriculum_stage`: 当前阶段名称（站立平衡/低速行走/全速行走）
- `curriculum_stage_index`: 阶段索引（0/1/2）
- `curriculum_progress`: 当前阶段进度（0-1，第3阶段为 None）

## 参数说明

### 环境参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `num_envs` | 并行环境数 | 标准：2048，高配：4096，测试：128 |
| `num_steps` | 每次rollout步数 | 64（平衡时序信息和显存） |
| `max_episode_steps` | episode最大步数 | 2000（行走任务）|

### PPO参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `num_epochs` | 每次update的epoch数 | 4 |
| `num_minibatches` | mini-batch数量 | 8 |
| `gamma` | 折扣因子 | 0.99 |
| `gae_lambda` | GAE lambda | 0.95 |
| `clip_epsilon` | PPO裁剪系数 | 0.2 |
| `max_grad_norm` | 梯度裁剪阈值 | 0.5 |

### 优化器参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `learning_rate` | 峰值学习率 | 3.0e-4 |
| `final_lr_fraction` | 最终LR比例 | 0.02（最终=6e-6） |

### 网络参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `hidden_dims` | 隐藏层维度 | 标准：[256,256,128]，测试：[128,128] |
| `shared_backbone` | 共享backbone | true（提高样本效率） |

## 最佳实践

```bash
# 日常训练（平衡性能和成本）
make train CONFIG=configs/train/train_default.yaml

# 追求最佳性能（长时间训练）
make train CONFIG=configs/train-10h/train_long.yaml

# 快速验证代码正确性
make train CONFIG=configs/quick_test/quick_test.yaml
```

## 常见问题

**Q: 如何禁用课程学习？**

A: 课程学习仅在 `env_type=walking` 时启用。如需禁用，可以设置 `env_type=velocity`（速度跟踪任务）。

**Q: 课程学习支持哪些任务？**

A: 当前支持 `walking` 任务。未来可扩展到 `jumping`、`climbing` 等任务，只需继承 `BaseCurriculum` 并实现 `_define_stages` 方法。

**Q: 训练中途切换配置会怎样？**

A: 课程学习状态不会保存到 checkpoint，重新加载时会根据当前步数自动恢复到对应阶段。

## 参考资料

- Isaac Lab Curriculum: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- Gait-Conditioned RL: https://arxiv.org/abs/2505.20619
