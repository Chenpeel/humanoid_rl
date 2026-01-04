# 配置文件使用指南

## 目录结构

```
configs/
├── train/                      # 正式训练配置（120G RAM + 24G GPU）
│   ├── stage0_standing.yaml    # 站立（164M步，3天）
│   ├── stage1_stepping.yaml    # 踏步（328M步）
│   ├── stage2_slow_walk.yaml   # 慢速行走（655M步）
│   ├── stage3_normal_walk.yaml # 正常行走（1.31B步，24天）★重点
│   ├── stage4_fast_walk.yaml   # 高速行走（819M步）
│   ├── stage5_terrain.yaml     # 地形适应（暂禁用）
│   └── stage6_robustness.yaml  # 鲁棒性（暂禁用）
│
├── test/                       # 测试配置（快速验证）
│   ├── test_quick.yaml         # 极简测试（128K步，<2分钟，单阶段）
│   ├── test_stage0.yaml        # 阶段0验证（8M步，~10分钟，单阶段）
│   ├── pipeline_stage0_standing.yaml   # 流水线测试-站立（32K步）
│   ├── pipeline_stage1_stepping.yaml   # 流水线测试-踏步（32K步）
│   ├── pipeline_stage2_slow_walk.yaml  # 流水线测试-慢走（32K步）
│   └── fast_test.yaml          # 旧版快速测试（保留兼容）
│
├── train.yaml                  # 单独训练（1080Ti优化，3.28B步）
├── train_walking.yaml          # 高配训练（RTX4090，10B步）
└── README.md                   # 本文档
```

## 配置类型说明

### 测试配置（configs/test/）

**用途**: 快速验证代码、调试、开发迭代

| 配置文件 | 环境数 | 训练步数 | 预计时间 | 资源需求 | 用途 |
|---------|-------|---------|---------|---------|------|
| test_quick.yaml | 256 | 128K | <2分钟 | 4-6GB | 单阶段最快验证 |
| test_stage0.yaml | 512 | 8M | ~10分钟 | 8-10GB | 单阶段流程验证 |
| pipeline_stage0-2.yaml | 128 | 32K×3 | <3分钟 | 6-8GB | **流水线3阶段测试** |
| fast_test.yaml | 512 | 262K | ~5分钟 | 8-10GB | 旧版测试 |

### 训练配置（configs/train/）

**用途**: 生产训练，获得可用模型

| 配置文件 | 环境数 | 训练步数 | 预计时间 | 阶段目标 |
|---------|-------|---------|---------|---------|
| stage0_standing.yaml | 1024 | 164M | 3天 | 学会站立平衡 |
| stage1_stepping.yaml | 1024 | 328M | 5-7天 | 原地踏步 |
| stage2_slow_walk.yaml | 1024 | 655M | 10-12天 | 小步行走 |
| stage3_normal_walk.yaml | 1024 | 1.31B | 24天 | ★正常行走（重点）|
| stage4_fast_walk.yaml | 1024 | 819M | 15-18天 | 高速行走 |

**资源要求**: 120GB RAM + 24GB GPU（1080Ti/RTX4090）

### 独立配置（configs/）

| 配置文件 | 用途 | 特点 |
|---------|------|------|
| train.yaml | 单独训练 | 1080Ti优化，3.28B步 |
| train_walking.yaml | 高配训练 | RTX4090专用，10B步 |

## 快速开始

### 方法1：使用Makefile（推荐）

```bash
# 测试命令
make test-quick       # 单阶段最快验证（<2分钟）
make test-stage0      # 单阶段完整验证（~10分钟）
make test-pipeline    # 流水线3阶段测试（<3分钟）★验证完整流程
make test-staged      # 流水线2阶段测试（验证切换逻辑）
make validate-config  # 检查配置语法

# 训练命令
make train            # 训练阶段0（站立）
make train-stage STAGE=0          # 单阶段训练
make train-all                    # 流水线训练所有阶段
make train-range FROM=2 TO=4      # 训练阶段2-4
```

### 方法2：直接使用Python脚本

**单阶段训练**:
```bash
python scripts/train.py --config configs/train/stage0_standing.yaml
```

**流水线训练**:
```bash
python scripts/train_staged.py --start-stage 0 --end-stage 4
```

**快速测试**:
```bash
# 单阶段快速测试
python scripts/train.py --config configs/test/test_quick.yaml

# 流水线快速测试（3阶段）
python scripts/train_staged.py --test-mode --start-stage 0 --end-stage 2
```

### 方法3：YAML + 命令行覆盖

```bash
# 使用测试配置，但修改环境数
python scripts/train.py --config configs/test/test_quick.yaml --num-envs 512

# 使用训练配置，但覆盖学习率
python scripts/train.py --config configs/train/stage0_standing.yaml --learning-rate 5e-4
```

## 配置文件结构

所有YAML配置文件包含以下关键段落:

```yaml
# ==================== 场景配置 ====================
scene: jiyuan_fit_flat         # 场景类型
env_type: walking              # 环境类型

# ==================== 环境配置 ====================
num_envs: 1024                 # 并行环境数
num_steps: 64                  # 每次rollout步数
max_episode_steps: 500         # episode最大步数

# ==================== 任务参数 ====================
cmd_x_range: [0.0, 1.0]        # 前向速度范围
cmd_y_range: [-0.3, 0.3]       # 侧向速度范围
cmd_yaw_range: [-0.5, 0.5]     # 角速度范围

# ==================== 奖励权重 ====================
reward_weights:
  forward_velocity: 1.5
  trunk_height: 0.5
  orientation: -0.3
  # ... 更多奖励项

# ==================== PPO超参数 ====================
num_epochs: 4
num_minibatches: 8
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
value_coef: 0.5
entropy_coef: 0.01
max_grad_norm: 0.5

# ==================== 训练配置 ====================
total_timesteps: 163840000     # 总训练步数
log_interval: 20
eval_interval: 100
save_interval: 50

# ==================== 优化器配置 ====================
learning_rate: 3.0e-4
final_lr_fraction: 0.02

# ==================== 网络配置 ====================
hidden_dims: [512, 512, 256]   # 网络结构
shared_backbone: true

# ==================== 视频录制配置 ====================
enable_video: true
video_interval: 100
video_frames: 180
video_camera: "track"

# ==================== 阶段切换配置 ====================
stage_config:
  stage_id: 0
  stage_name: "standing"
  min_iterations: 2500
  transition_criteria:
    min_reward: 0.6
    max_fall_rate: 0.05
```

## 命令行参数优先级

```
命令行参数 > YAML配置 > 代码默认值
```

**示例**:
```bash
# train.yaml中设置 num_envs: 1024
# 但命令行指定 --num-envs 512
# 最终使用: 512
python scripts/train.py --config configs/train.yaml --num-envs 512
```

## 参数调优指南

### 快速测试时

**目标**: 最快验证代码

推荐调整:
- `num_envs`: 256-512（最小化）
- `num_steps`: 8-16（极短rollout）
- `total_timesteps`: 128K-8M（快速完成）
- `enable_video`: false（禁用视频）
- `hidden_dims`: [128, 128, 64]（小网络）

### 正式训练时

**目标**: 获得可用模型

推荐设置:
- `num_envs`: 1024（充分并行）
- `num_steps`: 64（标准rollout）
- `total_timesteps`: 按阶段设置（164M-1.31B）
- `enable_video`: true（监控训练）
- `hidden_dims`: [512, 512, 256]（标准网络）

### 资源受限时

**场景**: GPU显存不足

调整策略:
1. 减少 `num_envs`（1024 → 512 → 256）
2. 减少 `hidden_dims`（[512,512,256] → [256,256,128]）
3. 调整 `num_minibatches`（8 → 16，减小batch size）

## 常见问题

### Q: 如何快速验证代码改动？

A: 根据需要选择不同的测试方式:
```bash
# 单阶段最快验证（<2分钟）
make test-quick

# 流水线完整测试（<3分钟，验证阶段切换）
make test-pipeline
```

### Q: 流水线测试和单阶段测试有什么区别？

A:
- **单阶段测试**（test_quick.yaml）：只测试一个训练阶段，验证基本训练逻辑
- **流水线测试**（pipeline_stage*.yaml）：测试完整的3阶段流程（standing → stepping → slow_walk），验证阶段切换逻辑和模型传递

推荐开发时使用流水线测试，确保修改不会破坏阶段切换机制。

### Q: 如何调整阶段训练顺序？

A: 使用train-range命令:
```bash
make train-range FROM=2 TO=4  # 只训练阶段2-4
```

### Q: 如何验证配置文件语法？

A: 使用validate-config命令:
```bash
make validate-config
```

### Q: 配置文件路径错误怎么办？

A: 确保使用正确的路径前缀:
- 正式训练: `configs/train/stage*.yaml`
- 测试配置: `configs/test/test*.yaml`
- 独立配置: `configs/train*.yaml`

## 高级用法

### 自定义奖励权重

编辑配置文件的 `reward_weights` 部分:

```yaml
reward_weights:
  forward_velocity: 2.0    # 提高前向速度权重
  trunk_height: 1.5        # 强调高度保持
  orientation: -1.0        # 加大姿态惩罚
```

### 启用增强奖励

**注意**: 以下奖励需要环境支持额外参数

```yaml
reward_weights:
  # 基础奖励（立即可用）
  stability: 0.5           # ✅ 综合稳定性

  # 增强奖励（需要环境支持，当前注释）
  # gait_periodicity: 0.6    # 需要phase参数
  # swing_trajectory: 0.4    # 需要phase参数
  # landing_impact: 0.2      # 需要landing_events
  # energy_efficiency: 0.1   # 需要joint_velocities
```

### 调整训练节奏

```yaml
log_interval: 10         # 更频繁的日志（vs 默认20）
save_interval: 25        # 更频繁的保存（vs 默认50）
eval_interval: 50        # 更频繁的评估（vs 默认100）
```

## 参考文档

- [分阶段训练指南](../docs/staged_training_guide.md)
- [项目结构](../.std_docs/struct.md)
- [技术细节](../.std_docs/technical_details.md)

---

**最后更新**: 2026-01-04
**维护者**: Jiyuan RL Team
