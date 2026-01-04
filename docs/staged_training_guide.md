# 分阶段训练指南

## 概述

本项目实现了四足机器人行走的分阶段训练方案，采用课程学习（Curriculum Learning）策略，从简单的站立平衡开始，逐步过渡到复杂的地形适应。

## 阶段划分

| 阶段 | 名称 | Iterations | 训练步数 | 时间估算 | 切换条件 |
|-----|------|-----------|---------|---------|---------|
| 0 | 站立平衡 | 0-200 | 0-209M | 10分钟 | mean_reward > 0.6, 摔倒率 < 5% |
| 1 | 原地踏步 | 200-500 | 209M-524M | 15分钟 | 步态对称性 > 0.5 |
| 2 | 小步行走 | 500-1000 | 524M-1.05B | 25分钟 | 速度跟踪 > 70% |
| 3 | 正常行走 | 1000-2000 | 1.05B-2.1B | 50分钟 | 速度跟踪 > 80% |
| 4 | 高速适应 | 2000-3000 | 2.1B-3.1B | 50分钟 | 最大速度 > 0.9 m/s |
| 5 | 地形适应 | 3000-5000 | 3.1B-5.2B | 1.7小时 | 粗糙地形摔倒率 < 15% |
| 6 | 鲁棒性提升 | 5000-10000 | 5.2B-10.5B | 4.2小时 | 综合性能达标 |

**总训练时间（完整流程）**: 约9小时

## 使用 Makefile 管理

### 基本命令

```bash
# 显示帮助
make help

# 安装依赖
make install

# 检查环境
make check-env
```

### 训练命令

```bash
# 快速开始（从阶段0开始）
make train

# 训练指定阶段
make train-stage STAGE=0  # 站立平衡
make train-stage STAGE=1  # 原地踏步
make train-stage STAGE=2  # 小步行走
make train-stage STAGE=3  # 正常行走
make train-stage STAGE=4  # 高速适应
make train-stage STAGE=5  # 地形适应
make train-stage STAGE=6  # 鲁棒性提升

# 从指定阶段开始连续训练
make train-from FROM_STAGE=0

# 自动化训练所有阶段（基于切换条件）
make train-all
```

### 评估和监控

```bash
# 评估模型
make eval CKPT=logs/stage0_*/checkpoints/best_model

# 启动TensorBoard
make tensorboard

# 检查模型
make inspect CKPT=logs/stage0_*/checkpoints/best_model
```

### 清理命令

```bash
make clean         # 清理日志和临时文件
make clean-cache   # 清理JAX缓存
make clean-logs    # 清理所有日志
make clean-all     # 清理所有
```

## 配置文件

每个阶段都有独立的配置文件：

| 阶段 | 配置文件 |
|-----|---------|
| 0 | `configs/stage0_standing.yaml` |
| 1 | `configs/stage1_stepping.yaml` |
| 2 | `configs/stage2_slow_walk.yaml` |
| 3 | `configs/stage3_normal_walk.yaml` |
| 4 | `configs/stage4_fast_walk.yaml` |
| 5 | `configs/stage5_terrain.yaml` |
| 6 | `configs/stage6_robustness.yaml` |

### 配置文件结构

```yaml
# 场景配置
scene: jiyuan_fit_flat  # 地形场景
env_type: walking       # 环境类型

# 环境配置
num_envs: 16384
num_steps: 64
max_episode_steps: 500

# 任务参数（课程学习核心）
cmd_x_range: [0.0, 0.0]      # 前向速度范围
cmd_y_range: [0.0, 0.0]      # 侧向速度范围
cmd_yaw_range: [0.0, 0.0]    # 角速度范围

# 奖励权重
reward_weights:
  forward_velocity: 0.0
  gait_symmetry: 0.0
  foot_clearance: 0.0
  trunk_height: 2.0
  # ...

# 阶段切换配置
stage_config:
  stage_id: 0
  stage_name: "standing"
  min_iterations: 150
  transition_criteria:
    min_reward: 0.6
    max_fall_rate: 0.05
```

## 直接使用 Python 脚本

```bash
# 自动化分阶段训练（从阶段0开始）
python scripts/train_staged.py

# 从指定阶段开始
python scripts/train_staged.py --start-stage 2

# 训练指定范围
python scripts/train_staged.py --start-stage 0 --end-stage 3

# 从检查点恢复
python scripts/train_staged.py --resume-checkpoint logs/stage0_*/checkpoints/best_model
```

## 阶段详解

### 阶段0：站立平衡

**目标**：学会保持站立，抵抗扰动

**关键参数**：
- 速度命令：0 m/s（纯站立）
- 地形：平坦
- Episode长度：500步

**奖励权重调整**：
- `trunk_height: 2.0` - 加倍高度保持奖励
- `orientation: -1.0` - 加强姿态惩罚
- `forward_velocity: 0.0` - 关闭速度奖励

**预期行为**：机器人学会站立不动，偶尔有小幅度的动作调整以保持平衡

---

### 阶段1：原地踏步

**目标**：在保持平衡的基础上，开始有节奏的踏步动作

**关键参数**：
- 速度命令：微小移动
- 地形：平坦

**奖励权重调整**：
- `gait_symmetry: 0.5` - 开启步态对称性
- `foot_clearance: 0.3` - 开启脚部抬高

**预期行为**：机器人开始原地踏步，形成左右脚交替的步态

---

### 阶段2：小步行走

**目标**：缓慢向前移动，建立速度跟踪意识

**关键参数**：
- 前向速度：0.0-0.3 m/s
- Episode长度：1000步

**奖励权重调整**：
- `forward_velocity: 1.0` - 开启前向速度奖励

**预期行为**：机器人开始缓慢向前移动，步态保持稳定

---

### 阶段3：正常行走

**目标**：中等速度行走，稳定步态

**关键参数**：
- 前向速度：-0.2 到 0.8 m/s
- Episode长度：2000步

**预期行为**：机器人能够稳定行走，速度变化流畅

---

### 阶段4：高速适应

**目标**：高速行走，扩展速度范围

**关键参数**：
- 前向速度：-0.5 到 1.2 m/s
- `forward_velocity: 2.0` - 加强速度奖励

**预期行为**：机器人能够快速行走，高速下保持稳定

---

### 阶段5：地形适应

**目标**：适应不平地形

**关键参数**：
- 地形：`jiyuan_fit_rough`（关键变化）
- 前向速度：-0.3 到 0.8 m/s

**奖励权重调整**：
- `foot_clearance: 0.4` - 加强脚部抬高
- `trunk_height: 0.8` - 加强高度保持
- `forward_velocity: 1.2` - 降低速度要求，优先稳定性

**预期行为**：机器人适应不平地形，脚部抬高避开障碍

---

### 阶段6：鲁棒性提升

**目标**：全面提升性能和鲁棒性

**关键参数**：
- 完整速度范围
- 粗糙地形
- 学习率降低：`2.0e-4`

**预期行为**：在各种条件下稳定，抵抗外部扰动

## 监控指标

### 关键指标

| 指标 | 说明 |
|-----|------|
| `mean_reward` | 平均奖励 |
| `episode_length` | 平均episode长度 |
| `fall_rate` | 摔倒率 |
| `velocity_tracking_accuracy` | 速度跟踪准确率 |
| `gait_symmetry_reward` | 步态对称性奖励 |
| `foot_clearance_reward` | 脚部抬高奖励 |

### TensorBoard

```bash
make tensorboard
# 访问 http://localhost:6006
```

## 故障排除

### 常见问题

1. **训练不收敛**
   - 检查学习率是否过高
   - 检查奖励权重是否合理
   - 查看TensorBoard曲线

2. **摔倒率过高**
   - 检查终止条件是否过于严格
   - 调整 `orientation` 和 `trunk_height` 惩罚

3. **速度跟踪不准确**
   - 增加 `forward_velocity` 权重
   - 检查命令范围是否合理

4. **显存不足**
   - 减少 `num_envs`
   - 减少 `num_steps`
   - 减少 `hidden_dims`

## 参考

- 原始训练脚本：`scripts/train.py`
- 分阶段训练脚本：`scripts/train_staged.py`
- 环境定义：`src/rl/envs/robot_envs.py`
- 奖励函数：`src/rl/rewards/walking_rewards.py`
