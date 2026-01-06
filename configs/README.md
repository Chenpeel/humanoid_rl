# 配置文件使用指南

## 核心设计理念 ⭐

**自动课程学习 + YAML 驱动 + 统一配置结构**

从 2026-01-06 起，项目采用**配置驱动**的自动三阶段课程学习框架：
- **阶段1（0-50k steps）**：站立平衡 - 学习保持直立不摔倒
- **阶段2（50k-150k steps）**：低速行走 - 学习基本步态和低速前进
- **阶段3（150k+ steps）**：全速行走 - 跟踪任意速度命令并优化性能

**优势**：
✅ **YAML 定义课程**：通过 `curriculum.yaml` 自由定义奖励权重和环境阶段，无需修改代码。
✅ **统一目录结构**：每个配置集拥有独立的目录，包含 `train.yaml`（超参数）和 `curriculum.yaml`（奖励策略）。
✅ **奖励分量可视化**：Rich 终端和 TensorBoard 会实时显示每个奖励分量的数值（如 `reward/forward_velocity`），极大方便调参。
✅ **训练日志记录阶段信息**：自动记录 `curriculum_stage`、`curriculum_progress` 等指标。

## 目录结构

项目严格遵守 `configs/<任务名>/` 的二级目录规范，每个目录下包含统一命名的配置文件：

```
configs/
├── train/                      # 标准训练配置（主版本）
│   ├── train.yaml              # PPO 训练与环境超参数
│   └── curriculum.yaml         # 三阶段课程学习权重定义
│
├── train-10h/                  # 长时间训练配置
│   └── train.yaml              # 优化显存，适合 1080Ti/4090 长时间运行
│
├── quick_test/                 # 快速测试配置
│   ├── train.yaml              # 小规模、短步数验证
│   └── curriculum.yaml         # 单阶段（站立阶段）快速验证权重
│
├── examples/                   # 示例与旧版本兼容配置
│   ├── velocity_legacy/        # 旧版速度跟踪（无课程）
│   └── walking_legacy/         # 旧版行走任务
│
└── README.md                   # 本文档
```

## 配置类型说明

### 1. 标准训练（train/）

**适用场景**：日常训练，平衡性能和资源消耗。

**使用方式**：
```bash
make train
# 或者手动指定：
python scripts/train.py --config configs/train/train.yaml
```

### 2. 长时间训练（train-10h/）

**适用场景**：高性能训练，追求最佳收敛效果（通常运行 10-24 小时）。

**使用方式**：
```bash
make train-long
```

### 3. 快速测试（quick_test/）

**适用场景**：代码逻辑调试、新奖励函数快速验证。

**使用方式**：
```bash
make train-test
```

## 课程学习机制

### 配置文件加载

在 `train.yaml` 中，通过以下字段指定课程定义：
```yaml
curriculum_file: configs/train/curriculum.yaml
```
- 如果指向的文件包含 `stages:` 列表，系统将运行**多阶段课程**。
- 如果仅包含扁平的权重字典，系统将运行**单阶段无限时训练**。

### 实时奖励分析 (New!)

训练过程中，Rich 终端将显示详细的奖励分量：
```
reward/forward_velocity: 0.85
reward/upright_bonus: 0.92
reward/action_rate: -0.01
```
这些分量也会以 `train/reward/xxx` 为路径同步记录到 TensorBoard。

## 参数说明

### 核心配置参数 (`train.yaml`)

| 参数 | 说明 |
|-----|------|
| `curriculum_file` | 指向该配置对应的奖励权重定义文件 |
| `env_config` | 覆盖课程阶段中的默认环境参数（如 `target_height`） |
| `num_envs` | 并行环境数（建议：1080Ti 256-1024，4090 2048-4096） |
| `total_timesteps` | 总训练步数 |

## 常见问题

**Q: 如何自定义奖励权重而不影响标准配置？**

A: 推荐在 `configs/quick_test/curriculum.yaml` 中修改权重，然后运行 `make train-test`。验证有效后再同步到 `train/curriculum.yaml`。

**Q: 为什么终端看不到奖励分量？**

A: 确保在 `logger.py` 的白名单中包含了 `reward/` 前缀（系统默认已包含）。

**Q: 课程学习是如何实现自动切换的？**

A: `src/rl/curriculum/base.py` 中的 `ConfigurableCurriculum` 会根据 `train_state.env_steps` 自动计算当前所属阶段，并实时更新环境的 `reward_weights`。

---
*更新日期：2026-01-06*