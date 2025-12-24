# 使用指南

本文档介绍项目的日常使用方法，包括任务训练、评估和开发指南。

## 项目状态

**当前版本**: v0.3.0
**最后更新**: 2024-12-24
**状态**: ✅ 已合并到 master，待云端测试

### 已实现功能

- ✅ 完整的 Isaac Lab + PyTorch + RSL_RL 集成
- ✅ 3个任务环境（测试、站立、速度跟踪）
- ✅ 16个奖励函数 + 14个终止条件
- ✅ 训练和评估脚本
- ✅ Sim2Real映射和模仿学习支持
- ✅ Makefile自动化命令

## 项目结构

```
rl/
├── assets/xmls/models/jiyuan/  # 机器人MJCF模型
├── dep/                         # 子模块依赖
│   ├── IsaacLab/               # Isaac Lab框架
│   └── rsl_rl/                 # RSL_RL训练器
├── isaaclab_rl/                # 主项目
│   ├── jiyuan_tasks/           # 任务定义
│   │   ├── envs/cfg/          # 环境配置
│   │   ├── managers/          # 奖励/终止/命令管理器
│   │   └── utils/             # 工具函数
│   ├── agents/rsl_rl/         # PPO配置
│   ├── scripts/               # 训练/评估脚本
│   ├── configs/               # 配置文件
│   └── Makefile               # 自动化命令
└── docs/                       # 文档
```

## 可用任务

### 1. test - 测试环境

最小化环境，用于验证基础功能。

```bash
# 快速测试
make train-test

# 或手动运行
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task test \
    --num_envs 64 \
    --max_iterations 10 \
    --headless
```

### 2. standing - 站立任务

训练机器人保持站立姿态。

```bash
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --num_envs 4096 \
    --headless
```

**目标**: 保持目标高度(0.35m)和直立姿态
**奖励**: 高度、姿态、关节限制、动作平滑性等

### 3. velocity - 速度跟踪

训练机器人跟踪速度指令。

```bash
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task velocity \
    --num_envs 4096 \
    --headless
```

**目标**: 跟踪x/y方向速度和yaw角速度
**命令**: 随机生成速度命令（范围可配置）

## 训练命令

### 基础训练

```bash
cd isaaclab_rl

# 使用默认配置
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing

# 自定义环境数量
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --num_envs 8192

# 无头模式（云端必需）
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --headless

# 指定最大迭代次数
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --max_iterations 5000
```

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称（test/standing/velocity） | standing |
| `--num_envs` | 并行环境数量 | 4096 |
| `--headless` | 无头模式（不显示GUI） | False |
| `--max_iterations` | 最大迭代次数 | 无限 |
| `--log_dir` | 日志目录 | logs/rsl_rl/{task} |
| `--seed` | 随机种子 | 42 |

### 使用配置文件

```bash
# 自定义训练配置
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task velocity \
    --config configs/train_config.yaml
```

### 恢复训练

```bash
# 从checkpoint恢复
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --resume \
    --load_run logs/rsl_rl/standing/2024-12-24_10-30-00
```

## 评估策略

### 加载Checkpoint

```bash
# 评估最新模型
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task standing \
    --checkpoint logs/rsl_rl/standing/model_1000.pt \
    --num_envs 1
```

### 录制视频

```bash
# 录制200帧视频
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task standing \
    --checkpoint logs/rsl_rl/standing/model_1000.pt \
    --video \
    --video_length 200
```

视频保存在 `videos/` 目录。

### 批量评估

```bash
# 评估所有保存的模型
for ckpt in logs/rsl_rl/standing/model_*.pt; do
    ../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
        --task standing \
        --checkpoint $ckpt \
        --num_envs 16 \
        --headless
done
```

## Makefile命令

### 子模块管理

```bash
make submodule-update        # 更新子模块（推荐）
make submodule-status        # 查看子模块状态
```

### 环境检查

```bash
make check-isaaclab          # 检查Isaac Lab环境
make check-env               # 检查所有环境
```

### 安装

```bash
make install                 # 安装项目（开发模式）
make install-dev             # 安装项目 + 开发工具
make verify                  # 验证安装
```

### 训练

```bash
make train                   # 开始训练（使用默认配置）
make train-test              # 快速测试训练（64 envs, 10 iters）
```

### 清理

```bash
make clean                   # 清理构建文件
make clean-logs              # 清理日志文件
```

## 开发指南

### 添加新任务环境

**1. 创建环境配置**

```python
# isaaclab_rl/jiyuan_tasks/envs/cfg/my_task_cfg.py
from isaaclab.envs import ManagerBasedRLEnvCfg
from .jiyuan_scene_cfg import JiyuanSceneCfg
from isaaclab.utils import configclass

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    # 复用场景配置
    scene: JiyuanSceneCfg = JiyuanSceneCfg()

    # 自定义观测、奖励、终止条件
    observations: ObservationsCfg = ...
    rewards: RewardsCfg = ...
    terminations: TerminationsCfg = ...
```

**2. 注册环境**

```python
# isaaclab_rl/jiyuan_tasks/__init__.py
import gymnasium as gym

gym.register(
    id="Isaac-Jiyuan-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MyTaskEnvCfg},
)

# 添加到任务列表
TASK_NAMES.append("mytask")
```

**3. 测试新环境**

```bash
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task mytask \
    --num_envs 64 \
    --max_iterations 10
```

### 添加新奖励函数

**1. 定义奖励函数**

```python
# isaaclab_rl/jiyuan_tasks/managers/rewards.py
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def my_custom_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """自定义奖励函数

    Args:
        env: 环境实例
        asset_cfg: 资产配置（默认为robot）

    Returns:
        奖励张量，形状 (num_envs,)
    """
    robot = env.scene[asset_cfg.name]

    # 计算奖励（示例：基于基座高度）
    base_height = robot.data.root_pos_w[:, 2]
    target_height = 0.35
    reward = -torch.abs(base_height - target_height)

    return reward
```

**2. 在环境配置中使用**

```python
# isaaclab_rl/jiyuan_tasks/envs/cfg/my_task_cfg.py
from isaaclab.managers import RewardTermCfg as RewTerm
from ..managers import rewards as mdp

@configclass
class RewardsCfg:
    # 使用自定义奖励
    my_custom = RewTerm(func=mdp.my_custom_reward, weight=1.0)

    # 可以组合多个奖励
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
```

### 修改环境参数

编辑对应的环境配置文件：

```python
# isaaclab_rl/jiyuan_tasks/envs/cfg/standing_env_cfg.py

@configclass
class StandingEnvCfg(ManagerBasedRLEnvCfg):
    # 修改仿真参数
    decimation = 4  # 控制频率 = sim_freq / decimation
    episode_length_s = 20.0  # 修改回合长度

    # 修改奖励权重
    @configclass
    class RewardsCfg:
        height = RewTerm(func=mdp.height_reward, weight=2.0)  # 增加权重
```

## 故障排查

### 问题1: GPU显存不足

```bash
# 减少并行环境数量
--num_envs 2048

# 或修改网络大小（agents/rsl_rl/ppo_cfg.py）
```

### 问题2: 训练速度慢

```bash
# 增加并行环境
--num_envs 8192

# 检查GPU利用率
watch -n 1 nvidia-smi
```

### 问题3: 奖励不收敛

```bash
# 1. 降低学习率（编辑 configs/train_config.yaml）
ppo:
  algorithm:
    learning_rate: 0.0001  # 从0.0003降低

# 2. 增加训练时间
--max_iterations 5000

# 3. 检查奖励权重是否合理
```

### 问题4: MJCF模型加载失败

```bash
# 检查文件路径
ls -la assets/xmls/models/jiyuan/index.xml

# 检查配置
grep "ISAAC_LAB_RL_ROOT" isaaclab_rl/jiyuan_tasks/envs/cfg/jiyuan_scene_cfg.py
```

## 更多资源

- [从零开始完整指南](GETTING_STARTED.md) - 详细安装和训练教程
- [高级功能指南](ADVANCED.md) - Sim2Real和模仿学习
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL_RL 文档](https://github.com/leggedrobotics/rsl_rl)

---

**维护**: Chenpeel (chenpeel@foxmail.com)
**版本**: v0.3.0
