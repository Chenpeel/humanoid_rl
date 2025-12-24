# Isaac Lab RL 双足机器人

使用 Isaac Lab + PyTorch + RSL_RL 训练 双足机器人的完整实现。

## 🎯 项目状态

**当前版本**: v0.3.0
**最后更新**: 2024-12-24
**项目状态**: ✅ 已合并到 master，待云端测试

### 已完成功能

- ✅ 完整的项目结构和包配置
- ✅ Isaac Lab 子模块集成 (dep/IsaacLab)
- ✅ RSL_RL 子模块集成 (dep/rsl_rl)
- ✅ Jiyuan 机器人 MJCF 模型加载（绝对路径）
- ✅ 3个任务环境实现（测试、站立、速度跟踪）
- ✅ 完整的奖励函数管理器（16个奖励函数）
- ✅ 完整的终止条件管理器（14个终止函数）
- ✅ 命令生成器（速度命令）
- ✅ 训练和评估脚本
- ✅ RSL_RL PPO 配置
- ✅ Sim2Real 映射和模仿学习支持
- ✅ 完善的 Makefile 和安装脚本
- ✅ 详细的文档和迁移指南

### 待验证功能（需云端 GPU 测试）

- ⏸️ 环境创建和运行验证
- ⏸️ 奖励函数 API 兼容性
- ⏸️ 完整训练流程（1000+ iterations）
- ⏸️ 性能基准测试（GPU 利用率、训练速度）
- ⏸️ 长时间训练稳定性（30M steps）

## 📁 项目结构

```
isaaclab_rl/
├── jiyuan_tasks/              # Jiyuan 机器人任务
│   ├── envs/                  # 环境定义
│   │   └── cfg/
│   │       ├── jiyuan_scene_cfg.py          # ✅ 场景和机器人配置
│   │       ├── jiyuan_test_env_cfg.py       # ✅ 测试环境
│   │       ├── standing_env_cfg.py          # ✅ 站立任务
│   │       └── velocity_tracking_env_cfg.py # ✅ 速度跟踪任务
│   ├── managers/              # MDP 管理器
│   │   ├── rewards.py         # ✅ 16个奖励函数
│   │   ├── terminations.py    # ✅ 14个终止条件
│   │   ├── commands.py        # ✅ 命令生成器
│   │   └── walking_rewards.py # ✅ 行走任务奖励
│   └── utils/                 # 工具函数
│       ├── math_utils.py      # ✅ 数学工具（四元数、欧拉角）
│       ├── sim2real.py        # ✅ Sim2Real 映射
│       ├── imitation.py       # ✅ 模仿学习
│       └── config_loader.py   # ✅ 配置加载器
├── agents/                    # RL 算法配置
│   └── rsl_rl/
│       └── ppo_cfg.py         # ✅ RSL_RL PPO 配置
├── scripts/                   # 脚本
│   ├── train.py               # ✅ 训练脚本（560行）
│   ├── play.py                # ✅ 评估脚本（374行）
│   └── install.sh             # ✅ 安装脚本
├── configs/                   # 配置文件
│   ├── train_config.yaml      # ✅ 训练配置
│   ├── robot_config.yaml      # ✅ 机器人配置
│   └── servo_config.yaml      # ✅ 舵机配置
├── docs/                      # 文档
│   ├── GETTING_STARTED.md     # 📝 从零开始完整指南（新增）
│   ├── MINIMAL_SETUP_GUIDE.md # ✅ 最小化安装指南
│   ├── CONDA_INSTALLATION.md  # ✅ Conda 安装指南
│   └── SIM2REAL_AND_ADVANCED_FEATURES.md # ✅ 高级特性
├── Makefile                   # ✅ Make 命令（子模块管理、训练等）
├── setup.py                   # ✅ 安装脚本
├── pyproject.toml             # ✅ 项目配置
└── README.md                  # 本文件
```

## 🚀 快速开始

### 方法 A: 使用 Makefile（推荐）

```bash
# 1. 确保在项目目录
cd isaaclab_rl

# 2. 更新子模块
make submodule-update

# 3. 检查 Isaac Lab 环境
make check-isaaclab

# 4. 安装 Isaac Lab（首次）
cd ../dep/IsaacLab
./isaaclab.sh --install
cd ../../isaaclab_rl

# 5. 安装项目
make install

# 6. 验证安装
make verify

# 7. 快速测试训练
make train-test
```

### 方法 B: 手动安装

详见 [从零开始完整指南](docs/GETTING_STARTED.md)

## 🎮 可用任务环境

### 1. Isaac-Jiyuan-Test-v0（测试环境）

**用途**: 最小可行环境，验证基础功能

```bash
# 测试命令
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task test --num_envs 64 --max_iterations 10
```

**特性**:
- 观测维度: 65 (base状态 + 关节状态 + 上一步动作)
- 动作维度: 16 (16个关节)
- 简单奖励: 存活 + 动作平滑性
- 用于快速验证环境加载和训练流程

### 2. Isaac-Jiyuan-Standing-v0（站立任务）

**用途**: 训练机器人保持站立姿态

```bash
# 训练命令
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --num_envs 4096 --headless
```

**特性**:
- 观测维度: 扩展观测（包含高度、姿态等）
- 目标: 保持目标高度 (0.35m) 和直立姿态
- 奖励: 高度、姿态、关节限制、动作平滑性等
- 终止: 摔倒、超时

### 3. Isaac-Jiyuan-Velocity-v0（速度跟踪任务）

**用途**: 训练机器人跟踪速度指令

```bash
# 训练命令
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task velocity --num_envs 4096 --headless
```

**特性**:
- 观测维度: 包含速度命令
- 目标: 跟踪 x/y 方向的速度和 yaw 角速度
- 命令: 随机生成速度命令（范围可配置）
- 奖励: 速度跟踪、姿态、能量消耗等

## 📊 训练示例

### 基础训练

```bash
# 使用默认配置训练站立任务
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing

# 自定义环境数量
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --num_envs 8192

# 无头模式（不显示GUI）
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --headless

# 指定日志目录
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --log_dir my_logs
```

### 评估训练好的策略

```bash
# 加载 checkpoint 评估
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task velocity \
    --checkpoint logs/rsl_rl/velocity/model_10000.pt \
    --num_envs 1

# 录制视频
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task velocity \
    --checkpoint logs/rsl_rl/velocity/model_10000.pt \
    --video \
    --video_length 200
```

### 使用配置文件训练

```bash
# 使用自定义训练配置
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task velocity \
    --config configs/train_config.yaml
```

## 🛠️ Makefile 命令参考

### 子模块管理

```bash
make submodule-init          # 初始化子模块
make submodule-update        # 更新子模块（推荐）
make submodule-status        # 查看子模块状态
make submodule-update-remote # 更新子模块到最新版本
```

### 环境检查

```bash
make check-isaaclab          # 检查 Isaac Lab 环境
make check-env               # 检查所有环境
```

### 安装命令

```bash
make install                 # 安装项目（开发模式）
make install-dev             # 安装项目 + 开发工具
make install-vis             # 安装项目 + 可视化工具
make install-all             # 安装所有依赖
```

### 训练命令

```bash
make train                   # 开始训练（使用默认配置）
make train-test              # 快速测试训练（64 envs, 10 iters）
```

### 验证和清理

```bash
make verify                  # 验证安装是否成功
make clean                   # 清理构建文件
make clean-logs              # 清理日志文件
```

### 开发工具（需安装 dev 依赖）

```bash
make format                  # 格式化代码
make test                    # 运行测试
```

## 📚 文档导航

### 入门文档

- **[从零开始完整指南](docs/GETTING_STARTED.md)** ⭐⭐⭐⭐⭐ - 完整的安装、测试、训练流程
- [最小化安装指南](docs/MINIMAL_SETUP_GUIDE.md) ⭐⭐⭐⭐ - 快速上手
- [Conda 安装指南](docs/CONDA_INSTALLATION.md) ⭐⭐⭐ - 使用 Conda 管理环境

### 使用文档

- [文档索引](docs/README.md) - 所有文档列表
- [Sim2Real 和高级特性](docs/SIM2REAL_AND_ADVANCED_FEATURES.md) - 真机部署和模仿学习

### 迁移文档（仅供参考）

- [迁移概览](../.migrate/README.md) - JAX 到 Isaac Lab 迁移记录
- [环境搭建](../.migrate/01-setup.md) - Isaac Lab 详细安装步骤
- [架构设计](../.migrate/02-architecture.md) - 系统架构说明
- [代码映射](../.migrate/03-code-mapping.md) - JAX 到 PyTorch 代码对照

## 🔧 开发指南

### 添加新任务环境

1. **创建环境配置**:

```python
# isaaclab_rl/jiyuan_tasks/envs/cfg/my_task_cfg.py
from isaaclab.envs import ManagerBasedRLEnvCfg
from .jiyuan_scene_cfg import JiyuanSceneCfg

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    # 复用场景配置
    scene: JiyuanSceneCfg = JiyuanSceneCfg()

    # 自定义观测、奖励、终止条件
    observations: ObservationsCfg = ...
    rewards: RewardsCfg = ...
    terminations: TerminationsCfg = ...
```

2. **注册环境**:

```python
# isaaclab_rl/jiyuan_tasks/__init__.py
gym.register(
    id="Isaac-Jiyuan-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MyTaskEnvCfg},
)
```

3. **测试新环境**:

```bash
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task mytask --num_envs 64 --max_iterations 10
```

### 添加新奖励函数

在 `jiyuan_tasks/managers/rewards.py` 中添加：

```python
def my_custom_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """自定义奖励函数

    Args:
        env: 环境实例
        asset_cfg: 资产配置（通常是机器人）

    Returns:
        奖励张量，形状 (num_envs,)
    """
    # 获取机器人数据
    robot = env.scene[asset_cfg.name]

    # 计算奖励
    reward = ...

    return reward
```

然后在环境配置中使用：

```python
@configclass
class RewardsCfg:
    my_custom = RewTerm(func=mdp.my_custom_reward, weight=1.0)
```

## 🛠️ 故障排查

### 常见问题

#### Q1: 导入 isaaclab_rl 时报错

```bash
# 确保安装了项目
cd isaaclab_rl
make install

# 或手动安装
pip install -e .
```

#### Q2: Isaac Lab 子模块不存在

```bash
# 更新子模块
cd isaaclab_rl
make submodule-update

# 或手动更新
git submodule update --init --recursive
```

#### Q3: MJCF 模型加载失败

```bash
# 检查文件是否存在
ls -la ../assets/xmls/models/jiyuan/index.xml

# 检查路径配置
grep -r "ISAAC_LAB_RL_ROOT" jiyuan_tasks/envs/cfg/jiyuan_scene_cfg.py
```

#### Q4: 训练时 GPU 利用率低

增加并行环境数量：

```bash
# 从 4096 增加到 8192
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task velocity --num_envs 8192
```

#### Q5: 云端 GPU 部署

参考 [从零开始完整指南](docs/GETTING_STARTED.md) 的云端部署章节

### 更多帮助

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL_RL 文档](https://github.com/leggedrobotics/rsl_rl)
- [项目 Issues](https://github.com/yourusername/jiyuan-rl/issues)

## 🎯 下一步计划

### 立即进行（云端测试）

- [ ] 在 autodl GPU 环境中验证环境创建
- [ ] 运行 1000+ iterations 训练测试
- [ ] 验证奖励函数 API 兼容性
- [ ] 性能基准测试（GPU 利用率、训练速度）

### 短期计划（1-2周）

- [ ] 修复云端测试发现的问题
- [ ] 完成 30M steps 长时间训练
- [ ] 录制训练视频
- [ ] 性能对比报告（vs JAX/MJX）

### 中期计划（1-2月）

- [ ] 添加更多任务（行走、跳跃等）
- [ ] 域随机化增强
- [ ] 真机部署测试
- [ ] 模仿学习验证

### 长期计划（3+月）

- [ ] 多机器人支持架构重构
- [ ] 课程学习
- [ ] 分层控制
- [ ] 真机性能优化

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 🙏 致谢

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - NVIDIA 的强化学习框架
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl) - ETH Zurich 的 PPO 实现
- [legged-loco](https://github.com/yang-zj1026/legged-loco) - 参考架构

## 📞 联系方式

- **作者**: Chenpeel
- **邮箱**: chenpeel@foxmail.com
- **问题反馈**: GitHub Issues

---

**文档版本**: v0.3.0
**最后更新**: 2024-12-24
**项目状态**: ✅ 已合并到 master，待云端测试
