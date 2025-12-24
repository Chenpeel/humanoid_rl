# 双足机器人强化学习训练框架

基于 **Isaac Lab** + **PyTorch** + **RSL_RL** 的双足机器人强化学习训练系统。

## 项目概览

本项目为机器人提供完整的强化学习训练解决方案，从仿真训练到真机部署的全流程支持。

### 核心特性

- ✅ **Isaac Lab 仿真环境** - 基于 NVIDIA Isaac Sim 的高性能物理仿真
- ✅ **RSL_RL PPO 训练器** - 成熟的强化学习算法实现
- ✅ **MJCF 模型支持** - 直接加载 MuJoCo XML 格式的机器人模型
- ✅ **Sim2Real 映射** - 仿真到真机的动作映射和校准
- ✅ **模仿学习支持** - 支持从专家演示数据学习
- ✅ **实时可视化** - Omniverse Viewer 实时3D可视化
- ✅ **并行环境训练** - 支持数千个并行环境加速训练

### 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 仿真引擎 | NVIDIA Isaac Sim | 2024.1.1+ |
| 环境框架 | Isaac Lab | v0.2.0+ |
| 深度学习 | PyTorch | 2.0+ |
| 强化学习 | RSL_RL | v2.2.1+ |
| 机器人模型 | MJCF (MuJoCo) | - |

## 快速开始

### 1. 环境要求

**硬件：**
- GPU: NVIDIA RTX 3060 及以上 (推荐 RTX 4070+)
- VRAM: 至少 8GB (推荐 12GB+)
- RAM: 至少 16GB
- 存储: 至少 50GB 可用空间

**软件：**
- Ubuntu 20.04 / 22.04
- NVIDIA Driver 525+
- CUDA 11.8 / 12.1

### 2. 安装步骤

#### 方法 A: 使用 Makefile（推荐）

```bash
# 1. 克隆仓库（包含子模块）
git clone --recursive https://github.com/Chenpeel/rl.git
cd rl

# 2. 更新子模块
cd isaaclab_rl
make submodule-update

# 3. 安装 Isaac Lab
cd ../dep/IsaacLab
./isaaclab.sh --install

# 4. 安装项目
cd ../../isaaclab_rl
make install

# 5. 验证安装
make verify
```

#### 方法 B: 手动安装

详见 [完整安装指南](docs/GETTING_STARTED.md)

### 3. 快速测试训练

```bash
cd isaaclab_rl

# 测试训练（64 个环境，10 次迭代）
make train-test

# 完整训练（4096 个环境）
make train
```

## 项目结构

```
jiyuan-rl/
├── assets/                      # 机器人资源文件
│   └── xmls/models/jiyuan/     # MJCF 模型文件
├── dep/                         # 依赖子模块
│   ├── IsaacLab/               # Isaac Lab 框架
│   └── rsl_rl/                 # RSL_RL 训练器
├── isaaclab_rl/                # 主项目目录
│   ├── jiyuan_tasks/           # 任务定义
│   │   ├── envs/cfg/          # 环境配置
│   │   ├── managers/          # 奖励/终止/命令管理器
│   │   └── utils/             # 工具函数
│   ├── agents/                 # 训练器配置
│   ├── scripts/                # 训练和评估脚本
│   ├── configs/                # 配置文件
│   └── docs/                   # 文档
├── scripts/                     # 通用脚本
└── docs/                        # 项目文档
```

## 文档导航

### 入门文档

- [从零开始完整指南](docs/GETTING_STARTED.md) 
- [使用指南](docs/USAGE.md) - 日常使用和开发
- [高级功能](docs/ADVANCED.md) - Sim2Real和模仿学习

### 文档索引

- [文档中心](docs/README.md) - 所有文档和阅读路径

## 使用示例

### 训练站立任务

```bash
cd isaaclab_rl

# 使用默认配置训练
./dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --num_envs 4096 \
    --headless

# 使用自定义配置
./dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --config configs/train_config.yaml
```

### 训练速度跟踪任务

```bash
./dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task velocity \
    --num_envs 4096 \
    --headless
```

### 评估训练好的策略

```bash
./dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task velocity \
    --checkpoint logs/rsl_rl/velocity/model_10000.pt \
    --num_envs 1
```

## 可用任务

| 任务名称 | 环境ID | 描述 |
|---------|--------|------|
| 测试环境 | `Isaac-Jiyuan-Test-v0` | 最小化测试环境 |
| 站立任务 | `Isaac-Jiyuan-Standing-v0` | 保持站立姿态 |
| 速度跟踪 | `Isaac-Jiyuan-Velocity-v0` | 跟踪指令速度 |

## 常见问题

### Q: 训练时 GPU 利用率低？

**A:** 增加并行环境数量：

```bash
# 从 4096 增加到 8192
./dep/IsaacLab/isaaclab.sh -p scripts/train.py --task velocity --num_envs 8192
```

### Q: MJCF 模型加载失败？

**A:** 检查文件路径：

```bash
# 验证 MJCF 文件存在
ls -la assets/xmls/models/jiyuan/index.xml

# 检查配置中的路径是否正确
grep -r "asset_path" isaaclab_rl/jiyuan_tasks/envs/cfg/
```

### Q: 如何在云端 GPU 运行？

**A:** 参考 [从零开始完整指南](docs/GETTING_STARTED.md) 的云端部署章节。

## 性能参考

在 NVIDIA RTX 4070 (12GB VRAM) 上：

- **训练速度**: ~50,000 steps/s (4096 envs)
- **GPU 利用率**: 85-95%
- **训练时间**: 约 4-6 小时达到 30M steps

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: 添加某个特性'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 版本历史

### v0.3.0 (当前版本)

- ✅ 完整迁移到 Isaac Lab + PyTorch + RSL_RL
- ✅ 支持 MJCF 模型加载
- ✅ 实现站立和速度跟踪任务
- ✅ 添加 Sim2Real 映射和模仿学习支持
- ✅ 完善文档和安装脚本

### v0.2.0 (已弃用)

- 基于 JAX/MJX 的实现（已迁移到 `jax` 分支）

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - 仿真环境框架
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl) - 强化学习算法库
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) - 物理仿真引擎

## 联系方式

- **作者**: Chenpeel
- **邮箱**: chenpeel@foxmail.com
- **问题反馈**: [GitHub Issues](https://github.com/yourusername/jiyuan-rl/issues)

---

⚡ 使用 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 驱动 | 🚀 基于 [RSL_RL](https://github.com/leggedrobotics/rsl_rl) 训练
