# 环境搭建和依赖安装

本文档详细说明如何安装和配置 Isaac Lab 环境，为迁移工作做好准备。

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 | 当前配置 |
|------|---------|---------|---------|
| **GPU** | NVIDIA GTX 1080 Ti (11GB) | NVIDIA A100 (40GB+) | GTX 1080 Ti 11GB |
| **CPU** | 8 核 | 16 核+ | - |
| **内存** | 16 GB | 32 GB+ | - |
| **硬盘** | 100 GB 可用空间 | 200 GB+ SSD | - |
| **操作系统** | Linux 6.14.0+ | Ubuntu 22.04 LTS | Linux 6.14.0-37-generic |

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **NVIDIA 驱动** | >= 535.104.05 | 支持 CUDA 12.1+ |
| **CUDA** | 12.1+ | Isaac Sim 要求 |
| **Python** | 3.10 | Isaac Lab 要求（不支持 3.11） |
| **Conda** | 最新版 | 用于环境隔离 |

## 安装步骤

### 第 1 步：安装 Isaac Sim

#### 方法 1：通过 Omniverse Launcher（推荐）

1. **下载 Omniverse Launcher**
```bash
# 访问 https://www.nvidia.com/en-us/omniverse/download/
# 下载 Linux 版本的 Launcher
```

2. **安装 Isaac Sim**
```bash
# 启动 Launcher
./omniverse-launcher-linux.AppImage

# 在 Launcher 中:
# 1. 登录 NVIDIA 账号
# 2. 进入 Exchange 标签页
# 3. 搜索 "Isaac Sim"
# 4. 安装 Isaac Sim 2024.1.1 或更新版本
```

#### 方法 2：通过 pip（开发者模式）

```bash
# 创建 conda 环境
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 安装 Isaac Sim（pip 版本）
pip install isaacsim==4.2.0.2  # 对应 Isaac Sim 2024.1.1
```

### 第 2 步：克隆 Isaac Lab

```bash
# 克隆 Isaac Lab 仓库
cd /home/chenpeel/work/repo/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 检出稳定版本（推荐）
git checkout v1.2.0  # 或最新稳定版
```

### 第 3 步：安装 Isaac Lab

```bash
# 激活 conda 环境
conda activate isaaclab

# 运行安装脚本
./isaaclab.sh --install

# 安装额外依赖
./isaaclab.sh --extra rsl_rl  # RSL_RL 库
./isaaclab.sh --extra robomimic  # （可选）模仿学习库
```

### 第 4 步：验证安装

```bash
# 验证 Isaac Sim 可用
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# 应该看到一个空的仿真场景窗口
# 如果成功打开，说明安装成功
```

### 第 5 步：配置 MJCF 支持

Isaac Lab 默认已支持 MJCF，无需额外配置。验证：

```bash
# 测试 MJCF 加载
python -c "from omni.isaac.lab.sim.spawners.from_files import MjcfFileCfg; print('MJCF support OK')"
```

## 依赖安装

### Python 依赖

Isaac Lab 的依赖已在安装脚本中处理，但可以手动验证：

```bash
# 验证核心依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import omni.isaac.lab; print('Isaac Lab OK')"
python -c "from rsl_rl.algorithms import PPO; print('RSL_RL OK')"
```

### 可选依赖

```bash
# 如果需要使用 TensorBoard
pip install tensorboard

# 如果需要使用 Weights & Biases
pip install wandb

# 如果需要 ONNX 导出
pip install onnx onnxruntime
```

## 环境配置

### 配置 GPU

```bash
# 检查 GPU 可用性
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 验证 PyTorch 可以使用 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 配置 Isaac Sim 环境变量

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
# Isaac Sim 路径（根据实际安装路径调整）
export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0"

# Isaac Lab 路径
export ISAACLAB_PATH="/home/chenpeel/work/repo/IsaacLab"

# 添加到 PATH
export PATH="${ISAACLAB_PATH}:${PATH}"
```

然后重新加载配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

## 项目环境配置

### 创建项目专用环境

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl

# 复制 Isaac Lab 的环境配置
cp ${ISAACLAB_PATH}/.python-version ./

# 创建虚拟环境（可选，如果不使用 conda）
python3.10 -m venv .venv
source .venv/bin/activate

# 或使用 conda
conda create -n jiyuan-isaaclab python=3.10 -y
conda activate jiyuan-isaaclab
```

### 安装项目依赖

```bash
# 安装 Isaac Lab 和 RSL_RL
pip install -e ${ISAACLAB_PATH}/source/extensions/omni.isaac.lab
pip install -e ${ISAACLAB_PATH}/source/extensions/omni.isaac.lab_tasks
pip install -e ${ISAACLAB_PATH}/source/extensions/omni.isaac.lab_assets

# 安装 RSL_RL
pip install rsl-rl

# 安装项目开发依赖
pip install pytest pytest-cov black isort flake8 mypy
```

## 验证安装

### 运行官方示例

```bash
cd ${ISAACLAB_PATH}

# 测试基础仿真
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# 测试机器人加载
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

# 测试强化学习环境
./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_rl_env.py

# 测试 RSL_RL 训练
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 --num_envs 16 --headless
```

### 检查清单

在开始迁移前，确保以下所有项都已完成：

- [ ] Isaac Sim 安装成功，可打开仿真场景
- [ ] Isaac Lab 安装成功，可运行官方示例
- [ ] RSL_RL 可正常导入和使用
- [ ] GPU 可被 PyTorch 识别和使用
- [ ] MJCF 支持可用
- [ ] conda/虚拟环境配置正确
- [ ] 项目目录结构已创建

## 常见问题

### 问题 1：Isaac Sim 无法启动

**症状**: 运行示例时报错 "Failed to load Isaac Sim"

**解决方案**:
```bash
# 检查 NVIDIA 驱动版本
nvidia-smi

# 更新驱动（如果版本过低）
sudo apt update
sudo apt install nvidia-driver-535  # 或更新版本

# 重启系统
sudo reboot
```

### 问题 2：GPU 内存不足

**症状**: 训练时报错 "CUDA out of memory"

**解决方案**:
```bash
# 减少并行环境数
# 在配置文件中调整 num_envs: 4096 -> 2048
```

### 问题 3：Python 版本冲突

**症状**: Isaac Lab 安装失败，提示 "Python 3.11 not supported"

**解决方案**:
```bash
# 卸载当前环境
conda deactivate
conda env remove -n isaaclab

# 重新创建环境（指定 Python 3.10）
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 重新安装 Isaac Lab
cd ${ISAACLAB_PATH}
./isaaclab.sh --install
```

### 问题 4：MJCF 加载失败

**症状**: 加载 MJCF 文件时报错

**解决方案**:
```bash
# 验证 MJCF 文件路径正确
ls -la /home/chenpeel/work/repo/jiyuan/rl/assets/xmls/models/jiyuan/index.xml

# 检查 MJCF 文件语法
# 使用 MuJoCo 官方工具验证
python -c "import mujoco; model = mujoco.MjModel.from_xml_path('path/to/index.xml'); print('MJCF OK')"
```

## 性能优化建议

### GPU 内存优化

```bash
# 在运行脚本前设置环境变量
export ISAAC_SIM_MEMORY_FRACTION=0.8  # 使用 80% GPU 内存
```

### 编译缓存

```bash
# 启用 Isaac Sim 编译缓存
export ISAAC_SIM_CACHE_DIR="${HOME}/.cache/isaac-sim"
mkdir -p ${ISAAC_SIM_CACHE_DIR}
```

## 下一步

环境搭建完成后：

1. 阅读 [02-architecture.md](./02-architecture.md) 了解项目架构
2. 阅读 [03-code-mapping.md](./03-code-mapping.md) 了解代码映射关系
3. 开始 [04-implementation-phases.md](./04-implementation-phases.md) 的阶段 1 实施

## 参考资料

- [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html)
- [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html)
- [RSL_RL GitHub](https://github.com/leggedrobotics/rsl_rl)
