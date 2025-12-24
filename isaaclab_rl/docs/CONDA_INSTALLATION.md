# Conda 环境快速安装指南

本指南专为 **Miniconda/Anaconda 用户**设计，提供最快的安装流程。

---

## 前提条件

✅ 已安装 Miniconda/Anaconda
✅ 已创建并激活 Conda 环境（如 `irl`）
✅ NVIDIA GPU + CUDA 驱动

---

## 快速安装（5 分钟）

### 步骤 1：激活 Conda 环境

```bash
conda activate irl
```

### 步骤 2：一键安装（推荐）

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
make install  # 或 bash scripts/quick_install.sh
```

这将自动安装：
- Isaac Sim (Pip 版本)
- Isaac Lab
- RSL_RL
- 本项目及其依赖

---

## 手动安装（逐步执行）

如果一键安装失败，可以手动执行以下步骤：

### 步骤 1：安装 Isaac Sim (Pip 版本)

```bash
conda activate irl

# 安装 Isaac Sim Python 包（约 5-8 GB，需要 10-20 分钟）
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
    isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
    --extra-index-url https://pypi.nvidia.com
```

**国内用户加速**：
```bash
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
    isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
    --extra-index-url https://pypi.nvidia.com \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 2：安装 Isaac Lab

```bash
cd ~/workspace  # 或任意工作目录

# 克隆 Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 安装（使用 pip，不需要 isaac-sim 路径）
pip install -e .
```

### 步骤 3：安装 RSL_RL

```bash
cd ~/workspace

# 克隆 RSL_RL
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl

# 安装
pip install -e .
```

### 步骤 4：安装本项目

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl

# 安装项目（开发模式）
pip install -e .

# 可选：安装额外依赖
pip install -e ".[vis]"   # 可视化工具
pip install -e ".[dev]"   # 开发工具
pip install -e ".[all]"   # 所有依赖
```

---

## 验证安装

### 快速验证

```bash
conda activate irl

# 验证所有组件
python -c "
import isaacsim
import omni.isaac.lab
import rsl_rl
from jiyuan_tasks import *
print('✅ 所有组件安装成功！')
"
```

### 完整测试

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl

# 测试配置加载
python -c "
from jiyuan_tasks.utils.config_loader import load_train_config, validate_train_config
cfg = load_train_config('configs/train_config.yaml')
validate_train_config(cfg)
print('✅ 配置文件验证通过！')
"

# 测试训练（快速测试，64 环境，10 迭代）
python scripts/train.py --config configs/train_config.yaml \
    --num_envs 64 --max_iterations 10
```

---

## 环境管理

### 查看已安装的包

```bash
conda activate irl
pip list | grep -E "isaac|rsl|biped"
```

### 更新项目

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
git pull
pip install -e . --upgrade
```

### 卸载

```bash
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
pip uninstall isaaclab-biped-rl -y
```

### 完全清理环境

```bash
conda deactivate
conda remove -n irl --all -y
```

---

## Conda 特定配置

### 设置环境变量（可选）

在 Conda 环境中自动设置环境变量：

```bash
conda activate irl

# 创建环境变量脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# 激活时设置环境变量
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
#!/bin/bash
export BIPED_RL_PATH="/home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl"
export PYTHONPATH="$BIPED_RL_PATH:$PYTHONPATH"
EOF

# 停用时清除环境变量
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh <<'EOF'
#!/bin/bash
unset BIPED_RL_PATH
EOF

# 重新激活环境使其生效
conda deactivate
conda activate irl

# 验证
echo $BIPED_RL_PATH
```

---

## 常见问题

### Q1: `pip install -e .` 报错 "No module named 'setuptools'"

**A**: 升级 setuptools

```bash
conda activate irl
pip install --upgrade setuptools pip wheel
pip install -e .
```

### Q2: Isaac Sim 安装失败或下载慢

**A**: 使用国内镜像

```bash
# 设置 pip 默认镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 重新安装
pip install isaacsim-rl --extra-index-url https://pypi.nvidia.com
```

### Q3: ImportError: libpython3.10.so.1.0

**A**: 确保使用 Conda 环境的 Python

```bash
conda activate irl
which python  # 应该输出 ~/miniconda3/envs/irl/bin/python

# 如果不对，重新创建环境
conda deactivate
conda remove -n irl --all -y
conda create -n irl python=3.10 -y
conda activate irl
```

### Q4: CUDA 版本不匹配

**A**: 安装对应 CUDA 版本的 PyTorch

```bash
conda activate irl

# 检查 CUDA 版本
nvidia-smi

# 安装对应版本 PyTorch
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### Q5: 如何在 Jupyter 中使用？

**A**: 安装 IPython kernel

```bash
conda activate irl
pip install ipykernel
python -m ipykernel install --user --name irl --display-name "Python (irl)"

# 在 Jupyter 中选择 "Python (irl)" kernel
```

---

## 便捷命令（Makefile）

在项目根目录创建 `Makefile`，使用 `make` 命令快速执行常见任务。

### 安装

```bash
make install        # 完整安装
make install-dev    # 安装开发工具
make install-vis    # 安装可视化工具
```

### 训练

```bash
make train          # 开始训练（使用默认配置）
make train-test     # 快速测试（64 envs, 10 iters）
make train-headless # 无头模式训练
```

### 清理

```bash
make clean          # 清理构建文件
make clean-logs     # 清理日志文件
```

查看所有可用命令：
```bash
make help
```

---

## 推荐工作流

### 首次安装

```bash
# 1. 创建并激活环境
conda create -n irl python=3.10 -y
conda activate irl

# 2. 一键安装
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
make install

# 3. 验证
python -c "from jiyuan_tasks import *; print('OK')"

# 4. 测试训练
make train-test
```

### 日常开发

```bash
# 激活环境
conda activate irl

# 进入项目目录
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl

# 修改代码...

# 运行训练
python scripts/train.py --config configs/train_config.yaml

# 或使用 make
make train
```

---

## 总结

### 最快安装方式

```bash
conda activate irl
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
make install
```

### 手动安装（3 步）

```bash
# 1. 安装 Isaac Sim
pip install isaacsim-rl --extra-index-url https://pypi.nvidia.com

# 2. 安装 Isaac Lab 和 RSL_RL
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && pip install -e .
git clone https://github.com/leggedrobotics/rsl_rl.git && cd rsl_rl && pip install -e .

# 3. 安装本项目
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
pip install -e .
```

---

**维护记录**：
- 2025-01-22：创建 Conda 环境快速安装指南
- 针对 Miniconda 用户优化
- 提供一键安装和手动安装两种方式
