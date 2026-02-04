# 双足机器人强化学习训练框架

基于 **Isaac Lab** + **PyTorch** + **RSL_RL** 的双足机器人强化学习训练系统。

## 项目概览

本项目为机器人提供完整的强化学习训练解决方案，从仿真训练到真机部署的全流程支持。

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
- NVIDIA Driver 550+
- CUDA 11.8 / 12.1

### 2. 使用步骤

```bash
# 1. 克隆仓库（包含子模块）
git clone --recursive https://github.com/Chenpeel/rl.git
cd rl

# 2. 更新子模块
make submodule-update

# 3. 安装 uv（用于管理 Python 版本）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者使用清华镜像 pip 安装
# $(which python) -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv --version
# 4. 创建并激活 Python3.11 虚拟环境
uv venv -p 3.11
.venv/bin/activate.bat  # Windows
# source .venv/bin/activate.fish # Linux fish shell
# 可查看其他 shell
# ls .venv/bin/activate*

# 4. 安装 Isaac Lab
cd dep/IsaacLab
./isaaclab.sh --install

# 4. 安装项目
cd ../../
make install

# 5. 验证安装
make verify

# 6. 运行训练
make train
```
