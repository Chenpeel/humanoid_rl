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
- NVIDIA Driver 550+
- CUDA 11.8 / 12.1

### 2. 使用步骤

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

# 6. 运行训练
make train
```
