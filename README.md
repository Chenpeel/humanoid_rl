# 双足机器人强化学习训练框架

>  基于 **Isaac Lab** + **PyTorch** + **RSL_RL** 的双足机器人强化学习训练系统。

## 项目概览

本项目为机器人提供完整的强化学习训练解决方案，从仿真训练到真机部署的全流程支持。

### 技术栈

| 组件     | 技术             | 版本      |
| -------- | ---------------- | --------- |
| 仿真引擎 | NVIDIA Isaac Sim | 2024.1.1+ |
| 环境框架 | Isaac Lab        | v0.2.0+   |
| 深度学习 | PyTorch          | 2.0+      |
| 强化学习 | RSL_RL           | v2.2.1+   |

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
# 1. 克隆仓库
git clone https://github.com/Chenpeel/humanoid_rl.git
cd humanoid_rl

# 2. 安装 uv（用于管理 Python 版本）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者使用清华镜像 pip 安装
# $(which python) -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv --version

# 3. 准备 Python 3.11
uv python install 3.11

# 4. 安装项目
make install
# Linux x86_64 / aarch64: 自动安装 dev + training
# macOS / 其他平台: 自动安装 dev

# 5 验证安装
make verify

# 6 运行训练（需 Linux x86_64 / aarch64）
make train
```


### 3. 自定模型

将自定义的 MJCF 模型文件放置在 `rl/robots/` 目录下，并在训练配置中指定模型名称即可。

> 编写urdf/mjcf的xml文件
> 使用可视化验证

```bash
make visualize-mjcf XML=robots/unitree_h1/scene.xml \
AUTORELOAD=0 #是否自动重载模型\
GRAVITY=1 #是否启动重力\
MODE=launch 
```

> 更多内容参考

```bash
make help
```
