# 从零开始完整指南

本指南将带您从零开始搭建 Isaac Lab 强化学习训练环境，并完成第一次训练。适合初次使用的用户，特别是在云端 GPU 环境（如 autodl）中部署。

## 📋 目录

- [环境准备](#环境准备)
- [安装步骤](#安装步骤)
- [验证安装](#验证安装)
- [首次训练](#首次训练)
- [评估策略](#评估策略)
- [云端GPU部署](#云端gpu部署)
- [常见问题](#常见问题)

---

## 环境准备

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 3060 (8GB) | RTX 4070+ (12GB+) |
| CPU | 4 核心 | 8+ 核心 |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB 可用空间 | 100GB+ SSD |

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 20.04 / 22.04 | 必须 |
| NVIDIA Driver | 525+ | 必须，支持 CUDA 11.8/12.1 |
| CUDA | 11.8 或 12.1 | 必须 |
| Python | 3.10 或 3.11 | 必须 |
| Git | 2.0+ | 必须，用于克隆仓库 |

### 检查现有环境

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 Python 版本
python3 --version

# 检查 Git 版本
git --version
```

---

## 安装步骤

### 步骤 1: 克隆项目仓库

```bash
# 选择一个工作目录
cd ~
mkdir -p work && cd work

# 克隆项目（包含子模块）
git clone --recursive https://github.com/Chenpeel/rl.git
cd rl

# 验证子模块已下载
ls dep/IsaacLab    # 应该看到 Isaac Lab 文件
ls dep/rsl_rl      # 应该看到 RSL_RL 文件
```

**如果子模块未下载：**

```bash
# 初始化并更新子模块
git submodule update --init --recursive
```

### 步骤 2: 安装 Isaac Sim

Isaac Sim 是 NVIDIA 的物理仿真引擎，Isaac Lab 依赖它运行。

#### 选项 A: 使用 Omniverse Launcher（推荐本地环境）

1. **下载 Omniverse Launcher**:
   - 访问: https://www.nvidia.com/en-us/omniverse/download/
   - 下载并安装 Omniverse Launcher

2. **安装 Isaac Sim**:
   - 打开 Launcher
   - 在 "Exchange" 标签中搜索 "Isaac Sim"
   - 安装 Isaac Sim 2024.1.1 或更高版本

#### 选项 B: 使用 pip 安装（推荐云端环境）

```bash
# 创建虚拟环境
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 安装 Isaac Sim pip 包（约 8GB）
pip install isaacsim==4.2.0.2
# 或使用清华镜像加速
pip install isaacsim==4.2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证 Isaac Sim 安装：**

```bash
# 测试导入
python -c "import isaacsim; print('✅ Isaac Sim 安装成功')"
```

### 步骤 3: 安装 Isaac Lab

```bash
# 进入 Isaac Lab 目录
cd dep/IsaacLab

# 运行安装脚本
./isaaclab.sh --install

# 安装 RSL_RL 扩展
./isaaclab.sh --extra rsl_rl
```

**安装过程说明：**
- `--install`: 安装 Isaac Lab 核心依赖
- `--extra rsl_rl`: 安装 RSL_RL 强化学习库

**验证 Isaac Lab 安装：**

```bash
# 测试 Isaac Lab
python -c "import isaaclab; print('✅ Isaac Lab 安装成功')"

# 测试 RSL_RL
python -c "from rsl_rl.algorithms import PPO; print('✅ RSL_RL 安装成功')"
```

### 步骤 4: 安装项目

```bash
# 返回项目根目录
cd ../../

# 进入 isaaclab_rl 目录
cd isaaclab_rl

# 使用 Makefile 安装（推荐）
make install

# 或手动安装
pip install -e .
```

**验证项目安装：**

```bash
# 测试导入
python -c "from isaaclab_rl import jiyuan_tasks; print('✅ 项目安装成功')"

# 查看已注册的环境
python -c "from isaaclab_rl.jiyuan_tasks import TASK_NAMES; print('可用任务:', TASK_NAMES)"
```

### 步骤 5: 验证完整安装

```bash
# 使用 Makefile 验证
make verify
```

**预期输出：**

```
✓ PyYAML 可用
✓ TensorBoard 可用
✓ Isaac Lab 可用
✓ RSL_RL 可用
✓ 项目包可用
```

---

## 验证安装

### 快速测试

运行一个最小化测试，确保环境可以正常创建：

```bash
cd isaaclab_rl

# 方法 1: 使用 Makefile
make train-test

# 方法 2: 直接运行
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task test \
    --num_envs 64 \
    --max_iterations 10 \
    --headless
```

**预期输出：**

```
[INFO] 注册任务: Isaac-Jiyuan-Test-v0
[INFO] 创建环境: 64 个并行环境
[INFO] 观测空间: (64, 65)
[INFO] 动作空间: (64, 16)
[INFO] 开始训练...
[INFO] Iteration 1/10: reward_mean=0.234
[INFO] Iteration 2/10: reward_mean=0.256
...
[INFO] ✅ 训练完成！
```

### 检查 GPU 使用

```bash
# 在另一个终端中运行
watch -n 1 nvidia-smi
```

**正常情况：**
- GPU 利用率: 70-95%
- 显存使用: 2-4GB (64 envs)
- 温度: 60-80°C

---

## 首次训练

现在开始真正的训练！我们将训练一个站立任务。

### 训练站立任务

```bash
cd isaaclab_rl

# 完整训练（4096 个环境，无头模式）
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --num_envs 4096 \
    --headless \
    --max_iterations 1000
```

**命令参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称（test/standing/velocity） | standing |
| `--num_envs` | 并行环境数量 | 4096 |
| `--headless` | 无头模式（不显示GUI） | False |
| `--max_iterations` | 最大迭代次数 | 无限 |
| `--log_dir` | 日志保存目录 | logs/rsl_rl/{task} |
| `--seed` | 随机种子 | 42 |

### 监控训练进度

**方法 1: 命令行输出**

```
[INFO] Iteration 100/1000
       reward_mean: 15.234
       episode_length_mean: 150.5
       fps: 45000
       time_elapsed: 120.5s
```

**方法 2: TensorBoard**

```bash
# 在新终端中启动 TensorBoard
tensorboard --logdir isaaclab_rl/logs

# 在浏览器中打开
# http://localhost:6006
```

**方法 3: 日志文件**

```bash
# 查看最新日志
tail -f logs/rsl_rl/standing/log.txt
```

### 训练时间预估

| 环境数量 | GPU | 训练速度 | 达到 30M steps 时间 |
|---------|-----|---------|-------------------|
| 4096 | RTX 4070 | ~50k steps/s | 4-6 小时 |
| 4096 | RTX 3060 | ~30k steps/s | 7-10 小时 |
| 8192 | RTX 4090 | ~80k steps/s | 3-4 小时 |
| 2048 | RTX 3060 | ~20k steps/s | 12-16 小时 |

### 中断和恢复训练

**中断训练：**

按 `Ctrl + C` 停止训练。训练进度会自动保存。

**恢复训练：**

```bash
# 从最后一个 checkpoint 恢复
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --resume \
    --load_run logs/rsl_rl/standing/YYYY-MM-DD_HH-MM-SS
```

---

## 评估策略

训练完成后，评估训练好的策略。

### 加载 Checkpoint 评估

```bash
# 评估最新的 checkpoint
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task standing \
    --checkpoint logs/rsl_rl/standing/model_1000.pt \
    --num_envs 1
```

**带可视化评估（本地环境）：**

```bash
# 打开 GUI 查看机器人行为
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task standing \
    --checkpoint logs/rsl_rl/standing/model_1000.pt \
    --num_envs 1
```

### 录制视频

```bash
# 录制 200 帧视频
../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task standing \
    --checkpoint logs/rsl_rl/standing/model_1000.pt \
    --video \
    --video_length 200
```

视频将保存到 `videos/` 目录。

### 批量评估多个 Checkpoints

```bash
# 评估所有保存的模型
for ckpt in logs/rsl_rl/standing/model_*.pt; do
    echo "评估: $ckpt"
    ../dep/IsaacLab/isaaclab.sh -p scripts/play.py \
        --task standing \
        --checkpoint $ckpt \
        --num_envs 16 \
        --num_steps 1000 \
        --headless
done
```

---

## 云端GPU部署

本节专门针对 autodl、恒源云等云端 GPU 平台。

### autodl 平台部署

#### 1. 创建实例

**推荐配置：**

| 项目 | 配置 |
|------|------|
| GPU | RTX 4090 (24GB) 或 RTX 4070 (12GB) |
| CPU | 8 核心+ |
| 内存 | 32GB+ |
| 硬盘 | 100GB+ |
| 镜像 | PyTorch 2.0+ / CUDA 11.8+ |

#### 2. 连接实例

```bash
# SSH 连接（autodl 提供）
ssh root@connect.autodl.com:12345

# 或使用 JupyterLab / VS Code
```

#### 3. 快速安装脚本

```bash
# 下载并运行一键安装脚本
wget https://raw.githubusercontent.com/Chenpeel/rl/master/isaaclab_rl/scripts/install.sh
chmod +x install.sh
./install.sh
```

#### 4. 配置无头模式

云端环境通常没有显示器，必须使用无头模式：

```bash
# 训练时始终添加 --headless
../dep/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task standing \
    --num_envs 4096 \
    --headless
```

#### 5. 远程监控

**使用 TensorBoard：**

```bash
# 启动 TensorBoard（映射端口）
tensorboard --logdir logs --port 6006 --host 0.0.0.0

# 在本地浏览器访问
# http://<实例IP>:6006
```

**使用 tmux/screen 保持训练：**

```bash
# 安装 tmux
apt-get install tmux

# 创建会话
tmux new -s training

# 运行训练
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --headless

# 分离会话: Ctrl + B, 然后按 D

# 重新连接
tmux attach -t training
```

#### 6. 数据传输

**下载训练日志：**

```bash
# 在本地机器上运行
scp -P 12345 -r root@connect.autodl.com:/root/work/rl/isaaclab_rl/logs ./
```

**上传 checkpoint：**

```bash
# 在本地机器上运行
scp -P 12345 model_best.pt root@connect.autodl.com:/root/work/rl/isaaclab_rl/logs/
```

### 恒源云平台部署

步骤与 autodl 类似，主要差异：

1. **镜像选择**: 选择 Ubuntu 22.04 + CUDA 11.8 镜像
2. **端口映射**: 在控制台配置 TensorBoard 端口（6006）
3. **存储**: 使用数据盘（/hy-tmp）存储日志和模型

---

## 常见问题

### 安装相关

#### Q1: 子模块克隆失败

**问题：**

```
fatal: clone of 'https://github.com/isaac-sim/IsaacLab.git' into submodule path 'dep/IsaacLab' failed
```

**解决方案：**

```bash
# 方法 1: 手动克隆子模块
cd dep
git clone https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/leggedrobotics/rsl_rl.git

# 方法 2: 使用 GitHub 镜像
git config --global url."https://ghproxy.com/https://github.com".insteadOf "https://github.com"
git submodule update --init --recursive
```

#### Q2: Isaac Lab 安装超时

**问题：**

```
ERROR: Could not install packages due to an OSError: [Errno 110] Connection timed out
```

**解决方案：**

```bash
# 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 重新安装
cd dep/IsaacLab
./isaaclab.sh --install
```

#### Q3: CUDA 版本不匹配

**问题：**

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案：**

```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 如果不匹配，重新安装对应版本的 PyTorch
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 训练相关

#### Q4: GPU 显存不足

**问题：**

```
RuntimeError: CUDA out of memory
```

**解决方案：**

```bash
# 减少并行环境数量
../dep/IsaacLab/isaaclab.sh -p scripts/train.py --task standing --num_envs 2048

# 或减少网络大小（编辑 agents/rsl_rl/ppo_cfg.py）
```

#### Q5: 训练速度慢

**问题：** GPU 利用率低于 50%

**解决方案：**

```bash
# 1. 增加并行环境数量
--num_envs 8192

# 2. 检查 CPU 瓶颈
htop  # 查看 CPU 使用率

# 3. 确保使用 GPU 版本的 PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

#### Q6: 训练不收敛

**问题：** 奖励值不增长或震荡

**解决方案：**

```bash
# 1. 调整学习率（编辑 configs/train_config.yaml）
ppo:
  algorithm:
    learning_rate: 0.0003  # 改为 0.0001

# 2. 增加训练迭代次数
--max_iterations 5000

# 3. 检查奖励函数权重（编辑 jiyuan_tasks/envs/cfg/*_env_cfg.py）
```

### 云端部署相关

#### Q7: 连接 SSH 超时

**解决方案：**

```bash
# 配置 SSH keep-alive
echo "ServerAliveInterval 60" >> ~/.ssh/config

# 或使用 mosh（更稳定）
apt-get install mosh
mosh root@connect.autodl.com
```

#### Q8: 硬盘空间不足

**解决方案：**

```bash
# 清理不必要的文件
make clean
make clean-logs

# 删除旧的 checkpoint
cd logs/rsl_rl/standing
ls -lt | tail -n +10 | awk '{print $9}' | xargs rm -f

# 使用外部存储（autodl 数据盘）
ln -s /root/autodl-tmp logs
```

---

## 下一步

完成首次训练后，您可以：

1. **尝试不同任务**:
   - 速度跟踪: `--task velocity`
   - 自定义任务: 参考 [开发指南](../README.md#开发指南)

2. **优化性能**:
   - 调整超参数: 编辑 `configs/train_config.yaml`
   - 调整奖励函数: 编辑 `jiyuan_tasks/envs/cfg/*_env_cfg.py`

3. **真机部署**:
   - 参考 [Sim2Real 指南](SIM2REAL_AND_ADVANCED_FEATURES.md)
   - 使用并联脚踝映射

4. **进阶功能**:
   - 模仿学习
   - 课程学习
   - 多机器人训练

## 获取帮助

- **文档**: [文档索引](README.md)
- **Issues**: [GitHub Issues](https://github.com/Chenpeel/rl/issues)
- **讨论**: [GitHub Discussions](https://github.com/Chenpeel/rl/discussions)

---

祝您训练顺利！🚀
