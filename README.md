# JRL - JAX强化学习训练库

基于JAX + MJX的双足机器人强化学习训练框架，采用自动三阶段课程学习策略。

## 技术栈

- **物理引擎**: MuJoCo (MJX)
- **深度学习**: JAX 0.4.20+ ， XAX
- **算法**: PPO (Proximal Policy Optimization)
- **课程学习**: 自动三阶段渐进式训练
- **硬件**: NVIDIA GPU with CUDA 11.8+

## 核心特性

### 🎓 自动课程学习

采用自动三阶段课程学习框架：

- **阶段1（0-50k steps）**：站立平衡 - 学习保持直立不摔倒
- **阶段2（50k-150k steps）**：低速行走 - 学习基本步态和低速前进
- **阶段3（150k+ steps）**：全速行走 - 跟踪任意速度命令并优化性能

**优势**：

- ✅ 奖励权重自动切换 - 无需手动配置多个阶段文件
- ✅ 环境参数自动调整 - 速度命令范围、目标高度自动适配
- ✅ 训练日志记录阶段信息 - curriculum_stage、progress等
- ✅ 可复用框架 - 未来可扩展到跳跃、攀爬等任务

## 快速开始

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者使用清华镜像 pip 安装
# $(which python) -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv --version
```

### 拉取 assets

```bash
# ubuntu
# 使用 lfs
sudo apt-get install git-lfs

git lfs pull
```

### ksim 训练

```bash
# 1) 同步依赖（包含 ksim extra）
make sync-ksim

# 2) 站立专训（先把 upright/height 练稳）
make train-ksim-stand

# 3) 行走训练（世界系速度命令 vx, vy, wz）
make train-ksim-walk
```

配置文件见 `configs/ksim/`，训练入口脚本为 `scripts/train_ksim.py`。

### 安装依赖

```bash
make install
```

### 训练

```bash
# 标准训练（2048 envs，200M steps，约8-12小时）
make train

# 长时间训练（4096 envs，500M steps，约24-48小时）
make train-long

# 快速测试（128 envs，1M steps，约5-10分钟）
make train-test
```

### 监控训练

```bash
make tensorboard
# 访问: http://localhost:6006
```

### 评估模型

```bash
make eval CKPT=logs/diy_train/ppo_*/checkpoints/best_model
make eval CKPT=... RENDER=0                 # 不渲染（最快）
make eval CKPT=... RENDER=10                # 每10步渲染（加速）
make eval CKPT=... SAVE_VIDEO=1 VIDEO_PATH=eval.mp4  # 保存视频（较慢）
```

### 仅可视化播放（不评估）

```bash
make play CKPT=logs/diy_train/ppo_*/checkpoints/best_model
# 或直接运行:
python scripts/play.py --checkpoint logs/diy_train/ppo_*/checkpoints/best_model --render 1 --realtime

# 保存视频（可选）
make play CKPT=... SAVE_VIDEO=1 VIDEO_PATH=play.mp4 VIDEO_FPS=50
# 指定模型XML路径（可选）
make play CKPT=... XML_PATH=assets/xmls/scenes/flat_terrain.xml
```

## 目录结构

```
jrl/
├── configs/                    # 配置文件
│   ├── train/                  # 标准训练配置
│   ├── train-10h/              # 长时间训练配置
│   └── quick_test/             # 快速测试配置
├── src/rl/                     # 源代码
│   ├── envs/                   # 环境实现
│   ├── models/                 # 神经网络模型
│   ├── rewards/                # 奖励函数（含三阶段权重）
│   ├── curriculum/             # 课程学习模块
│   ├── training/               # 训练逻辑
│   └── utils/                  # 工具函数
├── scripts/                    # 训练脚本
│   ├── train.py                # 主训练脚本（集成课程学习）
│   └── eval.py                 # 评估脚本
├── logs/                       # 日志目录
│   ├── diy_train/              # 训练日志（TensorBoard、检查点、视频等）
│   ├── ksim_train/             # ksim 训练输出（xax/ksim 默认 run_dir）
│   └── makelog/                # Make命令执行日志
├── assets/                     # 机器人资源
├── docs/                       # 项目文档
└── Makefile                    # 命令管理

```

## 日志结构

- **训练日志**: `logs/diy_train/ppo_[时间戳]/`
  - `events.out.tfevents.*` - TensorBoard事件文件
  - `checkpoints/` - 模型检查点
  - `videos/` - 训练过程视频
  - `config.yaml` - 训练配置快照

- **ksim 训练输出**: `logs/ksim_train/gaoda_jiyuan_task/run_###/`

- **Make执行日志**: `logs/makelog/[命令]_[时间戳].log`
  - 完整记录所有make命令的执行过程
  - 与终端输出完全一致（使用tee实现）

## 配置说明

详细配置说明请参考：[configs/README.md](configs/README.md)

## 常用命令

```bash
# 查看所有可用命令
make help

# 检查环境
make check-env

# 验证配置文件
make validate-config

# 查看项目信息
make info

# 清理缓存和日志
make clean-all
```

## 参考

- **Isaac Lab Curriculum**: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- **Gait-Conditioned RL**: https://arxiv.org/abs/2505.20619
