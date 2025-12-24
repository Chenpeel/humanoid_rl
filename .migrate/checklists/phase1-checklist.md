# 阶段 1 验收清单：环境搭建和基础验证

**时间范围**: 第 1-2 周
**目标**: 安装 Isaac Lab 并验证基础功能可正常工作

## 必备条件

在开始本阶段前，确保已完成：

- [ ] 已阅读 `.migrate/README.md`
- [ ] 已阅读 `.migrate/01-setup.md`
- [ ] 已阅读 `.migrate/02-architecture.md`
- [ ] 当前在 `feature/isaaclab-migration` 分支

## 任务清单

### Task 1.1: 安装 Isaac Sim

- [ ] **下载 Omniverse Launcher**
  - 访问 https://www.nvidia.com/en-us/omniverse/download/
  - 下载 Linux 版本

- [ ] **安装 Isaac Sim 2024.1.1+**
  - 通过 Launcher 安装 Isaac Sim
  - 验证版本 >= 2024.1.1

- [ ] **验证安装**
  ```bash
  # 启动 Isaac Sim，应该能打开窗口
  ${ISAAC_SIM_PATH}/isaac-sim.sh
  ```

**验收标准**:
- ✅ Isaac Sim 可正常启动
- ✅ 无 GPU 驱动错误
- ✅ 可在窗口中看到默认场景

**遇到问题**: 见 `.migrate/01-setup.md` 的"常见问题"章节

---

### Task 1.2: 克隆和安装 Isaac Lab

- [ ] **克隆 Isaac Lab 仓库**
  ```bash
  cd /home/chenpeel/work/repo/
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab
  git checkout v1.2.0  # 或最新稳定版
  ```

- [ ] **安装 Isaac Lab**
  ```bash
  conda create -n isaaclab python=3.10 -y
  conda activate isaaclab
  ./isaaclab.sh --install
  ```

- [ ] **安装 RSL_RL**
  ```bash
  ./isaaclab.sh --extra rsl_rl
  ```

**验收标准**:
- ✅ Isaac Lab 安装成功，无错误
- ✅ RSL_RL 可正常导入
  ```bash
  python -c "from rsl_rl.algorithms import PPO; print('RSL_RL OK')"
  ```

---

### Task 1.3: 运行官方示例

- [ ] **测试基础仿真**
  ```bash
  cd /home/chenpeel/work/repo/IsaacLab
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
  ```
  **预期**: 打开一个空的仿真场景窗口

- [ ] **测试机器人加载**
  ```bash
  ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py
  ```
  **预期**: 看到机器人模型在仿真中运动

- [ ] **测试强化学习环境**
  ```bash
  ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_rl_env.py
  ```
  **预期**: 看到 CartPole 环境重置和运行

- [ ] **测试 RSL_RL 训练**
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Cartpole-v0 --num_envs 16 --headless
  ```
  **预期**: 训练开始，TensorBoard 日志正常生成

**验收标准**:
- ✅ 所有示例可正常运行
- ✅ 无 CUDA 错误
- ✅ GPU 利用率 > 50%（训练时）

---

### Task 1.4: 验证 MJCF 支持

- [ ] **测试 MJCF 加载**
  ```bash
  cd /home/chenpeel/work/repo/jiyuan/rl
  python -c "
  from omni.isaac.lab.sim.spawners.from_files import MjcfFileCfg
  print('MJCF support OK')
  "
  ```

- [ ] **测试加载 Jiyuan MJCF 模型**
  ```bash
  python -c "
  import mujoco
  model = mujoco.MjModel.from_xml_path('assets/xmls/models/jiyuan/index.xml')
  print(f'Loaded robot with {model.nu} actuators, {model.nq} DOFs')
  "
  ```
  **预期输出**: `Loaded robot with 16 actuators, ...`

**验收标准**:
- ✅ MJCF 支持可用
- ✅ Jiyuan MJCF 可成功加载，无语法错误
- ✅ 执行器数量正确（16 个）

---

### Task 1.5: 创建项目目录结构

- [ ] **创建 isaaclab_rl 目录**
  ```bash
  cd /home/chenpeel/work/repo/jiyuan/rl
  mkdir -p isaaclab_rl/{jiyuan_tasks/{envs/cfg,managers,utils},agents/rsl_rl,scripts,tools,tests}
  ```

- [ ] **创建 __init__.py 文件**
  ```bash
  touch isaaclab_rl/__init__.py
  touch isaaclab_rl/jiyuan_tasks/__init__.py
  touch isaaclab_rl/jiyuan_tasks/envs/__init__.py
  touch isaaclab_rl/jiyuan_tasks/managers/__init__.py
  ```

- [ ] **创建 assets 软链接**
  ```bash
  ln -s ../../assets isaaclab_rl/assets
  ```

**验收标准**:
- ✅ 目录结构与 `.migrate/02-architecture.md` 一致
- ✅ assets 软链接可正常访问

---

### Task 1.6: 创建最小可行环境

- [ ] **创建场景配置文件**
  - 文件: `isaaclab_rl/jiyuan_tasks/envs/cfg/jiyuan_scene_cfg.py`
  - 内容: 定义 Jiyuan 机器人的场景配置
  - 参考: `.migrate/03-code-mapping.md` 中的"MJCF 模型配置"

- [ ] **创建环境基类**
  - 文件: `isaaclab_rl/jiyuan_tasks/envs/jiyuan_base_env.py`
  - 内容: 继承 `ManagerBasedRLEnv`
  - 参考: `.migrate/03-code-mapping.md` 中的"环境定义转换"

- [ ] **注册环境**
  - 文件: `isaaclab_rl/jiyuan_tasks/__init__.py`
  - 使用 `gym.register()` 注册环境

**验收标准**:
- ✅ 环境可通过 `gym.make("Isaac-Jiyuan-Test-v0")` 创建
- ✅ 无导入错误

---

### Task 1.7: 验证批量环境重置

- [ ] **测试环境创建**
  ```python
  import gymnasium as gym
  env = gym.make("Isaac-Jiyuan-Test-v0", num_envs=16)
  print(f"Created {env.num_envs} environments")
  ```

- [ ] **测试批量重置**
  ```python
  obs, info = env.reset()
  print(f"Observation shape: {obs.shape}")
  ```

- [ ] **测试单步执行**
  ```python
  action = torch.zeros(env.num_envs, env.action_space.shape[0])
  obs, reward, terminated, truncated, info = env.step(action)
  print(f"Reward shape: {reward.shape}")
  ```

**验收标准**:
- ✅ 可成功创建 16 个并行环境
- ✅ 批量重置成功，观测形状正确
- ✅ 单步执行成功，无错误

---

### Task 1.8: 验证实时可视化

- [ ] **启动非 headless 模式**
  ```bash
  python scripts/test_env.py --task Isaac-Jiyuan-Test-v0 --num_envs 4
  ```

- [ ] **检查 Omniverse 查看器**
  - 应该能看到 4 个机器人同时运行
  - 可以用鼠标旋转视角
  - 可以在窗口中看到机器人状态

- [ ] **测试相机切换**
  - 尝试切换到不同的相机视角（track, side, top）

**验收标准**:
- ✅ Omniverse 查看器可正常打开
- ✅ 可实时看到机器人运动
- ✅ 帧率 >= 30 FPS
- ✅ 无渲染错误或警告

---

### Task 1.9: 性能基准测试

- [ ] **测试 GPU 利用率**
  ```bash
  # 在训练时运行
  watch -n 1 nvidia-smi
  ```
  **预期**: GPU 利用率 > 80%

- [ ] **测试训练吞吐量**
  ```bash
  # 记录 10 秒内的步数
  # 计算 steps/second
  ```
  **预期**: > 100k steps/second（16 envs）

- [ ] **测试内存占用**
  ```bash
  # 观察 GPU 内存使用
  nvidia-smi --query-gpu=memory.used --format=csv
  ```
  **预期**: < 8GB（16 envs）

**验收标准**:
- ✅ GPU 利用率 > 80%
- ✅ 训练吞吐量满足预期
- ✅ 内存占用合理

---

## 最终验收

### 必须通过的所有检查

- [ ] Isaac Lab 安装成功，能运行官方示例
- [ ] Jiyuan MJCF 模型可在 Isaac Sim 中加载，无错误
- [ ] 最小环境可成功重置 16 个并行环境（目标 4096，先从小规模验证）
- [ ] 可在 Omniverse 查看器中实时查看机器人状态
- [ ] 无 CUDA 错误，GPU 利用率 > 80%
- [ ] 项目目录结构已创建并符合规范

### 可选加分项

- [ ] 测试更大规模环境（64, 256, 1024 envs）
- [ ] 录制演示视频
- [ ] 编写简单的单元测试

## 提交清单

阶段 1 完成后，提交代码：

```bash
# 1. 添加所有新文件
git add isaaclab_rl/
git add .migrate/

# 2. 提交
git commit -m "阶段 1 完成：环境搭建和基础验证

✅ 完成任务：
- Isaac Lab 安装和配置
- 运行官方示例验证
- MJCF 模型加载验证
- 创建项目目录结构
- 最小可行环境实现
- 批量环境重置验证
- 实时可视化验证
- 性能基准测试

验收清单：.migrate/checklists/phase1-checklist.md
"

# 3. 推送到远程（可选）
git push origin feature/isaaclab-migration
```

## 进入下一阶段

阶段 1 验收通过后：

1. 阅读 [../03-code-mapping.md](../03-code-mapping.md) 深入理解代码转换
2. 阅读 [./phase2-checklist.md](./phase2-checklist.md) 了解阶段 2 任务
3. 开始实施阶段 2：核心功能迁移

## 记录遇到的问题

如果遇到问题，记录到 `.migrate/07-troubleshooting.md`：

```markdown
### 问题：Isaac Sim 启动失败

**日期**: 2025-01-22
**症状**: 运行示例时报错 "Failed to load Isaac Sim"
**解决方案**: 更新 NVIDIA 驱动到 535.104.05
**参考**: https://forums.developer.nvidia.com/...
```

---

**预计完成时间**: 1-2 周
**当前状态**: [ ] 未开始 | [ ] 进行中 | [ ] 已完成
**完成日期**: _____________
