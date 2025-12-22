# JAX/MJX 到 Isaac Lab + PyTorch 迁移指南

## 概览

本目录包含从 JAX/MJX 强化学习框架迁移到 Isaac Lab + PyTorch + RSL_RL 的完整文档和指南。

**迁移目标**: 从 JAX/MJX 训练框架迁移到 Isaac Lab + PyTorch + RSL_RL
**机器人**: Jiyuan 双足机器人（16 执行器，自定义设计）
**时间规划**: 8+ 周深度优化
**训练任务**: VelocityTrackingEnv（速度跟踪）+ StandingEnv（站立平衡）

## 快速开始

### 查看当前分支状态

```bash
git branch
# 应该看到:
# - master (将成为 Isaac Lab 版本)
# - jax (原 JAX/MJX 实现，作为参考)
# - feature/isaaclab-migration (当前开发分支)
```

### 迁移文档结构

| 文档 | 内容 | 阅读优先级 |
|------|------|-----------|
| [01-setup.md](./01-setup.md) | 环境搭建和依赖安装 | ⭐⭐⭐⭐⭐ 必读 |
| [02-architecture.md](./02-architecture.md) | 架构设计和目录结构 | ⭐⭐⭐⭐⭐ 必读 |
| [03-code-mapping.md](./03-code-mapping.md) | MJX → Isaac Lab 代码映射 | ⭐⭐⭐⭐ 重要 |
| [04-implementation-phases.md](./04-implementation-phases.md) | 分阶段实施计划（8 周） | ⭐⭐⭐⭐ 重要 |
| [05-testing-strategy.md](./05-testing-strategy.md) | 测试和验证策略 | ⭐⭐⭐ 建议阅读 |
| [06-optimization-guide.md](./06-optimization-guide.md) | 性能优化指南 | ⭐⭐⭐ 建议阅读 |
| [07-troubleshooting.md](./07-troubleshooting.md) | 常见问题和解决方案 | ⭐⭐ 参考文档 |

### 阶段验收清单

每个阶段完成后，使用对应的验收清单进行自检：

- [checklists/phase1-checklist.md](./checklists/phase1-checklist.md) - 阶段 1：环境搭建和基础验证
- [checklists/phase2-checklist.md](./checklists/phase2-checklist.md) - 阶段 2：核心功能迁移
- [checklists/phase3-checklist.md](./checklists/phase3-checklist.md) - 阶段 3：完整功能实现
- [checklists/phase4-checklist.md](./checklists/phase4-checklist.md) - 阶段 4：优化和高级功能

## 迁移流程

### 阶段 0：准备工作（当前阶段）

**已完成**:
- ✅ 创建 `jax` 分支（保留原 JAX/MJX 实现）
- ✅ 创建 `feature/isaaclab-migration` 开发分支
- ✅ 创建 `.migrate` 目录和文档结构

**下一步**:
1. 阅读 [01-setup.md](./01-setup.md) 安装 Isaac Lab
2. 阅读 [02-architecture.md](./02-architecture.md) 了解目标架构
3. 开始阶段 1 的实施

### 阶段 1：环境搭建和基础验证（第 1-2 周）

**目标**:
- 安装 Isaac Lab 并验证基础功能
- 加载 Jiyuan MJCF 模型
- 创建最小可行环境

**详细计划**: 见 [04-implementation-phases.md](./04-implementation-phases.md)
**验收清单**: 见 [checklists/phase1-checklist.md](./checklists/phase1-checklist.md)

### 阶段 2：核心功能迁移（第 3-5 周）

**目标**:
- 实现观测和动作管理器
- 迁移奖励函数（JAX → PyTorch）
- 集成 RSL_RL PPO 训练器

**详细计划**: 见 [04-implementation-phases.md](./04-implementation-phases.md)
**验收清单**: 见 [checklists/phase2-checklist.md](./checklists/phase2-checklist.md)

### 阶段 3：完整功能实现（第 6-7 周）

**目标**:
- 完整训练循环验证（30M timesteps）
- 性能基准测试（与 MJX 对比）
- 视频录制和可视化

**详细计划**: 见 [04-implementation-phases.md](./04-implementation-phases.md)
**验收清单**: 见 [checklists/phase3-checklist.md](./checklists/phase3-checklist.md)

### 阶段 4：优化和高级功能（第 8+ 周）

**目标**:
- 视觉传感器集成（可选）
- 高级领域随机化
- Sim2Real 准备
- 课程学习

**详细计划**: 见 [04-implementation-phases.md](./04-implementation-phases.md)
**验收清单**: 见 [checklists/phase4-checklist.md](./checklists/phase4-checklist.md)

## Git 工作流

### 日常开发

```bash
# 1. 确保在开发分支
git checkout feature/isaaclab-migration

# 2. 定期提交代码
git add <modified_files>
git commit -m "实现 XXX 功能"

# 3. 遇到问题时，可以查看 jax 分支的原始实现
git diff jax -- src/rl/envs/robot_envs.py
```

### 阶段完成后

```bash
# 1. 更新验收清单
# 编辑 .migrate/checklists/phaseX-checklist.md

# 2. 提交阶段性成果
git add .
git commit -m "完成阶段 X：XXX

- 实现了 XXX
- 验证了 XXX
- 通过了验收标准

详见 .migrate/checklists/phaseX-checklist.md
"

# 3. 推送到远程（可选）
git push origin feature/isaaclab-migration
```

### 最终合并

当所有阶段完成并验证通过后：

```bash
# 1. 切换到 master
git checkout master

# 2. 合并迁移分支（使用 squash 保持历史简洁）
git merge --squash feature/isaaclab-migration

# 3. 创建最终提交
git commit -m "迁移到 Isaac Lab + PyTorch + RSL_RL

- 完整的 Isaac Lab 环境实现
- 使用 RSL_RL PPO 训练器
- 保留所有奖励函数和训练逻辑
- 性能提升 2-3 倍
- 详细迁移文档见 .migrate/

🤖 Generated with Claude Code
"

# 4. 推送到远程
git push origin master
git push origin jax  # 同时推送 jax 分支作为备份
```

## 关键决策和理由

### 为什么选择 Isaac Lab？

1. **完整的强化学习框架**: 包含 Omniverse 可视化、GPU 加速物理、活跃社区
2. **性能提升**: 训练速度提升 10-20 倍（相比 MJX）
3. **PyTorch 生态**: 团队熟悉度高，调试工具丰富
4. **可扩展性**: 支持视觉传感器、多地形、Sim2Real

### 为什么选择 RSL_RL？

1. **专业优化**: ETH Zurich 开发，专为四足/双足运动优化
2. **GPU 加速**: GPU 优化的 PPO 实现
3. **广泛验证**: 已在多篇论文中验证（ANYmal, Spot 等）
4. **无缝集成**: 与 Isaac Lab 官方集成

### 为什么保留 jax 分支？

1. **参考基线**: 迁移过程中可对比原始实现
2. **性能基准**: 用于验证迁移后的性能提升
3. **回滚保险**: 如果迁移遇到重大问题，可以回退
4. **知识保留**: 保留原始设计思路和实现细节

## 预期收益

| 维度 | 当前（MJX） | 迁移后（Isaac Lab） | 提升 |
|------|------------|-------------------|------|
| **训练速度** | 12+ 小时（30M steps） | 4-6 小时 | 2-3 倍 |
| **调试效率** | 命令行日志 + 离线视频 | 实时 3D 可视化 + 交互式调试 | 10 倍 |
| **可维护性** | JAX 学习曲线陡峭 | PyTorch 生态成熟 | 显著提升 |
| **扩展性** | 有限（纯物理） | 支持视觉传感器、Sim2Real | 质的飞跃 |

## 联系和支持

### 遇到问题时

1. **查阅文档**: 首先查看 [07-troubleshooting.md](./07-troubleshooting.md)
2. **对比原实现**: 查看 `jax` 分支的原始代码
3. **查看参考项目**: 见 [02-architecture.md](./02-architecture.md) 中的参考项目列表
4. **社区支持**:
   - [Isaac Lab GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions)
   - [NVIDIA 官方论坛](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67)

### 进度跟踪

建议创建 `.migrate/progress.md` 文件记录每日进度：

```markdown
# 迁移进度日志

## 2025-01-22（示例）
- ✅ 完成 Isaac Lab 安装
- ✅ 验证 MJCF 模型可加载
- ⏸️ 创建最小环境（遇到 GPU 内存问题，待解决）

## 2025-01-23（示例）
- ✅ 解决 GPU 内存问题（调整 num_envs 到 2048）
- ✅ 最小环境可成功运行
- ▶️ 开始实现观测管理器
```

## 总结

这是一个系统的、经过充分规划的迁移项目。关键成功因素：

1. **充分的准备**: 阅读所有文档，理解目标架构
2. **渐进式迁移**: 从最小环境开始，逐步增加复杂度
3. **持续验证**: 每个阶段都与 MJX 基线对比
4. **灵活调整**: 根据实际情况调整计划和优先级

祝迁移顺利！🚀
