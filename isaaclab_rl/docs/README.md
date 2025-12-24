# 文档索引

本目录包含项目的所有详细文档。根据您的需求选择合适的文档阅读。

## 📚 文档分类

### 🚀 入门文档（新手必读）

#### 1. [GETTING_STARTED.md](./GETTING_STARTED.md) ⭐⭐⭐⭐⭐

**最完整的从零开始指南**

- 详细的环境准备和硬件要求
- 分步安装教程（本地 + 云端）
- 首次训练完整流程
- 云端 GPU 部署专题（autodl/恒源云）
- 常见问题和故障排查

**适合对象**:
- 首次使用本项目的用户
- 需要在云端 GPU 部署的用户
- 遇到安装或训练问题的用户

**预计阅读时间**: 30-45 分钟

---

#### 2. [MINIMAL_SETUP_GUIDE.md](./MINIMAL_SETUP_GUIDE.md) ⭐⭐⭐⭐

**快速上手指南**

- 最小化安装步骤
- 配置系统说明
- 配置文件优先级（CLI > config > 预定义）
- 快速验证和测试

**适合对象**:
- 有经验的用户，想快速搭建环境
- 需要了解配置系统的用户
- 需要自定义配置的开发者

**预计阅读时间**: 15-20 分钟

---

#### 3. [CONDA_INSTALLATION.md](./CONDA_INSTALLATION.md) ⭐⭐⭐

**Conda 环境安装指南**

- 使用 Conda 管理环境
- 一键安装 Isaac Sim + Isaac Lab + RSL_RL
- 虚拟环境隔离和依赖管理

**适合对象**:
- 偏好使用 Conda 的用户
- 需要多环境隔离的用户
- 有 Python 包冲突问题的用户

**预计阅读时间**: 10-15 分钟

---

### 🎯 高级功能文档

#### 4. [SIM2REAL_AND_ADVANCED_FEATURES.md](./SIM2REAL_AND_ADVANCED_FEATURES.md) ⭐⭐⭐⭐

**Sim2Real 和高级特性**

- 3-DOF 并联脚踝机构集成
- 仿真到真机的动作映射
- 舵机校准和偏移配置
- 模仿学习（BVH/FBX）
- Walking 任务实现
- 域随机化

**适合对象**:
- 需要真机部署的用户
- 使用并联脚踝机构的用户
- 需要模仿学习的研究者
- 高级功能开发者

**预计阅读时间**: 25-30 分钟

---

## 🗺️ 推荐阅读路径

### 路径 A: 完全新手（推荐）

```
1. GETTING_STARTED.md（从零开始完整指南）
   ↓
2. 实际操作：安装 + 首次训练
   ↓
3. MINIMAL_SETUP_GUIDE.md（深入了解配置）
   ↓
4. SIM2REAL_AND_ADVANCED_FEATURES.md（高级功能）
```

**预计总时间**: 4-6 小时（包含实际操作）

---

### 路径 B: 有经验用户（快速上手）

```
1. MINIMAL_SETUP_GUIDE.md（快速安装）
   ↓
2. 实际操作：快速验证
   ↓
3. 根据需要查阅 GETTING_STARTED.md 的特定章节
```

**预计总时间**: 1-2 小时

---

### 路径 C: 云端 GPU 用户（重点推荐）

```
1. GETTING_STARTED.md 的"云端GPU部署"章节
   ↓
2. 实际操作：云端安装 + 训练
   ↓
3. GETTING_STARTED.md 的"常见问题"章节
```

**预计总时间**: 2-3 小时

---

### 路径 D: 真机部署用户

```
1. GETTING_STARTED.md（基础环境）
   ↓
2. 完成仿真训练
   ↓
3. SIM2REAL_AND_ADVANCED_FEATURES.md（Sim2Real 映射）
   ↓
4. 真机测试和调优
```

**预计总时间**: 1-2 天（包含训练和测试）

---

## 📖 文档详细说明

### GETTING_STARTED.md

**包含章节：**
- ✅ 环境准备（硬件/软件要求）
- ✅ 安装步骤（5 个详细步骤）
- ✅ 验证安装（快速测试）
- ✅ 首次训练（完整流程）
- ✅ 评估策略（加载和测试）
- ✅ 云端 GPU 部署（autodl/恒源云）
- ✅ 常见问题（8+ 个FAQ）

**特色：**
- 每个步骤都有预期输出
- 包含故障排查
- 专门的云端部署章节
- 详细的命令参数说明

---

### MINIMAL_SETUP_GUIDE.md

**包含章节：**
- ✅ 最小化安装流程
- ✅ 配置系统架构
- ✅ 配置文件说明
- ✅ 舵机校准流程
- ✅ 快速验证方法

**特色：**
- 简洁明了，适合快速参考
- 重点讲解配置系统
- 提供配置文件模板

---

### CONDA_INSTALLATION.md

**包含章节：**
- ✅ Conda 环境创建
- ✅ 依赖安装
- ✅ 环境激活和切换
- ✅ 常见问题

**特色：**
- 专门针对 Conda 用户
- 一键安装脚本
- 环境隔离最佳实践

---

### SIM2REAL_AND_ADVANCED_FEATURES.md

**包含章节：**
- ✅ 并联脚踝机构原理
- ✅ Sim2Real 动作映射
- ✅ 舵机配置和校准
- ✅ 模仿学习流程
- ✅ Walking 任务实现
- ✅ 域随机化配置

**特色：**
- 深入讲解机器人机构
- 真机部署完整流程
- 高级算法集成指南

---

## 🔗 相关资源

### 外部文档

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL_RL 文档](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Sim 文档](https://docs.omniverse.nvidia.com/isaacsim/)
- [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)

### 项目文档

- [项目主 README](../../README.md)
- [Isaac Lab RL README](../README.md)
- [迁移指南](.../migrate/README.md)

### 社区和支持

- [GitHub Issues](https://github.com/Chenpeel/rl/issues) - 报告问题和建议
- [GitHub Discussions](https://github.com/Chenpeel/rl/discussions) - 讨论和交流

---

## 📝 文档贡献

发现文档错误或有改进建议？欢迎贡献！

1. Fork 本仓库
2. 编辑文档（Markdown 格式）
3. 提交 Pull Request

**文档编写规范：**
- 使用简体中文
- 包含代码示例
- 提供预期输出
- 添加故障排查提示

---

## 📅 文档更新日志

### v0.3.0 (2024-12-24)

- ✅ 新增 [GETTING_STARTED.md](./GETTING_STARTED.md) - 完整的从零开始指南
- ✅ 更新所有文档以反映项目最新状态
- ✅ 添加云端 GPU 部署专题
- ✅ 重新组织文档结构

### v0.2.0 (2024-01-22)

- ✅ 初始版本的迁移文档
- ✅ MINIMAL_SETUP_GUIDE.md
- ✅ CONDA_INSTALLATION.md
- ✅ SIM2REAL_AND_ADVANCED_FEATURES.md

---

**文档维护者**: Chenpeel (chenpeel@foxmail.com)
**最后更新**: 2024-12-24
**文档版本**: v0.3.0
