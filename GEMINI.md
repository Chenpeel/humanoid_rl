# Jiyuan RL 项目开发规范

**本文档为 AI 助手(Gemini/Claude)开发本项目时的强制规范**

---

## 📁 目录组织架构（强制执行）

### 核心原则

**清晰的职责边界,避免混淆**

### 目录职责划分

#### 1. `scripts/` - 训练入口脚本

**职责**：仅包含训练相关的入口脚本

**允许存在**：
- ✅ `train.py` - 主训练脚本（支持多阶段课程学习）
- ✅ `eval.py` - 评估启动脚本
- ✅ `convert_urdf.py` - 完整的URDF转换流水线（包含镜像和模块化）

**严格禁止**：
- ❌ 通用工具
- ❌ 可视化工具
- ❌ 数据处理工具
- ❌ 检查点检查/修补工具

#### 2. `src/rl/` - 训练内部实现

**职责**：训练算法、环境、模型、奖励函数等核心实现

**允许存在**：
- ✅ `training/` - PPO算法实现
- ✅ `envs/` - 环境实现
- ✅ `models/` - 模型定义
- ✅ `rewards/` - 奖励函数
- ✅ `curriculum/` - 课程学习
- ✅ `utils/` - 训练内部工具（日志、检查点、模型导出等）

**严格禁止**：
- ❌ 独立运行的脚本（有`if __name__ == "__main__"`的文件）
- ❌ UI工具
- ❌ 与训练无关的工具

#### 3. `utils/` - 非训练相关工具

**职责**：与训练无关的独立工具

**允许存在**：
- ✅ `urdf2mjcf/` - URDF/MJCF转换工具
- ✅ `visualize_mjcf.py` - MJCF可视化工具（支持热刷新）
- ✅ `xml_tools/` - XML处理工具
- ✅ `inspect_checkpoint.py` - 检查点检查工具
- ✅ `patch_train_checkpoint.py` - 检查点修补工具
- ✅ `simplify_collision.py` - 碰撞简化工具

**严格禁止**：
- ❌ 训练逻辑
- ❌ 环境实现
- ❌ PPO算法相关代码

#### 4. `configs/` - 配置文件（统一YAML驱动）

**职责**：所有训练配置,统一使用 `train.yaml` + `curriculum.yaml` 结构

**强制规范**（2026-01-06起）：
- 每个配置目录必须包含：
  - `train.yaml` - 训练超参数和环境配置
  - `curriculum.yaml` - 奖励权重和课程学习阶段定义

**配置目录结构**：
```
configs/
├── train/                      # 标准训练配置
│   ├── train.yaml
│   └── curriculum.yaml
├── train-10h/                  # 长时间训练配置
│   ├── train.yaml
│   └── curriculum.yaml
├── quick_test/                 # 快速测试配置
│   ├── train.yaml
│   └── curriculum.yaml
└── examples/                   # 示例和遗留配置
    ├── velocity_legacy/
    └── walking_legacy/
```

**curriculum.yaml 支持两种模式**：
1. **简单模式**（单阶段）：扁平的奖励权重字典
2. **高级模式**（多阶段）：包含 `stages:` 列表的完整课程定义

---

## 🔧 文件放置决策树

### 新增Python文件时的判断流程

```
1. 这个文件是否会被用户直接运行?
   (是否有 if __name__ == "__main__")

   ├─ 否 → 放在 src/rl/ 对应子目录
   │        例如: 新的奖励函数 → src/rl/rewards/
   │
   └─ 是 → 继续判断
        │
        2. 这个文件是否涉及训练逻辑?
           (启动训练、评估、URDF转换流水线等)

           ├─ 是 → 放在 scripts/
           │        例如: train_new_task.py → scripts/
           │
           └─ 否 → 放在 utils/
                    例如: visualize_trajectory.py → utils/

3. 如果是训练内部的辅助工具?
   (模型导出、检查点管理等)

   → 放在 src/rl/utils/
     例如: export_onnx.py → src/rl/utils/
```

### 新增配置文件时的规范

```
1. 创建新配置目录: configs/<任务名>/
2. 必须同时包含:
   - train.yaml (训练超参数)
   - curriculum.yaml (奖励权重定义)
3. 在 configs/README.md 中添加说明
```

---

## ⚠️ 常见错误示例

### ❌ 错误示例 1：工具放错位置

```
# 错误：可视化工具放在scripts/
scripts/visualize_mjcf.py  ❌

# 正确：应该放在utils/
utils/visualize_mjcf.py    ✅
```

### ❌ 错误示例 2：检查点工具放错位置

```
# 错误：检查点检查工具放在scripts/
scripts/inspect_checkpoint.py  ❌

# 正确：应该放在utils/
utils/inspect_checkpoint.py    ✅
```

### ❌ 错误示例 3：模型导出放错位置

```
# 错误：模型导出工具放在scripts/
scripts/export_model.py  ❌

# 正确：应该放在src/rl/utils/
src/rl/utils/export_model.py  ✅
```

### ❌ 错误示例 4：配置文件结构不统一

```
# 错误：使用旧的配置结构
configs/my_task/
└── config.yaml  ❌

# 正确：使用统一的双文件结构
configs/my_task/
├── train.yaml        ✅
└── curriculum.yaml   ✅
```

---

## ✅ 开发检查清单

在提交代码前,请确认:

### 文件组织检查

- [ ] 所有训练入口脚本都在 `scripts/` 中
- [ ] 所有非训练工具都在 `utils/` 中
- [ ] 所有训练内部工具都在 `src/rl/utils/` 中
- [ ] 没有将可视化/检查点工具放在 `scripts/` 中

### 配置文件检查

- [ ] 新增配置目录包含 `train.yaml` 和 `curriculum.yaml`
- [ ] `train.yaml` 中正确指定了 `curriculum_file` 路径
- [ ] `curriculum.yaml` 遵循简单模式或高级模式规范
- [ ] 在 `configs/README.md` 中添加了说明

### 代码规范检查

- [ ] 遵循 JAX 纯函数设计原则
- [ ] 奖励函数返回 `(scalar, dict)` 元组
- [ ] 奖励分量使用 `reward/` 前缀命名
- [ ] 使用 `jp.clip` 和 `jp.nan_to_num` 确保数值稳定性
- [ ] 使用 `@jax.jit` 装饰计算密集型函数

---

## 📚 相关文档

详细技术规范请参阅:

- `.std_docs/struct.md` - 完整的项目结构和配置说明
- `.std_docs/coding_standards.md` - 详细的代码规范和JAX使用指南
- `configs/README.md` - 配置文件使用指南

---

**最后更新**: 2026-01-06
**维护者**: Jiyuan RL Team
