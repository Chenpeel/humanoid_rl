# Mirrored.xml 文件差异分析

## 概述

本文档详细描述了 `assets/xmls/mirrored.xml` 相对于 `assets/mjcf/mirrored.xml` 的添加和修改部分。这两个文件都是用于 Jiyuan 双足机器人强化学习训练的 MJCF 模型文件，但 `xmls/mirrored.xml` 包含了更多用于实际训练的增强功能。

## 文件基本信息

| 属性 | `assets/mjcf/mirrored.xml` | `assets/xmls/mirrored.xml` |
|------|---------------------------|---------------------------|
| 文件大小 | 399 行 | 887 行 |
| 主要用途 | 基础模型定义 | 增强的训练模型 |
| 新增功能 | 基础模型 | 肌腱约束、接触排除、传感器等 |

## 主要差异总结

### 1. 默认类定义增强 (`<default>` 部分)

**新增内容：**
```xml
<default class="thigh_equality">
    <equality solref="0.001 1" solimp="0.99 0.999 0.0001" />
</default>
```

**说明：**
- 添加了 `thigh_equality` 类，用于定义大腿部分的等式约束
- 配置了约束求解器参数：
  - `solref="0.001 1"`: 参考时间常数和阻尼比
  - `solimp="0.99 0.999 0.0001"`: 阻抗参数

### 2. 资产部分 (`<asset>` 部分)

**新增内容：**
- 添加了完整的材料定义：
  ```xml
  <material name="ground" rgba="0.8 0.8 0.8 1" />
  ```
- 添加了更多网格文件引用（完整的人体模型网格）

### 3. 世界体定义 (`<worldbody>` 部分)

**主要增强：**
- 添加了完整的光源配置：
  ```xml
  <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true" />
  ```
- 添加了地面几何体：
  ```xml
  <geom type="plane" size="10 10 0.1" pos="0 0 0" material="ground" condim="3" friction="1 0.005 0.0001" />
  ```
- 完整的机器人身体结构定义（从基础链接到各个关节和末端执行器）

### 4. 肌腱系统 (`<tendon>` 部分) - **全新添加**

**这是最重要的新增功能之一**，用于模拟踝关节的并联机构：

```xml
<!-- 踝关节并联机构：用tendon模拟刚性连杆，窄range + 高stiffness = 近似恒定长度 -->
<tendon>
    <!-- 右侧踝关节肌腱 -->
    <spatial name="right_ankle_tendon_1" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="right_ankle_1_3_attach" />
        <site site="right_foot_attach_1" />
    </spatial>
    <spatial name="right_ankle_tendon_2" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="right_ankle_2_3_attach" />
        <site site="right_foot_attach_2" />
    </spatial>
    <spatial name="right_ankle_tendon_3" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="right_ankle_3_3_attach" />
        <site site="right_foot_attach_3" />
    </spatial>
    
    <!-- 左侧踝关节肌腱（镜像） -->
    <spatial name="left_ankle_tendon_1" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="left_ankle_1_3_attach" />
        <site site="left_foot_attach_1" />
    </spatial>
    <spatial name="left_ankle_tendon_2" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="left_ankle_2_3_attach" />
        <site site="left_foot_attach_2" />
    </spatial>
    <spatial name="left_ankle_tendon_3" limited="true" range="0.034 0.038" stiffness="8000" damping="100">
        <site site="left_ankle_3_3_attach" />
        <site site="left_foot_attach_3" />
    </spatial>
</tendon>
```

**设计原理：**
- 使用空间肌腱模拟刚性连杆
- 窄范围 (`range="0.034 0.038"`) + 高刚度 (`stiffness="8000"`) = 近似恒定长度
- 添加阻尼 (`damping="100"`) 防止振荡
- 连接踝关节各个部分到脚部附着点

### 5. 执行器定义 (`<actuator>` 部分)

**新增内容：**
完整的电机执行器定义，包括所有关节：
- 髋关节（俯仰、偏航、滚动）
- 膝关节
- 踝关节（1-3号并联机构）
- 脚趾关节

**示例：**
```xml
<motor name="right_hip_pitch_engine_joint_ctrl" joint="right_hip_pitch_engine_joint" class="motor" />
```

### 6. 接触排除 (`<contact>` 部分) - **全新添加**

**这是另一个重要的新增功能**，用于排除相邻身体部件之间的接触，防止自碰撞：

```xml
<contact>
    <!-- 右侧身体接触排除 -->
    <exclude body1="right_hip_pitch_engine_link" body2="right_hip_roll_link" />
    <exclude body1="right_hip_pitch_engine_link" body2="right_hip_cube_link" />
    <exclude body1="right_hip_pitch_engine_link" body2="right_thigh_link" />
    <exclude body1="right_hip_pitch_engine_link" body2="right_hip_yaw_engine_link" />
    
    <!-- ... 更多排除规则 ... -->
    
    <!-- 左侧身体接触排除（镜像） -->
    <exclude body1="left_hip_pitch_engine_link" body2="left_hip_roll_link" />
    <exclude body1="left_hip_pitch_engine_link" body2="left_hip_cube_link" />
    <!-- ... 更多排除规则 ... -->
</contact>
```

**排除规则覆盖：**
- 所有相邻的机械连接部件
- 防止同一肢体内部的自碰撞
- 提高仿真稳定性和计算效率

### 7. 传感器系统 (`<sensor>` 部分) - **全新添加**

**用于强化学习观测的传感器：**

```xml
<sensor>
    <framepos name="base_link_site_pos" objtype="site" objname="base_link_site" />
    <framequat name="base_link_site_quat" objtype="site" objname="base_link_site" />
    <framelinvel name="base_link_site_linvel" objtype="site" objname="base_link_site" />
    <frameangvel name="base_link_site_angvel" objtype="site" objname="base_link_site" />
    <velocimeter name="base_link_site_vel" site="base_link_site" />
</sensor>
```

**传感器类型：**
- 位置传感器 (`framepos`)
- 姿态传感器 (`framequat`)
- 线速度传感器 (`framelinvel`)
- 角速度传感器 (`frameangvel`)
- 速度计 (`velocimeter`)

## 技术设计原理

### 1. 肌腱模拟并联机构
- **问题**：真实的踝关节使用并联机构（如 Stewart 平台）提供6自由度运动
- **解决方案**：使用6个高刚度肌腱模拟刚性连杆
- **优势**：在 MuJoCo 中实现近似刚性的连接，同时保持计算效率

### 2. 接触排除优化
- **问题**：复杂机器人模型容易产生自碰撞，导致仿真不稳定
- **解决方案**：显式排除所有已知的相邻部件接触
- **优势**：提高仿真稳定性，减少不必要的接触计算

### 3. 传感器集成
- **目的**：为强化学习提供丰富的观测信息
- **设计**：在基础链接上添加多种运动学传感器
- **应用**：用于状态估计、奖励计算和策略学习

## 文件用途说明

### `assets/mjcf/mirrored.xml`
- **用途**：基础模型定义文件
- **特点**：包含基本的机器人几何和运动学结构
- **适用场景**：模型验证、基础可视化

### `assets/xmls/mirrored.xml`
- **用途**：增强的训练模型文件
- **特点**：包含完整的仿真优化功能
- **适用场景**：强化学习训练、高性能仿真

## 对强化学习训练的影响

### 正面影响：
1. **仿真稳定性**：接触排除减少自碰撞，提高训练稳定性
2. **物理准确性**：肌腱系统提供更真实的踝关节动力学
3. **观测丰富性**：传感器提供全面的状态信息
4. **计算效率**：优化的接触处理减少计算开销

### 注意事项：
1. **肌腱参数调整**：高刚度可能导致数值稳定性问题
2. **接触排除完整性**：需要确保所有必要的排除都已定义
3. **传感器噪声**：实际应用中可能需要添加噪声模型

## 维护建议

1. **同步更新**：当修改基础模型时，需要同步更新两个文件
2. **参数调优**：肌腱刚度和接触参数可能需要根据具体任务调整
3. **验证测试**：任何修改后都应运行验证测试确保功能正常
4. **文档更新**：模型变更应及时更新本文档

## 总结

`assets/xmls/mirrored.xml` 在基础模型的基础上，通过添加肌腱系统、接触排除和传感器等高级功能，为强化学习训练提供了更加稳定、真实和高效的仿真环境。这些增强功能对于训练复杂的双足机器人运动控制策略至关重要。
