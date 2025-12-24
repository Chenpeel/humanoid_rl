# 双足机器人 机器人最小化配置指南

本指南说明如何最小化配置训练、仿真和实体机器人控制环境。

## 目录

1. [配置优先级机制](#配置优先级机制)
2. [快速开始](#快速开始)
3. [配置文件详解](#配置文件详解)
4. [训练环境配置](#训练环境配置)
5. [仿真环境配置](#仿真环境配置)
6. [实体机器人部署](#实体机器人部署)
7. [舵机校准流程](#舵机校准流程)
8. [常见问题](#常见问题)

---

## 配置优先级机制

**核心原则**：命令行参数 > 配置文件 > 代码预定义值

### 工作方式

1. **代码预定义值**（最低优先级）
   - 代码中的默认配置
   - 位置：`agents/rsl_rl/ppo_cfg.py`
   - 作用：提供基线配置，确保系统可运行

2. **配置文件**（中等优先级）
   - YAML 配置文件
   - 位置：`configs/train_config.yaml`, `configs/robot_config.yaml`, `configs/servo_config.yaml`
   - 作用：项目级配置，便于团队共享和版本管理

3. **命令行参数**（最高优先级）
   - 运行时传入的参数
   - 示例：`--num_envs 8192 --headless`
   - 作用：快速实验和临时调整，无需修改文件

### 示例

假设以下场景：
- 代码预定义：`num_envs = 4096`
- 配置文件：`num_envs: 2048`
- 命令行：`--num_envs 8192`

**最终结果**：`num_envs = 8192`（命令行优先级最高）

---

## 快速开始

### 1. 最小化训练（使用默认配置）

```bash
# 不使用配置文件，使用默认值
cd /home/chenpeel/work/repo/jiyuan/rl/isaaclab_rl
python scripts/train.py --task velocity --num_envs 4096
```

### 2. 使用配置文件训练（推荐）

```bash
# 使用配置文件
python scripts/train.py --config configs/train_config.yaml

# 命令行覆盖配置文件参数
python scripts/train.py --config configs/train_config.yaml --num_envs 8192 --headless
```

### 3. 评估训练策略

```bash
# 评估最新检查点
python scripts/play.py --task velocity --checkpoint logs/velocity_tracking/20250122_143000/model_30000.pt

# 确定性评估（无随机性）
python scripts/play.py --task velocity --checkpoint logs/velocity_tracking/20250122_143000/model_30000.pt --deterministic
```

### 4. 部署到实体机器人

```bash
# 在实体机器人代码中（ROS 节点）
# 1. 加载舵机配置
# 2. 加载策略模型
# 3. 创建 Sim2Real 映射器
# 4. 运行控制循环

# 详细步骤见"实体机器人部署"章节
```

---

## 配置文件详解

### 1. 训练配置 (`configs/train_config.yaml`)

**用途**：训练超参数、环境配置、奖励权重等

**关键参数**：

```yaml
# 任务选择
task: "velocity"  # 可选: velocity, standing, walking, test

# 环境配置
environment:
  num_envs: 4096           # 并行环境数（GPU 性能相关）
  episode_length_s: 20.0   # episode 时长（秒）
  headless: false          # 是否无头模式（服务器训练时设为 true）

# PPO 算法参数
ppo:
  algorithm:
    learning_rate: 0.001    # 学习率
    clip_param: 0.2         # PPO 裁剪参数
    entropy_coef: 0.01      # 熵系数
    gamma: 0.99             # 折扣因子
    lam: 0.95               # GAE lambda

  runner:
    max_iterations: 30000   # 最大训练迭代数
    num_steps_per_env: 24   # 每环境步数
    save_interval: 500      # 检查点保存间隔
    device: "cuda:0"        # 训练设备

  network:
    actor_hidden_dims: [512, 256, 128]   # Actor 网络架构
    critic_hidden_dims: [512, 256, 128]  # Critic 网络架构
    activation: "elu"                    # 激活函数

# 奖励权重（按任务）
rewards:
  velocity_tracking:
    track_lin_vel_xy: 1.5   # 线速度跟踪奖励
    track_ang_vel_z: 0.75   # 角速度跟踪奖励
    ankle_workspace: -1.0   # 脚踝工作空间约束（Sim2Real）
    # ... 其他奖励

# 领域随机化
domain_randomization:
  randomize_mass:
    enable: true
    range: [-0.2, 0.2]  # ±20% 质量变化
  # ... 其他随机化

# 终止条件
terminations:
  time_out: true
  fallen: true
  min_height: 0.15  # m
  max_tilt: 1.0     # rad (~57度)
```

**修改建议**：
- **训练速度慢**：降低 `num_envs`（如 2048），或增加 GPU 显存
- **策略不稳定**：降低 `learning_rate`（如 0.0005），增加 `num_learning_epochs`
- **希望更快收敛**：调整奖励权重，增加主要任务奖励的权重
- **Sim2Real 准备**：启用 `ankle_workspace` 奖励，启用领域随机化

### 2. 机器人配置 (`configs/robot_config.yaml`)

**用途**：机器人物理参数、并联脚踝几何参数、动作空间映射

**关键参数**：

```yaml
# 机器人基础信息
robot_info:
    name: "Jiyuan"
    version: "v1.0"
    total_mass: 5  # kg
    height: 0.35   # m（站立高度）

# 并联脚踝几何参数（3-parallel-dot）
parallel_ankle:
    # 几何参数（单位：米）
    l0: 0.02  # 平台半径
    l1: 0.01  # 动平台到O点的距离
    l2: 0.03  # 静平台到O点的距离

    # 工作空间限制（单位：弧度）
    workspace:
        roll_min: -0.5236   # -30度
        roll_max: 0.5236    # +30度
        pitch_min: -0.5236
        pitch_max: 0.5236
        yaw_min: -0.5236
        yaw_max: 0.5236

    # 安全边界（训练时的软约束）
    safety_margin: 0.1  # 弧度

# 动作空间映射
action_mapping:
    action_dim: 16
    ankle_indices:
        left:
            roll: 9
            pitch: 10
            yaw: 11
        right:
            roll: 12
            pitch: 13
            yaw: 14
    # 其他关节索引
    other_joints:
        left_hip_yaw: 0
        left_hip_roll: 1
        # ... 完整关节列表

# Sim2Real 参数
sim2real:
    enable_parallel_ankle_mapping: true
    filtering:
        enable: true
        alpha: 0.7  # 低通滤波系数
```

**修改场景**：
- **更换机器人模型**：修改 `robot_info` 和 `action_mapping`
- **实测脚踝参数**：从 CAD 或 ROS 配置读取 `l0, l1, l2` 并更新
- **调整工作空间**：根据实测调整 `workspace` 限制
- **滤波调优**：根据实体机器人抖动情况调整 `alpha`（越大越平滑）

### 3. 舵机配置 (`configs/servo_config.yaml`)

**用途**：舵机物理参数、角度偏移、校准数据

**关键参数**：

```yaml
# 舵机通用参数
servo_general:
  position_mapping:
    min_us: 500      # 最小脉宽 (微秒)
    max_us: 2500     # 最大脉宽 (微秒)
    min_angle: 0.0   # 最小角度 (弧度)
    max_angle: 3.141592653589793  # 最大角度 (弧度, π)

  default_speed: 100  # 默认运动时间 (毫秒)

# 左脚踝并联机构舵机配置
left_ankle:
  servo_1:
    id: 9               # 舵机硬件ID
    offset: 0.0         # 角度偏移 (弧度)，用于校准
    direction: 1        # 方向：1=正向, -1=反向
    name: "左脚踝舵机1"

  servo_2:
    id: 10
    offset: 0.0
    direction: 1
    name: "左脚踝舵机2"

  servo_3:
    id: 11
    offset: 0.0
    direction: 1
    name: "左脚踝舵机3"

# 右脚踝并联机构舵机配置
right_ankle:
  servo_1:
    id: 12
    offset: 0.0
    direction: 1
    name: "右脚踝舵机1"
  # ... 类似配置

# 舵机校准记录（可选）
calibration:
  date: "2025-01-22"
  method: "手动调整"
  values:
    9:  { measured_offset: 0.0, note: "左脚踝1 - 已校准" }
    10: { measured_offset: 0.0, note: "左脚踝2 - 已校准" }
    11: { measured_offset: 0.0, note: "左脚踝3 - 已校准" }
    12: { measured_offset: 0.0, note: "右脚踝1 - 已校准" }
    13: { measured_offset: 0.0, note: "右脚踝2 - 已校准" }
    14: { measured_offset: 0.0, note: "右脚踝3 - 已校准" }

# 安全限制
safety:
  max_temperature: 80  # 摄氏度
  max_current: 2.0     # 安培
  min_voltage: 6.0     # 伏特
  max_voltage: 8.4     # 伏特
```

**修改场景**：
- **舵机校准**：更新 `calibration.values` 中的 `measured_offset`
- **更换舵机型号**：修改 `position_mapping` 参数
- **反向安装**：设置 `direction: -1`
- **调整速度**：修改 `default_speed`

---

## 训练环境配置

### 方式 1：使用配置文件（推荐）

```bash
# 1. 编辑配置文件
nano configs/train_config.yaml

# 2. 运行训练
python scripts/train.py --config configs/train_config.yaml
```

### 方式 2：命令行覆盖

```bash
# 保持配置文件不变，临时修改参数
python scripts/train.py --config configs/train_config.yaml \
    --num_envs 8192 \
    --max_iterations 50000 \
    --device cuda:1 \
    --headless
```

### 方式 3：完全使用命令行（不推荐）

```bash
# 不使用配置文件（向后兼容）
python scripts/train.py --task velocity --num_envs 4096 --seed 42
```

### 训练参数调优建议

#### GPU 显存不足
```bash
# 降低并行环境数
python scripts/train.py --config configs/train_config.yaml --num_envs 2048
```

#### 训练速度慢
```bash
# 启用无头模式（关闭可视化）
python scripts/train.py --config configs/train_config.yaml --headless
```

#### 策略不收敛
1. 检查奖励函数配置（`configs/train_config.yaml` 中的 `rewards` 部分）
2. 降低学习率：修改 `ppo.algorithm.learning_rate: 0.0005`
3. 增加训练时长：`--max_iterations 50000`

#### 录制训练视频
```bash
python scripts/train.py --config configs/train_config.yaml --video
```

---

## 仿真环境配置

### Isaac Sim 可视化设置

```bash
# 非无头模式（显示 GUI）
python scripts/train.py --config configs/train_config.yaml

# 无头模式（服务器训练）
python scripts/train.py --config configs/train_config.yaml --headless
```

### 环境参数调整

在 `configs/train_config.yaml` 中修改：

```yaml
environment:
  num_envs: 4096           # 并行环境数
  episode_length_s: 20.0   # episode 时长
  headless: false          # GUI 显示

  # 速度跟踪环境特定配置
  velocity_tracking:
    target_velocity_range: [-1.5, 1.5]        # m/s
    target_angular_velocity_range: [-1.5, 1.5]  # rad/s
    command_resample_time: 10.0               # 命令重采样时间（秒）
```

### 领域随机化配置

```yaml
domain_randomization:
  # 质量随机化
  randomize_mass:
    enable: true
    range: [-0.2, 0.2]  # ±20%

  # 执行器增益随机化
  randomize_actuator_gains:
    enable: true
    stiffness_range: [0.9, 1.1]
    damping_range: [0.9, 1.1]

  # 关节摩擦力随机化
  randomize_friction:
    enable: true
    range: [0.0, 0.01]

  # 外部扰动（推力）
  external_push:
    enable: true
    interval_range_s: [10.0, 15.0]
    velocity_range:
      x: [-0.5, 0.5]
      y: [-0.5, 0.5]
```

**调整建议**：
- Sim2Real 前期训练：启用所有随机化
- 调试训练流程：暂时禁用随机化（`enable: false`）
- 策略过拟合：增加随机化范围

---

## 实体机器人部署

### 架构设计

```
训练（仿真）                    部署（实体机器人）
-----------------              -------------------
策略输出动作                    策略输出动作
    ↓                              ↓
16 维动作向量                   16 维动作向量
    ↓                              ↓
[仿真: 直接使用]               [实体: Sim2Real 映射]
                                   ↓
                              ParallelAnkleMapper
                                   ↓
                              舵机命令（脉宽+速度）
                                   ↓
                              ROS 舵机控制节点
                                   ↓
                              实体舵机执行
```

### 部署步骤

#### 步骤 1：准备配置文件

确保以下配置文件已正确配置：
- `configs/robot_config.yaml` - 并联脚踝几何参数
- `configs/servo_config.yaml` - 舵机偏移和校准数据

#### 步骤 2：导出策略模型

```bash
# 导出为 ONNX 格式（可选，用于非 Python 部署）
python scripts/export_policy.py --checkpoint logs/velocity_tracking/20250122_143000/model_30000.pt --format onnx

# 或直接使用 PyTorch 模型（.pt 文件）
```

#### 步骤 3：创建 ROS 部署节点

在 ROS 工作空间中创建部署节点（示例）：

```python
#!/usr/bin/env python3
"""
双足机器人 机器人实体部署节点

订阅传感器数据 → 策略推理 → Sim2Real 映射 → 舵机控制
"""

import rospy
import torch
import numpy as np

from jiyuan_tasks.utils.sim2real import ParallelAnkleMapper
from jiyuan_tasks.utils.config_loader import load_robot_config, load_servo_config

# 导入 ROS 消息类型（根据实际情况调整）
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray


class JiyuanDeploymentNode:
    """双足机器人 机器人部署节点"""

    def __init__(self):
        rospy.init_node("jiyuan_deployment", anonymous=False)

        # 1. 加载配置
        self.robot_config = load_robot_config("configs/robot_config.yaml")
        self.servo_config = load_servo_config("configs/servo_config.yaml", apply_calibration=True)

        # 2. 加载策略模型
        checkpoint_path = rospy.get_param("~checkpoint_path", "logs/velocity_tracking/model_30000.pt")
        self.policy = self.load_policy(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"策略模型已加载: {checkpoint_path}")

        # 3. 创建 Sim2Real 映射器
        ankle_cfg = self.robot_config.parallel_ankle
        self.ankle_mapper = ParallelAnkleMapper(
            ankle_indices=self.robot_config.action_mapping.ankle_indices,
            l0=ankle_cfg.l0,
            l1=ankle_cfg.l1,
            l2=ankle_cfg.l2,
            enable_filtering=self.robot_config.sim2real.filtering.enable,
            filter_alpha=self.robot_config.sim2real.filtering.alpha,
        )
        rospy.loginfo("Sim2Real 映射器已初始化")

        # 4. 订阅传感器数据
        rospy.Subscriber("/jiyuan/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/jiyuan/imu", Imu, self.imu_callback)

        # 5. 发布舵机命令
        self.servo_cmd_pub = rospy.Publisher("/jiyuan/servo_commands", Float64MultiArray, queue_size=1)

        # 6. 状态变量
        self.joint_pos = np.zeros(16)
        self.joint_vel = np.zeros(16)
        self.imu_data = None

        # 7. 控制循环
        control_freq = self.robot_config.control.control_frequency  # Hz
        self.timer = rospy.Timer(rospy.Duration(1.0 / control_freq), self.control_loop)

        rospy.loginfo(f"部署节点已启动，控制频率: {control_freq} Hz")

    def load_policy(self, checkpoint_path):
        """加载策略模型"""
        # TODO: 实现策略加载（根据实际模型结构）
        checkpoint = torch.load(checkpoint_path)
        # policy = ActorCritic(...)
        # policy.load_state_dict(checkpoint['model_state_dict'])
        # return policy
        raise NotImplementedError("策略加载尚未实现")

    def joint_state_callback(self, msg):
        """关节状态回调"""
        self.joint_pos = np.array(msg.position)
        self.joint_vel = np.array(msg.velocity)

    def imu_callback(self, msg):
        """IMU 回调"""
        self.imu_data = msg

    def get_observation(self):
        """构建观测向量（需要与训练时的观测空间一致）"""
        # TODO: 实现观测构建
        # 示例：[base_ang_vel, projected_gravity, commands, joint_pos, joint_vel, actions_prev]
        obs = np.zeros(61)  # 根据实际观测空间调整
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)

    def control_loop(self, event):
        """控制循环（每个控制周期调用一次）"""
        # 1. 获取观测
        obs = self.get_observation()

        # 2. 策略推理
        with torch.no_grad():
            action = self.policy(obs.to(self.device))  # (1, 16)
            action = action.cpu().numpy()[0]  # (16,)

        # 3. Sim2Real 映射（脚踝部分）
        left_servo_cmds = self.ankle_mapper.map_action(action, 'left')
        right_servo_cmds = self.ankle_mapper.map_action(action, 'right')

        # 4. 其他关节直接使用动作值（或添加偏移）
        # TODO: 处理其他关节

        # 5. 应用舵机偏移（从 servo_config 读取）
        # TODO: 应用偏移

        # 6. 发布舵机命令
        servo_commands = Float64MultiArray()
        # servo_commands.data = [...]  # 组装命令
        # self.servo_cmd_pub.publish(servo_commands)

    def run(self):
        """运行节点"""
        rospy.spin()


if __name__ == "__main__":
    try:
        node = JiyuanDeploymentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
```

#### 步骤 4：运行部署节点

```bash
# 在 ROS 工作空间中
rosrun jiyuan_control deployment_node.py _checkpoint_path:=/path/to/model_30000.pt
```

### Sim2Real 关键点

1. **观测空间一致性**：
   - 实体机器人的观测必须与训练时完全一致（维度、归一化、噪声）
   - 检查 IMU 坐标系是否与仿真一致

2. **动作空间映射**：
   - 脚踝关节：使用 `ParallelAnkleMapper` 映射到舵机命令
   - 其他关节：直接使用动作值（或根据实际情况调整）

3. **舵机偏移应用**：
   - 从 `servo_config.yaml` 读取校准偏移
   - 在发送舵机命令前应用偏移

4. **控制频率匹配**：
   - 确保实体控制频率与训练时一致（通常 50 Hz）
   - 从 `robot_config.yaml` 读取 `control.control_frequency`

5. **低通滤波**：
   - `ParallelAnkleMapper` 已内置低通滤波
   - 根据实际抖动情况调整 `filter_alpha`

---

## 舵机校准流程

### 目的

由于舵机安装误差、齿轮间隙等原因，实际舵机零点可能与理论值存在偏差。校准流程用于测量并记录这些偏移量。

### 校准步骤

#### 步骤 1：准备工作

1. 将机器人固定在支架上（避免摔倒）
2. 确保所有舵机已通电且正常工作
3. 准备水平仪或角度测量工具

#### 步骤 2：单个舵机校准

对于每个需要校准的舵机（以左脚踝舵机1为例）：

```bash
# 1. 发送零位命令到舵机
ros topic pub /jiyuan/servo_commands std_msgs/Float64MultiArray "data: [1500, 1500, 1500, ...]"  # 1500us = 中位

# 2. 测量实际角度
# 使用水平仪或角度计测量舵机实际角度

# 3. 计算偏移
# 偏移 = 理论零位 - 实际测量值
# 例如：理论零位 = 90°，实际测量 = 92°，则偏移 = -2° = -0.0349 rad

# 4. 记录到 servo_config.yaml
nano configs/servo_config.yaml
# 更新 left_ankle.servo_1.offset: -0.0349
```

#### 步骤 3：批量校准（推荐）

创建校准脚本 `scripts/calibrate_servos.py`：

```python
#!/usr/bin/env python3
"""
舵机校准脚本

交互式校准所有舵机，自动生成 servo_config.yaml
"""

import yaml
import numpy as np

def calibrate_servo(servo_id, servo_name):
    """校准单个舵机"""
    print(f"\n=== 校准舵机 {servo_id}: {servo_name} ===")
    print("1. 发送零位命令到舵机（1500us）")
    print("2. 使用角度计测量实际角度")

    # 等待用户输入
    measured_angle_deg = float(input("请输入测量的角度（度）: "))
    theoretical_angle_deg = 90.0  # 假设零位为90度

    # 计算偏移（弧度）
    offset_deg = theoretical_angle_deg - measured_angle_deg
    offset_rad = np.deg2rad(offset_deg)

    print(f"计算偏移: {offset_deg:.2f}° = {offset_rad:.4f} rad")

    return offset_rad

def main():
    """主校准流程"""
    print("=" * 60)
    print("双足机器人 机器人舵机校准工具")
    print("=" * 60)

    # 加载现有配置
    config_path = "configs/servo_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 校准左脚踝
    print("\n【左脚踝校准】")
    for servo_key in ["servo_1", "servo_2", "servo_3"]:
        servo_id = config["left_ankle"][servo_key]["id"]
        servo_name = config["left_ankle"][servo_key]["name"]
        offset = calibrate_servo(servo_id, servo_name)
        config["left_ankle"][servo_key]["offset"] = float(offset)
        config["calibration"]["values"][servo_id] = {
            "measured_offset": float(offset),
            "note": f"{servo_name} - 已校准"
        }

    # 校准右脚踝
    print("\n【右脚踝校准】")
    for servo_key in ["servo_1", "servo_2", "servo_3"]:
        servo_id = config["right_ankle"][servo_key]["id"]
        servo_name = config["right_ankle"][servo_key]["name"]
        offset = calibrate_servo(servo_id, servo_name)
        config["right_ankle"][servo_key]["offset"] = float(offset)
        config["calibration"]["values"][servo_id] = {
            "measured_offset": float(offset),
            "note": f"{servo_name} - 已校准"
        }

    # 更新校准日期
    from datetime import datetime
    config["calibration"]["date"] = datetime.now().strftime("%Y-%m-%d")

    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print("\n" + "=" * 60)
    print(f"✓ 校准完成！配置已保存到: {config_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

运行校准：

```bash
python scripts/calibrate_servos.py
```

#### 步骤 4：验证校准

```bash
# 发送零位命令，检查机器人姿态是否标准
# 如果仍有偏差，重复校准流程
```

---

## 常见问题

### Q1: 训练时 GPU 显存不足怎么办？

**解决方案**：

```bash
# 方案 1: 降低并行环境数
python scripts/train.py --config configs/train_config.yaml --num_envs 2048

# 方案 2: 使用更小的网络
# 修改 configs/train_config.yaml:
ppo:
  network:
    actor_hidden_dims: [256, 128, 64]  # 原本 [512, 256, 128]
    critic_hidden_dims: [256, 128, 64]
```

### Q2: 如何快速验证配置文件是否正确？

**解决方案**：

```python
# 创建测试脚本 scripts/test_config.py
from jiyuan_tasks.utils.config_loader import (
    load_train_config,
    load_robot_config,
    load_servo_config,
    validate_train_config,
    validate_robot_config,
)

# 测试训练配置
train_cfg = load_train_config("configs/train_config.yaml")
validate_train_config(train_cfg)
print("✓ 训练配置验证通过")

# 测试机器人配置
robot_cfg = load_robot_config("configs/robot_config.yaml")
validate_robot_config(robot_cfg)
print("✓ 机器人配置验证通过")

# 测试舵机配置
servo_cfg = load_servo_config("configs/servo_config.yaml", apply_calibration=True)
print("✓ 舵机配置加载成功")
```

运行测试：

```bash
python scripts/test_config.py
```

### Q3: 命令行参数没有覆盖配置文件怎么办？

**可能原因**：
1. 参数名拼写错误
2. 配置文件中没有对应的键

**调试方法**：

```bash
# 启用详细输出，检查覆盖信息
python scripts/train.py --config configs/train_config.yaml --num_envs 8192
# 应该看到: ✓ 命令行覆盖: environment.num_envs = 8192
```

### Q4: 如何在不修改配置文件的情况下测试不同参数？

**解决方案**：

```bash
# 使用命令行覆盖
python scripts/train.py --config configs/train_config.yaml \
    --num_envs 1024 \
    --max_iterations 10000 \
    --device cuda:1 \
    --headless
```

### Q5: 实体机器人部署时，脚踝动作不正确怎么办？

**可能原因**：
1. 并联脚踝几何参数（l0, l1, l2）不正确
2. 舵机偏移未正确应用
3. ROS 运动学求解器版本不匹配

**调试步骤**：

```bash
# 1. 检查几何参数
nano configs/robot_config.yaml
# 对比 ROS 配置文件 parallel_3dof_controller/config/kinematics_params.yaml

# 2. 检查舵机偏移
python scripts/calibrate_servos.py  # 重新校准

# 3. 测试运动学映射
python -c "
from jiyuan_tasks.utils.sim2real import ParallelAnkleMapper
import numpy as np

mapper = ParallelAnkleMapper(
    ankle_indices={'left': [9, 10, 11], 'right': [12, 13, 14]},
    l0=0.02, l1=0.01, l2=0.03
)

# 测试零位
action = np.zeros(16)
cmds = mapper.map_action(action, 'left')
print('零位舵机命令:', cmds)
"
```

### Q6: 如何调整奖励权重？

**步骤**：

1. 编辑 `configs/train_config.yaml`：

```yaml
rewards:
  velocity_tracking:
    track_lin_vel_xy: 2.0  # 增加线速度跟踪的重要性（原本 1.5）
    track_ang_vel_z: 0.5   # 降低角速度跟踪的重要性（原本 0.75）
    ankle_workspace: -2.0  # 增加工作空间惩罚（原本 -1.0）
```

2. 重新训练：

```bash
python scripts/train.py --config configs/train_config.yaml
```

3. 观察 TensorBoard 曲线：

```bash
tensorboard --logdir=logs
```

### Q7: 如何添加新的任务（如 walking）？

**步骤**：

1. 在 `configs/train_config.yaml` 中添加 walking 任务的奖励权重：

```yaml
rewards:
  # ... 其他任务

  # 行走任务
  walking:
    track_lin_vel_xy: 1.5
    gait_symmetry: 0.5
    feet_air_time: 0.3
    foot_clearance: 0.2
    stumbling: -2.0
    drag: -1.0
    # ... 完整奖励列表
```

2. 使用 walking 任务训练：

```bash
# 修改 configs/train_config.yaml 中的 task: "walking"
# 或使用命令行覆盖
python scripts/train.py --config configs/train_config.yaml --task walking
```

---

## 总结

### 核心原则

1. **配置优先级**：命令行 > 配置文件 > 预定义
2. **最小化修改**：优先使用命令行覆盖，避免频繁修改配置文件
3. **配置版本管理**：将配置文件纳入 Git 管理
4. **舵机偏移管理**：定期校准，记录校准日期
5. **Sim2Real 准备**：训练时启用工作空间约束和领域随机化

### 推荐工作流程

1. **训练阶段**：使用 `configs/train_config.yaml`，快速实验时使用命令行覆盖
2. **调优阶段**：修改配置文件中的奖励权重和超参数
3. **部署准备**：校准舵机，更新 `servo_config.yaml`
4. **实体部署**：使用 `ParallelAnkleMapper` + ROS 节点

### 下一步

- 查阅 [SIM2REAL_AND_ADVANCED_FEATURES.md](./SIM2REAL_AND_ADVANCED_FEATURES.md) 了解高级功能
- 查阅 [QUICKSTART.md](./QUICKSTART.md) 了解快速开始
- 查阅 [ARCHITECTURE.md](./ARCHITECTURE.md) 了解架构设计

---

**维护记录**：
- 2025-01-22：初始版本
- 配置优先级机制文档化
- 舵机校准流程详细说明
