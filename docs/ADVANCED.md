# Sim2Real 和高级功能指南

本指南说明如何使用 3-parallel-dot 脚踝映射、行走任务和模仿学习功能。

## 目录

1. [3-parallel-dot 脚踝结构集成](#1-3-parallel-dot-脚踝结构集成)
2. [行走任务配置](#2-行走任务配置)
3. [模仿学习（BVH/FBX）](#3-模仿学习bvhfbx)
4. [完整工作流程](#4-完整工作流程)

---

## 1. 3-parallel-dot 脚踝结构集成

### 问题背景

实体机器人脚踝使用 **3-DOF 并联机构**（3-parallel-dot），而仿真中使用简化的串联模型。

**并联结构特性：**
- 输入：RPY 姿态角 (roll, pitch, yaw)
- 输出：3 个舵机位置命令
- 工作空间：±30° (roll/pitch/yaw)
- ROS 映射算法：`parallel_3dof_controller/kinematics_solver.py`

### 推荐策略：训练时简化，部署时映射

**训练阶段（仿真）：**
```python
# 1. 使用简化的串联关节模型（如当前 MJCF）
# 2. 策略直接输出 ankle RPY 角度
# 3. 添加工作空间软约束

from isaaclab_rl.jiyuan_tasks.utils import sim2real

# 在奖励配置中添加
@configclass
class RewardsCfg:
    # ... 其他奖励 ...

    # 脚踝工作空间约束（Sim2Real 准备）
    ankle_workspace = RewTerm(
        func=sim2real.ankle_workspace_penalty,
        weight=-1.0,
        params={
            "ankle_indices": {
                "left": [9, 10, 11],    # 左脚踝 roll/pitch/yaw 索引
                "right": [12, 13, 14],  # 右脚踝 roll/pitch/yaw 索引
            # 这并不是写死的
            },
            "margin": 0.1,  # 安全边界（弧度）
        },
    )
```

**部署阶段（实体机器人）：**
```python
from isaaclab_rl.jiyuan_tasks.utils.sim2real import ParallelAnkleMapper

# 1. 创建映射器
mapper = ParallelAnkleMapper(
    ankle_indices={
        "left": [9, 10, 11],    # 左脚踝在动作空间中的索引
        "right": [12, 13, 14],  # 右脚踝
    },
    l0=0.02,  # 平台半径（从 ROS 配置读取）
    l1=0.01,  # 动平台距离
    l2=0.03,  # 静平台距离
    enable_filtering=True,  # 启用低通滤波
    filter_alpha=0.7,  # 滤波系数
)

# 2. 策略推理
obs = get_observation_from_sensors()
action = policy.get_action(obs)  # 形状 (action_dim,)

# 3. 映射到舵机命令
left_servo_cmds = mapper.map_action(action, 'left')
right_servo_cmds = mapper.map_action(action, 'right')

# 4. 发送到硬件
for cmd in left_servo_cmds + right_servo_cmds:
    servo_controller.set_position(
        id=cmd['id'],
        position=cmd['position'],
        speed=cmd['speed']
    )
```

### 关键参数说明

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `ankle_indices` | 脚踝关节在动作空间中的索引 | - | **必须根据实际模型设置** |
| `l0, l1, l2` | 并联机构几何参数 (m) | 0.02, 0.01, 0.03 | **从 ROS 配置或 CAD 获取** |
| `enable_filtering` | 是否启用低通滤波 | True | 实体机器人建议启用 |
| `filter_alpha` | 滤波系数（0-1） | 0.7 | 越大越平滑，但响应越慢 |
| `margin` | 工作空间安全边界（弧度） | 0.1 | 根据实测调整 |

### 工作空间约束验证

训练后检查策略是否遵守工作空间限制：

```python
import matplotlib.pyplot as plt

# 收集 1000 步数据
actions_history = []
for step in range(1000):
    action = policy.get_action(obs)
    actions_history.append(action)
    obs, _, _, _, _ = env.step(action)

actions_history = np.array(actions_history)

# 提取脚踝角度
left_ankle_rpy = actions_history[:, [9, 10, 11]]
right_ankle_rpy = actions_history[:, [12, 13, 14]]

# 绘制分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
    axes[0, i].hist(np.degrees(left_ankle_rpy[:, i]), bins=50)
    axes[0, i].set_title(f'Left Ankle {name}')
    axes[0, i].axvline(-30, color='r', linestyle='--', label='Limit')
    axes[0, i].axvline(30, color='r', linestyle='--')

    axes[1, i].hist(np.degrees(right_ankle_rpy[:, i]), bins=50)
    axes[1, i].set_title(f'Right Ankle {name}')
    axes[1, i].axvline(-30, color='r', linestyle='--')
    axes[1, i].axvline(30, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('ankle_workspace_distribution.png')
print(f"✅ 工作空间分布图已保存")
```

---

## 2. 行走任务配置

### 行走 vs 速度跟踪

| 特性 | 速度跟踪 (Velocity Tracking) | 行走 (Walking) |
|------|------------------------------|----------------|
| 主要目标 | 跟踪任意速度命令 | 稳定的步态生成 |
| 步态质量 | 不强制要求 | 要求对称、周期性 |
| 奖励函数 | 速度误差为主 | 步态质量 + 速度跟踪 |
| 使用场景 | 通用运动控制 | 类人行走、参数化步态 |

### 创建行走环境

```python
from isaaclab_rl.jiyuan_tasks.envs.cfg import VelocityTrackingEnvCfg
from isaaclab_rl.jiyuan_tasks.managers import walking_rewards
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

@configclass
class WalkingEnvCfg(VelocityTrackingEnvCfg):
    """行走任务环境配置

    基于速度跟踪环境，增强步态质量奖励。
    """

    @configclass
    class RewardsCfg:
        # 主要目标：速度跟踪
        track_lin_vel_xy = RewTerm(
            func=rewards.track_lin_vel_xy_exp,
            weight=1.5,
            params={"std": 0.5},
        )

        track_ang_vel_z = RewTerm(
            func=rewards.track_ang_vel_z_exp,
            weight=0.5,
            params={"std": 0.5},
        )

        # 步态质量（核心）
        gait_symmetry = RewTerm(
            func=walking_rewards.gait_symmetry_reward,
            weight=0.5,
            params={"sensor_cfg_name": "contact_forces"},
        )

        feet_air_time = RewTerm(
            func=walking_rewards.feet_air_time_reward,
            weight=0.3,
            params={"target_air_time": 0.5},  # 目标离地时间
        )

        foot_clearance = RewTerm(
            func=walking_rewards.foot_clearance_reward,
            weight=0.2,
            params={"target_clearance": 0.05},  # 目标抬脚高度 5cm
        )

        # 步态惩罚
        stumbling = RewTerm(
            func=walking_rewards.stumbling_penalty,
            weight=-2.0,
        )

        drag = RewTerm(
            func=walking_rewards.drag_penalty,
            weight=-1.0,
        )

        # 躯干稳定（行走专用）
        trunk_height = RewTerm(
            func=walking_rewards.trunk_height_reward,
            weight=0.5,
            params={"target_height": 0.35, "tolerance": 0.05},
        )

        trunk_tilt = RewTerm(
            func=walking_rewards.trunk_orientation_penalty,
            weight=-0.5,
            params={"max_tilt": 0.3},  # 允许 ~17° 倾斜
        )

        trunk_lin_vel_z = RewTerm(
            func=walking_rewards.trunk_lin_vel_z_penalty,
            weight=-1.0,
        )

        # 能量效率
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        joint_powers = RewTerm(func=rewards.joint_powers_l1, weight=-2.0e-5)

        # 存活
        alive = RewTerm(func=mdp.is_alive, weight=0.5)

# 注册环境
gym.register(
    id="Isaac-Jiyuan-Walking-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": WalkingEnvCfg},
)
```

### 环境状态跟踪（可选增强）

某些行走奖励函数需要环境维护额外状态：

```python
class WalkingEnv(ManagerBasedRLEnv):
    """行走环境（增强版）"""

    def __init__(self, cfg: WalkingEnvCfg):
        super().__init__(cfg)

        # 步态跟踪状态
        self.feet_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.feet_stance_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.feet_in_swing_phase = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.feet_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self.step_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_time = torch.zeros(self.num_envs, device=self.device)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)

        # 更新步态状态
        self._update_gait_state()

        return obs, rewards, terminated, truncated, info

    def _update_gait_state(self):
        """更新步态跟踪状态"""
        # 检测脚部接触
        contact_forces = self.scene.sensors["contact_forces"].data.net_forces_w
        contact_detected = torch.norm(contact_forces, dim=-1) > 1.0

        # 更新离地时间
        self.feet_air_time += self.dt * (~contact_detected).float()
        self.feet_air_time *= (~contact_detected).float()  # 接触时清零

        # 更新支撑时间
        self.feet_stance_time += self.dt * contact_detected.float()
        self.feet_stance_time *= contact_detected.float()  # 离地时清零

        # 更新脚部位置（需要从机器人状态提取）
        # TODO: 实现脚部位置提取

        # 更新 episode 时间
        self.episode_time += self.dt
```

---

## 3. 模仿学习（BVH/FBX）

### 使用场景

- **学习人类步态**：从动作捕捉数据学习自然的行走模式
- **复杂动作序列**：跳跃、转身、蹲下等
- **数据驱动生成**：基于参考动作库的运动生成

### 工作流程

```
BVH/FBX 文件 → 骨骼重定向 → ReferenceMotion → 模仿奖励 → 训练
```

### 步骤 1：准备参考动作

```python
from isaaclab_rl.jiyuan_tasks.utils.imitation import ReferenceMotion
import numpy as np

# 手动创建示例动作（实际应从 BVH/FBX 加载）
num_frames = 120  # 2秒 @ 60fps
num_joints = 16

# 创建简单的行走轨迹（示例）
t = np.linspace(0, 2*np.pi, num_frames)
joint_positions = np.zeros((num_frames, num_joints))

# 髋关节摆动（简化示例）
joint_positions[:, 0] = 0.3 * np.sin(t)  # 左髋
joint_positions[:, 1] = 0.3 * np.sin(t + np.pi)  # 右髋

# 膝关节弯曲
joint_positions[:, 2] = 0.5 + 0.2 * np.cos(t)  # 左膝
joint_positions[:, 3] = 0.5 + 0.2 * np.cos(t + np.pi)  # 右膝

# 创建参考动作对象
reference_motion = ReferenceMotion(
    joint_positions=joint_positions,
    fps=60.0
)

print(f"参考动作时长: {reference_motion.duration:.2f}s")
```

### 步骤 2：配置模仿学习环境

```python
from isaaclab_rl.jiyuan_tasks.utils.imitation import (
    pose_matching_reward,
    root_pose_matching_reward,
    update_motion_phase,
)

@configclass
class ImitationWalkEnvCfg(VelocityTrackingEnvCfg):
    """模仿学习行走环境"""

    @configclass
    class RewardsCfg:
        # 模仿奖励（主要目标）
        pose_matching = RewTerm(
            func=pose_matching_reward,
            weight=2.0,
            params={
                "reference_motion": reference_motion,
                "phase_variable_name": "motion_phase",
                "weight_pos": 1.0,
                "weight_vel": 0.1,
            },
        )

        root_pose_matching = RewTerm(
            func=root_pose_matching_reward,
            weight=1.0,
            params={
                "reference_motion": reference_motion,
            },
        )

        # 正则化奖励（避免过拟合）
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        joint_powers = RewTerm(func=rewards.joint_powers_l1, weight=-1.0e-5)
```

### 步骤 3：环境集成

```python
class ImitationWalkEnv(ManagerBasedRLEnv):
    """模仿学习环境"""

    def __init__(self, cfg: ImitationWalkEnvCfg):
        super().__init__(cfg)

        # 初始化相位变量
        self.motion_phase = torch.zeros(self.num_envs, device=self.device)
        self.motion_duration = reference_motion.duration

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)

        # 更新动作相位
        update_motion_phase(self, self.dt, loop=True)

        return obs, rewards, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        # 随机初始化相位（增加多样性）
        self.motion_phase = torch.rand(self.num_envs, device=self.device) * self.motion_duration

        return obs, info
```

### 步骤 4：BVH/FBX 加载（待实现）

```python
# TODO: 实现 BVH 加载和重定向
from isaaclab_rl.jiyuan_tasks.utils.imitation import load_motion_from_bvh

# 加载 BVH 文件
reference_motion = load_motion_from_bvh("path/to/walk_cycle.bvh")

# 应用骨骼重定向（人体 → 机器人）
# 这需要定义关节映射:
joint_mapping = {
    "human_left_hip": "robot_left_hip_pitch",
    "human_right_hip": "robot_right_hip_pitch",
    # ... 完整映射
}

# retarget_motion(reference_motion, joint_mapping)
```

**推荐库：**
- `python-bvh`: `pip install bvh`
- `pybvh`: `pip install pybvh`
- `Autodesk FBX SDK`: 用于 FBX 文件

---

## 4. 完整工作流程

### 工作流程 A：标准速度跟踪 → Sim2Real

```bash
# 1. 训练速度跟踪策略（带工作空间约束）
python scripts/train.py --task velocity --num_envs 4096

# 2. 评估训练结果
python scripts/play.py --task velocity --deterministic

# 3. 验证工作空间分布
python scripts/analyze_ankle_workspace.py --checkpoint logs/.../model_30000.pt

# 4. 导出策略（ONNX 或 JIT）
python scripts/export_policy.py --checkpoint logs/.../model_30000.pt --format onnx

# 5. 部署到实体机器人
# 在实体机器人代码中：
# - 加载策略：policy = onnx.load("policy.onnx")
# - 创建映射器：mapper = ParallelAnkleMapper(...)
# - 运行控制循环：action = policy(obs) → servo_commands = mapper.map_action(action)
```

### 工作流程 B：行走任务 → 步态优化

```bash
# 1. 训练行走策略（强化步态质量）
python scripts/train.py --task walking --num_envs 4096

# 2. 评估步态质量
python scripts/play.py --task walking --num_episodes 50

# 3. 分析步态指标
python scripts/analyze_gait.py --checkpoint logs/.../model_20000.pt
# 输出：步态周期、对称性、能量效率等

# 4. 调优奖励权重
# 根据分析结果调整 WalkingEnvCfg 中的权重，重新训练
```

### 工作流程 C：模仿学习 → 参考动作

```bash
# 1. 准备参考动作（BVH → ReferenceMotion）
python scripts/convert_bvh.py --input walk.bvh --output walk.pkl

# 2. 训练模仿策略
python scripts/train_imitation.py --reference walk.pkl --num_envs 2048

# 3. 评估模仿质量
python scripts/play_imitation.py --checkpoint logs/.../model_15000.pt --show_reference

# 4. 混合训练（模仿 + 任务奖励）
# 在环境配置中同时启用 pose_matching 和 track_lin_vel_xy
python scripts/train.py --task imitation_velocity --num_envs 4096
```

---

## 总结

### 核心设计原则

1. **训练时简化，部署时映射** - 仿真使用串联模型，实体使用并联映射
2. **软约束优于硬约束** - 用奖励惩罚引导，而非硬性限制
3. **模块化和可组合** - 行走、模仿等功能可独立或组合使用
4. **渐进式训练** - 站立 → 速度跟踪 → 行走 → 模仿学习

### 下一步

- [ ] 实现 BVH/FBX 加载器和重定向工具
- [ ] 添加步态分析脚本
- [ ] 优化行走奖励函数权重
- [ ] 实体机器人部署验证
- [ ] 收集实际行走数据并微调
