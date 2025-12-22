# 代码映射关系：MJX → Isaac Lab

本文档详细说明如何将现有的 JAX/MJX 代码转换为 Isaac Lab + PyTorch 代码。

## 核心概念映射

| MJX 概念 | Isaac Lab 概念 | 说明 |
|---------|---------------|------|
| `EnvState` (dataclass) | 环境内部状态（自动管理） | Isaac Lab 不需要显式定义状态类 |
| `MJXBaseEnv.reset()` | `ManagerBasedRLEnv.reset()` | 返回观测和信息字典 |
| `MJXBaseEnv.step()` | `ManagerBasedRLEnv.step()` | 返回标准 Gym 元组 |
| `_get_obs()` | `ObservationManager` | 配置驱动的观测收集 |
| `_compute_reward()` | `RewardManager` | 配置驱动的奖励计算 |
| `_is_done()` | `TerminationManager` | 配置驱动的终止检查 |
| `_sample_command()` | `CommandManager` | 配置驱动的命令生成 |
| `jax.vmap()` | 原生并行（无需显式） | Isaac Lab 自动批处理 |
| `jax.jit()` | `torch.jit.script()` / `torch.compile()` | PyTorch 编译机制 |

## 文件映射清单

### 环境定义

**原始文件**: `src/rl/envs/mjx_base_env.py` (250 行)

**目标文件**: `isaaclab_rl/jiyuan_tasks/envs/jiyuan_base_env.py` (约 100 行)

**转换要点**:
```python
# MJX (原始)
class MJXBaseEnv:
    def reset(self, rng: jax.Array) -> EnvState:
        pipeline_state = self._reset_pipeline(rng)
        obs = self._get_obs(pipeline_state, jp.zeros(self.nu))
        return EnvState(pipeline_state=pipeline_state, obs=obs, ...)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        for _ in range(self.frame_skip):
            pipeline_state = mjx.step(model, state.pipeline_state)
        obs = self._get_obs(pipeline_state, action)
        reward = self._compute_reward(state, action, pipeline_state)
        done = self._is_done(state, pipeline_state)
        return EnvState(...)

# Isaac Lab (目标)
from omni.isaac.lab.envs import ManagerBasedRLEnv

class JiyuanBaseEnv(ManagerBasedRLEnv):
    """继承 ManagerBasedRLEnv，大部分功能由管理器处理"""

    cfg: JiyuanEnvCfg

    def __init__(self, cfg: JiyuanEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.robot = self.scene["robot"]  # 获取机器人资产

    # reset(), step() 等方法由父类自动实现
    # 通过配置文件中的 Managers 定义行为
```

**关键差异**:
1. 不需要手动实现 `reset()` 和 `step()`
2. 状态管理由 Isaac Lab 自动处理
3. 通过配置类定义环境行为

---

**原始文件**: `src/rl/envs/robot_envs.py::VelocityTrackingEnv` (476 行)

**目标文件**: `isaaclab_rl/jiyuan_tasks/envs/cfg/velocity_tracking_cfg.py` (约 300 行)

**转换要点**:
```python
# MJX (原始) - 命令式编程
def _get_obs(self, pipeline_state, action):
    qpos = pipeline_state.qpos
    qvel = pipeline_state.qvel
    base_quat = qpos[3:7]
    joint_pos = qpos[self.actuator_qpos_indices]
    # ... 拼接观测
    return jp.concatenate([base_quat, base_linvel, ...])

def _compute_reward(self, state, action, pipeline_state):
    # 手动计算各项奖励
    lin_vel_error = jp.sum((actual_vel - command) ** 2)
    reward_tracking = jp.exp(-lin_vel_error / 0.25)
    reward_alive = 1.0
    reward_total = w1 * reward_tracking + w2 * reward_alive + ...
    return reward_total

# Isaac Lab (目标) - 声明式配置
from omni.isaac.lab.managers import ObservationGroupCfg, RewardTermCfg

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_quat = ObservationTermCfg(func=mdp.base_quat_w, noise=Unif(-0.02, 0.02))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unif(-0.1, 0.1))
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unif(-0.01, 0.01))
        # ... 其他观测项
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5)
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # ... 其他奖励项
```

**关键差异**:
1. 从命令式代码变为声明式配置
2. 观测和奖励函数单独定义，通过配置组合
3. 自动处理批量并行

### 奖励函数

**原始文件**: `src/rl/rewards/standing_rewards.py` (245 行)

**目标文件**: `isaaclab_rl/jiyuan_tasks/managers/rewards.py` (约 250 行)

**转换要点**:
```python
# MJX (原始)
import jax.numpy as jp

def compute_height_reward(torso_z: jax.Array, target_height: float, tolerance: float = 0.05) -> jax.Array:
    height_error = jp.abs(torso_z - target_height)
    return jp.exp(-height_error / tolerance)

def compute_orientation_reward(quat: jax.Array, tolerance: float = 0.1) -> jax.Array:
    roll, pitch, _ = quat_to_euler(quat)
    orientation_error = jp.square(roll) + jp.square(pitch)
    return jp.exp(-orientation_error / tolerance)

# Isaac Lab (目标)
import torch

def height_reward(env: ManagerBasedRLEnv, target_height: float = 0.3, tolerance: float = 0.05) -> torch.Tensor:
    """高度保持奖励

    Args:
        env: 环境实例（自动传入）
        target_height: 目标高度
        tolerance: 容差参数

    Returns:
        奖励张量 shape: (num_envs,)
    """
    torso_z = env.scene["robot"].data.root_pos_w[:, 2]
    height_error = torch.abs(torso_z - target_height)
    return torch.exp(-height_error / tolerance)

def orientation_reward(env: ManagerBasedRLEnv, tolerance: float = 0.1) -> torch.Tensor:
    """姿态稳定奖励"""
    quat = env.scene["robot"].data.root_quat_w
    roll, pitch, _ = quat_to_euler_xyz(quat)
    orientation_error = torch.square(roll) + torch.square(pitch)
    return torch.exp(-orientation_error / tolerance)
```

**关键差异**:
1. 函数签名：`(state, ...) → (env, ...)`（第一个参数变为 env）
2. 张量库：`jp.` → `torch.`
3. 数据访问：从 `pipeline_state` 变为 `env.scene["robot"].data`
4. 类型注解：`jax.Array` → `torch.Tensor`

### 网络定义

**原始文件**: `src/rl/models/networks.py::ActorCriticNetwork` (174 行)

**目标文件**: RSL_RL 内置网络（通过配置定义）

**转换要点**:
```python
# MJX (原始) - Flax
import flax.linen as nn

class ActorCriticNetwork(nn.Module):
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 256)

    @nn.compact
    def __call__(self, obs):
        # Backbone
        x = nn.Dense(self.hidden_dims[0])(obs)
        x = nn.swish(x)
        x = nn.Dense(self.hidden_dims[1])(x)
        x = nn.swish(x)
        # Actor head
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        # Critic head
        value = nn.Dense(1)(x)
        return mean, log_std, value

# Isaac Lab (目标) - RSL_RL 配置
from rsl_rl.modules import ActorCritic

# 通过配置定义（不需要手动实现）
@configclass
class JiyuanPPORunnerCfg(RslRlPpoRunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        # 网络架构
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",  # RSL_RL 默认使用 ELU
        init_noise_std=1.0,
        # ... PPO 超参数
    )
```

**关键差异**:
1. 不需要手动实现网络类（使用 RSL_RL 内置）
2. 通过配置类定义网络架构
3. Flax → PyTorch（底层，但对用户透明）
4. **注意**: 激活函数从 Swish 变为 ELU（权重不兼容，需从头训练）

### 训练循环

**原始文件**: `scripts/train.py` + `src/rl/training/ppo_trainer.py` (541 行)

**目标文件**: `isaaclab_rl/scripts/train.py` (约 80 行)

**转换要点**:
```python
# MJX (原始) - 手动实现训练循环
import jax
from rl.training.ppo_trainer import PPOTrainer, PPOConfig

# 创建环境
env = create_velocity_tracking_env(...)
env_state = env.batch_reset(rng, batch_size=4096)

# 创建网络和训练器
network = ActorCriticNetwork(...)
config = PPOConfig(num_envs=4096, num_steps=64, ...)
trainer = PPOTrainer(config, env, network, optimizer)

# 手动训练循环
train_step_jit = jax.jit(trainer.train_step)
for update in range(num_updates):
    train_state, env_state, info = train_step_jit(train_state, env_state)
    if update % 10 == 0:
        logger.log(info)
    if update % 100 == 0:
        save_checkpoint(train_state, f"model_{update}.pkl")

# Isaac Lab (目标) - 使用 RSL_RL 封装
import gymnasium as gym
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunner

# 创建环境（Gym API）
env = gym.make("Isaac-Jiyuan-Velocity-Tracking-v0", num_envs=4096)
env = RslRlVecEnvWrapper(env)

# 创建 RSL_RL 训练器（自动管理网络、优化器、训练循环）
runner = RslRlOnPolicyRunner(
    env=env,
    train_cfg=JiyuanPPORunnerCfg(),
    log_dir="logs/velocity_tracking",
    device="cuda:0"
)

# 训练（一行代码）
runner.learn(num_learning_iterations=30_000_000 // (4096 * 64), init_at_random_ep_len=True)
```

**关键差异**:
1. 训练循环由 RSL_RL 自动管理
2. 检查点保存/加载自动化
3. TensorBoard 日志自动记录
4. 代码量减少 ~85%

## 配置文件映射

### 训练配置

**原始文件**: `configs/train.yaml`

**目标位置**: `isaaclab_rl/agents/rsl_rl/ppo_cfg.py` (Python 配置类)

**转换要点**:
```yaml
# MJX (原始) - YAML
num_envs: 4096
num_steps: 64
num_epochs: 32
num_minibatches: 32
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
# ...
```

```python
# Isaac Lab (目标) - Python 配置类
from rsl_rl.runners import OnPolicyRunnerCfg
from rsl_rl.algorithms import PPOCfg

@configclass
class JiyuanPPORunnerCfg(OnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 30_000_000 // (4096 * 64)  # 约 114 次迭代
    save_interval = 50
    experiment_name = "jiyuan_velocity_tracking"
    empirical_normalization = False
    policy = PPOCfg(
        num_learning_epochs=32,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        gamma=0.99,
        lam=0.95,  # GAE lambda
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
    )
```

**关键差异**:
1. YAML → Python dataclass（类型安全）
2. 参数名略有不同（如 `gae_lambda` → `lam`）
3. 配置类可继承和扩展

## 数据类型转换

### 张量操作

| MJX (JAX) | Isaac Lab (PyTorch) | 说明 |
|-----------|---------------------|------|
| `jp.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` | 创建张量 |
| `jp.zeros((10, 3))` | `torch.zeros(10, 3)` | 创建零张量 |
| `jp.concatenate([a, b], axis=1)` | `torch.cat([a, b], dim=1)` | 拼接张量 |
| `jp.linalg.norm(x)` | `torch.norm(x)` | 范数计算 |
| `jp.exp(x)` | `torch.exp(x)` | 指数函数 |
| `jp.clip(x, -1, 1)` | `torch.clamp(x, -1, 1)` | 裁剪 |
| `jp.where(cond, a, b)` | `torch.where(cond, a, b)` | 条件选择 |
| `jp.nan_to_num(x)` | `torch.nan_to_num(x)` | NaN 处理 |

### 随机数生成

| MJX (JAX) | Isaac Lab (PyTorch) |
|-----------|---------------------|
| `jax.random.PRNGKey(42)` | `torch.manual_seed(42)` |
| `jax.random.split(rng, 3)` | `torch.Generator().manual_seed(...)` |
| `jax.random.uniform(rng, shape)` | `torch.rand(shape)` |
| `jax.random.normal(rng, shape)` | `torch.randn(shape)` |

## 关键转换步骤总结

### 步骤 1：复制奖励函数（最简单）

```bash
# 1. 复制文件
cp src/rl/rewards/standing_rewards.py \
   isaaclab_rl/jiyuan_tasks/managers/rewards.py

# 2. 全局替换
# - import jax.numpy as jp → import torch
# - jp. → torch.
# - jax.Array → torch.Tensor
# - 函数签名添加 env 参数

# 3. 修改数据访问
# - 从 env.scene["robot"].data 获取状态
```

### 步骤 2：创建环境配置（中等难度）

```bash
# 1. 创建配置文件
touch isaaclab_rl/jiyuan_tasks/envs/cfg/velocity_tracking_cfg.py

# 2. 定义场景配置（JiyuanSceneCfg）
# - 加载 MJCF 模型
# - 定义执行器参数

# 3. 定义 MDP 配置
# - ObservationsCfg: 从 _get_obs() 映射
# - RewardsCfg: 从 _compute_reward() 映射
# - ActionsCfg: 动作空间定义
# - TerminationsCfg: 从 _is_done() 映射
# - EventCfg: 从 _reset_pipeline() 映射
```

### 步骤 3：创建训练脚本（最简单）

```bash
# 1. 复制 Isaac Lab 官方模板
cp ${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl/train.py \
   isaaclab_rl/scripts/train.py

# 2. 修改环境名称
# "Isaac-Cartpole-v0" → "Isaac-Jiyuan-Velocity-Tracking-v0"

# 3. 修改配置路径
# 指向 JiyuanPPORunnerCfg
```

## 验证转换正确性

### 单元测试对比

```python
# 测试奖励函数数值一致性
import jax.numpy as jp
import torch
from src.rl.rewards import standing_rewards as mjx_rewards
from isaaclab_rl.jiyuan_tasks.managers import rewards as isaaclab_rewards

# 创建测试数据
torso_z_jax = jp.array([0.3, 0.35, 0.25, 0.4])
torso_z_torch = torch.tensor([0.3, 0.35, 0.25, 0.4])

# 计算奖励
reward_mjx = mjx_rewards.compute_height_reward(torso_z_jax, target_height=0.3)
reward_isaaclab = isaaclab_rewards.height_reward(
    mock_env_with_height(torso_z_torch), target_height=0.3
)

# 验证数值一致性
assert torch.allclose(
    torch.from_numpy(np.array(reward_mjx)),
    reward_isaaclab,
    atol=1e-6
), "Height reward mismatch!"
```

## 下一步

完成代码映射理解后：

1. 阅读 [04-implementation-phases.md](./04-implementation-phases.md) 了解分阶段实施计划
2. 开始阶段 1 的环境搭建
3. 使用 [checklists/phase2-checklist.md](./checklists/phase2-checklist.md) 验证核心功能迁移
