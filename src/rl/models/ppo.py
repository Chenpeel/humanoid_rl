"""
PPO算法核心组件
实现GAE、PPO损失函数等
"""

import jax
import jax.numpy as jp
from typing import Tuple, Dict
from flax import struct


@struct.dataclass
class PPOBatch:
    """PPO训练批次数据

    JAX数组，可JIT编译
    """
    obs: jax.Array          # 观测 (batch, obs_dim)
    actions: jax.Array      # 动作 (batch, action_dim)
    old_log_probs: jax.Array  # 旧策略的对数概率 (batch,)
    advantages: jax.Array   # 优势函数 (batch,)
    returns: jax.Array      # 回报 (batch,)
    values: jax.Array       # 价值估计 (batch,)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """计算广义优势估计（GAE）

    GAE(δ_t) = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards: 奖励序列 (T,)
        values: 价值估计序列 (T+1,) - 包含最后一个next_value
        dones: 终止标志序列 (T,)
        gamma: 折扣因子
        gae_lambda: GAE的lambda参数

    Returns:
        advantages: 优势函数 (T,)
        returns: 回报 (T,)
    """
    T = len(rewards)
    advantages = jp.zeros(T)
    last_gae = 0.0

    # 反向计算GAE
    for t in reversed(range(T)):
        # TD误差: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        next_value = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE: A_t = δ_t + (γλ)A_{t+1}(1-done)
        last_gae = delta + gamma * gae_lambda * last_gae * (1.0 - dones[t])
        advantages = advantages.at[t].set(last_gae)

    # 回报 = 优势 + 价值
    returns = advantages + values[:-1]

    return advantages, returns


def compute_gae_scan(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """使用jax.lax.scan计算GAE（更高效）

    Args:
        rewards: 奖励序列 (T,)
        values: 价值估计序列 (T+1,)
        dones: 终止标志序列 (T,)
        gamma: 折扣因子
        gae_lambda: GAE的lambda参数

    Returns:
        advantages: 优势函数 (T,)
        returns: 回报 (T,)
    """
    def scan_fn(carry, inp):
        """scan函数：从后往前计算GAE"""
        last_gae = carry
        reward, value, next_value, done = inp

        # TD误差
        delta = reward + gamma * next_value * (1.0 - done) - value

        # GAE
        gae = delta + gamma * gae_lambda * last_gae * (1.0 - done)

        return gae, gae

    # 准备输入：(reward, value, next_value, done)
    inputs = (rewards, values[:-1], values[1:], dones)

    # 反向扫描
    _, advantages = jax.lax.scan(
        scan_fn,
        init=0.0,
        xs=jax.tree.map(lambda x: x[::-1], inputs),
        reverse=False,
    )

    # 反转回来
    advantages = advantages[::-1]

    # 回报 = 优势 + 价值
    returns = advantages + values[:-1]

    return advantages, returns


def ppo_loss(
    params,
    network,
    batch: PPOBatch,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算PPO损失函数

    L = L^CLIP + c_1 * L^VF - c_2 * S[π](s)

    其中：
    - L^CLIP: PPO裁剪的策略损失
    - L^VF: 价值函数损失（MSE）
    - S[π]: 策略熵（探索奖励）

    Args:
        params: 网络参数
        network: Actor-Critic网络
        batch: 训练批次数据
        clip_epsilon: PPO裁剪参数（通常0.1-0.3）
        value_coef: 价值函数损失系数
        entropy_coef: 熵正则化系数

    Returns:
        total_loss: 总损失
        info: 各项损失的详细信息字典
    """
    # 前向传播：获取新策略的log_prob和value
    mean, log_std, values_pred = network.apply(params, batch.obs)

    # 安全裁剪log_std（防止极端值）
    log_std = jp.clip(log_std, -5.0, 2.0)  # 缩小范围提高稳定性
    std = jp.exp(log_std)
    std = jp.maximum(std, 1e-6)  # 防止除零

    # 计算新策略的对数概率（数值稳定版本）
    log_probs = -0.5 * jp.sum(
        ((batch.actions - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi),
        axis=-1
    )
    # 裁剪对数概率防止极端值
    log_probs = jp.clip(log_probs, -100.0, 100.0)

    # 1. PPO裁剪的策略损失
    # ratio = π_new / π_old = exp(log π_new - log π_old)
    log_ratio = log_probs - batch.old_log_probs
    # 裁剪log_ratio防止exp溢出
    log_ratio = jp.clip(log_ratio, -20.0, 20.0)
    ratio = jp.exp(log_ratio)

    # 优势标准化（改进数值稳定性）
    advantages_mean = batch.advantages.mean()
    advantages_std = batch.advantages.std()
    # 使用更大的epsilon防止除零
    advantages_std = jp.maximum(advantages_std, 1e-4)
    advantages_normalized = (batch.advantages - advantages_mean) / advantages_std
    # 裁剪标准化后的优势值
    advantages_normalized = jp.clip(advantages_normalized, -10.0, 10.0)

    # PPO裁剪目标
    surr1 = ratio * advantages_normalized
    surr2 = jp.clip(ratio, 1.0 - clip_epsilon, 1.0 +
                    clip_epsilon) * advantages_normalized
    policy_loss = -jp.mean(jp.minimum(surr1, surr2))

    # 2. 价值函数损失（MSE）
    value_loss = jp.mean((values_pred - batch.returns) ** 2)

    # 3. 策略熵（鼓励探索）
    # H[π] = E[-log π] = E[0.5 * (log(2πσ^2) + 1)]
    entropy = 0.5 * jp.mean(jp.sum(log_std + 0.5 *
                            jp.log(2 * jp.pi * jp.e), axis=-1))

    # 总损失
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # 返回详细信息
    info = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        # 近似KL散度
        'approx_kl': jp.mean((log_probs - batch.old_log_probs) ** 2) / 2,
        'clip_fraction': jp.mean(jp.abs(ratio - 1.0) > clip_epsilon),  # 被裁剪的比例
        'ratio_mean': jp.mean(ratio),
        'ratio_std': jp.std(ratio),
        'advantages_mean': jp.mean(batch.advantages),
        'advantages_std': jp.std(batch.advantages),
    }

    return total_loss, info


def normalize_advantages(advantages: jax.Array) -> jax.Array:
    """标准化优势函数

    Args:
        advantages: 优势函数 (batch,)

    Returns:
        标准化后的优势函数 (batch,)
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def explained_variance(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """计算解释方差（用于评估价值函数的拟合质量）

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Args:
        y_pred: 预测值 (batch,)
        y_true: 真实值 (batch,)

    Returns:
        解释方差（接近1表示拟合好）
    """
    var_y = jp.var(y_true)
    return 1.0 - jp.var(y_true - y_pred) / (var_y + 1e-8)


# ==================== JIT编译的版本 ====================

# JIT编译的GAE计算（使用scan版本）
compute_gae_jit = jax.jit(compute_gae_scan, static_argnums=(3, 4))

# JIT编译的PPO损失计算
ppo_loss_jit = jax.jit(ppo_loss, static_argnums=(1, 3, 4, 5))


# ==================== 便捷函数 ====================

def prepare_ppo_batch(
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    old_log_probs: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> PPOBatch:
    """准备PPO训练批次

    Args:
        obs: 观测序列 (T, obs_dim)
        actions: 动作序列 (T, action_dim)
        rewards: 奖励序列 (T,)
        dones: 终止标志 (T,)
        values: 价值估计 (T+1,) - 包含next_value
        old_log_probs: 旧策略对数概率 (T,)
        gamma: 折扣因子
        gae_lambda: GAE lambda

    Returns:
        PPOBatch数据
    """
    # 计算GAE和returns
    advantages, returns = compute_gae_scan(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # 创建batch
    batch = PPOBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        values=values[:-1],  # 去掉最后一个next_value
    )

    return batch
