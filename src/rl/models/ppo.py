"""
PPO算法核心组件
实现GAE、PPO损失函数等
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jp
from flax import struct

# ============================================================================================
# ======================================= 数据结构 ============================================
# ============================================================================================


@struct.dataclass
class PPOBatch:
    """PPO训练批次数据

    JAX数组，可JIT编译
    """

    obs: jax.Array  # 观测 (batch, obs_dim)
    actions: jax.Array  # 动作 (batch, action_dim)
    old_log_probs: jax.Array  # 旧策略的对数概率 (batch,)
    advantages: jax.Array  # 优势函数 (batch,)
    returns: jax.Array  # 回报 (batch,)
    values: jax.Array  # 价值估计 (batch,)


# ============================================================================================
# ===================================== END: 数据结构 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 优势估计 (GAE) =======================================
# ============================================================================================


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """计算广义优势估计（GAE） - 循环版本"""
    T = len(rewards)
    advantages = jp.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae * (1.0 - dones[t])
        advantages = advantages.at[t].set(last_gae)

    returns = advantages + values[:-1]
    return advantages, returns


# --------------------------------------------------------------------------------------------


def compute_gae_scan(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """使用jax.lax.scan计算GAE（更高效）"""

    def scan_fn(carry, inp):
        last_gae = carry
        reward, value, next_value, done = inp

        reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        value = jp.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        next_value = jp.nan_to_num(next_value, nan=0.0, posinf=100.0, neginf=-100.0)
        done = jp.clip(done, 0.0, 1.0)

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * last_gae * (1.0 - done)
        gae = jp.nan_to_num(gae, nan=0.0, posinf=100.0, neginf=-100.0)

        return gae, gae

    inputs = (rewards, values[:-1], values[1:], dones)
    _, advantages = jax.lax.scan(
        scan_fn,
        init=0.0,
        xs=jax.tree.map(lambda x: x[::-1], inputs),
        reverse=False,
    )

    advantages = advantages[::-1]
    returns = advantages + values[:-1]

    return advantages, returns


# ============================================================================================
# ===================================== END: 优势估计 (GAE) ====================================
# ============================================================================================


# ============================================================================================
# ======================================= PPO损失函数 =========================================
# ============================================================================================


def ppo_loss(
    params,
    network,
    batch: PPOBatch,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算PPO损失函数"""
    mean, log_std, values_pred = network.apply(params, batch.obs)

    log_std = jp.clip(log_std, -5.0, 2.0)
    std = jp.exp(log_std)
    std = jp.maximum(std, 1e-6)

    log_probs = -0.5 * jp.sum(
        ((batch.actions - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
    )
    log_probs = jp.clip(log_probs, -100.0, 100.0)

    # 1. 策略损失
    log_ratio = log_probs - batch.old_log_probs
    log_ratio = jp.clip(log_ratio, -20.0, 20.0)
    ratio = jp.exp(log_ratio)

    advantages_mean = batch.advantages.mean()
    advantages_std = batch.advantages.std()
    advantages_std = jp.maximum(
        advantages_std, jp.maximum(jp.abs(advantages_mean) * 0.01, 1e-3)
    )
    advantages_normalized = (batch.advantages - advantages_mean) / advantages_std
    advantages_normalized = jp.clip(advantages_normalized, -10.0, 10.0)

    surr1 = ratio * advantages_normalized
    surr2 = (
        jp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_normalized
    )
    policy_loss = -jp.mean(jp.minimum(surr1, surr2))

    # 2. 价值函数损失
    value_loss = jp.mean((values_pred - batch.returns) ** 2)

    # 3. 策略熵
    entropy = 0.5 * jp.mean(jp.sum(log_std + 0.5 * jp.log(2 * jp.pi * jp.e), axis=-1))

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    total_loss = jp.where(
        jp.isnan(total_loss) | jp.isinf(total_loss),
        jp.array(1e6),
        total_loss,
    )

    info = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jp.mean((log_probs - batch.old_log_probs) ** 2) / 2,
        "clip_fraction": jp.mean(jp.abs(ratio - 1.0) > clip_epsilon),
        "ratio_mean": jp.mean(ratio),
        "ratio_std": jp.std(ratio),
        "advantages_mean": jp.mean(batch.advantages),
        "advantages_std": jp.std(batch.advantages),
    }

    return total_loss, info


# ============================================================================================
# ===================================== END: PPO损失函数 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 工具与JIT版本 =======================================
# ============================================================================================


def normalize_advantages(advantages: jax.Array) -> jax.Array:
    """标准化优势函数"""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


# --------------------------------------------------------------------------------------------


def explained_variance(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """计算解释方差"""
    var_y = jp.var(y_true)
    return 1.0 - jp.var(y_true - y_pred) / (var_y + 1e-8)


# --------------------------------------------------------------------------------------------


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
    """准备PPO训练批次"""
    advantages, returns = compute_gae_scan(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    batch = PPOBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        values=values[:-1],
    )

    return batch


# --------------------------------------------------------------------------------------------

# JIT编译的版本
compute_gae_jit = jax.jit(compute_gae_scan, static_argnums=(3, 4))
ppo_loss_jit = jax.jit(ppo_loss, static_argnums=(1, 3, 4, 5))

# ============================================================================================
# ===================================== END: 工具与JIT版本 =====================================
# ============================================================================================
