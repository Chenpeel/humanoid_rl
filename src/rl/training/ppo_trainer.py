"""
PPO训练器
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp

from ..envs.mjx_base_env import EnvState
from ..models.ppo import PPOBatch, compute_gae_scan, ppo_loss
from .train_state import TrainState


@dataclass
class PPOConfig:
    """PPO训练配置"""

    # 环境配置
    num_envs: int = 2048  # 并行环境数
    num_steps: int = 100  # 每次rollout的步数

    # PPO超参数
    num_epochs: int = 4  # 每次update的epoch数
    num_minibatches: int = 8  # mini-batch数量
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO裁剪系数
    value_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵正则化系数
    max_grad_norm: float = 0.5  # 梯度裁剪

    # 训练配置
    total_timesteps: int = 10_000_000  # 总训练步数
    log_interval: int = 10  # 日志记录间隔（更新次数）
    eval_interval: int = 100  # 评估间隔（更新次数）

    @property
    def batch_size(self) -> int:
        """总批次大小"""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Mini-batch大小"""
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """总更新次数"""
        return self.total_timesteps // self.batch_size


class PPOTrainer:
    """PPO训练器"""

    def __init__(
        self,
        config: PPOConfig,
        env,
        network,
        optimizer,
    ):
        """初始化PPO训练器

        Args:
            config: PPO配置
            env: MJX环境
            network: Actor-Critic网络
            optimizer: Optax优化器
        """
        self.config = config
        self.env = env
        self.network = network
        self.optimizer = optimizer

    def collect_trajectory(
        self,
        train_state: TrainState,
        env_state: EnvState,
    ) -> Tuple[PPOBatch, EnvState, Dict[str, Any]]:
        """收集轨迹数据（rollout）

        Args:
            train_state: 训练状态
            env_state: 环境状态

        Returns:
            (PPOBatch, 新的env_state, info字典)
        """

        def scan_fn(carry, _):
            """扫描函数：执行一步环境交互"""
            state, e_state, rng = carry

            # 获取动作和价值（一次前向传播）
            rng, action_rng = jax.random.split(rng)
            mean, log_std, value = self.network.apply(state.params, e_state.obs)

            # 数值稳定性保护
            log_std = jp.clip(log_std, -5.0, 2.0)
            std = jp.exp(log_std)
            std = jp.maximum(std, 1e-6)  # 防止除零

            # 采样动作
            action = mean + std * jax.random.normal(action_rng, mean.shape)
            action = jp.clip(action, -10.0, 10.0)  # 裁剪动作防止极端值

            # 计算log概率（增强数值稳定性）
            action_diff = jp.clip((action - mean) / std, -100.0, 100.0)
            log_prob = -0.5 * jp.sum(
                action_diff ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
            )
            log_prob = jp.clip(log_prob, -1000.0, 100.0)  # 裁剪 log_prob

            # 环境步进（e_state已经是批量状态）
            new_e_state = self.env.batch_step(e_state, action)

            # 收集转移数据
            transition = {
                "obs": e_state.obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": new_e_state.reward,
                "done": new_e_state.done,
            }

            return (state, new_e_state, rng), transition

        # 执行rollout
        rng, _ = train_state.split_rng()
        _, transitions = jax.lax.scan(
            scan_fn,
            init=(train_state, env_state, rng),
            xs=None,
            length=self.config.num_steps,
        )

        # 提取数据 (num_steps, num_envs, ...)
        obs = transitions["obs"]
        actions = transitions["action"]
        log_probs = transitions["log_prob"]
        values = transitions["value"]
        rewards = transitions["reward"]
        dones = transitions["done"]

        # 计算最后一步的价值（bootstrap）
        # 注意：env_state.obs是批量观测，需要为每个环境计算价值
        _, _, last_value = self.network.apply(train_state.params, env_state.obs)

        # 拼接价值序列
        values_with_last = jp.concatenate([values, last_value[None, :]], axis=0)

        # 计算GAE（对每个环境分别计算）
        advantages, returns = jax.vmap(
            lambda r, v, d: compute_gae_scan(
                r, v, d, self.config.gamma, self.config.gae_lambda
            ),
            in_axes=1,
            out_axes=1,  # 对每个环境分别计算
        )(rewards, values_with_last, dones)

        # Flatten batch (num_steps * num_envs, ...)
        obs_flat = obs.reshape(-1, *obs.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        log_probs_flat = log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values.reshape(-1)

        # 创建PPOBatch
        batch = PPOBatch(
            obs=obs_flat,
            actions=actions_flat,
            old_log_probs=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            values=values_flat,
        )

        # 统计信息
        info = {
            "mean_reward": jp.mean(rewards),
            "mean_value": jp.mean(values),
            "mean_advantage": jp.mean(advantages),
        }

        return batch, env_state, info

    def update_policy(
        self,
        train_state: TrainState,
        batch: PPOBatch,
    ) -> Tuple[TrainState, Dict[str, Any]]:
        """更新策略（多个epoch + mini-batch）- 使用JAX优化循环

        Args:
            train_state: 训练状态
            batch: PPO批次数据

        Returns:
            (更新后的train_state, info字典)
        """

        def loss_fn(params, mb_batch):
            """损失函数"""
            loss, info = ppo_loss(
                params,
                self.network,
                mb_batch,
                clip_epsilon=self.config.clip_epsilon,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef,
            )
            return loss, info

        def update_minibatch(carry, mb_indices):
            """单个mini-batch的更新 - 用于scan"""
            state = carry

            # 提取mini-batch
            mb_batch = PPOBatch(
                obs=batch.obs[mb_indices],
                actions=batch.actions[mb_indices],
                old_log_probs=batch.old_log_probs[mb_indices],
                advantages=batch.advantages[mb_indices],
                returns=batch.returns[mb_indices],
                values=batch.values[mb_indices],
            )

            # 计算梯度并更新
            (loss, info), grads = jax.value_and_grad(
                lambda p: loss_fn(p, mb_batch), has_aux=True
            )(state.params)

            # ✅ 梯度NaN保护：防止NaN梯度污染参数
            # 将梯度中的NaN/Inf替换为0（相当于跳过这次更新）
            grads = jax.tree.map(
                lambda g: jp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                grads
            )

            # 应用梯度
            state = state.apply_gradients(grads=grads, optimizer=self.optimizer)

            return state, info

        def update_epoch(carry, _):
            """单个epoch的更新 - 用于scan"""
            state = carry

            # 打乱数据
            rng, new_rng = jax.random.split(state.rng)
            state = state.replace(rng=new_rng)
            perm = jax.random.permutation(rng, self.config.batch_size)

            # 将索引分成mini-batches
            mb_indices = perm.reshape(self.config.num_minibatches, self.config.minibatch_size)

            # 使用scan更新所有mini-batches
            state, infos = jax.lax.scan(update_minibatch, state, mb_indices)

            # 平均所有mini-batch的指标
            avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)

            return state, avg_info

        # 使用scan执行多个epochs
        train_state, infos = jax.lax.scan(
            update_epoch,
            train_state,
            None,
            length=self.config.num_epochs
        )

        # 平均所有epoch的指标
        avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)

        return train_state, avg_info

    def train_step(
        self,
        train_state: TrainState,
        env_state: EnvState,
    ) -> Tuple[TrainState, EnvState, Dict[str, Any]]:
        """完整训练步（collect + update）

        Args:
            train_state: 训练状态
            env_state: 环境状态

        Returns:
            (更新后的train_state, 新的env_state, info字典)
        """
        # 收集轨迹
        batch, env_state, collect_info = self.collect_trajectory(train_state, env_state)

        # 更新策略
        train_state, update_info = self.update_policy(train_state, batch)

        # 增加环境步数
        train_state = train_state.increment_env_steps(self.config.batch_size)

        # 合并信息
        info = {**collect_info, **update_info}

        return train_state, env_state, info


# ==================== 纯函数版本（用于 JIT 持久化缓存） ====================

def create_train_step_fn(config: PPOConfig, env, network, optimizer):
    """创建纯函数版本的 train_step（支持持久化缓存）

    通过闭包捕获配置和环境，避免对象 ID 变化导致缓存失效。

    Args:
        config: PPO 配置
        env: MJX 环境
        network: Actor-Critic 网络
        optimizer: Optax 优化器

    Returns:
        train_step_fn: 纯函数，签名为 (train_state, env_state) -> (train_state, env_state, info)
    """

    def collect_trajectory(train_state: TrainState, env_state: EnvState):
        """收集轨迹数据（rollout）"""

        def scan_fn(carry, _):
            """单步环境交互"""
            state, e_state, rng = carry

            # 获取动作和价值
            rng, action_rng = jax.random.split(rng)
            mean, log_std, value = network.apply(state.params, e_state.obs)

            # 数值稳定性保护
            log_std = jp.clip(log_std, -5.0, 2.0)
            std = jp.exp(log_std)
            std = jp.maximum(std, 1e-6)  # 防止除零

            # 采样动作
            action = mean + std * jax.random.normal(action_rng, mean.shape)
            action = jp.clip(action, -10.0, 10.0)  # 裁剪动作防止极端值

            # 计算 log 概率（增强数值稳定性）
            action_diff = jp.clip((action - mean) / std, -100.0, 100.0)
            log_prob = -0.5 * jp.sum(
                action_diff ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
            )
            log_prob = jp.clip(log_prob, -1000.0, 100.0)  # 裁剪 log_prob

            # 环境步进
            new_e_state = env.batch_step(e_state, action)

            # 收集转移数据
            transition = {
                "obs": e_state.obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": new_e_state.reward,
                "done": new_e_state.done,
            }

            return (state, new_e_state, rng), transition

        # 执行 rollout
        rng, _ = train_state.split_rng()
        _, transitions = jax.lax.scan(
            scan_fn,
            init=(train_state, env_state, rng),
            xs=None,
            length=config.num_steps,
        )

        # 提取数据 (num_steps, num_envs, ...)
        obs = transitions["obs"]
        actions = transitions["action"]
        log_probs = transitions["log_prob"]
        values = transitions["value"]
        rewards = transitions["reward"]
        dones = transitions["done"]

        # 计算最后一步的价值（bootstrap）
        _, _, last_value = network.apply(train_state.params, env_state.obs)

        # 调试日志：检查 obs 和 value 是否包含 NaN/Inf
        jax.debug.print(
            "[DEBUG] obs: min={min}, max={max}, has_nan={nan}, has_inf={inf}",
            min=env_state.obs.min(),
            max=env_state.obs.max(),
            nan=jp.isnan(env_state.obs).any(),
            inf=jp.isinf(env_state.obs).any()
        )
        jax.debug.print(
            "[DEBUG] last_value: min={min}, max={max}, has_nan={nan}, has_inf={inf}",
            min=last_value.min(),
            max=last_value.max(),
            nan=jp.isnan(last_value).any(),
            inf=jp.isinf(last_value).any()
        )

        # 拼接价值序列
        values_with_last = jp.concatenate([values, last_value[None, :]], axis=0)

        # 计算 GAE（对每个环境分别计算）
        advantages, returns = jax.vmap(
            lambda r, v, d: compute_gae_scan(
                r, v, d, config.gamma, config.gae_lambda
            ),
            in_axes=1,
            out_axes=1,
        )(rewards, values_with_last, dones)

        # Flatten batch (num_steps * num_envs, ...)
        obs_flat = obs.reshape(-1, *obs.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        log_probs_flat = log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values.reshape(-1)

        # 创建 PPOBatch
        batch = PPOBatch(
            obs=obs_flat,
            actions=actions_flat,
            old_log_probs=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            values=values_flat,
        )

        # 统计信息
        info = {
            "mean_reward": jp.mean(rewards),
            "mean_value": jp.mean(values),
            "mean_advantage": jp.mean(advantages),
        }

        return batch, env_state, info

    def update_policy(train_state: TrainState, batch: PPOBatch):
        """更新策略（多个 epoch + mini-batch）"""

        def loss_fn(params, mb_batch):
            """损失函数"""
            loss, info = ppo_loss(
                params,
                network,
                mb_batch,
                clip_epsilon=config.clip_epsilon,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
            )
            return loss, info

        def update_minibatch(carry, mb_indices):
            """单个 mini-batch 的更新 - 用于 scan"""
            state = carry

            # 提取 mini-batch
            mb_batch = PPOBatch(
                obs=batch.obs[mb_indices],
                actions=batch.actions[mb_indices],
                old_log_probs=batch.old_log_probs[mb_indices],
                advantages=batch.advantages[mb_indices],
                returns=batch.returns[mb_indices],
                values=batch.values[mb_indices],
            )

            # 计算梯度并更新
            (loss, info), grads = jax.value_and_grad(
                lambda p: loss_fn(p, mb_batch), has_aux=True
            )(state.params)

            # ✅ 梯度NaN保护：防止NaN梯度污染参数
            # 将梯度中的NaN/Inf替换为0（相当于跳过这次更新）
            grads = jax.tree.map(
                lambda g: jp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                grads
            )

            # 应用梯度
            state = state.apply_gradients(grads=grads, optimizer=optimizer)

            return state, info

        def update_epoch(carry, _):
            """单个 epoch 的更新 - 用于 scan"""
            state = carry

            # 打乱数据
            rng, new_rng = jax.random.split(state.rng)
            state = state.replace(rng=new_rng)
            perm = jax.random.permutation(rng, config.batch_size)

            # 将索引分成 mini-batches
            mb_indices = perm.reshape(config.num_minibatches, config.minibatch_size)

            # 使用 scan 更新所有 mini-batches
            state, infos = jax.lax.scan(update_minibatch, state, mb_indices)

            # 平均所有 mini-batch 的指标
            avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)

            return state, avg_info

        # 使用 scan 执行多个 epochs
        train_state, infos = jax.lax.scan(
            update_epoch,
            train_state,
            None,
            length=config.num_epochs
        )

        # 平均所有 epoch 的指标
        avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)

        return train_state, avg_info

    def train_step_fn(train_state: TrainState, env_state: EnvState):
        """完整训练步（collect + update）- 纯函数"""
        # 收集轨迹
        batch, env_state, collect_info = collect_trajectory(train_state, env_state)

        # 更新策略
        train_state, update_info = update_policy(train_state, batch)

        # 增加环境步数
        train_state = train_state.increment_env_steps(config.batch_size)

        # 合并信息
        info = {**collect_info, **update_info}

        return train_state, env_state, info

    return train_step_fn
