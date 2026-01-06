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

# ============================================================================================
# ======================================= PPO配置 =============================================
# ============================================================================================


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


# ============================================================================================
# ===================================== END: PPO配置 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= PPO训练器 ============================================
# ============================================================================================


class PPOTrainer:
    """PPO训练器"""

    def __init__(
        self,
        config: PPOConfig,
        env,
        network,
        optimizer,
    ):
        """初始化PPO训练器"""
        self.config = config
        self.env = env
        self.network = network
        self.optimizer = optimizer

    # --------------------------------------------------------------------------------------------

    def collect_trajectory(
        self,
        train_state: TrainState,
        env_state: EnvState,
    ) -> Tuple[PPOBatch, EnvState, Dict[str, Any]]:
        """收集轨迹数据（rollout）"""

        def scan_fn(carry, _):
            """扫描函数：执行一步环境交互"""
            state, e_state, rng = carry

            rng, action_rng = jax.random.split(rng)
            mean, log_std, value = self.network.apply(state.params, e_state.obs)

            log_std = jp.clip(log_std, -5.0, 2.0)
            std = jp.exp(log_std)
            std = jp.maximum(std, 1e-6)

            action = mean + std * jax.random.normal(action_rng, mean.shape)
            action = jp.clip(action, -10.0, 10.0)

            action_diff = jp.clip((action - mean) / std, -100.0, 100.0)
            log_prob = -0.5 * jp.sum(
                action_diff**2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
            )
            log_prob = jp.clip(log_prob, -1000.0, 100.0)

            new_e_state = self.env.batch_step(e_state, action)

            transition = {
                "obs": e_state.obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": new_e_state.reward,
                "done": new_e_state.done,
            }

            return (state, new_e_state, rng), transition

        rng, _ = train_state.split_rng()
        _, transitions = jax.lax.scan(
            scan_fn,
            init=(train_state, env_state, rng),
            xs=None,
            length=self.config.num_steps,
        )

        obs = transitions["obs"]
        actions = transitions["action"]
        log_probs = transitions["log_prob"]
        values = transitions["value"]
        rewards = transitions["reward"]
        dones = transitions["done"]

        _, _, last_value = self.network.apply(train_state.params, env_state.obs)
        values_with_last = jp.concatenate([values, last_value[None, :]], axis=0)

        advantages, returns = jax.vmap(
            lambda r, v, d: compute_gae_scan(
                r, v, d, self.config.gamma, self.config.gae_lambda
            ),
            in_axes=1,
            out_axes=1,
        )(rewards, values_with_last, dones)

        obs_flat = obs.reshape(-1, *obs.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        log_probs_flat = log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values.reshape(-1)

        batch = PPOBatch(
            obs=obs_flat,
            actions=actions_flat,
            old_log_probs=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            values=values_flat,
        )

        info = {
            "mean_reward": jp.mean(rewards),
            "mean_value": jp.mean(values),
            "mean_advantage": jp.mean(advantages),
        }

        if env_state.info:
            for key, value in env_state.info.items():
                if key not in info:
                    info[key] = jp.mean(value)

        return batch, env_state, info

    # --------------------------------------------------------------------------------------------

    def update_policy(
        self,
        train_state: TrainState,
        batch: PPOBatch,
    ) -> Tuple[TrainState, Dict[str, Any]]:
        """更新策略（多个epoch + mini-batch）"""

        def loss_fn(params, mb_batch):
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
            state = carry
            mb_batch = PPOBatch(
                obs=batch.obs[mb_indices],
                actions=batch.actions[mb_indices],
                old_log_probs=batch.old_log_probs[mb_indices],
                advantages=batch.advantages[mb_indices],
                returns=batch.returns[mb_indices],
                values=batch.values[mb_indices],
            )

            (loss, info), grads = jax.value_and_grad(
                lambda p: loss_fn(p, mb_batch), has_aux=True
            )(state.params)

            grads = jax.tree.map(
                lambda g: jp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            state = state.apply_gradients(grads=grads, optimizer=self.optimizer)
            return state, info

        def update_epoch(carry, _):
            state = carry
            rng, new_rng = jax.random.split(state.rng)
            state = state.replace(rng=new_rng)
            perm = jax.random.permutation(rng, self.config.batch_size)
            mb_indices = perm.reshape(
                self.config.num_minibatches, self.config.minibatch_size
            )
            state, infos = jax.lax.scan(update_minibatch, state, mb_indices)
            avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)
            return state, avg_info

        train_state, infos = jax.lax.scan(
            update_epoch, train_state, None, length=self.config.num_epochs
        )
        avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)
        return train_state, avg_info

    # --------------------------------------------------------------------------------------------

    def train_step(
        self,
        train_state: TrainState,
        env_state: EnvState,
    ) -> Tuple[TrainState, EnvState, Dict[str, Any]]:
        """完整训练步（collect + update）"""
        batch, env_state, collect_info = self.collect_trajectory(train_state, env_state)
        train_state, update_info = self.update_policy(train_state, batch)
        train_state = train_state.increment_env_steps(self.config.batch_size)
        info = {**collect_info, **update_info}
        return train_state, env_state, info


# ============================================================================================
# ===================================== END: PPO训练器 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 纯函数版本 ==========================================
# ============================================================================================


def create_train_step_fn(config: PPOConfig, env, network, optimizer):
    """创建纯函数版本的 train_step（支持持久化缓存）"""

    def collect_trajectory(train_state: TrainState, env_state: EnvState):
        def scan_fn(carry, _):
            state, e_state, rng = carry
            rng, action_rng = jax.random.split(rng)
            mean, log_std, value = network.apply(state.params, e_state.obs)

            log_std = jp.clip(log_std, -5.0, 2.0)
            std = jp.exp(log_std)
            std = jp.maximum(std, 1e-6)

            action = mean + std * jax.random.normal(action_rng, mean.shape)
            action = jp.clip(action, -10.0, 10.0)

            action_diff = jp.clip((action - mean) / std, -100.0, 100.0)
            log_prob = -0.5 * jp.sum(
                action_diff**2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
            )
            log_prob = jp.clip(log_prob, -1000.0, 100.0)

            new_e_state = env.batch_step(e_state, action)

            transition = {
                "obs": e_state.obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": new_e_state.reward,
                "done": new_e_state.done,
            }
            return (state, new_e_state, rng), transition

        rng, _ = train_state.split_rng()
        _, transitions = jax.lax.scan(
            scan_fn,
            init=(train_state, env_state, rng),
            xs=None,
            length=config.num_steps,
        )

        obs = transitions["obs"]
        actions = transitions["action"]
        log_probs = transitions["log_prob"]
        values = transitions["value"]
        rewards = transitions["reward"]
        dones = transitions["done"]

        _, _, last_value = network.apply(train_state.params, env_state.obs)
        values_with_last = jp.concatenate([values, last_value[None, :]], axis=0)

        advantages, returns = jax.vmap(
            lambda r, v, d: compute_gae_scan(r, v, d, config.gamma, config.gae_lambda),
            in_axes=1,
            out_axes=1,
        )(rewards, values_with_last, dones)

        obs_flat = obs.reshape(-1, *obs.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        log_probs_flat = log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values.reshape(-1)

        batch = PPOBatch(
            obs=obs_flat,
            actions=actions_flat,
            old_log_probs=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            values=values_flat,
        )

        info = {
            "mean_reward": jp.mean(rewards),
            "mean_value": jp.mean(values),
            "mean_advantage": jp.mean(advantages),
        }

        if env_state.info:
            for key, value in env_state.info.items():
                if key not in info:
                    info[key] = jp.mean(value)

        return batch, env_state, info

    def update_policy(train_state: TrainState, batch: PPOBatch):
        def loss_fn(params, mb_batch):
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
            state = carry
            mb_batch = PPOBatch(
                obs=batch.obs[mb_indices],
                actions=batch.actions[mb_indices],
                old_log_probs=batch.old_log_probs[mb_indices],
                advantages=batch.advantages[mb_indices],
                returns=batch.returns[mb_indices],
                values=batch.values[mb_indices],
            )

            (loss, info), grads = jax.value_and_grad(
                lambda p: loss_fn(p, mb_batch), has_aux=True
            )(state.params)

            grads = jax.tree.map(
                lambda g: jp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
            )
            state = state.apply_gradients(grads=grads, optimizer=optimizer)
            return state, info

        def update_epoch(carry, _):
            state = carry
            rng, new_rng = jax.random.split(state.rng)
            state = state.replace(rng=new_rng)
            perm = jax.random.permutation(rng, config.batch_size)
            mb_indices = perm.reshape(config.num_minibatches, config.minibatch_size)
            state, infos = jax.lax.scan(update_minibatch, state, mb_indices)
            avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)
            return state, avg_info

        train_state, infos = jax.lax.scan(
            update_epoch, train_state, None, length=config.num_epochs
        )
        avg_info = jax.tree.map(lambda x: jp.mean(x, axis=0), infos)
        return train_state, avg_info

    def train_step_fn(train_state: TrainState, env_state: EnvState):
        batch, env_state, collect_info = collect_trajectory(train_state, env_state)
        train_state, update_info = update_policy(train_state, batch)
        train_state = train_state.increment_env_steps(config.batch_size)
        info = {**collect_info, **update_info}
        return train_state, env_state, info

    return train_step_fn


# ============================================================================================
# ===================================== END: 纯函数版本 ========================================
# ============================================================================================
