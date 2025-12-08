"""
PPO训练器
"""

import jax
import jax.numpy as jp
from flax import struct
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..models.ppo import compute_gae_scan, ppo_loss, PPOBatch
from ..envs.mjx_base_env import EnvState
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
            mean, log_std, value = self.network.apply(
                state.params, e_state.obs)
            std = jp.exp(log_std)
            action = mean + std * jax.random.normal(action_rng, mean.shape)

            # 计算log概率
            log_prob = -0.5 * jp.sum(
                ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi),
                axis=-1
            )

            # 环境步进
            new_e_state = jax.vmap(self.env.step)(e_state, action)

            # 收集转移数据
            transition = {
                'obs': e_state.obs,
                'action': action,
                'log_prob': log_prob,
                'value': value,
                'reward': new_e_state.reward,
                'done': new_e_state.done,
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
        obs = transitions['obs']
        actions = transitions['action']
        log_probs = transitions['log_prob']
        values = transitions['value']
        rewards = transitions['reward']
        dones = transitions['done']

        # 计算最后一步的价值（bootstrap）
        _, _, last_value = self.network.apply(
            train_state.params, env_state.obs)

        # 拼接价值序列
        values_with_last = jp.concatenate(
            [values, last_value[None, :]], axis=0)

        # 计算GAE
        advantages, returns = jax.vmap(
            lambda r, v, d: compute_gae_scan(
                r, v, d, self.config.gamma, self.config.gae_lambda),
            in_axes=1, out_axes=1  # 对每个环境分别计算
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
            'mean_reward': jp.mean(rewards),
            'mean_value': jp.mean(values),
            'mean_advantage': jp.mean(advantages),
        }

        return batch, env_state, info

    def update_policy(
        self,
        train_state: TrainState,
        batch: PPOBatch,
    ) -> Tuple[TrainState, Dict[str, Any]]:
        """更新策略（多个epoch + mini-batch）

        Args:
            train_state: 训练状态
            batch: PPO批次数据

        Returns:
            (更新后的train_state, info字典)
        """
        def loss_fn(params):
            """损失函数"""
            loss, info = ppo_loss(
                params,
                self.network,
                batch,
                clip_epsilon=self.config.clip_epsilon,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef,
            )
            return loss, info

        # 累积指标
        total_info = {}

        # 多个epoch
        for epoch in range(self.config.num_epochs):
            # 打乱数据
            # perm_rng, train_state = train_state.split_rng()
            rng, new_rng = jax.random.split(train_state.rng)
            train_state = train_state.replace(rng=new_rng)
            perm_rng = rng
            perm = jax.random.permutation(perm_rng, self.config.batch_size)

            # Mini-batch更新
            for i in range(self.config.num_minibatches):
                start = i * self.config.minibatch_size
                end = start + self.config.minibatch_size
                mb_indices = perm[start:end]

                # 提取mini-batch
                mb_batch = PPOBatch(
                    obs=batch.obs[mb_indices],
                    actions=batch.actions[mb_indices],
                    old_log_probs=batch.old_log_probs[mb_indices],
                    advantages=batch.advantages[mb_indices],
                    returns=batch.returns[mb_indices],
                    values=batch.values[mb_indices],
                )

                # 计算梯度
                (loss, info), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(train_state.params)

                # 应用梯度
                train_state = train_state.apply_gradients(
                    grads=grads, optimizer=self.optimizer)

                # 累积信息
                if not total_info:
                    total_info = {k: v for k, v in info.items()}
                else:
                    for k, v in info.items():
                        total_info[k] += v

        # 平均指标
        num_updates = self.config.num_epochs * self.config.num_minibatches
        avg_info = {k: v / num_updates for k, v in total_info.items()}

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
        batch, env_state, collect_info = self.collect_trajectory(
            train_state, env_state)

        # 更新策略
        train_state, update_info = self.update_policy(train_state, batch)

        # 增加环境步数
        train_state = train_state.increment_env_steps(self.config.batch_size)

        # 合并信息
        info = {**collect_info, **update_info}

        return train_state, env_state, info
