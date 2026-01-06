"""
Flax神经网络实现
使用flax.linen构建Actor-Critic网络
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jp
from flax import linen as nn
from rich.console import Console

console = Console()


# ============================================================================================
# ======================================= 基础模块 ============================================
# ============================================================================================

class MLP(nn.Module):
    """多层感知机（MLP）基础网络

    Args:
        features: 每层的神经元数量列表，例如 [256, 256]
        activation: 激活函数，默认为tanh
        activate_final: 是否在最后一层应用激活函数
    """

    features: Sequence[int]
    activation: Callable = nn.tanh
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        """前向传播

        Args:
            x: 输入张量 (batch, input_dim)

        Returns:
            输出张量 (batch, output_dim)
        """
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1 or self.activate_final:
                x = self.activation(x)
        return x

# ============================================================================================
# ===================================== END: 基础模块 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= Actor网络 ============================================
# ============================================================================================

class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络

    输出动作分布的均值和对数标准差

    Args:
        action_dim: 动作空间维度
        hidden_dims: 隐藏层维度列表，例如 [256, 256]
        activation: 激活函数
        log_std_min: log_std的最小值（防止标准差过小）
        log_std_max: log_std的最大值（防止标准差过大）
    """

    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.tanh
    log_std_min: float = -5.0  # exp(-5) ≈ 0.0067
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs):
        """前向传播"""
        # Backbone网络
        x = MLP(features=self.hidden_dims, activation=self.activation)(obs)

        # 输出均值
        mean = nn.Dense(self.action_dim)(x)

        # 输出log_std（学习的参数）
        log_std = nn.Dense(self.action_dim)(x)

        # 裁剪log_std到合理范围
        log_std = jp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    # --------------------------------------------------------------------------------------------

    def get_action(self, obs, rng_key):
        """采样动作"""
        mean, log_std = self(obs)
        std = jp.exp(log_std)

        # 从正态分布采样
        eps = jax.random.normal(rng_key, shape=mean.shape)
        action = mean + eps * std

        return action, mean, log_std

    # --------------------------------------------------------------------------------------------

    def get_log_prob(self, obs, action):
        """计算动作的对数概率"""
        mean, log_std = self(obs)
        std = jp.exp(log_std)

        # 计算对数概率（多维独立正态分布）
        log_prob = -0.5 * jp.sum(
            ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
        )

        return log_prob

# ============================================================================================
# ===================================== END: Actor网络 =========================================
# ============================================================================================


# ============================================================================================
# ======================================= Critic网络 ===========================================
# ============================================================================================

class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络

    输出状态价值V(s)

    Args:
        hidden_dims: 隐藏层维度列表，例如 [256, 256]
        activation: 激活函数
    """

    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, obs):
        """前向传播"""
        # Backbone网络
        x = MLP(features=self.hidden_dims, activation=self.activation)(obs)

        # 输出标量价值
        value = nn.Dense(1)(x)

        # 压缩最后一维
        value = jp.squeeze(value, axis=-1)

        return value

# ============================================================================================
# ===================================== END: Critic网络 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= Actor-Critic网络 =====================================
# ============================================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic联合网络

    支持共享backbone或分离backbone

    Args:
        action_dim: 动作空间维度
        shared_backbone: 是否共享backbone（True=共享，False=分离）
        hidden_dims: 隐藏层维度列表
        activation: 激活函数
        log_std_min: log_std的最小值
        log_std_max: log_std的最大值
    """

    action_dim: int
    shared_backbone: bool = True
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.tanh
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def setup(self):
        """初始化网络层"""
        if self.shared_backbone:
            self.backbone = MLP(features=self.hidden_dims, activation=self.activation)
            self.actor_head = nn.Dense(self.action_dim)
            self.actor_log_std_head = nn.Dense(self.action_dim)
            self.critic_head = nn.Dense(1)
        else:
            self.actor_backbone = MLP(
                features=self.hidden_dims, activation=self.activation
            )
            self.critic_backbone = MLP(
                features=self.hidden_dims, activation=self.activation
            )
            self.actor_head = nn.Dense(self.action_dim)
            self.actor_log_std_head = nn.Dense(self.action_dim)
            self.critic_head = nn.Dense(1)

    # --------------------------------------------------------------------------------------------

    def __call__(self, obs):
        """前向传播"""
        if self.shared_backbone:
            features = self.backbone(obs)
            mean = self.actor_head(features)
            log_std = self.actor_log_std_head(features)
            value = self.critic_head(features)
        else:
            actor_features = self.actor_backbone(obs)
            critic_features = self.critic_backbone(obs)
            mean = self.actor_head(actor_features)
            log_std = self.actor_log_std_head(actor_features)
            value = self.critic_head(critic_features)

        log_std = jp.clip(log_std, self.log_std_min, self.log_std_max)
        value = jp.squeeze(value, axis=-1)

        return mean, log_std, value

    # --------------------------------------------------------------------------------------------

    def get_action_and_value(self, obs, rng_key):
        """同时获取动作和价值"""
        mean, log_std, value = self(obs)
        std = jp.exp(log_std)

        eps = jax.random.normal(rng_key, shape=mean.shape)
        action = mean + eps * std

        return action, mean, log_std, value

    # --------------------------------------------------------------------------------------------

    def get_log_prob_and_value(self, obs, action):
        """计算动作的对数概率和状态价值"""
        mean, log_std, value = self(obs)
        std = jp.exp(log_std)

        log_prob = -0.5 * jp.sum(
            ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
        )

        return log_prob, value

# ============================================================================================
# ===================================== END: Actor-Critic网络 ==================================
# ============================================================================================


# ============================================================================================
# ======================================= 工具函数 ============================================
# ============================================================================================

def create_actor_critic(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    shared_backbone: bool = True,
    activation: Callable = nn.tanh,
    verbose: bool = True,
) -> ActorCriticNetwork:
    """创建Actor-Critic网络（便捷函数）"""
    network = ActorCriticNetwork(
        action_dim=action_dim,
        shared_backbone=shared_backbone,
        hidden_dims=hidden_dims,
        activation=activation,
    )

    if verbose:
        console.print(f"[green]✓ 创建Actor-Critic网络[/green]")
        console.print(f"  观测维度: {obs_dim}")
        console.print(f"  动作维度: {action_dim}")
        console.print(f"  隐藏层: {hidden_dims}")
        console.print(f"  共享backbone: {shared_backbone}")

    return network

# --------------------------------------------------------------------------------------------

def count_parameters(params) -> int:
    """统计参数数量"""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# ============================================================================================
# ===================================== END: 工具函数 ==========================================
# ============================================================================================