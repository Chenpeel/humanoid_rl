"""
训练状态管理
"""

import jax
import jax.numpy as jp
from flax import struct
from typing import Any, Optional
import optax


@struct.dataclass
class TrainState:
    """训练状态数据类
    
    Attributes:
        step: 当前训练步数
        params: 网络参数（flax params dict）
        opt_state: 优化器状态
        rng: JAX随机数生成器状态
        env_steps: 环境交互总步数
    """
    step: int
    params: Any  # flax params pytree
    opt_state: optax.OptState
    rng: jax.Array
    env_steps: int = 0
    
    def apply_gradients(self, *, grads, optimizer):
        """应用梯度更新参数
        
        Args:
            grads: 梯度pytree
            optimizer: Optax优化器
            
        Returns:
            更新后的TrainState
        """
        updates, new_opt_state = optimizer.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
    
    def increment_env_steps(self, n_steps: int):
        """增加环境交互步数
        
        Args:
            n_steps: 增加的步数
            
        Returns:
            更新后的TrainState
        """
        return self.replace(env_steps=self.env_steps + n_steps)
    
    def split_rng(self):
        """分割RNG并更新状态
        
        Returns:
            (新的rng_key, 更新后的TrainState)
        """
        rng, new_rng = jax.random.split(self.rng)
        return rng, self.replace(rng=new_rng)


def create_train_state(
    network,
    optimizer: optax.GradientTransformation,
    obs_shape: tuple,
    rng: jax.Array,
) -> TrainState:
    """创建初始训练状态
    
    Args:
        network: Flax网络模块
        optimizer: Optax优化器
        obs_shape: 观测空间形状
        rng: 初始随机数种子
        
    Returns:
        初始化的TrainState
    """
    # 分割RNG
    rng, init_rng = jax.random.split(rng)
    
    # 创建假观测初始化网络
    dummy_obs = jp.zeros((1, *obs_shape))
    params = network.init(init_rng, dummy_obs)
    
    # 初始化优化器状态
    opt_state = optimizer.init(params)
    
    return TrainState(
        step=0,
        params=params,
        opt_state=opt_state,
        rng=rng,
        env_steps=0,
    )
