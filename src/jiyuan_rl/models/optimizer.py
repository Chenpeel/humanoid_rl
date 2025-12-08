"""
Optax优化器配置
"""

import optax
from typing import Optional, Callable


def create_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    adam_epsilon: float = 1e-5,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    use_lr_schedule: bool = False,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
) -> optax.GradientTransformation:
    """创建Adam优化器（带梯度裁剪）
    
    Args:
        learning_rate: 学习率
        max_grad_norm: 梯度裁剪的最大范数
        adam_epsilon: Adam的epsilon参数
        adam_b1: Adam的beta1参数
        adam_b2: Adam的beta2参数
        use_lr_schedule: 是否使用学习率调度
        total_steps: 总训练步数（用于学习率调度）
        warmup_steps: 预热步数
        
    Returns:
        Optax优化器
    """
    # 学习率调度
    if use_lr_schedule and total_steps is not None:
        # 线性衰减学习率（带预热）
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=learning_rate * 0.1,
        )
    else:
        # 常量学习率
        schedule = learning_rate
    
    # 组合优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # 梯度裁剪
        optax.adam(
            learning_rate=schedule,
            eps=adam_epsilon,
            b1=adam_b1,
            b2=adam_b2,
        ),
    )
    
    return optimizer


def create_ppo_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    """创建PPO默认优化器（便捷函数）
    
    Args:
        learning_rate: 学习率（PPO通常使用3e-4）
        max_grad_norm: 梯度裁剪阈值（PPO通常使用0.5）
        
    Returns:
        Optax优化器
    """
    return create_optimizer(
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        use_lr_schedule=False,
    )


def create_optimizer_with_schedule(
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    warmup_steps: int = 10000,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    """创建带学习率调度的优化器（便捷函数）
    
    Args:
        learning_rate: 初始学习率
        total_steps: 总训练步数
        warmup_steps: 预热步数
        max_grad_norm: 梯度裁剪阈值
        
    Returns:
        Optax优化器
    """
    return create_optimizer(
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        use_lr_schedule=True,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
