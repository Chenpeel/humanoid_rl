"""
Optax优化器配置
"""

from typing import Callable, Optional

import optax

# ============================================================================================
# ======================================= 优化器创建函数 ========================================
# ============================================================================================


def create_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    adam_epsilon: float = 1e-5,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    use_lr_schedule: bool = False,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    schedule_type: str = "cosine",
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
        schedule_type: 调度类型，可选 "cosine" 或 "linear"

    Returns:
        Optax优化器
    """
    # 学习率调度
    if use_lr_schedule and total_steps is not None:
        if schedule_type == "cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                end_value=learning_rate * 0.1,
            )
        elif schedule_type == "linear":
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=learning_rate,
                        transition_steps=warmup_steps,
                    ),
                    optax.linear_schedule(
                        init_value=learning_rate,
                        end_value=learning_rate * 0.1,
                        transition_steps=total_steps - warmup_steps,
                    ),
                ],
                boundaries=[warmup_steps],
            )
        else:
            raise ValueError(f"未知的调度类型: {schedule_type}")
    else:
        schedule = learning_rate

    # 组合优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(
            learning_rate=schedule,
            eps=adam_epsilon,
            b1=adam_b1,
            b2=adam_b2,
        ),
    )

    return optimizer


# ============================================================================================
# ===================================== END: 优化器创建函数 =====================================
# ============================================================================================


# ============================================================================================
# ======================================= 便捷包装函数 ==========================================
# ============================================================================================


def create_ppo_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    """创建PPO默认优化器"""
    return create_optimizer(
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        use_lr_schedule=False,
    )


# --------------------------------------------------------------------------------------------


def create_optimizer_with_schedule(
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    warmup_steps: int = 10000,
    max_grad_norm: float = 0.5,
    schedule_type: str = "cosine",
) -> optax.GradientTransformation:
    """创建带学习率调度的优化器"""
    return create_optimizer(
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        use_lr_schedule=True,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        schedule_type=schedule_type,
    )


# --------------------------------------------------------------------------------------------


def create_ppo_optimizer_cosine(
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    warmup_steps: int = 10000,
    max_grad_norm: float = 0.5,
    final_lr_fraction: float = 0.1,
) -> optax.GradientTransformation:
    """创建带余弦退火调度的PPO优化器"""
    if warmup_steps >= total_steps:
        warmup_steps = max(1, total_steps // 2)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        warmup_steps = max(1, total_steps - 1)
        decay_steps = total_steps - warmup_steps

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=learning_rate * final_lr_fraction,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=schedule),
    )

    return optimizer


# --------------------------------------------------------------------------------------------


def create_ppo_optimizer_linear(
    learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    warmup_steps: int = 10000,
    max_grad_norm: float = 0.5,
    final_lr_fraction: float = 0.1,
) -> optax.GradientTransformation:
    """创建带线性衰减调度的PPO优化器"""
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
            optax.linear_schedule(
                init_value=learning_rate,
                end_value=learning_rate * final_lr_fraction,
                transition_steps=total_steps - warmup_steps,
            ),
        ],
        boundaries=[warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=schedule),
    )

    return optimizer


# ============================================================================================
# ===================================== END: 便捷包装函数 =======================================
# ============================================================================================
