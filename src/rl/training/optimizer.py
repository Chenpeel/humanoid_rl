"""
统一优化器配置模块

提供统一的优化器创建接口，支持多种学习率调度策略，
兼容标准训练框架和ksim训练框架。

设计原则：
1. 配置驱动：所有参数通过配置类管理
2. 扩展性：支持多种调度策略，易于添加新策略
3. 兼容性：兼容现有训练框架
4. 安全性：包含梯度裁剪、NaN处理等安全措施
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

import jax.numpy as jnp
import optax


class ScheduleType(str, Enum):
    """学习率调度类型"""
    CONSTANT = "constant"      # 常数学习率
    COSINE = "cosine"          # 余弦退火
    LINEAR = "linear"          # 线性衰减
    EXPONENTIAL = "exponential"  # 指数衰减
    STEP = "step"              # 阶梯衰减
    WARMUP_COSINE = "warmup_cosine"  # 预热+余弦退火
    WARMUP_LINEAR = "warmup_linear"  # 预热+线性衰减


class OptimizerType(str, Enum):
    """优化器类型"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    LAMB = "lamb"


@dataclass
class OptimizerConfig:
    """优化器统一配置类

    所有优化器参数通过此类统一管理，支持YAML配置。
    """

    # 基础配置
    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # Adam特定参数
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # 学习率调度配置
    schedule_type: ScheduleType = ScheduleType.WARMUP_COSINE
    total_steps: Optional[int] = None  # 总优化步数
    warmup_steps: int = 0              # 预热步数
    final_lr_fraction: float = 0.1     # 最终学习率比例

    # 余弦退火参数
    cosine_decay_min: float = 0.0      # 余弦退火最小学习率比例

    # 指数衰减参数
    exponential_decay_rate: float = 0.96
    exponential_decay_steps: int = 1000

    # 阶梯衰减参数
    step_decay_rate: float = 0.1
    step_decay_steps: int = 10000

    # 安全措施
    clip_gradients: bool = True
    handle_nans: bool = True
    zero_nans: bool = True

    # 调试选项
    log_lr_changes: bool = False

    def __post_init__(self):
        """后初始化验证"""
        if self.learning_rate <= 0:
            raise ValueError(f"学习率必须大于0: {self.learning_rate}")

        if self.max_grad_norm <= 0:
            raise ValueError(f"梯度裁剪范数必须大于0: {self.max_grad_norm}")

        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError(f"总步数必须大于0: {self.total_steps}")

        if self.warmup_steps < 0:
            raise ValueError(f"预热步数不能为负数: {self.warmup_steps}")

        if self.total_steps is not None and self.warmup_steps >= self.total_steps:
            # 自动调整：如果预热步数超过总步数，设为总步数的一半
            self.warmup_steps = max(1, self.total_steps // 2)


def create_learning_rate_schedule(
    config: OptimizerConfig,
    learning_rate: Optional[float] = None,
) -> optax.Schedule:
    """创建学习率调度

    Args:
        config: 优化器配置
        learning_rate: 可选的学习率覆盖

    Returns:
        Optax学习率调度
    """
    lr = learning_rate if learning_rate is not None else config.learning_rate

    if config.schedule_type == ScheduleType.CONSTANT:
        return lr

    if config.total_steps is None:
        raise ValueError(f"使用调度类型 {config.schedule_type} 时必须提供 total_steps")

    total_steps = config.total_steps
    warmup_steps = config.warmup_steps

    if config.schedule_type == ScheduleType.WARMUP_COSINE:
        # 确保warmup_steps不超过total_steps
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps // 2)

        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            warmup_steps = max(1, total_steps - 1)
            decay_steps = total_steps - warmup_steps

        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=lr * config.final_lr_fraction,
        )

    elif config.schedule_type == ScheduleType.WARMUP_LINEAR:
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps // 2)

        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=lr,
                    transition_steps=warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=lr,
                    end_value=lr * config.final_lr_fraction,
                    transition_steps=total_steps - warmup_steps,
                ),
            ],
            boundaries=[warmup_steps],
        )

    elif config.schedule_type == ScheduleType.COSINE:
        return optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=total_steps,
            alpha=config.final_lr_fraction,
        )

    elif config.schedule_type == ScheduleType.LINEAR:
        return optax.linear_schedule(
            init_value=lr,
            end_value=lr * config.final_lr_fraction,
            transition_steps=total_steps,
        )

    elif config.schedule_type == ScheduleType.EXPONENTIAL:
        return optax.exponential_decay(
            init_value=lr,
            transition_steps=config.exponential_decay_steps,
            decay_rate=config.exponential_decay_rate,
            staircase=False,
        )

    elif config.schedule_type == ScheduleType.STEP:
        return optax.exponential_decay(
            init_value=lr,
            transition_steps=config.step_decay_steps,
            decay_rate=config.step_decay_rate,
            staircase=True,
        )

    else:
        raise ValueError(f"未知的调度类型: {config.schedule_type}")


def create_optimizer_from_config(
    config: OptimizerConfig,
    learning_rate: Optional[float] = None,
) -> optax.GradientTransformation:
    """根据配置创建优化器

    Args:
        config: 优化器配置
        learning_rate: 可选的学习率覆盖

    Returns:
        Optax优化器
    """
    # 创建学习率调度
    lr_schedule = create_learning_rate_schedule(config, learning_rate)

    # 构建优化器链
    chain_components = []

    # 1. NaN处理
    if config.handle_nans and config.zero_nans:
        chain_components.append(optax.zero_nans())

    # 2. 梯度裁剪
    if config.clip_gradients:
        chain_components.append(optax.clip_by_global_norm(config.max_grad_norm))

    # 3. 权重衰减（仅对AdamW有效）
    if config.weight_decay > 0 and config.optimizer_type == OptimizerType.ADAMW:
        chain_components.append(optax.add_decayed_weights(config.weight_decay))

    # 4. 核心优化器
    if config.optimizer_type == OptimizerType.ADAM:
        chain_components.append(
            optax.adam(
                learning_rate=lr_schedule,
                eps=config.adam_epsilon,
                b1=config.adam_beta1,
                b2=config.adam_beta2,
            )
        )

    elif config.optimizer_type == OptimizerType.ADAMW:
        chain_components.append(
            optax.adamw(
                learning_rate=lr_schedule,
                eps=config.adam_epsilon,
                b1=config.adam_beta1,
                b2=config.adam_beta2,
                weight_decay=config.weight_decay,
            )
        )

    elif config.optimizer_type == OptimizerType.SGD:
        chain_components.append(
            optax.sgd(
                learning_rate=lr_schedule,
                momentum=None,
                nesterov=False,
            )
        )

    elif config.optimizer_type == OptimizerType.RMSPROP:
        chain_components.append(
            optax.rmsprop(
                learning_rate=lr_schedule,
                decay=0.9,
                eps=config.adam_epsilon,
                momentum=None,
                nesterov=False,
            )
        )

    elif config.optimizer_type == OptimizerType.LAMB:
        chain_components.append(
            optax.lamb(
                learning_rate=lr_schedule,
                b1=config.adam_beta1,
                b2=config.adam_beta2,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        )

    else:
        raise ValueError(f"未知的优化器类型: {config.optimizer_type}")

    # 组合所有组件
    return optax.chain(*chain_components)


def create_ppo_optimizer(
    learning_rate: float = 3e-4,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    schedule_type: Union[str, ScheduleType] = ScheduleType.WARMUP_COSINE,
    final_lr_fraction: float = 0.1,
) -> optax.GradientTransformation:
    """创建PPO优化器（便捷函数）

    Args:
        learning_rate: 学习率
        total_steps: 总优化步数
        warmup_steps: 预热步数
        max_grad_norm: 梯度裁剪范数
        schedule_type: 调度类型
        final_lr_fraction: 最终学习率比例

    Returns:
        Optax优化器
    """
    if isinstance(schedule_type, str):
        schedule_type = ScheduleType(schedule_type)

    config = OptimizerConfig(
        learning_rate=learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        schedule_type=schedule_type,
        final_lr_fraction=final_lr_fraction,
    )

    return create_optimizer_from_config(config)


def create_ksim_optimizer(
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
) -> optax.GradientTransformation:
    """创建ksim兼容的优化器（保持向后兼容）

    Args:
        learning_rate: 学习率
        grad_clip: 梯度裁剪
        total_steps: 总优化步数
        warmup_steps: 预热步数

    Returns:
        Optax优化器
    """
    config = OptimizerConfig(
        learning_rate=learning_rate,
        max_grad_norm=grad_clip,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        schedule_type=ScheduleType.WARMUP_COSINE if total_steps else ScheduleType.CONSTANT,
        handle_nans=True,
        zero_nans=True,
    )

    return create_optimizer_from_config(config)


def optimizer_config_from_dict(config_dict: Dict[str, Any]) -> OptimizerConfig:
    """从字典创建优化器配置

    Args:
        config_dict: 配置字典

    Returns:
        优化器配置对象
    """
    # 处理字符串枚举转换
    if "optimizer_type" in config_dict and isinstance(config_dict["optimizer_type"], str):
        config_dict["optimizer_type"] = OptimizerType(config_dict["optimizer_type"])

    if "schedule_type" in config_dict and isinstance(config_dict["schedule_type"], str):
        config_dict["schedule_type"] = ScheduleType(config_dict["schedule_type"])

    return OptimizerConfig(**config_dict)


# 向后兼容的别名
create_ppo_optimizer_cosine = create_ppo_optimizer
create_ppo_optimizer_linear = lambda **kwargs: create_ppo_optimizer(
    schedule_type=ScheduleType.WARMUP_LINEAR, **kwargs
)
