"""PPOCfg —— PPO 算法配置。

继承 RSL-RL 的默认 PPO 配置，提供双足机器人训练的推荐超参数。
"""

from __future__ import annotations

from rsl_rl.modules import ActorCritic, EmpiricalNormalization


class PPOCfg:
    """PPO 训练配置。

    可通过 YAML 覆盖以下参数：
    """

    # ---- 算法 ----
    algorithm_class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

    # ---- 网络 ----
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"

    # ---- 训练规模 ----
    num_envs: int = 4096
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500

    # ---- 日志 ----
    experiment_name: str = "humanoid_rl"
    run_name: str = ""
    logger: str = "tensorboard"

    # ---- 域随机化 ----
    randomize_dynamics: bool = False
    randomize_friction: tuple[float, float] = (0.5, 1.5)
    randomize_mass: tuple[float, float] = (0.8, 1.2)
    randomize_kp: tuple[float, float] = (0.8, 1.2)
    randomize_kd: tuple[float, float] = (0.8, 1.2)

    # ---- 观测归一化 ----
    normalize_observations: bool = True
    normalize_values: bool = True
    observation_normalization: EmpiricalNormalization | None = None
