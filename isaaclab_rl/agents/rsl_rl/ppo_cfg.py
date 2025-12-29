"""
RSL_RL PPO 超参数配置

直接使用 Isaac Lab 官方的配置类,确保与 RSL_RL 完全兼容。

参考:
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab 官方配置: isaaclab_rl/rsl_rl/rl_cfg.py
"""

from __future__ import annotations

from isaaclab.utils import configclass
# 直接导入 Isaac Lab 官方的配置类
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# 预定义配置（用于不同任务）
##


@configclass
class VelocityTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """速度跟踪任务的 PPO 配置

    针对速度跟踪任务优化的超参数。
    """

    # 实验配置
    experiment_name = "jiyuan_velocity_tracking"
    seed = 42
    device = "cuda:0"

    # 训练配置
    num_steps_per_env = 24
    max_iterations = 30000  # 30K iterations × 24 steps × 4096 envs ≈ 2.95B steps
    save_interval = 500

    # 日志配置
    logger = "tensorboard"

    # 观测组映射
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # Actor-Critic 网络配置
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        # 学习参数
        learning_rate=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        # PPO 特定参数
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # GAE 参数
        gamma=0.99,
        lam=0.95,
        # 自适应学习率
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class StandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """站立任务的 PPO 配置

    站立任务较简单,可以使用更少的训练步数和更小的网络。
    """

    # 实验配置
    experiment_name = "jiyuan_standing"
    seed = 42
    device = "cuda:0"

    # 训练配置
    num_steps_per_env = 24
    max_iterations = 10000  # 10K iterations × 24 steps × 4096 envs ≈ 983M steps
    save_interval = 500

    # 日志配置
    logger = "tensorboard"

    # 观测组映射
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # Actor-Critic 网络配置（稍小一些）
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,  # 站立任务探索较少
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # PPO 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        # 学习参数
        learning_rate=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        # PPO 特定参数
        clip_param=0.2,
        entropy_coef=0.005,  # 站立任务熵系数更小
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # GAE 参数
        gamma=0.99,
        lam=0.95,
        # 自适应学习率
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


##
# 导出配置实例
##

# 速度跟踪任务配置
VELOCITY_TRACKING_PPO_CFG = VelocityTrackingPPORunnerCfg()

# 站立任务配置
STANDING_PPO_CFG = StandingPPORunnerCfg()
