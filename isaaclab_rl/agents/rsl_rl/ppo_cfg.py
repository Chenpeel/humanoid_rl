"""
RSL_RL PPO 超参数配置

统一网络架构配置,各任务仅调整任务特定参数。

设计理念:
- 基础配置 (JiyuanBasePPORunnerCfg): 定义统一的网络架构和核心超参数
- 任务配置: 继承基础配置,仅覆盖任务特定参数 (学习率、探索噪声等)
- 目的: 方便流水线式训练中模型迁移和管理

参考:
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab 官方配置: isaaclab_rl/rsl_rl/rl_cfg.py
- JAX 分支配置: configs/train/stage0_standing.yaml
"""

from __future__ import annotations

from isaaclab.utils import configclass
# 直接导入 Isaac Lab 官方的配置类
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# 基础配置 (统一网络架构)
##


@configclass
class JiyuanBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Jiyuan 机器人 PPO 基础配置

    定义统一的网络架构和核心超参数,各任务继承后仅调整任务特定参数。

    网络架构设计:
    - Actor/Critic: [512, 512, 256] (对齐 JAX 分支)
    - 激活函数: elu
    - 共享架构: 方便课程学习和模型迁移

    核心超参数:
    - learning_rate: 3.0e-4 (P0修复: 降低学习率避免策略崩溃)
    - num_steps_per_env: 32 (P1优化: 增加batch size提高梯度质量)
    - 其他 PPO 参数保持标准配置
    """

    # 设备配置
    seed = 42
    device = "cuda:0"

    # 训练配置 (默认值,可被子类覆盖)
    num_steps_per_env = 32  # P1优化: 增加batch size
    save_interval = 500

    # 日志配置
    logger = "tensorboard"

    # 观测组映射
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # 统一网络架构 (所有任务共享)
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,  # 默认探索噪声,子类可覆盖
        actor_hidden_dims=[512, 512, 256],  # P1优化: 对齐JAX分支,增加网络容量
        critic_hidden_dims=[512, 512, 256],  # P1优化: 对齐JAX分支
        activation="elu",
    )

    # 统一 PPO 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        # 学习参数
        learning_rate=3.0e-4,  # P0修复: 降低学习率,避免策略崩溃
        num_learning_epochs=5,
        num_mini_batches=8,  # P2A优化: 从4增加到8,更细粒度学习,参考Isaac Lab最佳实践
        # PPO 特定参数
        clip_param=0.2,
        entropy_coef=0.01,  # 默认熵系数,子类可覆盖
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
# 任务特定配置 (仅调整必要参数)
##


@configclass
class StandingPPORunnerCfg(JiyuanBasePPORunnerCfg):
    """站立任务的 PPO 配置

    继承基础配置,仅调整站立任务特定参数:
    - init_noise_std: 0.5 (站立任务探索较少)
    - entropy_coef: 0.005 (站立任务熵系数更小)
    - max_iterations: 10000 (站立任务训练步数较少)
    """

    # 实验配置
    experiment_name = "jiyuan_standing"

    # 训练配置
    max_iterations = 10000  # 10K iterations × 32 steps × 4096 envs ≈ 1.31B steps

    # 网络配置: 覆盖探索噪声
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,  # 站立任务探索较少
        actor_hidden_dims=[512, 512, 256],  # 继承基础配置
        critic_hidden_dims=[512, 512, 256],  # 继承基础配置
        activation="elu",
    )

    # PPO 配置: 覆盖熵系数
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        # 学习参数
        learning_rate=3.0e-4,  # 继承基础配置
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


@configclass
class VelocityTrackingPPORunnerCfg(JiyuanBasePPORunnerCfg):
    """速度跟踪任务的 PPO 配置

    继承基础配置,仅调整速度跟踪任务特定参数:
    - max_iterations: 30000 (速度跟踪任务训练步数较多)
    - init_noise_std: 1.0 (保持默认探索)
    - entropy_coef: 0.01 (保持默认熵系数)
    """

    # 实验配置
    experiment_name = "jiyuan_velocity_tracking"

    # 训练配置
    max_iterations = 30000  # 30K iterations × 32 steps × 4096 envs ≈ 3.93B steps

    # 网络配置和 PPO 配置完全继承基础配置,无需覆盖


##
# 导出配置实例
##

# 站立任务配置
STANDING_PPO_CFG = StandingPPORunnerCfg()

# 速度跟踪任务配置
VELOCITY_TRACKING_PPO_CFG = VelocityTrackingPPORunnerCfg()
