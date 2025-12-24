"""
RSL_RL PPO 超参数配置


参考:
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab 示例: source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion
- 原始训练配置: configs/train.yaml (jax 分支)
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class RslRlPpoAlgorithmCfg:
    """PPO 算法超参数配置

    基于 RSL_RL 的 PPO 实现。
    """

    # 训练超参数
    value_loss_coef = 1.0
    """价值函数损失系数"""

    use_clipped_value_loss = True
    """是否使用裁剪的价值损失"""

    clip_param = 0.2
    """PPO 裁剪参数 (epsilon)"""

    entropy_coef = 0.01
    """熵正则化系数（鼓励探索）"""

    num_learning_epochs = 5
    """每次更新的训练轮数"""

    num_mini_batches = 4
    """每次更新的 mini-batch 数量"""

    learning_rate = 1.0e-3
    """学习率"""

    schedule = "adaptive"
    """学习率调度策略: 'fixed', 'adaptive'"""

    gamma = 0.99
    """折扣因子"""

    lam = 0.95
    """GAE lambda 参数"""

    desired_kl = 0.01
    """期望的 KL 散度（用于自适应学习率）"""

    max_grad_norm = 1.0
    """梯度裁剪范数"""


@configclass
class RslRlPpoRunnerCfg:
    """PPO 训练器配置

    控制训练循环、数据收集和日志记录。
    """

    # 运行器参数
    seed = 42
    """随机种子"""

    device = "cuda:0"
    """训练设备: 'cpu', 'cuda:0', 'cuda:1', ..."""

    num_steps_per_env = 24
    """每个环境每次收集的步数（Horizon）"""

    max_iterations = 30000
    """最大训练迭代数

    总训练步数 = max_iterations × num_steps_per_env × num_envs
    例如: 30000 × 24 × 4096 = 2.95B 步
    """

    # 日志和保存
    save_interval = 500
    """保存检查点的间隔（迭代数）"""

    experiment_name = "jiyuan_velocity_tracking"
    """实验名称（用于日志目录）"""

    run_name = ""
    """运行名称（如果为空，使用时间戳）"""

    logger = "tensorboard"
    """日志记录器: 'tensorboard', 'wandb'"""

    # TensorBoard 配置
    log_dir = "logs"
    """日志根目录"""

    # 评估
    empirical_normalization = False
    """是否使用经验归一化（适用于非对称 actor-critic）"""


@configclass
class ActorCriticNetworkCfg:
    """Actor-Critic 网络架构配置

    使用 RSL_RL 的 MLP 网络。
    """

    # Actor 网络
    actor_hidden_dims = [512, 256, 128]
    """Actor 网络隐藏层维度列表"""

    # Critic 网络
    critic_hidden_dims = [512, 256, 128]
    """Critic 网络隐藏层维度列表"""

    # 激活函数
    activation = "elu"
    """激活函数: 'elu', 'relu', 'leaky_relu', 'tanh'"""

    # 初始化
    init_noise_std = 1.0
    """动作分布的初始标准差"""


@configclass
class RslRlOnPolicyRunnerCfg:
    """完整的 RSL_RL On-Policy Runner 配置

    结合算法、训练器和网络配置。
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "jiyuan_velocity_tracking"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False

    # 算法配置
    algorithm_class_name = "PPO"
    policy: ActorCriticNetworkCfg = ActorCriticNetworkCfg()
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg()

    def __post_init__(self):
        """后处理配置"""
        # 计算总训练步数
        # 注意：num_envs 从环境配置中获取
        pass


##
# 预定义配置（用于不同任务）
##


@configclass
class VelocityTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """速度跟踪任务的 PPO 配置

    针对速度跟踪任务优化的超参数。
    """

    experiment_name = "jiyuan_velocity_tracking"
    num_steps_per_env = 24
    max_iterations = 30000  # 30K iterations × 24 steps × 4096 envs ≈ 2.95B steps

    def __post_init__(self):
        """配置验证"""
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4

        # 网络架构
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        self.policy.init_noise_std = 1.0


@configclass
class StandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """站立任务的 PPO 配置

    站立任务较简单，可以使用更少的训练步数。
    """

    experiment_name = "jiyuan_standing"
    num_steps_per_env = 24
    max_iterations = 10000  # 10K iterations × 24 steps × 4096 envs ≈ 983M steps

    def __post_init__(self):
        """配置验证"""
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.005  # 站立任务熵系数更小
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4

        # 网络架构（稍小一些）
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        self.policy.activation = "elu"
        self.policy.init_noise_std = 0.5  # 站立任务探索较少


##
# 导出配置实例
##

# 速度跟踪任务配置
VELOCITY_TRACKING_PPO_CFG = VelocityTrackingPPORunnerCfg()

# 站立任务配置
STANDING_PPO_CFG = StandingPPORunnerCfg()
