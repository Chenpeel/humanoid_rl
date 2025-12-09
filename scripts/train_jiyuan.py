"""
机器人PPO训练主脚本
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
from rich import box
from rich.console import Console
from rich.panel import Panel

from jiyuan_rl.envs.jiyuan_mjx_env import JiyuanMJXEnv, create_jiyuan_env
from jiyuan_rl.models.networks import ActorCriticNetwork, count_parameters
from jiyuan_rl.models.optimizer import create_ppo_optimizer
from jiyuan_rl.training.logger import Logger, MetricsLogger
from jiyuan_rl.training.ppo_trainer import PPOConfig, PPOTrainer
from jiyuan_rl.training.train_state import create_train_state

# ==================== JAX配置 (必须在导入jax之前) ====================
# 禁用预分配，避免显存占满
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # 使用95%的显存

# 启用JAX编译缓存 (使用绝对路径)
cache_path = os.path.join(os.getcwd(), ".tmp")
os.makedirs(cache_path, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

# 启用多核编译 (使用8个线程)
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") +
    " --xla_gpu_force_compilation_parallelism=8  --xla_gpu_autotune_level=4"
)


warnings.filterwarnings("ignore", category=Warning)


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


console = Console()


def print_config(config: PPOConfig):
    """打印训练配置"""
    from rich.table import Table

    table = Table(title="训练配置", box=box.ROUNDED)
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green", justify="right")

    # 环境配置
    table.add_row("并行环境数", str(config.num_envs))
    table.add_row("Rollout步数", str(config.num_steps))
    table.add_row("总批次大小", str(config.batch_size))

    # PPO超参数
    table.add_row("Epoch数", str(config.num_epochs))
    table.add_row("Mini-batch数", str(config.num_minibatches))
    table.add_row("Mini-batch大小", str(config.minibatch_size))
    table.add_row("折扣因子γ", str(config.gamma))
    table.add_row("GAE λ", str(config.gae_lambda))
    table.add_row("裁剪系数ε", str(config.clip_epsilon))
    table.add_row("价值损失系数", str(config.value_coef))
    table.add_row("熵系数", str(config.entropy_coef))
    table.add_row("梯度裁剪", str(config.max_grad_norm))

    # 训练配置
    table.add_row("总时间步", f"{config.total_timesteps:,}")
    table.add_row("总更新次数", f"{config.num_updates:,}")
    table.add_row("日志间隔", str(config.log_interval))

    console.print(table)


def main():
    """主训练函数"""
    console.print(
        Panel.fit(
            "[bold green]机器人 PPO 训练[/bold green]\n[dim]JAX + MJX + Flax实现[/dim]",
            border_style="green",
        )
    )

    # ==================== 配置 ====================
    console.print("\n[bold cyan]1. 加载配置[/bold cyan]")

    config = PPOConfig(
        # 环境配置
        num_envs=1024,  # 并行环境数
        num_steps=200,  # Rollout步数
        # PPO超参数
        num_epochs=16,
        num_minibatches=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        # 训练配置
        total_timesteps=10_000_000,
        log_interval=10,
        eval_interval=100,
    )

    print_config(config)

    # ==================== 设备检测 ====================
    console.print("\n[bold cyan]2. 检测计算设备[/bold cyan]")
    devices = jax.devices()
    console.print(f"可用设备: {devices}")
    console.print(f"默认后端: {jax.default_backend()}")

    # ==================== 创建环境 ====================
    console.print("\n[bold cyan]3. 创建MJX环境[/bold cyan]")

    scene_path = "../assets/xmls/scene.xml"
    if not os.path.exists(os.path.join(os.path.dirname(__file__), scene_path)):
        pass
    env = create_jiyuan_env(xml_path=scene_path)
    console.print(f"✓ 环境创建完成")
    console.print(f"  观测维度: {env.observation_size}")
    console.print(f"  动作维度: {env.action_size}")

    # ==================== 创建网络 ====================
    console.print("\n[bold cyan]4. 创建Actor-Critic网络[/bold cyan]")

    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=True,
        hidden_dims=(512, 512),
    )

    # 初始化网络以统计参数
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jp.zeros((1, env.observation_size))
    params = network.init(init_rng, dummy_obs)
    num_params = count_parameters(params)

    console.print(f"✓ 网络创建完成")
    console.print(f"  参数数量: {num_params:,}")
    console.print(f"  共享backbone: True")
    console.print(f"  隐藏层: [256, 256]")

    # ==================== 创建优化器 ====================
    console.print("\n[bold cyan]5. 创建Optax优化器[/bold cyan]")

    optimizer = create_ppo_optimizer(
        learning_rate=1e-5,
        max_grad_norm=config.max_grad_norm,
    )

    console.print(f"✓ 优化器创建完成")
    console.print(f"  类型: Adam")
    console.print(f"  学习率: 1e-5")
    console.print(f"  梯度裁剪: {config.max_grad_norm}")

    # ==================== 创建训练状态 ====================
    console.print("\n[bold cyan]6. 初始化训练状态[/bold cyan]")

    train_state = create_train_state(
        network=network,
        optimizer=optimizer,
        obs_shape=(env.observation_size,),
        rng=rng,
    )

    console.print(f"✓ 训练状态初始化完成")
    console.print(f"  初始步数: {train_state.step}")
    console.print(f"  初始环境步数: {train_state.env_steps}")

    # ==================== 初始化环境 ====================
    console.print("\n[bold cyan]7. 初始化批量环境[/bold cyan]")

    rng, reset_rng = jax.random.split(rng)
    env_state = env.batch_reset(reset_rng, config.num_envs)

    console.print(f"✓ 环境初始化完成")
    console.print(f"  环境数量: {config.num_envs}")
    console.print(f"  观测形状: {env_state.obs.shape}")

    # ==================== 创建日志系统 ====================
    console.print("\n[bold cyan]8. 创建日志系统[/bold cyan]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/jiyuan_ppo_{timestamp}"
    logger = Logger(log_dir=log_dir, use_tensorboard=True, use_rich=True)

    console.print(f"✓ 日志系统创建完成")
    console.print(f"  日志目录: {log_dir}")

    # ==================== 创建训练器 ====================
    console.print("\n[bold cyan]9. 创建PPO训练器[/bold cyan]")

    trainer = PPOTrainer(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )

    console.print(f"✓ 训练器创建完成")

    # ==================== JIT编译 ====================
    console.print("\n[bold cyan]10. JIT编译[/bold cyan]")
    console.print("正在编译 JAX 计算图，第一次运行可能需要几分钟...")

    # 使用jax.jit加速训练步
    train_step_jit = jax.jit(trainer.train_step)

    # 触发一次编译
    t0 = time.time()
    train_state, env_state, info = train_step_jit(train_state, env_state)
    compile_time = time.time() - t0
    console.print(f"✓ 编译完成 (耗时: {compile_time:.2f}s)")

    # ==================== 开始训练 ====================
    logger.print_section("开始训练")

    metrics_logger = MetricsLogger()

    progress = logger.create_progress_bar(
        total=config.num_updates, description="PPO训练"
    )

    try:
        with progress:
            task = progress.add_task("[cyan]训练中...", total=config.num_updates)

            # 记录第一次更新的指标
            metrics_logger.log_dict(info)
            progress.update(task, advance=1)

            # 从第二次更新开始循环 (因为第一次已经作为编译预热运行了)
            for update in range(1, config.num_updates):
                # 执行训练步
                train_state, env_state, info = train_step_jit(
                    train_state, env_state)

                # 记录指标
                metrics_logger.log_dict(info)

                # 更新进度条
                progress.update(task, advance=1)

                # 定期日志
                if (update + 1) % config.log_interval == 0:
                    avg_metrics = metrics_logger.get_averages()
                    avg_metrics["steps_since_last_log"] = config.log_interval

                    # TensorBoard日志
                    logger.log_scalars(
                        metrics=avg_metrics,
                        step=train_state.step,
                        prefix="train",
                    )

                    # 终端输出
                    logger.print_training_status(
                        step=train_state.step,
                        total_steps=config.num_updates,
                        env_steps=train_state.env_steps,
                        metrics=avg_metrics,
                    )

                    # 重置指标累积器
                    metrics_logger.reset()

        # ==================== 训练完成 ====================
        logger.print_summary(
            "✓ 训练完成！\n"
            f"总步数: {train_state.step}\n"
            f"总环境步数: {train_state.env_steps:,}",
            style="green",
        )

    except KeyboardInterrupt:
        logger.print_summary("训练被用户中断", style="yellow")

    except Exception as e:
        logger.print_summary(f"训练出错: {e}", style="red")
        import traceback

        console.print(traceback.format_exc())

    finally:
        logger.close()
        console.print("\n[dim]日志已保存[/dim]")


if __name__ == "__main__":
    main()
