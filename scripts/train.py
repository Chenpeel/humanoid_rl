"""
PPO训练主脚本
"""

import os
import sys
import time
import warnings
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# ==================== JAX配置 (必须在导入jax之前) ====================
# 🔧 指定使用 GPU device:0（第一张显卡，AutoDL 单卡环境）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 启用JAX编译缓存 (使用绝对路径，确保持久化)
cache_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".jax_cache"))
os.makedirs(cache_path, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

# 🔧 增强缓存配置
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"  # 缓存所有编译结果
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"  # 缓存所有编译

# 最大化显存使用（V100 32GB 可以用 0.95-0.98）
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # V100 32GB 用 95%

# 启用编译优化（V100 用最高级别）
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_enable_latency_hiding_scheduler=true"
    + " --xla_gpu_enable_highest_priority_async_stream=true"
    + " --xla_gpu_autotune_level=1"  # V100 用最高级别1
    + " --xla_gpu_deterministic_ops=false"  # 非确定性换取速度
)

warnings.filterwarnings("ignore", category=Warning)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ==================== 延迟导入 (必须在环境变量设置之后) ====================
# 注意: 为了防止格式化工具将 import 移到文件顶部
# 这样可以确保 import jax 发生在 os.environ 配置之后
if True:
    # 导入JAX和其他依赖
    import jax
    import jax.numpy as jp
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rl.envs import VelocityTrackingEnv, create_velocity_tracking_env
    from rl.models.networks import ActorCriticNetwork, count_parameters
    from rl.models.optimizer import create_ppo_optimizer_cosine
    from rl.training.logger import Logger, MetricsLogger
    from rl.training.ppo_trainer import PPOConfig, PPOTrainer, create_train_step_fn
    from rl.training.train_state import create_train_state
    from rl.utils.performance_monitor import PerformanceMonitor, benchmark_train_step
    from rl.utils.checkpoint import create_checkpoint_manager


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


def load_config_from_yaml(config_path: str) -> dict:
    """从YAML文件加载配置

    Args:
        config_path: YAML配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        console.print(f"[yellow]警告: 配置文件不存在: {config_path}[/yellow]")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    console.print(f"[green]✓ 从 {config_path} 加载配置[/green]")
    return config


def main():
    """主训练函数"""
    # 第一阶段：解析 --config 参数
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="YAML配置文件路径"
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # 加载YAML配置
    yaml_config = load_config_from_yaml(pre_args.config)

    # 第二阶段：解析所有参数，使用YAML中的值作为默认值
    parser = argparse.ArgumentParser(
        description="PPO训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser]
    )

    # 场景配置
    parser.add_argument(
        "--scene",
        type=str,
        default=yaml_config.get("scene", "flat_terrain"),
        choices=["flat_terrain", "rough_terrain"],
        help="选择训练场景"
    )

    # 环境配置
    parser.add_argument("--num-envs", type=int,
                        default=yaml_config.get("num_envs", 7680),
                        help="并行环境数")
    parser.add_argument("--num-steps", type=int,
                        default=yaml_config.get("num_steps", 64),
                        help="每次rollout的步数")

    # PPO超参数
    parser.add_argument("--num-epochs", type=int,
                        default=yaml_config.get("num_epochs", 4),
                        help="每次update的epoch数")
    parser.add_argument("--num-minibatches", type=int,
                        default=yaml_config.get("num_minibatches", 4),
                        help="mini-batch数量")
    parser.add_argument("--gamma", type=float,
                        default=yaml_config.get("gamma", 0.99),
                        help="折扣因子")
    parser.add_argument("--gae-lambda", type=float,
                        default=yaml_config.get("gae_lambda", 0.95),
                        help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float,
                        default=yaml_config.get("clip_epsilon", 0.2),
                        help="PPO裁剪系数")
    parser.add_argument("--value-coef", type=float,
                        default=yaml_config.get("value_coef", 0.5),
                        help="价值损失系数")
    parser.add_argument("--entropy-coef", type=float,
                        default=yaml_config.get("entropy_coef", 0.01),
                        help="熵正则化系数")
    parser.add_argument("--max-grad-norm", type=float,
                        default=yaml_config.get("max_grad_norm", 0.5),
                        help="梯度裁剪阈值")

    # 训练配置
    parser.add_argument("--total-timesteps", type=int,
                        default=yaml_config.get(
                            "total_timesteps", 200_000_000),
                        help="总训练步数")
    parser.add_argument("--log-interval", type=int,
                        default=yaml_config.get("log_interval", 100),
                        help="日志记录间隔")
    parser.add_argument("--eval-interval", type=int,
                        default=yaml_config.get("eval_interval", 500),
                        help="评估间隔")
    parser.add_argument("--save-interval", type=int,
                        default=yaml_config.get("save_interval", 100),
                        help="模型保存间隔")

    # 优化器配置
    parser.add_argument("--learning-rate", type=float,
                        default=yaml_config.get("learning_rate", 1e-3),
                        help="峰值学习率")
    parser.add_argument("--final-lr-fraction", type=float,
                        default=yaml_config.get("final_lr_fraction", 0.02),
                        help="最终学习率相对峰值的比例")

    # 网络配置
    parser.add_argument("--hidden-dims", type=int, nargs="+",
                        default=yaml_config.get(
                            "hidden_dims", [512, 512, 256]),
                        help="隐藏层维度列表")
    parser.add_argument("--shared-backbone", action="store_true",
                        default=yaml_config.get("shared_backbone", True),
                        help="是否使用共享backbone")
    parser.add_argument("--no-shared-backbone", dest="shared_backbone",
                        action="store_false",
                        help="不使用共享backbone")

    # 视频录制配置
    parser.add_argument(
        "--enable-video",
        action="store_true",
        default=yaml_config.get("enable_video", False),
        help="启用训练过程视频录制"
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=yaml_config.get("video_interval", 200),
        help="视频录制间隔（单位：updates）"
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=yaml_config.get("video_frames", 180),
        help="每个视频的帧数（默认 180帧 = 3秒@60fps）"
    )
    parser.add_argument(
        "--video-camera",
        type=str,
        default=yaml_config.get("video_camera", "track"),
        help="录制视频使用的相机名称"
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            f"[bold green]PPO 训练[/bold green]\n"
            f"[dim]JAX + MJX + Flax实现[/dim]\n"
            f"[yellow]场景: {args.scene}[/yellow]\n"
            f"[dim]配置: {pre_args.config}[/dim]",
            border_style="green",
        )
    )

    # ==================== 配置 ====================
    console.print("\n[bold cyan]1. 加载配置[/bold cyan]")

    config = PPOConfig(
        # 环境配置
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        # PPO超参数
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        # 训练配置
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    print_config(config)

    # ==================== 设备检测 ====================
    console.print("\n[bold cyan]2. 检测计算设备[/bold cyan]")
    devices = jax.devices()
    console.print(f"可用设备: {devices}")
    console.print(f"默认后端: {jax.default_backend()}")

    # ==================== 创建环境 ====================
    console.print("\n[bold cyan]3. 创建MJX环境[/bold cyan]")

    scene_path = f"assets/xmls/scenes/{args.scene}.xml"
    if not os.path.exists(os.path.join(os.path.dirname(__file__), scene_path)):
        pass
    env = create_velocity_tracking_env(xml_path=scene_path)
    console.print(f"✓ 环境创建完成")
    console.print(f"  观测维度: {env.observation_size}")
    console.print(f"  动作维度: {env.action_size}")

    # ==================== 创建网络 ====================
    console.print("\n[bold cyan]4. 创建Actor-Critic网络[/bold cyan]")

    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=args.shared_backbone,
        hidden_dims=tuple(args.hidden_dims),
    )

    # 初始化网络以统计参数
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jp.zeros((1, env.observation_size))
    params = network.init(init_rng, dummy_obs)
    num_params = count_parameters(params)

    # 🔍 诊断：检查网络参数是否包含 NaN
    params_flat = jax.tree_util.tree_leaves(params)
    params_has_nan = any(jp.isnan(p).any() for p in params_flat)
    params_has_inf = any(jp.isinf(p).any() for p in params_flat)

    # 测试前向传播
    mean, log_std, value = network.apply(params, dummy_obs)
    forward_has_nan = jp.isnan(mean).any() or jp.isnan(
        log_std).any() or jp.isnan(value).any()

    console.print(f"✓ 网络创建完成")
    console.print(f"  参数数量: {num_params:,}")
    console.print(f"  共享backbone: {args.shared_backbone}")
    console.print(f"  隐藏层: {args.hidden_dims}")
    console.print(
        f"[yellow]  参数诊断: has_nan={params_has_nan}, has_inf={params_has_inf}[/yellow]")
    console.print(
        f"[yellow]  前向传播诊断: has_nan={forward_has_nan}, mean_range=[{mean.min():.4f}, {mean.max():.4f}], log_std_range=[{log_std.min():.4f}, {log_std.max():.4f}][/yellow]")

    if params_has_nan or params_has_inf:
        console.print(f"[red]⚠️  警告：网络参数包含 NaN 或 Inf！[/red]")
    if forward_has_nan:
        console.print(f"[red]⚠️  警告：前向传播输出包含 NaN！[/red]")

    # ==================== 创建优化器 ====================
    console.print("\n[bold cyan]5. 创建带余弦退火的Optax优化器[/bold cyan]")

    # 计算调度参数
    total_updates = config.num_updates
    console.print(
        f"[dim]总更新次数计算: {config.total_timesteps:,} / ({config.num_envs} × {config.num_steps}) = {total_updates:,}[/dim]"
    )

    # 确保有足够的更新次数用于调度
    if total_updates < 50:
        console.print(
            f"[red]错误: 总更新次数({total_updates})过少！需要增加总训练步数或减少batch size[/red]"
        )
        # 自动调整：将总训练步数增加到保证至少100次更新
        required_timesteps = config.batch_size * 100
        console.print(
            f"[yellow]自动调整: 将total_timesteps增加到{required_timesteps:,}[/yellow]"
        )
        config.total_timesteps = required_timesteps
        total_updates = config.num_updates

    # 计算warmup_steps：5%的步数作为预热（增强稳定性）
    warmup_steps = max(10, total_updates // 20)
    console.print(
        f"[dim]warmup_steps: {warmup_steps:,} (约{warmup_steps / total_updates * 100:.1f}%)[/dim]"
    )

    optimizer = create_ppo_optimizer_cosine(
        learning_rate=args.learning_rate,
        total_steps=total_updates,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        final_lr_fraction=args.final_lr_fraction,
    )

    console.print(f"✓ 优化器创建完成")
    console.print(f"  调度类型: 余弦退火 + Warmup")
    console.print(f"  峰值学习率: {args.learning_rate:.2e}")
    console.print(
        f"  预热步数: {warmup_steps:,} ({warmup_steps / total_updates * 100:.1f}%)"
    )
    console.print(f"  总更新次数: {total_updates:,}")
    console.print(
        f"  最终学习率: {args.learning_rate * args.final_lr_fraction:.2e}")
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

    # 🔍 诊断：检查环境重置后的观测是否包含 NaN
    obs_has_nan = jp.isnan(env_state.obs).any()
    obs_has_inf = jp.isinf(env_state.obs).any()
    obs_min = env_state.obs.min()
    obs_max = env_state.obs.max()

    console.print(f"✓ 环境初始化完成")
    console.print(f"  环境数量: {config.num_envs}")
    console.print(f"  观测形状: {env_state.obs.shape}")
    console.print(
        f"[yellow]  观测诊断: has_nan={obs_has_nan}, has_inf={obs_has_inf}, min={obs_min:.4f}, max={obs_max:.4f}[/yellow]")

    if obs_has_nan or obs_has_inf:
        console.print(f"[red]⚠️  警告：环境重置后观测包含 NaN 或 Inf！[/red]")
        # 打印第一个环境的观测值用于调试
        console.print(f"[dim]第一个环境的观测: {env_state.obs[0]}[/dim]")

    # ==================== 创建日志系统 ====================
    console.print("\n[bold cyan]8. 创建日志系统[/bold cyan]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ppo_{timestamp}"
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

    # ==================== 创建视频录制器（可选） ====================
    video_recorder = None

    if args.enable_video:
        console.print("\n[bold cyan]10. 创建视频录制器[/bold cyan]")
        from rl.utils.renderer import VideoRecorder

        try:
            video_recorder = VideoRecorder(
                mujoco_model=env.mj_model,
                output_dir=f"{log_dir}/videos",
                width=1920,
                height=1080,
                fps=60,
                camera_name=args.video_camera,
            )

            console.print(f"✓ 视频录制器创建完成")
            console.print(f"  输出目录: {log_dir}/videos")
            console.print(f"  录制间隔: 每 {args.video_interval} updates")
            console.print(
                f"  视频时长: {args.video_frames / 60:.1f}秒 ({args.video_frames}帧@60fps)")

        except Exception as e:
            console.print(f"[yellow]警告: 视频录制器创建失败: {e}[/yellow]")
            console.print(f"[yellow]将禁用视频录制，继续训练[/yellow]")
            video_recorder = None

    # ==================== JIT编译 ====================
    step_number = "11" if args.enable_video else "10"
    console.print(f"\n[bold cyan]{step_number}. JIT编译[/bold cyan]")

    # 检查缓存状态
    cache_files = list(Path(cache_path).glob("*.cache"))
    if cache_files:
        console.print(f"[dim]找到 {len(cache_files)} 个缓存文件，将加速编译[/dim]")
    else:
        console.print("[dim]首次编译，将创建缓存以加速后续训练[/dim]")

    console.print("正在编译 JAX 计算图，第一次运行可能需要几分钟...")

    # ✅ 使用纯函数版本（支持持久化缓存）
    train_step_fn = create_train_step_fn(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )
    train_step_jit = jax.jit(train_step_fn)

    # 触发一次编译
    t0 = time.time()
    train_state, env_state, info = train_step_jit(train_state, env_state)
    jax.block_until_ready(train_state)  # 确保编译完成
    compile_time = time.time() - t0
    console.print(f"✓ 编译完成 (耗时: {compile_time:.2f}s)")

    # ==================== 性能基准测试（已跳过） ====================
    # console.print("\n[bold cyan]11. 跳过性能基准测试（直接开始训练）[/bold cyan]")

    # 创建性能监控器
    perf_monitor = PerformanceMonitor()
    perf_monitor.set_compile_time(compile_time)

    # ==================== 开始训练 ====================
    logger.print_section("开始训练")

    metrics_logger = MetricsLogger()
    perf_monitor.start()  # 启动性能监控

    # ==================== 视频录制辅助函数 ====================
    def _record_training_video(
        video_recorder,
        env_state,
        env,
        train_state,
        update: int,
        num_frames: int,
        network,
    ):
        """录制训练视频片段

        Args:
            video_recorder: VideoRecorder 实例
            env_state: 当前环境状态（批量，shape: (num_envs, ...)）
            env: 环境实例
            train_state: 训练状态
            update: 当前更新次数
            num_frames: 录制帧数
            network: Actor-Critic 网络
        """
        from rich.console import Console

        console = Console()
        console.print(
            f"\n[yellow]📹 录制视频中（Update {update}）...[/yellow]", end="")

        try:
            video_recorder.start_recording()

            # 选择第一个环境（env_idx=0）
            env_idx = 0
            current_full_state = env_state  # 保持完整批量状态

            # 录制指定帧数
            for frame_idx in range(num_frames):
                # 提取单个环境的 MJX 数据
                single_mjx_data = jax.tree_map(
                    lambda x: x[env_idx],
                    current_full_state.pipeline_state
                )

                # 准备指标信息
                metrics = {
                    "episode_reward": float(current_full_state.reward[env_idx]),
                    "update": update,
                    "env_steps": int(train_state.env_steps),
                }

                # 提取速度信息（如果环境提供）
                if hasattr(current_full_state, "info") and current_full_state.info:
                    info_dict = current_full_state.info

                    if "command" in info_dict:
                        cmd = info_dict["command"]
                        if hasattr(cmd, "__getitem__"):  # 检查是否可索引
                            cmd_single = cmd[env_idx] if len(
                                cmd.shape) > 1 else cmd
                            metrics.update({
                                "cmd_vx": float(cmd_single[0]),
                                "cmd_vy": float(cmd_single[1]),
                                "cmd_vyaw": float(cmd_single[2]),
                            })

                    if "actual_velocity" in info_dict:
                        vel = info_dict["actual_velocity"]
                        if hasattr(vel, "__getitem__"):
                            vel_single = vel[env_idx] if len(
                                vel.shape) > 1 else vel
                            metrics.update({
                                "actual_vx": float(vel_single[0]),
                                "actual_vy": float(vel_single[1]),
                                "actual_vyaw": float(vel_single[2]),
                            })

                # 添加帧到视频
                video_recorder.add_frame_from_mjx(single_mjx_data, metrics)

                # 执行一步仿真（使用确定性策略）
                obs = current_full_state.obs[env_idx:env_idx+1]  # (1, obs_dim)
                mean, log_std, value = network.apply(train_state.params, obs)
                action = mean[0]  # 使用均值作为确定性动作

                # 提取单个环境状态
                single_state = jax.tree_map(
                    lambda x: x[env_idx], current_full_state)

                # 执行步进
                new_single_state = env.step(single_state, action)

                # 检测 episode 终止，自动重置
                if new_single_state.done:
                    rng, reset_rng = jax.random.split(train_state.rng)
                    new_single_state = env.reset(reset_rng)

                # 更新批量状态（仅更新 env_idx=0）
                current_full_state = jax.tree_map(
                    lambda full_arr, single_val: (
                        full_arr.at[env_idx].set(single_val)
                        if hasattr(full_arr, "at") else full_arr
                    ),
                    current_full_state,
                    new_single_state
                )

            # 保存视频
            video_path = video_recorder.save_video(f"update_{update:06d}")
            console.print(f" [green]✓ 完成！[/green]")
            console.print(f"   保存至: {video_path}")

        except Exception as e:
            console.print(f" [red]✗ 失败[/red]")
            console.print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 纯函数训练循环（隔离UI代码） ====================
    def pure_train_loop(
        train_state, env_state, info,
        update_callback=None,  # 回调函数用于UI更新
        video_recorder=None,    # 可选的视频录制器
        video_config=None,      # 视频配置字典
    ):
        """纯函数训练循环，不依赖UI对象（避免触发JIT重新编译）

        Args:
            train_state: 训练状态
            env_state: 环境状态
            info: 初始信息
            update_callback: 可选的回调函数，签名为 callback(update, info)
            video_recorder: 可选的视频录制器
            video_config: 视频配置字典 {"interval": int, "frames": int}

        Returns:
            (train_state, env_state, info): 最终训练状态
        """
        # 记录第一次更新
        metrics_logger.log_dict(info)
        if update_callback:
            update_callback(0, info)

        # 训练循环
        for update in range(1, config.num_updates):
            # ✅ 纯函数调用，无UI依赖
            train_state, env_state, info = train_step_jit(
                train_state, env_state)

            # 确保计算完成（用于准确的性能测量）
            jax.block_until_ready(train_state)

            # 记录性能指标
            perf_metrics = perf_monitor.step(config.batch_size)
            info.update(perf_metrics)

            # 计算当前学习率（近似值）
            current_step = update
            if current_step < warmup_steps:
                current_lr = args.learning_rate * \
                    (current_step / warmup_steps)
            else:
                progress_ratio = (current_step - warmup_steps) / (
                    total_updates - warmup_steps
                )
                current_lr = 0.5 * args.learning_rate * \
                    (1 + jp.cos(jp.pi * progress_ratio))

            info["learning_rate"] = float(current_lr)

            # 记录指标
            metrics_logger.log_dict(info)

            # ✅ 通过回调更新UI（隔离UI逻辑）
            if update_callback:
                update_callback(update, info)

            # ==================== 视频录制逻辑 ====================
            if video_recorder and video_config and update > 0:
                if update % video_config["interval"] == 0:
                    _record_training_video(
                        video_recorder=video_recorder,
                        env_state=env_state,
                        env=env,
                        train_state=train_state,
                        update=update,
                        num_frames=video_config["frames"],
                        network=network,
                    )

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

            # 定期保存检查点
            if (update + 1) % args.save_interval == 0:
                avg_metrics_for_save = metrics_logger.get_averages()
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    train_state=train_state,
                    step=train_state.step,
                    metrics=avg_metrics_for_save,
                )
                console.print(f"[dim]💾 检查点已保存: {checkpoint_path}[/dim]")

        return train_state, env_state, info

    # ==================== UI层：创建显示并绑定回调 ====================
    training_display = logger.create_training_display(
        total=config.num_updates,
        steps_per_epoch=1,  # 每个update作为一个epoch
        description="PPO训练"
    )

    try:
        with training_display:
            # 定义UI更新回调（与训练逻辑完全分离）
            def on_update(update, info):
                training_display.update(epoch=update, step=1, metrics=info)

            # 定义视频配置（如果启用）
            video_config = None
            if video_recorder:
                video_config = {
                    "interval": args.video_interval,
                    "frames": args.video_frames,
                }

            # 执行纯函数训练循环
            train_state, env_state, info = pure_train_loop(
                train_state, env_state, info,
                update_callback=on_update,
                video_recorder=video_recorder,  # 传入录制器
                video_config=video_config,       # 传入配置
            )

        # ==================== 训练完成 ====================
        # 保存最终检查点
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            train_state=train_state,
            step=train_state.step,
            metrics=metrics_logger.get_averages(),
            force=True,  # 强制保存
        )
        console.print(f"[green]💾 最终检查点已保存: {final_checkpoint_path}[/green]")

        # 显示最佳模型信息
        best_info = checkpoint_manager.get_best_model_info()
        if best_info:
            console.print(f"[green]🏆 最佳模型信息:[/green]")
            for key, value in best_info.items():
                console.print(f"  {key}: {value}")

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

        # 清理视频录制器
        if video_recorder:
            video_recorder.close()

        console.print("\n[dim]日志已保存[/dim]")


if __name__ == "__main__":
    main()
