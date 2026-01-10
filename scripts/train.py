"""
PPO训练主脚本
"""

import site
import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import yaml

# ============================================================================================
# ======================================= JAX环境配置 =========================================
# ============================================================================================

# 🔧 指定使用 GPU device:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 启用JAX编译缓存
cache_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".jax_cache")
)
os.makedirs(cache_path, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

# 🔧 增强缓存配置
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

# 最大化显存使用
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

# 设置 CUDA 数据目录 (Triton)

site_packages = site.getsitepackages()[0]
triton_cuda_dir = os.path.join(
    site_packages, "triton", "backends", "nvidia", "lib")
if os.path.exists(triton_cuda_dir):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") +
        f" --xla_gpu_cuda_data_dir={triton_cuda_dir}"
    )

# 启用编译优化
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_enable_latency_hiding_scheduler=true"
    + " --xla_gpu_enable_highest_priority_async_stream=true"
    + " --xla_gpu_autotune_level=1"
    + " --xla_gpu_deterministic_ops=false"
    + " --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
)

# 忽略警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=Warning)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================================
# ===================================== END: JAX环境配置 =========================================
# ============================================================================================


# ============================================================================================
# ======================================= 延迟导入 ============================================
# ============================================================================================

if True:
    import jax
    import jax.numpy as jp
    from rich import box
    from rich.console import Console
    from rich.panel import Panel

    from rl.curriculum import ConfigurableCurriculum, WalkingCurriculum
    from rl.envs import (
        StandingEnv,
        VelocityTrackingEnv,
        WalkingEnv,
        create_standing_env,
        create_velocity_tracking_env,
        create_walking_env,
    )
    from rl.models.networks import ActorCriticNetwork, count_parameters
    from rl.models.optimizer import create_ppo_optimizer_cosine
    from rl.training.logger import (Logger, MetricsLogger,
                                    create_training_display, print_summary)
    from rl.training.ppo_trainer import (PPOConfig, PPOTrainer,
                                         create_train_step_fn)
    from rl.training.train_state import create_train_state
    from rl.utils.checkpoint import create_checkpoint_manager
    from rl.utils.performance_monitor import PerformanceMonitor

console = Console()

# ============================================================================================
# ===================================== END: 延迟导入 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 辅助函数 ============================================
# ============================================================================================


def print_config(config: PPOConfig):
    """打印训练配置"""
    from rich.table import Table

    table = Table(title="训练配置", box=box.ROUNDED)
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green", justify="right")

    table.add_row("并行环境数", str(config.num_envs))
    table.add_row("Rollout步数", str(config.num_steps))
    table.add_row("总批次大小", str(config.batch_size))

    table.add_row("Epoch数", str(config.num_epochs))
    table.add_row("Mini-batch数", str(config.num_minibatches))
    table.add_row("Mini-batch大小", str(config.minibatch_size))
    table.add_row("折扣因子γ", str(config.gamma))
    table.add_row("GAE λ", str(config.gae_lambda))
    table.add_row("裁剪系数ε", str(config.clip_epsilon))
    table.add_row("价值损失系数", str(config.value_coef))
    table.add_row("熵系数", str(config.entropy_coef))
    table.add_row("梯度裁剪", str(config.max_grad_norm))

    table.add_row("总时间步", f"{config.total_timesteps:,}")
    table.add_row("总更新次数", f"{config.num_updates:,}")
    table.add_row("日志间隔", str(config.log_interval))

    console.print(table)


# --------------------------------------------------------------------------------------------


def load_config_from_yaml(config_path: str) -> dict:
    """从YAML文件加载配置"""
    if not os.path.exists(config_path):
        console.print(f"[yellow]警告: 配置文件不存在: {config_path}[/yellow]")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    console.print(f"[green]✓ 从 {config_path} 加载配置[/green]")
    return config


# --------------------------------------------------------------------------------------------


def _record_training_video(
    video_recorder,
    env_state,
    env,
    train_state,
    update: int,
    num_frames: int,
    network,
):
    """录制训练视频片段"""
    from rich.console import Console

    console = Console()
    console.print(f"\n[yellow]📹 录制视频中（Update {update}）...[/yellow]", end="")

    try:
        video_recorder.start_recording()
        env_idx = 0
        current_full_state = env_state
        video_rng = jax.random.fold_in(train_state.rng, update)

        for frame_idx in range(num_frames):
            single_mjx_data = jax.tree_map(
                lambda x: x[env_idx], current_full_state.pipeline_state
            )

            metrics = {
                "episode_reward": float(current_full_state.reward[env_idx]),
                "update": update,
                "env_steps": int(train_state.env_steps),
            }

            if hasattr(current_full_state, "info") and current_full_state.info:
                info_dict = current_full_state.info
                if "command" in info_dict:
                    cmd = info_dict["command"]
                    if hasattr(cmd, "__getitem__") and hasattr(cmd, "shape"):
                        cmd_single = cmd[env_idx] if len(
                            cmd.shape) > 1 else cmd
                        metrics.update(
                            {
                                "cmd_vx": float(cmd_single[0]),
                                "cmd_vy": float(cmd_single[1]),
                                "cmd_vyaw": float(cmd_single[2]),
                            }
                        )
                if "actual_velocity" in info_dict:
                    vel = info_dict["actual_velocity"]
                    if hasattr(vel, "__getitem__") and hasattr(vel, "shape"):
                        vel_single = vel[env_idx] if len(
                            vel.shape) > 1 else vel
                        metrics.update(
                            {
                                "actual_vx": float(vel_single[0]),
                                "actual_vy": float(vel_single[1]),
                                "actual_vyaw": float(vel_single[2]),
                            }
                        )

            video_recorder.add_frame_from_mjx(single_mjx_data, metrics)

            obs = current_full_state.obs[env_idx: env_idx + 1]
            mean, log_std, value = network.apply(train_state.params, obs)
            action = mean[0]

            single_state = jax.tree_map(
                lambda x: x[env_idx], current_full_state)
            new_single_state = env.step(single_state, action)

            if new_single_state.done:
                video_rng, reset_rng = jax.random.split(video_rng)
                new_single_state = env.reset(reset_rng)

            current_full_state = jax.tree_map(
                lambda full_arr, single_val: (
                    full_arr.at[env_idx].set(single_val)
                    if hasattr(full_arr, "at")
                    else full_arr
                ),
                current_full_state,
                new_single_state,
            )

        video_path = video_recorder.save_video(f"update_{update:06d}")
        console.print(f" [green]✓ 完成！[/green]")
        console.print(f"   保存至: {video_path}")

    except Exception as e:
        console.print(f" [red]✗ 失败[/red]")
        console.print(f"   错误: {e}")
        import traceback

        traceback.print_exc()


# ============================================================================================
# ===================================== END: 辅助函数 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 主训练逻辑 ==========================================
# ============================================================================================


def main():
    """主训练函数"""
    # -------------------------------- 1. 解析参数 --------------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/train/train.yaml",
        help="YAML配置文件路径",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()
    yaml_config = load_config_from_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description="PPO训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )

    # 场景配置
    parser.add_argument(
        "--scene",
        type=str,
        default=yaml_config.get("scene", "flat_terrain"),
        choices=["flat_terrain", "rough_terrain"],
        help="选择训练场景",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="直接指定XML场景文件路径",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default=yaml_config.get("env_type", "velocity"),
        choices=["velocity", "walking", "standing"],
        help="环境类型",
    )

    # 环境参数
    parser.add_argument(
        "--num-envs",
        type=int,
        default=yaml_config.get("num_envs", 7680),
        help="并行环境数",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=yaml_config.get("num_steps", 64),
        help="每次rollout的步数",
    )

    # PPO参数
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=yaml_config.get("num_epochs", yaml_config.get("num-epochs", 4)),
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=yaml_config.get("num_minibatches", 4)
    )
    parser.add_argument("--gamma", type=float,
                        default=yaml_config.get("gamma", 0.99))
    parser.add_argument(
        "--gae-lambda", type=float, default=yaml_config.get("gae_lambda", 0.95)
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=yaml_config.get("clip_epsilon", 0.2)
    )
    parser.add_argument(
        "--value-coef", type=float, default=yaml_config.get("value_coef", 0.5)
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=yaml_config.get("entropy_coef", 0.01)
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=yaml_config.get("max_grad_norm", 0.5)
    )

    # 训练流程参数
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=yaml_config.get("total_timesteps", 200_000_000),
    )
    parser.add_argument(
        "--log-interval", type=int, default=yaml_config.get("log_interval", 100)
    )
    parser.add_argument(
        "--eval-interval", type=int, default=yaml_config.get("eval_interval", 500)
    )
    parser.add_argument(
        "--save-interval", type=int, default=yaml_config.get("save_interval", 100)
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从已有检查点继续训练（支持传入 checkpoint 文件，或 run/checkpoints 目录）",
    )

    # 优化器与网络
    parser.add_argument(
        "--learning-rate", type=float, default=yaml_config.get("learning_rate", 1e-3)
    )
    parser.add_argument(
        "--final-lr-fraction",
        type=float,
        default=yaml_config.get("final_lr_fraction", 0.02),
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=yaml_config.get("hidden_dims", [512, 512, 256]),
    )
    parser.add_argument(
        "--shared-backbone",
        action="store_true",
        default=yaml_config.get("shared_backbone", True),
    )
    parser.add_argument(
        "--no-shared-backbone", dest="shared_backbone", action="store_false"
    )

    # 视频录制
    parser.add_argument(
        "--enable-video",
        action="store_true",
        default=yaml_config.get("enable_video", False),
    )
    parser.add_argument(
        "--video-interval", type=int, default=yaml_config.get("video_interval", 200)
    )
    parser.add_argument(
        "--video-frames", type=int, default=yaml_config.get("video_frames", 180)
    )
    parser.add_argument(
        "--video-camera", type=str, default=yaml_config.get("video_camera", "track")
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            f"[bold green]PPO 训练[/bold green]\n"
            f"[dim]JAX + MJX + Flax实现[/dim]\n"
            f"[yellow]场景: {args.scene}[/yellow]\n"
            f"[dim]配置: {pre_args.config}[/dim",
            border_style="green",
        )
    )

    # -------------------------------- 2. 初始化配置与环境 --------------------------------
    console.print("\n[bold cyan]1. 加载配置[/bold cyan]")
    config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )
    print_config(config)

    console.print("\n[bold cyan]2. 检测计算设备[/bold cyan]")
    console.print(f"可用设备: {jax.devices()}")
    console.print(f"默认后端: {jax.default_backend()}")

    console.print("\n[bold cyan]3. 创建MJX environment[/bold cyan]")

    # 获取机器人配置
    robot_name = yaml_config.get("robot_name", None)  # 从配置文件读取robot_name
    if robot_name:
        from rl.utils.robot_config import resolve_scene_path
        xml_path = str(resolve_scene_path(robot_name))
        console.print(f"使用机器人模型: [bold cyan]{robot_name}[/bold cyan]")
        console.print(f"解析场景路径: [bold green]{xml_path}[/bold green]")
    else:
        # 向后兼容：使用旧的路径逻辑
        if args.xml_path:
            xml_path = args.xml_path
            console.print(f"使用自定义XML路径: {xml_path}")
        else:
            scene_file_map = {
                "flat_terrain": "flat_terrain",
                "rough_terrain": "rough_terrain",
            }
            scene_file = scene_file_map.get(args.scene, args.scene)
            xml_path = f"assets/xmls/scenes/{scene_file}.xml"
            console.print(f"使用预设场景: {args.scene}")

    t0 = time.time()
    if args.env_type == "walking":
        env = create_walking_env(xml_path=xml_path, robot_name=robot_name)
        env_create_time = time.time() - t0
        console.print(f"✓ WalkingEnv 创建完成 (耗时: {env_create_time:.2f}s)")
    elif args.env_type == "standing":
        reward_weights = yaml_config.get("reward_weights")
        if reward_weights is not None and not isinstance(reward_weights, dict):
            console.print(
                "[yellow]警告: reward_weights 不是字典，已忽略（standing env 可用）[/yellow]"
            )
            reward_weights = None
        env = create_standing_env(
            xml_path=xml_path, robot_name=robot_name, reward_weights=reward_weights
        )
        env_create_time = time.time() - t0
        console.print(f"✓ StandingEnv 创建完成 (耗时: {env_create_time:.2f}s)")
    else:
        env = create_velocity_tracking_env(xml_path=xml_path, robot_name=robot_name)
        env_create_time = time.time() - t0
        console.print(f"✓ VelocityTrackingEnv 创建完成 (耗时: {env_create_time:.2f}s)")

    # 应用 YAML env_config（对standing/velocity尤其重要；walking 后续可能被 curriculum 覆盖）
    env_config = yaml_config.get("env_config", {})
    if isinstance(env_config, dict) and env_config:
        for key, value in env_config.items():
            if hasattr(env, key):
                setattr(env, key, value)

    console.print(f"  观测维度: {env.observation_size}")
    console.print(f"  动作维度: {env.action_size}")

    # -------------------------------- 3. 课程学习 --------------------------------
    curriculum = None
    if args.env_type == "walking":
        console.print("\n[bold cyan]3.5. 初始化课程学习[/bold cyan]")
        t0 = time.time()
        curriculum_file = yaml_config.get("curriculum_file") or yaml_config.get(
            "weights_file"
        )

        if curriculum_file:
            file_path = os.path.join(os.getcwd(), curriculum_file)
            if os.path.exists(file_path):
                config_content = load_config_from_yaml(file_path)
                env_config = yaml_config.get("env_config", {})

                curriculum = ConfigurableCurriculum(
                    config=config_content,
                    env_config=env_config,
                    default_stage_name="CustomStage",
                )
                console.print(
                    f"[yellow]使用自定义课程学习 (源: {curriculum_file})[/yellow]")

                if "stages" in config_content:
                    console.print(
                        f"  检测到多阶段定义 ({len(config_content['stages'])} 个阶段)")
                elif env_config:
                    console.print(f"  环境配置: {env_config}")
            else:
                console.print(
                    f"[red]错误: 找不到课程文件 {file_path}，回退到标准 WalkingCurriculum[/red]"
                )
                curriculum = WalkingCurriculum()
        else:
            try:
                curriculum = WalkingCurriculum()
                console.print("使用标准行走课程学习 (WalkingCurriculum)")
            except FileNotFoundError as e:
                console.print(f"[red]错误: {e}[/red]")
                console.print(
                    "[red]无法加载标准课程配置，请检查 configs/train/curriculum.yaml 是否存在[/red]"
                )
                sys.exit(1)

        curriculum_init_time = time.time() - t0
        console.print(f"✓ 课程学习模块创建完成 (耗时: {curriculum_init_time:.2f}s)")
        console.print(f"  阶段数: {len(curriculum.stages)}")
        for i, stage in enumerate(curriculum.stages):
            console.print(
                f"  阶段{i+1}: {stage.name} ({stage.step_range[0]:,}-{stage.step_range[1]:,} steps)"
            )
        curriculum.apply_to_env(env, current_step=0)
        console.print(f"[dim]已应用初始阶段: {curriculum.stages[0].name}[/dim]")
    elif yaml_config.get("curriculum_file") or yaml_config.get("weights_file"):
        console.print(
            "[dim]提示: 当前 env_type!=walking，已忽略 curriculum_file/weights_file（仅walking支持课程学习）[/dim]"
        )

    # -------------------------------- 4. 网络与优化器 --------------------------------
    console.print("\n[bold cyan]4. 创建Actor-Critic网络[/bold cyan]")
    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=args.shared_backbone,
        hidden_dims=tuple(args.hidden_dims),
    )

    t0 = time.time()
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jp.zeros((1, env.observation_size))
    params = network.init(init_rng, dummy_obs)
    num_params = count_parameters(params)

    params_flat = jax.tree_util.tree_leaves(params)
    params_has_nan = any(jp.isnan(p).any() for p in params_flat)
    params_has_inf = any(jp.isinf(p).any() for p in params_flat)
    mean, log_std, value = network.apply(params, dummy_obs)
    forward_has_nan = (
        jp.isnan(mean).any() or jp.isnan(
            log_std).any() or jp.isnan(value).any()
    )
    network_init_time = time.time() - t0

    console.print(
        f"✓ 网络创建完成 (参数: {num_params:,}, 耗时: {network_init_time:.2f}s)")
    if params_has_nan or params_has_inf:
        console.print(f"[red]⚠️  警告：网络参数包含 NaN 或 Inf！[/red]")
    if forward_has_nan:
        console.print(f"[red]⚠️  警告：前向传播输出包含 NaN！[/red]")

    console.print("\n[bold cyan]5. 创建优化器[/bold cyan]")
    total_updates = config.num_updates
    if total_updates < 50:
        required_timesteps = config.batch_size * 100
        console.print(
            f"[yellow]自动调整: 将total_timesteps增加到{required_timesteps:,}[/yellow]"
        )
        config.total_timesteps = required_timesteps
        total_updates = config.num_updates

    warmup_steps = max(10, total_updates // 20)
    optimizer = create_ppo_optimizer_cosine(
        learning_rate=args.learning_rate,
        total_steps=total_updates,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        final_lr_fraction=args.final_lr_fraction,
    )
    console.print(f"✓ 优化器创建完成 (Warmup: {warmup_steps:,})")

    # -------------------------------- 5. 初始化状态 --------------------------------
    console.print("\n[bold cyan]6. 初始化训练状态[/bold cyan]")
    t0 = time.time()
    train_state = create_train_state(
        network=network,
        optimizer=optimizer,
        obs_shape=(env.observation_size,),
        rng=rng,
    )
    if args.resume_from:
        from flax import serialization

        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            candidates = [
                resume_path / "best_model" / "best_model",
                resume_path / "models",
            ]
            chosen = None
            if candidates[0].exists():
                chosen = candidates[0]
            elif candidates[1].exists() and candidates[1].is_dir():
                ckpts = sorted(
                    candidates[1].glob("checkpoint_*"),
                    key=lambda p: int(p.name.split("_")[1]),
                )
                if ckpts:
                    chosen = ckpts[-1]
            if chosen is None:
                ckpts = sorted(
                    resume_path.glob("checkpoint_*"),
                    key=lambda p: int(p.name.split("_")[1]),
                )
                if ckpts:
                    chosen = ckpts[-1]
            if chosen is None:
                raise FileNotFoundError(f"未找到可恢复的检查点: {resume_path}")
            resume_path = chosen

        with open(resume_path, "rb") as f:
            ckpt_data = serialization.from_bytes(None, f.read())

        if isinstance(ckpt_data, dict) and "params" in ckpt_data:
            # Full resume if optimizer/rng are present, otherwise params-only.
            if "opt_state" in ckpt_data and "rng" in ckpt_data:
                train_state = train_state.replace(
                    step=int(ckpt_data.get("step", 0)),
                    env_steps=int(ckpt_data.get("env_steps", 0)),
                    params=ckpt_data["params"],
                    opt_state=ckpt_data["opt_state"],
                    rng=ckpt_data["rng"],
                )
                console.print(f"[yellow]↻ 从检查点恢复训练: {resume_path}[/yellow]")
            else:
                train_state = train_state.replace(
                    step=int(ckpt_data.get("step", 0)),
                    env_steps=int(ckpt_data.get("env_steps", 0)),
                    params=ckpt_data["params"],
                )
                console.print(
                    f"[yellow]↻ 从参数文件恢复（不含opt_state/rng）: {resume_path}[/yellow]"
                )
        else:
            raise ValueError(f"检查点格式不正确: {resume_path}")

    train_state_init_time = time.time() - t0
    console.print(f"✓ 训练状态初始化完成 (耗时: {train_state_init_time:.2f}s)")

    console.print("\n[bold cyan]7. 初始化批量环境[/bold cyan]")
    rng, reset_rng = jax.random.split(rng)
    t0 = time.time()
    env_state = env.batch_reset(reset_rng, config.num_envs)
    env_reset_time = time.time() - t0
    obs_has_nan = jp.isnan(env_state.obs).any()
    obs_has_inf = jp.isinf(env_state.obs).any()
    console.print(
        f"✓ 环境初始化完成 (Env: {config.num_envs}, 耗时: {env_reset_time:.2f}s)")
    if obs_has_nan or obs_has_inf:
        console.print(f"[red]⚠️  警告：环境重置后观测包含 NaN 或 Inf！[/red]")

    # -------------------------------- 6. 日志与工具 --------------------------------
    console.print("\n[bold cyan]8. 创建日志系统[/bold cyan]")
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/train/ppo_{timestamp}"
    logger = Logger(log_dir=log_dir, use_tensorboard=True, use_rich=True)
    logger_init_time = time.time() - t0
    console.print(f"✓ 日志系统创建完成 ({log_dir}, 耗时: {logger_init_time:.2f}s)")

    console.print("\n[bold cyan]9. 创建检查点管理器[/bold cyan]")
    t0 = time.time()
    checkpoint_manager = create_checkpoint_manager(
        log_dir=log_dir,
        max_to_keep=5,
        keep_best=True,
        metric_name="mean_reward",
        metric_mode="max",
    )
    checkpoint_init_time = time.time() - t0
    console.print(f"✓ 检查点管理器创建完成 (耗时: {checkpoint_init_time:.2f}s)")

    console.print("\n[bold cyan]10. 创建PPO训练器[/bold cyan]")
    t0 = time.time()
    trainer = PPOTrainer(config=config, env=env,
                         network=network, optimizer=optimizer)
    trainer_init_time = time.time() - t0
    console.print(f"✓ 训练器创建完成 (耗时: {trainer_init_time:.2f}s)")

    video_recorder = None
    if args.enable_video:
        console.print("\n[bold cyan]11. 创建视频录制器[/bold cyan]")
        from rl.utils.renderer import VideoRecorder

        try:
            t0 = time.time()
            video_recorder = VideoRecorder(
                mujoco_model=env.mj_model,
                output_dir=f"{log_dir}/videos",
                width=1920,
                height=1080,
                fps=60,
                camera_name=args.video_camera,
            )
            video_recorder_init_time = time.time() - t0
            console.print(f"✓ 视频录制器创建完成 (耗时: {video_recorder_init_time:.2f}s)")
        except Exception as e:
            console.print(f"[yellow]警告: 视频录制器创建失败: {e}[/yellow]")
            video_recorder = None

    # -------------------------------- 7. JIT编译 --------------------------------
    step_number = "12" if args.enable_video else "11"
    console.print(f"\n[bold cyan]{step_number}. JIT编译[/bold cyan]")

    train_step_fn = create_train_step_fn(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )
    train_step_jit = jax.jit(train_step_fn)

    console.print("[dim]正在编译 JAX 计算图...[/dim]")

    # 使用 Live 显示实时更新的编译计时
    from rich.live import Live
    from rich.text import Text
    import threading

    t0 = time.time()
    stop_timer = threading.Event()

    def get_compile_timer_text():
        elapsed = time.time() - t0
        return Text(f"编译中... 已用时: {elapsed:.1f}s", style="bold cyan")

    with Live(get_compile_timer_text(), console=console, refresh_per_second=10) as live:
        def update_compile_timer():
            while not stop_timer.is_set():
                live.update(get_compile_timer_text())
                time.sleep(0.1)

        timer_thread = threading.Thread(
            target=update_compile_timer, daemon=True)
        timer_thread.start()

        train_state, env_state, info = train_step_jit(train_state, env_state)
        jax.block_until_ready(train_state)

        stop_timer.set()
        timer_thread.join(timeout=0.5)

    compile_time = time.time() - t0
    console.print(f"✓ 编译完成 (耗时: {compile_time:.2f}s)")

    # -------------------------------- 8. 训练循环 --------------------------------
    console.print("\n[bold cyan]开始训练[/bold cyan]")
    perf_monitor = PerformanceMonitor()
    perf_monitor.set_compile_time(compile_time)
    metrics_logger = MetricsLogger()
    perf_monitor.start()

    # 纯函数循环逻辑
    def pure_train_loop(
        train_state,
        env_state,
        info,
        update_callback=None,
        video_recorder=None,
        video_config=None,
    ):
        nonlocal train_step_jit
        metrics_logger.log_dict(info)
        if update_callback:
            update_callback(0, info)

        for update in range(1, config.num_updates):
            if curriculum is not None:
                prev_stage = curriculum.current_stage
                curriculum.apply_to_env(env, train_state.env_steps)
                if curriculum.current_stage != prev_stage:
                    console.print(
                        "[yellow]检测到课程阶段切换：重新编译JIT以应用新的环境/奖励配置...[/yellow]"
                    )
                    t_compile = time.time()
                    train_step_fn = create_train_step_fn(
                        config=config,
                        env=env,
                        network=network,
                        optimizer=optimizer,
                    )
                    train_step_jit = jax.jit(train_step_fn)
                    try:
                        train_step_jit.lower(train_state, env_state).compile()
                        console.print(
                            f"[green]✓ 阶段切换编译完成 (耗时: {time.time() - t_compile:.2f}s)[/green]"
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]警告: 阶段切换预编译失败，将在下一次调用时自动编译: {e}[/yellow]"
                        )

            if update == 1:
                # 首次迭代显示
                from rich.console import Console
                from rich.live import Live
                from rich.text import Text
                import threading

                _console = Console()
                _console.print("\n[yellow]⚙️  首次循环迭代中...[/yellow]")

                # 使用 Live 显示实时更新的计时
                t0 = time.time()
                stop_timer = threading.Event()

                def get_timer_text():
                    elapsed = time.time() - t0
                    return Text(f"正在执行... 已用时: {elapsed:.1f}s", style="bold yellow")

                with Live(get_timer_text(), console=_console, refresh_per_second=10) as live:
                    def update_timer():
                        while not stop_timer.is_set():
                            live.update(get_timer_text())
                            time.sleep(0.1)

                    timer_thread = threading.Thread(
                        target=update_timer, daemon=True)
                    timer_thread.start()

                    train_state, env_state, info = train_step_jit(
                        train_state, env_state
                    )
                    jax.block_until_ready(train_state)

                    stop_timer.set()
                    timer_thread.join(timeout=0.5)

                first_iter_time = time.time() - t0
                _console.print(
                    f"[green]✓ 首次迭代完成 (耗时: {first_iter_time:.2f}s)[/green]\n"
                )
            else:
                # 正常迭代
                train_state, env_state, info = train_step_jit(
                    train_state, env_state)
                jax.block_until_ready(train_state)

            # 记录指标
            perf_metrics = perf_monitor.step(config.batch_size)
            info.update(perf_metrics)

            # 计算学习率
            current_step = update
            if current_step < warmup_steps:
                current_lr = args.learning_rate * \
                    (current_step / max(1, warmup_steps))
            else:
                denominator = max(1, total_updates - warmup_steps)
                progress_ratio = (current_step - warmup_steps) / denominator
                current_lr = (
                    0.5 * args.learning_rate *
                    (1 + jp.cos(jp.pi * progress_ratio))
                )
            info["learning_rate"] = float(current_lr)

            # 课程学习信息
            if curriculum is not None:
                stage_info = curriculum.get_stage_info(train_state.env_steps)
                info["curriculum_stage"] = stage_info["stage_name"]
                info["curriculum_stage_index"] = stage_info["stage_index"]
                if stage_info["progress"] is not None:
                    info["curriculum_progress"] = stage_info["progress"]

            metrics_logger.log_dict(info)
            if update_callback:
                update_callback(update, info)

            # 视频录制
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

            # 日志记录
            if (update + 1) % config.log_interval == 0:
                avg_metrics = metrics_logger.get_averages()
                avg_metrics["steps_since_last_log"] = config.log_interval
                logger.log_scalars(
                    metrics=avg_metrics, step=train_state.step, prefix="train"
                )
                metrics_logger.reset()

            # 保存检查点
            if (update + 1) % args.save_interval == 0:
                avg_metrics_for_save = metrics_logger.get_averages()
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    train_state=train_state,
                    step=train_state.step,
                    metrics=avg_metrics_for_save,
                )
                console.print(f"[dim]💾 检查点已保存: {checkpoint_path}[/dim]")

        return train_state, env_state, info

    # 启动UI循环
    training_display = create_training_display(
        console=console,
        total=config.num_updates,
        steps_per_epoch=1,
        description="PPO训练",
    )

    try:
        with training_display:

            def on_update(update, info):
                training_display.update(epoch=update, step=1, metrics=info)

            video_config = None
            if video_recorder:
                video_config = {
                    "interval": args.video_interval,
                    "frames": args.video_frames,
                }

            train_state, env_state, info = pure_train_loop(
                train_state,
                env_state,
                info,
                update_callback=on_update,
                video_recorder=video_recorder,
                video_config=video_config,
            )

        # 结束处理
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            train_state=train_state,
            step=train_state.step,
            metrics=metrics_logger.get_averages(),
            force=True,
        )
        console.print(f"[green]💾 最终检查点已保存: {final_checkpoint_path}[/green]")

        best_info = checkpoint_manager.get_best_model_info()
        if best_info:
            console.print(f"[green]🏆 最佳模型信息:[/green]")
            for key, value in best_info.items():
                console.print(f"  {key}: {value}")

        print_summary(
            "✓ 训练完成！\n"
            f"总步数: {train_state.step}\n"
            f"总环境步数: {train_state.env_steps:,}",
            style="green",
            console=console,
        )

    except KeyboardInterrupt:
        print_summary("训练被用户中断", style="yellow", console=console)
    except Exception as e:
        print_summary(f"训练出错: {e}", style="red", console=console)
        import traceback

        console.print(traceback.format_exc())
    finally:
        logger.close()
        if video_recorder:
            video_recorder.close()
        console.print("\n[dim]日志已保存[/dim]")


if __name__ == "__main__":
    main()
