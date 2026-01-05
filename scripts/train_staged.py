"""
自动化分阶段训练脚本

功能：
1. 自动加载阶段配置
2. 监控训练指标
3. 自动切换到下一阶段
4. 保存每个阶段的最佳模型
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

# ==================== JAX配置 (必须在导入jax之前) ====================
# 🔧 指定使用 GPU device:0（第一张显卡）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 多 GPU 训练

# 启用JAX编译缓存 (使用绝对路径，确保持久化)
cache_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".jax_cache"))
os.makedirs(cache_path, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

# 🔧 增强缓存配置
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"  # 缓存所有编译结果
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"  # 缓存所有编译

# 最大化显存使用
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# 导入JAX
import jax
import jax.numpy as jp

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ==================== 阶段配置 ====================

# 配置文件基础路径
CONFIG_BASE_DIR = "configs/train"
TEST_CONFIG_BASE_DIR = "configs/test"
QUICK_TEST_CONFIG_BASE_DIR = "configs/quick_test"
TEN_HOUR_CONFIG_BASE_DIR = "configs/train-10h"

STAGE_CONFIGS = [
    {
        "stage_id": 0,
        "name": "standing",
        "config_file": f"{CONFIG_BASE_DIR}/stage0_standing.yaml",
        "description": "站立平衡",
        "min_iterations": 2500,     # 164M步 (3天训练优化)
        "transition_criteria": {
            "min_reward": 0.6,
            "max_fall_rate": 0.05,
        },
    },
    {
        "stage_id": 1,
        "name": "stepping",
        "config_file": f"{CONFIG_BASE_DIR}/stage1_stepping.yaml",
        "description": "原地踏步",
        "min_iterations": 5000,     # 328M步
        "transition_criteria": {
            "min_gait_symmetry": 0.5,
            "min_foot_clearance": 0.3,
        },
    },
    {
        "stage_id": 2,
        "name": "slow_walk",
        "config_file": f"{CONFIG_BASE_DIR}/stage2_slow_walk.yaml",
        "description": "小步行走",
        "min_iterations": 10000,    # 655M步
        "transition_criteria": {
            "min_velocity_tracking": 0.7,
        },
    },
    {
        "stage_id": 3,
        "name": "normal_walk",
        "config_file": f"{CONFIG_BASE_DIR}/stage3_normal_walk.yaml",
        "description": "正常行走",
        "min_iterations": 20000,    # 1.31B步 (重点阶段,40%时间)
        "transition_criteria": {
            "min_velocity_tracking": 0.8,
        },
    },
    {
        "stage_id": 4,
        "name": "fast_walk",
        "config_file": f"{CONFIG_BASE_DIR}/stage4_fast_walk.yaml",
        "description": "高速适应",
        "min_iterations": 12500,    # 819M步
        "transition_criteria": {
            "min_velocity_tracking": 0.75,
            "min_max_velocity": 0.9,
        },
    },
    # 暂时禁用stage5和stage6,聚焦核心步态训练
    # {
    #     "stage_id": 5,
    #     "name": "terrain_adaptation",
    #     "config_file": "configs/stage5_terrain.yaml",
    #     "description": "地形适应",
    #     "min_iterations": 4800,
    #     "transition_criteria": {
    #         "min_velocity_ratio": 0.7,
    #         "max_fall_rate": 0.15,
    #     },
    # },
    # {
    #     "stage_id": 6,
    #     "name": "robustness",
    #     "config_file": "configs/stage6_robustness.yaml",
    #     "description": "鲁棒性提升",
    #     "min_iterations": 9500,
    #     "transition_criteria": {
    #         "min_velocity_tracking": 0.8,
    #         "max_fall_rate": 0.1,
    #     },
    # },
]

TEN_HOUR_STAGE_CONFIGS = [
    {
        "stage_id": 0,
        "name": "standing_10h",
        "config_file": f"{TEN_HOUR_CONFIG_BASE_DIR}/stage0_standing.yaml",
        "description": "站立平衡 (10h版)",
        "min_iterations": 100,
        "transition_criteria": {
            "min_reward": 0.5,
            "max_fall_rate": 0.1,
        },
    },
    {
        "stage_id": 1,
        "name": "stepping_10h",
        "config_file": f"{TEN_HOUR_CONFIG_BASE_DIR}/stage1_stepping.yaml",
        "description": "原地踏步 (10h版)",
        "min_iterations": 300,
        "transition_criteria": {
            "min_gait_symmetry": 0.4,
            "min_foot_clearance": 0.2,
        },
    },
    {
        "stage_id": 2,
        "name": "slow_walk_10h",
        "config_file": f"{TEN_HOUR_CONFIG_BASE_DIR}/stage2_slow_walk.yaml",
        "description": "小步行走 (10h版)",
        "min_iterations": 600,
        "transition_criteria": {
            "min_velocity_tracking": 0.6,
        },
    },
    {
        "stage_id": 3,
        "name": "normal_walk_10h",
        "config_file": f"{TEN_HOUR_CONFIG_BASE_DIR}/stage3_normal_walk.yaml",
        "description": "正常行走 (10h版)",
        "min_iterations": 800,
        "transition_criteria": {
            "min_velocity_tracking": 0.7,
        },
    },
    {
        "stage_id": 4,
        "name": "fast_walk_10h",
        "config_file": f"{TEN_HOUR_CONFIG_BASE_DIR}/stage4_fast_walk.yaml",
        "description": "高速适应 (10h版)",
        "min_iterations": 200,
        "transition_criteria": {
            "min_velocity_tracking": 0.65,
        },
    },
]

# ==================== 测试阶段配置（流水线快速测试）====================

TEST_STAGE_CONFIGS = [
    {
        "stage_id": 0,
        "name": "test_standing",
        "config_file": f"{TEST_CONFIG_BASE_DIR}/pipeline_stage0_standing.yaml",
        "description": "快速站立测试",
        "min_iterations": 10,
        "transition_criteria": {
            "min_reward": 0.0,  # 无条件切换
        },
    },
    {
        "stage_id": 1,
        "name": "test_stepping",
        "config_file": f"{TEST_CONFIG_BASE_DIR}/pipeline_stage1_stepping.yaml",
        "description": "快速踏步测试",
        "min_iterations": 10,
        "transition_criteria": {
            "min_gait_symmetry": 0.0,
        },
    },
    {
        "stage_id": 2,
        "name": "test_slow_walk",
        "config_file": f"{TEST_CONFIG_BASE_DIR}/pipeline_stage2_slow_walk.yaml",
        "description": "快速慢走测试",
        "min_iterations": 10,
        "transition_criteria": {
            "min_velocity_tracking": 0.0,
        },
    },
]

# ==================== 极速测试阶段配置（无需重复编译）====================

QUICK_TEST_STAGE_CONFIGS = [
    {
        "stage_id": 0,
        "name": "quick_standing",
        "config_file": f"{QUICK_TEST_CONFIG_BASE_DIR}/stage0_standing.yaml",
        "description": "极速站立测试",
        "min_iterations": 16,
        "transition_criteria": {
            "min_reward": 0.0,  # 无条件切换
        },
    },
    {
        "stage_id": 1,
        "name": "quick_stepping",
        "config_file": f"{QUICK_TEST_CONFIG_BASE_DIR}/stage1_stepping.yaml",
        "description": "极速踏步测试",
        "min_iterations": 16,
        "transition_criteria": {
            "min_gait_symmetry": 0.0,
        },
    },
    {
        "stage_id": 2,
        "name": "quick_slow_walk",
        "config_file": f"{QUICK_TEST_CONFIG_BASE_DIR}/stage2_slow_walk.yaml",
        "description": "极速慢走测试",
        "min_iterations": 16,
        "transition_criteria": {
            "min_velocity_tracking": 0.0,
        },
    },
]


# ==================== 工具函数 ====================

def load_stage_config(config_path: str) -> Dict:
    """加载阶段配置文件,支持向后兼容旧路径"""
    if not os.path.exists(config_path):
        # 尝试旧路径兼容性
        legacy_path = config_path.replace("configs/train/", "configs/")
        if os.path.exists(legacy_path):
            console.print(f"[yellow]警告: 使用旧路径 {legacy_path}，建议更新配置[/yellow]")
            config_path = legacy_path
        else:
            console.print(f"[red]错误: 配置文件不存在: {config_path}[/red]")
            sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def print_stage_info(stage_config: Dict):
    """打印阶段信息"""
    console.print(Panel.fit(
        f"[bold green]阶段 {stage_config['stage_id']}: {stage_config['description']}[/bold green]\n"
        f"[dim]配置文件: {stage_config['config_file']}[/dim]",
        border_style="green",
    ))


def check_transition_criteria(
    stage_config: Dict,
    metrics: Dict[str, float],
    current_iteration: int,
) -> bool:
    """检查是否满足切换到下一阶段的条件

    Args:
        stage_config: 当前阶段配置
        metrics: 当前训练指标
        current_iteration: 当前迭代次数

    Returns:
        是否应该切换到下一阶段
    """
    # 检查最小迭代次数
    if current_iteration < stage_config["min_iterations"]:
        return False

    # 检查切换条件
    criteria = stage_config.get("transition_criteria", {})

    for key, threshold in criteria.items():
        if key.startswith("min_"):
            metric_name = key[4:]  # 去掉 "min_"
            if metrics.get(metric_name, 0) < threshold:
                return False
        elif key.startswith("max_"):
            metric_name = key[4:]  # 去掉 "max_"
            if metrics.get(metric_name, float('inf')) > threshold:
                return False

    return True


# ==================== 主训练函数 ====================

def train_stage(
    stage_config: Dict,
    previous_checkpoint: Optional[str] = None,
    start_from_stage: int = 0,
    use_jit: bool = True,
) -> str:
    """训练单个阶段

    Args:
        stage_config: 阶段配置
        previous_checkpoint: 上一阶段的检查点路径
        start_from_stage: 起始阶段ID

    Returns:
        最佳模型检查点路径
    """
    print_stage_info(stage_config)

    # 加载YAML配置
    yaml_config = load_stage_config(stage_config["config_file"])

    # 构建命令行参数
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", stage_config["config_file"],
    ]

    # 如果有上一阶段的检查点，添加恢复参数
    if previous_checkpoint and start_from_stage > stage_config["stage_id"]:
        cmd.extend(["--resume", previous_checkpoint])
        console.print(f"[dim]从检查点恢复: {previous_checkpoint}[/dim]")

    # 执行训练（导入train.py的主函数）
    # 为了简化，这里直接调用train.py的main函数
    # 实际使用时可以通过subprocess调用

    # 导入训练模块
    from rl.envs import create_walking_env
    from rl.models.networks import ActorCriticNetwork
    from rl.models.optimizer import create_ppo_optimizer_cosine
    from rl.training.logger import Logger, MetricsLogger, create_training_display
    from rl.training.ppo_trainer import PPOConfig, PPOTrainer
    from rl.training.train_state import create_train_state
    from rl.utils.checkpoint import create_checkpoint_manager
    from rl.utils.performance_monitor import PerformanceMonitor

    # 解析配置
    scene = yaml_config.get("scene", "jiyuan_fit_flat")
    env_type = yaml_config.get("env_type", "walking")

    # 环境配置
    num_envs = yaml_config.get("num_envs", 16384)
    num_steps = yaml_config.get("num_steps", 64)

    # 命令范围
    cmd_x_range = tuple(yaml_config.get("cmd_x_range", [0.0, 0.0]))
    cmd_y_range = tuple(yaml_config.get("cmd_y_range", [0.0, 0.0]))
    cmd_yaw_range = tuple(yaml_config.get("cmd_yaw_range", [0.0, 0.0]))

    # 奖励权重
    reward_weights = yaml_config.get("reward_weights", None)

    # PPO配置
    num_epochs = yaml_config.get("num_epochs", 4)
    num_minibatches = yaml_config.get("num_minibatches", 8)
    gamma = yaml_config.get("gamma", 0.99)
    gae_lambda = yaml_config.get("gae_lambda", 0.95)
    clip_epsilon = yaml_config.get("clip_epsilon", 0.2)
    value_coef = yaml_config.get("value_coef", 0.5)
    entropy_coef = yaml_config.get("entropy_coef", 0.01)
    max_grad_norm = yaml_config.get("max_grad_norm", 0.5)

    # 训练配置
    total_timesteps = yaml_config.get("total_timesteps", 100_000_000)
    log_interval = yaml_config.get("log_interval", 10)
    save_interval = yaml_config.get("save_interval", 100)

    # 优化器配置
    learning_rate = yaml_config.get("learning_rate", 3.0e-4)
    final_lr_fraction = yaml_config.get("final_lr_fraction", 0.02)

    # 网络配置
    hidden_dims = yaml_config.get("hidden_dims", [256, 256, 128])
    shared_backbone = yaml_config.get("shared_backbone", True)

    # 场景路径
    scene_file_map = {
        "flat_terrain": "flat_terrain",
        "rough_terrain": "rough_terrain",
        "jiyuan_fit_flat": "jiyuan_fit_flat_terrain",
        "jiyuan_fit_rough": "jiyuan_fit_rough_terrain",
    }
    scene_file = scene_file_map.get(scene, scene)
    scene_path = f"assets/xmls/scenes/{scene_file}.xml"

    console.print(f"[cyan]场景: {scene}[/cyan]")
    console.print(f"[cyan]环境类型: {env_type}[/cyan]")
    console.print(f"[cyan]命令范围: x={cmd_x_range}, y={cmd_y_range}, yaw={cmd_yaw_range}[/cyan]")

    # 根据环境类型创建对应的环境
    if env_type == "standing":
        from rl.envs import create_standing_env
        env = create_standing_env(
            xml_path=scene_path,
            cmd_x_range=cmd_x_range,
            cmd_y_range=cmd_y_range,
            cmd_yaw_range=cmd_yaw_range,
            reward_weights=reward_weights,
        )
    else:  # 默认创建行走环境
        target_height = yaml_config.get("target_height", 0.35)
        env = create_walking_env(
            xml_path=scene_path,
            cmd_x_range=cmd_x_range,
            cmd_y_range=cmd_y_range,
            cmd_yaw_range=cmd_yaw_range,
            target_height=target_height,
            reward_weights=reward_weights,
        )

    # 创建网络
    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=shared_backbone,
        hidden_dims=tuple(hidden_dims),
    )

    # 创建配置
    config = PPOConfig(
        num_envs=num_envs,
        num_steps=num_steps,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        total_timesteps=total_timesteps,
        log_interval=log_interval,
    )

    # 创建优化器
    total_updates = config.num_updates
    # 对于极小的 total_updates，使用更小的 warmup
    if total_updates < 20:
        warmup_steps = max(1, total_updates // 4)  # 25% warmup
    else:
        warmup_steps = max(10, total_updates // 20)  # 5% warmup

    optimizer = create_ppo_optimizer_cosine(
        learning_rate=learning_rate,
        total_steps=total_updates,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        final_lr_fraction=final_lr_fraction,
    )

    # 创建训练状态
    rng = jax.random.PRNGKey(42)
    train_state = create_train_state(
        network=network,
        optimizer=optimizer,
        obs_shape=(env.observation_size,),
        rng=rng,
    )

    # 如果有上一阶段检查点，加载它
    if previous_checkpoint and start_from_stage > stage_config["stage_id"]:
        console.print(f"[yellow]从检查点加载: {previous_checkpoint}[/yellow]")
        # TODO: 实现检查点加载逻辑

    # 创建日志系统
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/stage{stage_config['stage_id']}_{timestamp}"
    logger = Logger(log_dir=log_dir, use_tensorboard=True, use_rich=False)

    # 创建训练显示（使用TrainingDisplay替代简单的Progress）
    training_display = create_training_display(
        console=console,
        total=config.num_updates,
        steps_per_epoch=1,
        description=f"阶段 {stage_config['stage_id']} - {stage_config['name']}"
    )

    # 创建检查点管理器
    checkpoint_manager = create_checkpoint_manager(
        log_dir=log_dir,
        max_to_keep=3,
        keep_best=True,
        metric_name="mean_reward",
        metric_mode="max",
    )

    # 创建训练器
    trainer = PPOTrainer(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )

    # JIT编译训练函数
    from rl.training.ppo_trainer import create_train_step_fn
    train_step_fn = create_train_step_fn(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )

    if use_jit:
        train_step_jit = jax.jit(train_step_fn)

        # 触发编译（带进度提示）
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # 在screen中禁用SpinnerColumn，避免显示异常
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ]
        if not os.environ.get("STY"):  # STY由screen设置
            progress_columns.insert(0, SpinnerColumn())

        with Progress(
            *progress_columns,
            console=console,
        ) as progress:
            compile_task = progress.add_task(
                f"[yellow]编译 JIT 函数（阶段 {stage_config['stage_id']}）...",
                total=None
            )
            rng, reset_rng = jax.random.split(rng)
            env_state = env.batch_reset(reset_rng, config.num_envs)
            train_state, env_state, info = train_step_jit(train_state, env_state)
            jax.block_until_ready(train_state)
            progress.update(compile_task, completed=True)

        console.print("[green]✓ 编译完成[/green]")
    else:
        train_step_jit = train_step_fn  # 不使用JIT，直接用原函数
        console.print("[yellow]⚠ Eager模式（禁用JIT）：不需要编译，但训练较慢[/yellow]")
        rng, reset_rng = jax.random.split(rng)
        env_state = env.batch_reset(reset_rng, config.num_envs)

    # 创建指标记录器
    metrics_logger = MetricsLogger()
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()

    # 训练循环
    console.print(f"[bold green]开始训练阶段 {stage_config['stage_id']}[/bold green]")

    with training_display:
        for update in range(config.num_updates):
            # 训练一步
            if update == 0:
                console.print(f"[dim]开始第一次训练迭代（可能需要额外的初始化时间）...[/dim]")

            train_state, env_state, info = train_step_jit(train_state, env_state)
            jax.block_until_ready(train_state)

            if update == 0:
                console.print(f"[green]✓ 第一次迭代完成[/green]")

            # 记录指标
            perf_metrics = perf_monitor.step(config.batch_size)
            info.update(perf_metrics)
            metrics_logger.log_dict(info)

            # 更新训练显示（包含详细指标和进度条）
            training_display.update(epoch=update, metrics=info)

            # 定期日志
            if (update + 1) % log_interval == 0:
                avg_metrics = metrics_logger.get_averages()
                logger.log_scalars(metrics=avg_metrics, step=train_state.step, prefix="train")
                metrics_logger.reset()

            # 定期保存
            if (update + 1) % save_interval == 0:
                avg_metrics = metrics_logger.get_averages()
                checkpoint_manager.save_checkpoint(
                    train_state=train_state,
                    step=train_state.step,
                    metrics=avg_metrics,
                )

            # 检查是否应该切换到下一阶段
            if check_transition_criteria(
                stage_config,
                info,
                current_iteration=update + 1,
            ):
                console.print(f"[green]✓ 达到切换条件，准备进入下一阶段[/green]")
                break

    # 保存最终检查点
    final_checkpoint = checkpoint_manager.save_checkpoint(
        train_state=train_state,
        step=train_state.step,
        metrics=metrics_logger.get_averages(),
        force=True,
    )

    logger.close()

    return final_checkpoint


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动化分阶段训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        choices=range(7),
        help="起始阶段（0-6）",
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        default=6,
        choices=range(7),
        help="结束阶段（0-6）",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="使用测试配置（快速流水线测试，3阶段<3分钟）",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="使用极速测试配置（无需重复编译，3阶段<30秒）",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="禁用JIT编译",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["standard", "10h"],
        help="训练配置方案：standard=标准3天方案，10h=10小时快速方案",
    )

    args = parser.parse_args()

    # 根据测试模式选择配置
    if args.quick_test:
        stage_configs = QUICK_TEST_STAGE_CONFIGS
        console.print(Panel.fit(
            f"[bold green]极速测试模式（无需重复编译）[/bold green]\n"
            f"[dim]起始阶段: {args.start_stage}[/dim]\n"
            f"[dim]结束阶段: min({args.end_stage}, 2)[/dim]\n"
            f"[dim]预计时间: 首次~30秒（含编译），后续<10秒[/dim]\n"
            f"[dim]策略: 固定网络结构和环境参数，复用JAX编译缓存[/dim]",
            border_style="green",
        ))
        # 极速测试模式最多到阶段2
        end_stage = min(args.end_stage, 2)
    elif args.test_mode:
        stage_configs = TEST_STAGE_CONFIGS
        console.print(Panel.fit(
            f"[bold yellow]流水线快速测试模式[/bold yellow]\n"
            f"[dim]起始阶段: {args.start_stage}[/dim]\n"
            f"[dim]结束阶段: min({args.end_stage}, 2)[/dim]\n"
            f"[dim]预计时间: <3分钟[/dim]",
            border_style="yellow",
        ))
        # 测试模式最多到阶段2
        end_stage = min(args.end_stage, 2)
    elif args.profile == "10h":
        stage_configs = TEN_HOUR_STAGE_CONFIGS
        console.print(Panel.fit(
            f"[bold cyan]10小时快速训练方案[/bold cyan]\n"
            f"[dim]起始阶段: {args.start_stage}[/dim]\n"
            f"[dim]结束阶段: {args.end_stage}[/dim]\n"
            f"[dim]预计时间: ~12-14小时[/dim]\n"
            f"[dim]配置路径: configs/train-10h/[/dim]",
            border_style="cyan",
        ))
        end_stage = args.end_stage
    else:
        stage_configs = STAGE_CONFIGS
        console.print(Panel.fit(
            f"[bold green]标准分阶段训练[/bold green]\n"
            f"[dim]起始阶段: {args.start_stage}[/dim]\n"
            f"[dim]结束阶段: {args.end_stage}[/dim]",
            border_style="green",
        ))
        end_stage = args.end_stage

    # 获取要训练的阶段
    stages_to_train = [
        s for s in stage_configs
        if args.start_stage <= s["stage_id"] <= end_stage
    ]

    if not stages_to_train:
        console.print("[red]错误: 没有要训练的阶段[/red]")
        sys.exit(1)

    console.print(f"[green]将训练 {len(stages_to_train)} 个阶段[/green]")

    # 逐个训练阶段
    previous_checkpoint = args.resume_checkpoint
    start_from_stage = args.start_stage

    for stage_config in stages_to_train:
        try:
            # 训练当前阶段
            best_checkpoint = train_stage(
                stage_config=stage_config,
                previous_checkpoint=previous_checkpoint,
                start_from_stage=start_from_stage,
                use_jit=not args.no_jit,  # 根据参数决定是否使用JIT
            )

            # 更新检查点路径用于下一阶段
            previous_checkpoint = best_checkpoint
            start_from_stage = stage_config["stage_id"]

            console.print(f"[green]✓ 阶段 {stage_config['stage_id']} 完成[/green]")
            console.print(f"[dim]最佳模型: {best_checkpoint}[/dim]")

        except KeyboardInterrupt:
            console.print(f"[yellow]训练被用户中断（阶段 {stage_config['stage_id']}）[/yellow]")
            console.print(f"[dim]可以使用 --resume-checkpoint {previous_checkpoint} 恢复训练[/dim]")
            sys.exit(0)

        except Exception as e:
            console.print(f"[red]阶段 {stage_config['stage_id']} 训练出错: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            sys.exit(1)

    console.print(Panel.fit(
        "[bold green]所有阶段训练完成！[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
