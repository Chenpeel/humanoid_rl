#!/usr/bin/env python3
"""
训练脚本

使用 RSL_RL PPO 算法训练机器人。

支持的任务:
- velocity: 速度跟踪任务（主要训练任务）
- standing: 站立平衡任务（预训练）
- walking: 行走任务（专用步态训练）
- rough: 粗糙地形任务（复杂地形训练）
- flat: 平坦地形任务（简化训练）

配置优先级:
    命令行参数 > 配置文件 > 代码预定义值

运行方式:
    # 使用配置文件训练（推荐）
    python scripts/train.py --config configs/train_config.yaml

    # 命令行覆盖配置文件参数
    python scripts/train.py --config configs/train_config.yaml --num_envs 8192 --headless

    # 不使用配置文件（向后兼容）
    python scripts/train.py --task velocity --num_envs 4096

    # 行走任务训练
    python scripts/train.py --task walking --num_envs 4096

    # 粗糙地形训练
    python scripts/train.py --task rough --num_envs 4096

    # 平坦地形训练
    python scripts/train.py --task flat --num_envs 4096

    # 从检查点恢复
    python scripts/train.py --config configs/train_config.yaml --resume --load_run run_20250122_143000

参考:
- Isaac Lab CLI: isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ============================================================================
# 必须先启动 Isaac Sim 应用（在导入 Isaac Lab 之前）
# ============================================================================
from isaaclab.app import AppLauncher

# 创建参数解析器（用于 AppLauncher）
app_launcher_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(app_launcher_parser)
app_launcher_args, remaining_args = app_launcher_parser.parse_known_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# ============================================================================
# 现在可以安全地导入 Isaac Lab 和其他模块
# ============================================================================
import gymnasium as gym
import torch

# Isaac Lab 导入
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml

# RSL_RL 导入
from rsl_rl.runners import OnPolicyRunner

# 项目导入
from jiyuan_tasks import *  # 注册环境
from jiyuan_tasks.utils.config_loader import (
    load_train_config,
    load_robot_config,
    validate_train_config,
    get_default_config_path,
    ConfigDict,
)
from agents.rsl_rl import (
    VELOCITY_TRACKING_PPO_CFG,
    STANDING_PPO_CFG,
)


##
# 任务到环境ID的映射
##

TASK_ENV_MAP = {
    "velocity": "Isaac-Jiyuan-Velocity-v0",
    "standing": "Isaac-Jiyuan-Standing-v0",
    "walking": "Isaac-Jiyuan-Walking-v0",
    "rough": "Isaac-Jiyuan-Rough-v0",
    "flat": "Isaac-Jiyuan-Flat-v0",
    "test": "Isaac-Jiyuan-Test-v0",
}

TASK_PPO_CFG_MAP = {
    "velocity": VELOCITY_TRACKING_PPO_CFG,
    "standing": STANDING_PPO_CFG,
    "walking": VELOCITY_TRACKING_PPO_CFG,
    "rough": VELOCITY_TRACKING_PPO_CFG,
    "flat": VELOCITY_TRACKING_PPO_CFG,
    "test": STANDING_PPO_CFG,
}


##
# 辅助函数
##


def parse_args():
    """解析命令行参数

    注意：AppLauncher 参数已经在前面解析过了，这里只解析训练相关参数
    """
    parser = argparse.ArgumentParser(
        description="训练双足机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[app_launcher_parser],  # 继承 AppLauncher 参数
    )

    # 配置文件（新增）
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置文件路径 (YAML 格式)。如果不指定，使用命令行参数或默认值",
    )

    # 任务选择
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="训练任务 (可选: velocity, standing, walking, rough, flat, test)。如果使用配置文件，可以在配置文件中指定",
    )

    # 环境参数
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="并行环境数。覆盖配置文件中的值",
    )

    # 训练参数
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="最大训练迭代数。覆盖配置文件中的值",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子。覆盖配置文件中的值",
    )

    # 注意：--device 参数已由 AppLauncher 提供，不需要重复定义

    # 检查点相关
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复训练",
    )

    parser.add_argument(
        "--load_run",
        type=str,
        default="",
        help="要加载的运行目录名（位于 logs/{task}/ 下），如果为空且 --resume，则加载最新的",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="检查点文件名（如果为空，加载最新的）",
    )

    # 日志相关
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志根目录。覆盖配置文件中的值",
    )

    # 注意：--headless 参数已由 AppLauncher 提供，不需要重复定义

    parser.add_argument(
        "--video",
        action="store_true",
        help="录制训练视频",
    )

    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="视频长度（步数）。覆盖配置文件中的值",
    )

    parser.add_argument(
        "--video_interval",
        type=int,
        default=None,
        help="视频录制间隔（迭代数）。覆盖配置文件中的值",
    )

    args = parser.parse_args()
    return args


def build_cli_overrides(args) -> dict:
    """根据命令行参数构建配置覆盖字典

    Args:
        args: 命令行参数

    Returns:
        dict: 配置覆盖字典，键使用点号分隔的嵌套路径
    """
    overrides = {}

    # 任务
    if args.task is not None:
        overrides["task"] = args.task

    # 环境参数
    if args.num_envs is not None:
        overrides["environment.num_envs"] = args.num_envs

    if args.headless:
        overrides["environment.headless"] = True

    # PPO 训练参数
    if args.max_iterations is not None:
        overrides["ppo.runner.max_iterations"] = args.max_iterations

    if args.seed is not None:
        overrides["ppo.runner.seed"] = args.seed

    if args.device is not None:
        overrides["ppo.runner.device"] = args.device

    if args.log_dir is not None:
        overrides["ppo.runner.log_dir"] = args.log_dir

    # 视频录制
    if args.video:
        overrides["logging.record_video.enable"] = True

    if args.video_length is not None:
        overrides["logging.record_video.length"] = args.video_length

    if args.video_interval is not None:
        overrides["logging.record_video.interval"] = args.video_interval

    return overrides


def load_config_with_cli(args) -> ConfigDict:
    """加载配置并应用命令行覆盖

    Args:
        args: 命令行参数

    Returns:
        ConfigDict: 最终配置
    """
    # 构建命令行覆盖
    cli_overrides = build_cli_overrides(args)

    # 如果指定了配置文件，加载配置文件
    if args.config:
        print(f"\n[INFO] 加载配置文件: {args.config}")
        config = load_train_config(args.config, cli_overrides=cli_overrides)
    else:
        # 不使用配置文件，使用默认值 + 命令行参数
        print("\n[INFO] 未指定配置文件，使用默认值和命令行参数")

        # 默认配置
        default_config = {
            "task": "velocity",
            "environment": {
                "num_envs": 8192,
                "episode_length_s": 20.0,
                "headless": False,
            },
            "ppo": {
                "algorithm": {
                    "learning_rate": 0.001,
                },
                "runner": {
                    "seed": 42,
                    "device": "cuda:0",
                    "max_iterations": 30000,
                    "save_interval": 500,
                    "log_dir": "logs",
                },
            },
        }

        config = ConfigDict(default_config)

        # 应用命令行覆盖
        if cli_overrides:
            from jiyuan_tasks.utils.config_loader import apply_cli_overrides

            config = apply_cli_overrides(config, cli_overrides)

    # 验证配置
    try:
        validate_train_config(config)
    except ValueError as e:
        print(f"\n[ERROR] 配置验证失败: {e}")
        sys.exit(1)

    return config


def apply_config_to_ppo(ppo_cfg, config: ConfigDict):
    """将配置文件中的参数应用到 PPO 配置对象

    Args:
        ppo_cfg: RSL_RL PPO 配置对象
        config: 训练配置
    """
    if "ppo" not in config:
        return

    ppo_config = config.ppo

    # 应用算法参数
    if "algorithm" in ppo_config:
        alg = ppo_config.algorithm
        if "learning_rate" in alg:
            ppo_cfg.algorithm.learning_rate = alg.learning_rate
        if "clip_param" in alg:
            ppo_cfg.algorithm.clip_param = alg.clip_param
        if "entropy_coef" in alg:
            ppo_cfg.algorithm.entropy_coef = alg.entropy_coef
        if "value_loss_coef" in alg:
            ppo_cfg.algorithm.value_loss_coef = alg.value_loss_coef
        if "gamma" in alg:
            ppo_cfg.algorithm.gamma = alg.gamma
        if "lam" in alg:
            ppo_cfg.algorithm.lam = alg.lam
        if "num_learning_epochs" in alg:
            ppo_cfg.algorithm.num_learning_epochs = alg.num_learning_epochs
        if "num_mini_batches" in alg:
            ppo_cfg.algorithm.num_mini_batches = alg.num_mini_batches

    # 应用训练器参数
    if "runner" in ppo_config:
        runner = ppo_config.runner
        if "max_iterations" in runner:
            ppo_cfg.max_iterations = runner.max_iterations
        if "seed" in runner:
            ppo_cfg.seed = runner.seed
        if "device" in runner:
            ppo_cfg.device = runner.device
        if "num_steps_per_env" in runner:
            ppo_cfg.num_steps_per_env = runner.num_steps_per_env
        if "save_interval" in runner:
            ppo_cfg.save_interval = runner.save_interval

    # 应用网络架构参数
    if "network" in ppo_config:
        net = ppo_config.network
        if "actor_hidden_dims" in net:
            ppo_cfg.policy.actor_hidden_dims = net.actor_hidden_dims
        if "critic_hidden_dims" in net:
            ppo_cfg.policy.critic_hidden_dims = net.critic_hidden_dims
        if "activation" in net:
            ppo_cfg.policy.activation = net.activation
        if "init_noise_std" in net:
            ppo_cfg.policy.init_noise_std = net.init_noise_std


def create_runner(env: ManagerBasedRLEnv, ppo_cfg, config: ConfigDict, args):
    """创建 RSL_RL PPO 训练器

    Args:
        env: Isaac Lab 环境
        ppo_cfg: PPO 配置
        config: 训练配置
        args: 命令行参数

    Returns:
        OnPolicyRunner 实例
    """
    # 从配置中获取日志目录
    log_dir_root = config.ppo.runner.get("log_dir", "logs")

    # 创建日志目录
    log_root_path = os.path.join(log_dir_root, ppo_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # 如果恢复训练，确定运行目录
    if args.resume:
        if args.load_run:
            # 指定了运行目录
            log_dir = os.path.join(log_root_path, args.load_run)
        else:
            # 加载最新的运行
            runs = [d for d in os.listdir(log_root_path) if os.path.isdir(os.path.join(log_root_path, d))]
            if not runs:
                raise ValueError(f"在 {log_root_path} 中未找到任何运行目录")
            runs.sort()
            log_dir = os.path.join(log_root_path, runs[-1])

        print(f"[INFO] 从运行目录恢复: {log_dir}")
        resume_path = log_dir
    else:
        # 新的训练运行
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        resume_path = None
        print(f"[INFO] 创建新运行目录: {log_dir}")

    os.makedirs(log_dir, exist_ok=True)

    # 保存配置到日志目录
    config_path = os.path.join(log_dir, "train_config.yaml")
    dump_yaml(config_path, dict(config))
    print(f"[INFO] 训练配置已保存到: {config_path}")

    ppo_config_path = os.path.join(log_dir, "ppo_config.yaml")
    dump_yaml(ppo_config_path, ppo_cfg.__dict__)
    print(f"[INFO] PPO 配置已保存到: {ppo_config_path}")

    # 创建 RSL_RL 训练器
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=ppo_cfg.device)

    # 如果恢复训练，加载检查点
    if args.resume:
        if args.checkpoint:
            # 指定了检查点文件
            checkpoint_path = os.path.join(resume_path, args.checkpoint)
        else:
            # 加载最新的检查点
            checkpoint_path = os.path.join(resume_path, "model_*.pt")
            import glob

            checkpoints = glob.glob(checkpoint_path)
            if not checkpoints:
                raise ValueError(f"在 {resume_path} 中未找到检查点文件")
            checkpoints.sort()
            checkpoint_path = checkpoints[-1]

        print(f"[INFO] 加载检查点: {checkpoint_path}")
        runner.load(checkpoint_path)

    return runner


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()

    print("=" * 80)
    print(f"双足机器人训练脚本")
    print("=" * 80)
    print(f"配置优先级: 命令行参数 > 配置文件 > 代码预定义值")
    print("=" * 80)

    # 加载配置
    config = load_config_with_cli(args)

    # 打印配置信息
    print(f"\n最终配置:")
    print(f"  - 任务: {config.task}")
    print(f"  - 环境ID: {TASK_ENV_MAP[config.task]}")
    print(f"  - 并行环境数: {config.environment.num_envs}")
    print(f"  - 设备: {config.ppo.runner.device}")
    print(f"  - 随机种子: {config.ppo.runner.seed}")
    print(f"  - 无头模式: {config.environment.headless}")
    print(f"  - 恢复训练: {args.resume}")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(config.ppo.runner.seed)

    # 获取环境配置类
    print(f"\n[INFO] 加载环境配置: {TASK_ENV_MAP[config.task]}")
    # 从注册信息中获取配置入口点
    env_spec = gym.spec(TASK_ENV_MAP[config.task])
    env_cfg_entry_point = env_spec.kwargs["env_cfg_entry_point"]

    # 动态导入配置类
    module_path, class_name = env_cfg_entry_point.rsplit(":", 1)
    module = __import__(module_path, fromlist=[class_name])
    env_cfg = getattr(module, class_name)

    # 应用命令行参数覆盖环境配置
    env_cfg.scene.num_envs = config.environment.num_envs
    env_cfg.episode_length_s = config.environment.get("episode_length_s", env_cfg.episode_length_s)

    # 创建环境
    print(f"\n[INFO] 创建环境: {TASK_ENV_MAP[config.task]}")
    env = gym.make(
        TASK_ENV_MAP[config.task],
        cfg=env_cfg,
        render_mode="rgb_array" if config.get("logging", {}).get("record_video", {}).get("enable", False) else None,
    )

    print(f"[INFO] 环境创建成功")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 并行环境数: {env.num_envs}")

    # 获取 PPO 配置
    ppo_cfg = TASK_PPO_CFG_MAP[config.task]

    # 应用配置文件中的参数到 PPO 配置
    apply_config_to_ppo(ppo_cfg, config)

    print(f"\n[INFO] 使用 PPO 配置: {ppo_cfg.experiment_name}")
    print(f"  - 最大迭代数: {ppo_cfg.max_iterations}")
    print(f"  - 每环境步数: {ppo_cfg.num_steps_per_env}")
    print(f"  - 总训练步数: {ppo_cfg.max_iterations * ppo_cfg.num_steps_per_env * config.environment.num_envs:,}")
    print(f"  - 学习率: {ppo_cfg.algorithm.learning_rate}")
    print(f"  - 网络架构: {ppo_cfg.policy.actor_hidden_dims}")

    # 创建训练器
    print(f"\n[INFO] 创建 RSL_RL PPO 训练器")
    runner = create_runner(env, ppo_cfg, config, args)

    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"按 Ctrl+C 可以安全地停止训练并保存检查点")
    print("=" * 80 + "\n")

    try:
        runner.learn(num_learning_iterations=ppo_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")
    finally:
        print("[INFO] 保存最终检查点")
        runner.save(os.path.join(runner.log_dir, "model_final.pt"))
        print(f"[INFO] 检查点已保存到: {runner.log_dir}")

    # 关闭环境
    env.close()

    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"日志目录: {runner.log_dir}")
    print(f"使用 TensorBoard 查看训练曲线:")
    print(f"  tensorboard --logdir={config.ppo.runner.get('log_dir', 'logs')}")
    print("=" * 80)

    # 关闭 Isaac Sim 应用
    simulation_app.close()


if __name__ == "__main__":
    main()
