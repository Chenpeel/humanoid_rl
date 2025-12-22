#!/usr/bin/env python3
"""
Jiyuan 机器人训练脚本

使用 RSL_RL PPO 算法训练 Jiyuan 双足机器人。

支持的任务:
- velocity: 速度跟踪任务（主要训练任务）
- standing: 站立平衡任务（预训练）

运行方式:
    # 速度跟踪任务（4096个并行环境）
    python scripts/train.py --task velocity --num_envs 4096

    # 站立任务
    python scripts/train.py --task standing --num_envs 4096

    # 从检查点恢复
    python scripts/train.py --task velocity --resume --load_run run_20250122_143000

    # 无头模式（不显示GUI，用于服务器训练）
    python scripts/train.py --task velocity --headless

参考:
- Isaac Lab CLI: isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py
- RSL_RL: https://github.com/leggedrobotics/rsl_rl
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import torch

# Isaac Lab 导入
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

# RSL_RL 导入
from rsl_rl.runners import OnPolicyRunner

# 项目导入
from isaaclab_rl import jiyuan_tasks  # 注册环境
from isaaclab_rl.agents.rsl_rl import (
    VELOCITY_TRACKING_PPO_CFG,
    STANDING_PPO_CFG,
)


##
# 任务到环境ID的映射
##

TASK_ENV_MAP = {
    "velocity": "Isaac-Jiyuan-Velocity-v0",
    "standing": "Isaac-Jiyuan-Standing-v0",
    "test": "Isaac-Jiyuan-Test-v0",
}

TASK_PPO_CFG_MAP = {
    "velocity": VELOCITY_TRACKING_PPO_CFG,
    "standing": STANDING_PPO_CFG,
    "test": STANDING_PPO_CFG,  # 测试环境使用站立配置
}


##
# 辅助函数
##


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练 Jiyuan 机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 任务选择
    parser.add_argument(
        "--task",
        type=str,
        default="velocity",
        choices=list(TASK_ENV_MAP.keys()),
        help="训练任务 (默认: velocity)",
    )

    # 环境参数
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="并行环境数 (默认: 4096)",
    )

    # 训练参数
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="最大训练迭代数（如果不指定，使用 PPO 配置中的默认值）",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )

    # 设备选择
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="训练设备 (默认: cuda:0)",
    )

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
        default="logs",
        help="日志根目录 (默认: logs)",
    )

    # 可视化
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式（不显示 GUI）",
    )

    parser.add_argument(
        "--video",
        action="store_true",
        help="录制训练视频",
    )

    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="视频长度（步数）",
    )

    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="视频录制间隔（迭代数）",
    )

    args = parser.parse_args()
    return args


def create_runner(env: ManagerBasedRLEnv, ppo_cfg, args):
    """创建 RSL_RL PPO 训练器

    Args:
        env: Isaac Lab 环境
        ppo_cfg: PPO 配置
        args: 命令行参数

    Returns:
        OnPolicyRunner 实例
    """
    # 创建日志目录
    log_root_path = os.path.join(args.log_dir, ppo_cfg.experiment_name)
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
    config_path = os.path.join(log_dir, "config.yaml")
    dump_yaml(config_path, ppo_cfg.__dict__)
    print(f"[INFO] 配置已保存到: {config_path}")

    # 更新 PPO 配置中的参数
    if args.max_iterations is not None:
        ppo_cfg.max_iterations = args.max_iterations

    ppo_cfg.seed = args.seed
    ppo_cfg.device = args.device

    # 创建 RSL_RL 训练器
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=log_dir, device=args.device)

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
    print(f"Jiyuan 机器人训练脚本")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"环境ID: {TASK_ENV_MAP[args.task]}")
    print(f"并行环境数: {args.num_envs}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print(f"无头模式: {args.headless}")
    print(f"恢复训练: {args.resume}")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 创建环境
    print(f"\n[INFO] 创建环境: {TASK_ENV_MAP[args.task]}")
    env = gym.make(
        TASK_ENV_MAP[args.task],
        num_envs=args.num_envs,
        headless=args.headless,
    )

    print(f"[INFO] 环境创建成功")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 并行环境数: {env.num_envs}")

    # 获取 PPO 配置
    ppo_cfg = TASK_PPO_CFG_MAP[args.task]
    print(f"\n[INFO] 使用 PPO 配置: {ppo_cfg.experiment_name}")
    print(f"  - 最大迭代数: {ppo_cfg.max_iterations}")
    print(f"  - 每环境步数: {ppo_cfg.num_steps_per_env}")
    print(f"  - 总训练步数: {ppo_cfg.max_iterations * ppo_cfg.num_steps_per_env * args.num_envs:,}")
    print(f"  - 学习率: {ppo_cfg.algorithm.learning_rate}")
    print(f"  - 网络架构: {ppo_cfg.policy.actor_hidden_dims}")

    # 创建训练器
    print(f"\n[INFO] 创建 RSL_RL PPO 训练器")
    runner = create_runner(env, ppo_cfg, args)

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
    print(f"  tensorboard --logdir={args.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
