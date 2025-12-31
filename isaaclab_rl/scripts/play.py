#!/usr/bin/env python3
"""
Jiyuan 机器人策略评估脚本

加载训练好的策略并在 Isaac Sim 中可视化和录制视频。

运行方式:
    # 评估速度跟踪策略（GUI可视化）
    python scripts/play.py --task velocity --checkpoint logs/jiyuan_velocity_tracking/20250122_143000/model_30000.pt

    # 评估站立策略
    python scripts/play.py --task standing --checkpoint logs/jiyuan_standing/20250122_150000/model_10000.pt

    # 自动加载最新的检查点
    python scripts/play.py --task velocity

    # 评估多个环境（对比不同初始化）
    python scripts/play.py --task velocity --num_envs 4

    # 录制视频（自动保存到 logs/<实验名>/<运行ID>/videos/play/）
    python scripts/play.py --task velocity --video --video_length 500 --num_envs 1

    # 使用确定性策略（无探索噪声）
    python scripts/play.py --task velocity --deterministic

功能说明:
    - 自动加载最新检查点：不指定 --checkpoint 时自动查找最新模型
    - 视频录制：使用 --video 参数启用，视频保存到检查点目录下的 videos/play/ 文件夹
    - 支持多环境并行评估（用于对比不同随机种子）
    - 支持确定性/随机策略评估

参考:
- Isaac Lab CLI: isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py
"""

import argparse
import glob
import os
import sys
from pathlib import Path

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

# RSL_RL 导入
from rsl_rl.runners import OnPolicyRunner

# 项目导入
import jiyuan_tasks  # 注册环境
from agents.rsl_rl import (
    VELOCITY_TRACKING_PPO_CFG,
    STANDING_PPO_CFG,
)


##
# 任务映射（与 train.py 保持一致）
##

TASK_ENV_MAP = {
    "velocity": "Isaac-Jiyuan-Velocity-v0",
    "standing": "Isaac-Jiyuan-Standing-v0",
    "test": "Isaac-Jiyuan-Test-v0",
}

TASK_PPO_CFG_MAP = {
    "velocity": VELOCITY_TRACKING_PPO_CFG,
    "standing": STANDING_PPO_CFG,
    "test": STANDING_PPO_CFG,
}


##
# 辅助函数
##


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="评估 Jiyuan 机器人策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 任务选择
    parser.add_argument(
        "--task",
        type=str,
        default="velocity",
        choices=list(TASK_ENV_MAP.keys()),
        help="评估任务 (默认: velocity)",
    )

    # 环境参数
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="并行环境数（用于对比不同初始化）(默认: 1)",
    )

    # 检查点
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="检查点文件路径（如果为空，自动加载最新的）",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="日志根目录（用于查找检查点）(默认: logs)",
    )

    # 评估参数
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="评估的 episode 数量 (默认: 10)",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="使用确定性策略（无探索噪声）",
    )

    # 设备选择
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="评估设备 (默认: cuda:0)",
    )

    # 视频录制
    parser.add_argument(
        "--video",
        action="store_true",
        help="录制视频",
    )

    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="视频长度（步数）(默认: 500)",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos",
        help="视频保存路径 (默认: videos)",
    )

    args = parser.parse_args()
    return args


def find_latest_checkpoint(log_dir: str, task: str) -> str:
    """查找最新的检查点文件

    Args:
        log_dir: 日志根目录
        task: 任务名称

    Returns:
        检查点文件路径
    """
    # 获取 PPO 配置以确定实验名称
    ppo_cfg = TASK_PPO_CFG_MAP[task]
    experiment_dir = os.path.join(log_dir, ppo_cfg.experiment_name)

    if not os.path.exists(experiment_dir):
        raise ValueError(f"实验目录不存在: {experiment_dir}")

    # 查找所有运行目录
    runs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
    if not runs:
        raise ValueError(f"在 {experiment_dir} 中未找到任何运行目录")

    # 按时间排序，取最新的
    runs.sort()
    latest_run = os.path.join(experiment_dir, runs[-1])

    # 查找所有检查点
    checkpoints = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not checkpoints:
        raise ValueError(f"在 {latest_run} 中未找到检查点文件")

    # 按文件名排序，取最新的
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]

    print(f"[INFO] 自动找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint


def evaluate_policy(env: ManagerBasedRLEnv, runner: OnPolicyRunner, args):
    """评估策略性能

    Args:
        env: Isaac Lab 环境
        runner: RSL_RL 训练器（包含策略）
        args: 命令行参数

    Returns:
        评估统计信息
    """
    # 获取策略网络
    policy = runner.get_inference_policy(device=args.device)

    # 评估统计
    episode_rewards = []
    episode_lengths = []

    # 重置环境
    obs, _ = env.reset()

    current_episode_reward = torch.zeros(env.num_envs, device=args.device)
    current_episode_length = torch.zeros(env.num_envs, device=args.device)

    num_completed_episodes = 0
    step_count = 0

    print(f"\n[INFO] 开始评估，目标 episode 数: {args.num_episodes}")
    print(f"[INFO] 确定性策略: {args.deterministic}")

    # 如果录制视频，限制步数
    if args.video:
        max_steps = args.video_length
        print(f"[INFO] 视频录制模式：最多运行 {max_steps} 步")
    else:
        max_steps = None

    print("=" * 80)

    while num_completed_episodes < args.num_episodes:
        # 视频录制时，在达到步数后退出
        if max_steps is not None and step_count >= max_steps:
            print(f"\n[INFO] 达到视频长度限制 ({max_steps} 步)，结束录制")
            break
        # 获取动作
        with torch.no_grad():
            actions = policy(obs, deterministic=args.deterministic)

        # 执行动作
        obs, rewards, terminated, truncated, _ = env.step(actions)

        # 累积奖励
        current_episode_reward += rewards
        current_episode_length += 1
        step_count += 1

        # 检查完成的 episode
        dones = terminated | truncated
        if dones.any():
            # 记录完成的 episode
            for env_id in torch.where(dones)[0]:
                episode_rewards.append(current_episode_reward[env_id].item())
                episode_lengths.append(current_episode_length[env_id].item())
                num_completed_episodes += 1

                if num_completed_episodes % 5 == 0:
                    print(f"  完成 {num_completed_episodes}/{args.num_episodes} episodes")

            # 重置完成的环境
            current_episode_reward[dones] = 0
            current_episode_length[dones] = 0

    # 计算统计信息
    if len(episode_rewards) > 0:
        episode_rewards = torch.tensor(episode_rewards)
        episode_lengths = torch.tensor(episode_lengths)

        stats = {
            "mean_reward": episode_rewards.mean().item(),
            "std_reward": episode_rewards.std().item(),
            "min_reward": episode_rewards.min().item(),
            "max_reward": episode_rewards.max().item(),
            "mean_length": episode_lengths.mean().item(),
            "total_steps": step_count,
        }
    else:
        # 视频模式可能没有完成的 episode
        stats = {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "mean_length": 0.0,
            "total_steps": step_count,
        }

    return stats


def main():
    """主评估函数"""
    # 解析参数
    args = parse_args()

    print("=" * 80)
    print(f"Jiyuan 机器人策略评估脚本")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"环境ID: {TASK_ENV_MAP[args.task]}")
    print(f"并行环境数: {args.num_envs}")
    print(f"设备: {args.device}")
    print("=" * 80)

    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.log_dir, args.task)

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"检查点文件不存在: {checkpoint_path}")

    print(f"\n[INFO] 加载检查点: {checkpoint_path}")

    # 创建环境
    print(f"\n[INFO] 创建环境: {TASK_ENV_MAP[args.task]}")

    # 如果需要录制视频，设置 render_mode
    render_mode = "rgb_array" if args.video else None

    env = gym.make(
        TASK_ENV_MAP[args.task],
        num_envs=args.num_envs,
        headless=False,  # 评估时始终显示 GUI
        render_mode=render_mode,
    )

    # 如果需要录制视频，包装环境
    if args.video:
        log_dir = os.path.dirname(checkpoint_path)
        video_folder = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_folder, exist_ok=True)

        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # 从第一步开始录制
            "video_length": args.video_length,
            "disable_logger": True,
        }

        print(f"\n[INFO] 启用视频录制")
        print(f"  - 视频保存路径: {video_folder}")
        print(f"  - 视频长度: {args.video_length} 步")

        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO] 环境创建成功")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 并行环境数: {env.num_envs}")

    # 获取 PPO 配置
    ppo_cfg = TASK_PPO_CFG_MAP[args.task]

    # 创建训练器（用于加载策略）
    print(f"\n[INFO] 创建 RSL_RL 训练器")
    runner = OnPolicyRunner(env, ppo_cfg, log_dir=None, device=args.device)

    # 加载检查点
    print(f"[INFO] 加载策略权重")
    runner.load(checkpoint_path)

    # 评估策略
    print("\n" + "=" * 80)
    print("评估策略")
    print("=" * 80)

    stats = evaluate_policy(env, runner, args)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)

    if args.video:
        print(f"视频录制完成！")
        log_dir = os.path.dirname(checkpoint_path)
        video_folder = os.path.join(log_dir, "videos", "play")
        print(f"视频保存路径: {video_folder}")
        print(f"录制步数: {stats['total_steps']}")
    else:
        print(f"Episode 数量: {args.num_episodes}")
        print(f"平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"奖励范围: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"平均 episode 长度: {stats['mean_length']:.1f}")
        print(f"总步数: {stats['total_steps']}")

    print("=" * 80)

    # 关闭环境
    env.close()

    print("\n评估完成")


if __name__ == "__main__":
    main()
