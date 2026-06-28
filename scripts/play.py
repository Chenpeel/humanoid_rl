"""play.py —— 策略评估入口。

用法：
    python scripts/play.py robot=h1 checkpoint=latest
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Humanoid RL 策略评估脚本"
    )
    parser.add_argument("--robot", type=str, default="h1", help="机器人名称")
    parser.add_argument("--checkpoint", type=str, default="latest", help="checkpoint 路径或 'latest'")
    parser.add_argument("--num_envs", type=int, default=1, help="评估环境数（通常 1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--record", action="store_true", help="录制视频")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Humanoid RL Evaluation")
    print(f"  Robot: {args.robot}  |  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # ==================================================================
    # 1. 加载配置
    # ==================================================================
    from humanoid_rl.utils import load_robot_config

    robot_cfg = load_robot_config(args.robot)

    # 使用 velocity_tracking 环境（能展示完整步态）
    from humanoid_rl.envs.cfg.velocity_cfg import VelocityTrackingEnvCfg

    env_cfg = VelocityTrackingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.robot_name = args.robot
    env_cfg.robot_asset_path = robot_cfg["asset"]["path"]
    env_cfg.robot_joint_names = robot_cfg["joints"]["names"]
    env_cfg.robot_default_joint_pos = robot_cfg["joints"]["default_positions"]
    env_cfg.robot_stiffness = robot_cfg["pd"]["stiffness"]
    env_cfg.robot_damping = robot_cfg["pd"]["damping"]

    # ==================================================================
    # 2. 构建环境
    # ==================================================================
    print(f"\n📦 构建评估环境: {args.robot}")
    from humanoid_rl.envs import BaseEnv

    env = BaseEnv(cfg=env_cfg)

    # ==================================================================
    # 3. 加载策略
    # ==================================================================
    import torch

    if args.checkpoint == "latest":
        # 找到最新的 checkpoint
        log_dir = os.path.join("logs", f"{args.robot}_velocity_tracking")
        checkpoints = sorted(
            [f for f in os.listdir(log_dir) if f.endswith(".pt")] if os.path.exists(log_dir) else []
        )
        if not checkpoints:
            log_dir = os.path.join("logs", f"{args.robot}_standing")
            checkpoints = sorted(
                [f for f in os.listdir(log_dir) if f.endswith(".pt")] if os.path.exists(log_dir) else []
            )
        if not checkpoints:
            print("❌ 未找到 checkpoint！请先训练模型。")
            sys.exit(1)
        checkpoint_path = os.path.join(log_dir, checkpoints[-1])
    else:
        checkpoint_path = args.checkpoint

    print(f"\n🧠 加载策略: {checkpoint_path}")
    policy = torch.jit.load(checkpoint_path)
    policy.eval()

    # ==================================================================
    # 4. 评估循环
    # ==================================================================
    print(f"\n🎮 开始评估...\n")
    obs_dict = env.reset()
    obs = obs_dict["policy"]

    total_reward = 0.0
    total_steps = 0

    try:
        while True:
            with torch.no_grad():
                actions = policy(obs)

            obs_dict, rewards, dones, infos = env.step(actions)
            obs = obs_dict["policy"]

            total_reward += rewards.mean().item()
            total_steps += 1

            if dones.any():
                print(f"  Episode done at step {total_steps}, avg reward: {total_reward / total_steps:.3f}")
                total_reward = 0.0
                total_steps = 0

    except KeyboardInterrupt:
        print("\n\n⏹️  评估中断")

    env.close()


if __name__ == "__main__":
    main()
