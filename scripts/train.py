"""train.py —— 训练入口。

用法：
    python scripts/train.py robot=h1 task=standing num_envs=4096

使用 Hydra 配置管理，按 robot + task 两层组合。
"""

from __future__ import annotations

import argparse
import os
import sys

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Humanoid RL 训练脚本"
    )
    parser.add_argument("--robot", type=str, default="h1", help="机器人名称 (h1, h2, ...)")
    parser.add_argument("--task", type=str, default="standing", help="任务类型 (standing, walking, velocity_tracking)")
    parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数")
    parser.add_argument("--max_iterations", type=int, default=10000, help="最大训练迭代数")
    parser.add_argument("--headless", type=bool, default=True, help="无头模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--checkpoint", type=str, default=None, help="续训 checkpoint 路径")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Humanoid RL Training")
    print(f"  Robot: {args.robot}  |  Task: {args.task}")
    print(f"  Envs: {args.num_envs}  |  Max Iters: {args.max_iterations}")
    print("=" * 60)

    # ==================================================================
    # 1. 加载机器人配置
    # ==================================================================
    from humanoid_rl.utils import load_robot_config

    robot_cfg = load_robot_config(args.robot)

    # ==================================================================
    # 2. 构建任务环境配置
    # ==================================================================
    from humanoid_rl.envs.cfg.standing_cfg import StandingEnvCfg
    from humanoid_rl.envs.cfg.walking_cfg import WalkingEnvCfg
    from humanoid_rl.envs.cfg.velocity_cfg import VelocityTrackingEnvCfg

    task_cfg_map = {
        "standing": StandingEnvCfg,
        "walking": WalkingEnvCfg,
        "velocity_tracking": VelocityTrackingEnvCfg,
    }
    cfg_cls = task_cfg_map.get(args.task)
    if cfg_cls is None:
        raise ValueError(f"Unknown task: {args.task}. Choose from {list(task_cfg_map.keys())}")

    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args.num_envs

    # 注入机器人特定参数（PD 增益、关节名称等）
    env_cfg.robot_name = args.robot
    env_cfg.robot_asset_path = robot_cfg["asset"]["path"]
    env_cfg.robot_joint_names = robot_cfg["joints"]["names"]
    env_cfg.robot_default_joint_pos = robot_cfg["joints"]["default_positions"]
    env_cfg.robot_stiffness = robot_cfg["pd"]["stiffness"]
    env_cfg.robot_damping = robot_cfg["pd"]["damping"]

    # ==================================================================
    # 3. 构建环境
    # ==================================================================
    print(f"\n📦 构建环境: {args.robot} / {args.task}")
    from humanoid_rl.envs import BaseEnv

    env = BaseEnv(cfg=env_cfg)

    # ==================================================================
    # 4. 构建 PPO Agent
    # ==================================================================
    print(f"\n🧠 构建 PPO Agent")
    from humanoid_rl.agents import PPOCfg
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner

    ppo_cfg = PPOCfg()
    ppo_cfg.num_envs = args.num_envs
    ppo_cfg.max_iterations = args.max_iterations
    ppo_cfg.experiment_name = f"{args.robot}_{args.task}"

    runner = OnPolicyRunner(env, ppo_cfg, ppo_cfg, ppo_cfg)

    # ==================================================================
    # 5. 训练
    # ==================================================================
    print(f"\n🚀 开始训练...\n")
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"\n✅ 训练完成!")
    print(f"   模型保存至: logs/{ppo_cfg.experiment_name}/")


if __name__ == "__main__":
    main()
