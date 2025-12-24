#!/usr/bin/env python3
"""
测试 Jiyuan 环境的基础功能

此脚本用于验证：
1. 环境可以成功创建
2. 环境可以成功重置
3. 环境可以执行步骤
4. 观测和动作空间正确

运行方式:
    python scripts/test_env.py
    python scripts/test_env.py --num_envs 4
    python scripts/test_env.py --headless
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch


def test_env_creation(env_id: str, num_envs: int = 16):
    """测试环境创建"""
    print(f"\n{'='*60}")
    print(f"测试 1: 创建环境 '{env_id}'")
    print(f"{'='*60}")

    try:
        import gymnasium as gym
        from isaaclab_rl import jiyuan_tasks  # 触发环境注册

        env = gym.make(env_id, num_envs=num_envs)
        print(f"✅ 成功创建环境，并行环境数: {env.num_envs}")

        # 打印环境信息
        print(f"\n观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")

        return env

    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_env_reset(env):
    """测试环境重置"""
    print(f"\n{'='*60}")
    print(f"测试 2: 环境重置")
    print(f"{'='*60}")

    try:
        obs, info = env.reset()
        print(f"✅ 环境重置成功")
        print(f"   观测形状: {obs.shape}")
        print(f"   观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"   信息字典键: {list(info.keys())}")

        return obs

    except Exception as e:
        print(f"❌ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_env_step(env, num_steps: int = 10):
    """测试环境步进"""
    print(f"\n{'='*60}")
    print(f"测试 3: 环境步进 ({num_steps} 步)")
    print(f"{'='*60}")

    try:
        for step in range(num_steps):
            # 生成随机动作
            action = torch.rand(env.num_envs, env.action_space.shape[0]) * 2 - 1

            # 执行步骤
            obs, reward, terminated, truncated, info = env.step(action)

            if step == 0:
                print(f"✅ 首步执行成功")
                print(f"   观测形状: {obs.shape}")
                print(f"   奖励形状: {reward.shape}")
                print(f"   终止形状: {terminated.shape}")
                print(f"   截断形状: {truncated.shape}")

            # 打印进度
            if (step + 1) % 5 == 0:
                avg_reward = reward.mean().item()
                num_done = (terminated | truncated).sum().item()
                print(f"   步骤 {step+1}/{num_steps}: 平均奖励={avg_reward:.3f}, 完成={num_done}")

        print(f"\n✅ 所有 {num_steps} 步执行成功")
        return True

    except Exception as e:
        print(f"❌ 环境步进失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_properties(env):
    """测试环境属性"""
    print(f"\n{'='*60}")
    print(f"测试 4: 环境属性检查")
    print(f"{'='*60}")

    try:
        # 检查基础属性
        print(f"环境配置类: {env.cfg.__class__.__name__}")
        print(f"并行环境数: {env.num_envs}")
        print(f"观测维度: {env.observation_space.shape[0]}")
        print(f"动作维度: {env.action_space.shape[0]}")

        # 检查场景
        print(f"\n场景实体:")
        for entity_name in env.scene.keys():
            print(f"  - {entity_name}")

        # 检查机器人
        if "robot" in env.scene:
            robot = env.scene["robot"]
            print(f"\n机器人信息:")
            print(f"  - DOF 数量: {robot.num_joints}")
            print(f"  - Body 数量: {robot.num_bodies}")

        print(f"\n✅ 环境属性检查通过")
        return True

    except Exception as e:
        print(f"❌ 环境属性检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="测试 Jiyuan 环境")
    parser.add_argument(
        "--env_id",
        type=str,
        default="Isaac-Jiyuan-Test-v0",
        help="环境ID (默认: Isaac-Jiyuan-Test-v0)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="并行环境数 (默认: 16)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="测试步数 (默认: 10)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式（不显示GUI）",
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Jiyuan 环境测试脚本")
    print(f"{'#'*60}")
    print(f"环境ID: {args.env_id}")
    print(f"并行环境数: {args.num_envs}")
    print(f"测试步数: {args.num_steps}")
    print(f"无头模式: {args.headless}")

    # 运行测试
    results = {}

    # 测试1: 创建环境
    env = test_env_creation(args.env_id, args.num_envs)
    results["creation"] = env is not None

    if env is None:
        print("\n❌ 环境创建失败，测试终止")
        return 1

    # 测试2: 重置环境
    obs = test_env_reset(env)
    results["reset"] = obs is not None

    if obs is None:
        print("\n❌ 环境重置失败，测试终止")
        return 1

    # 测试3: 步进环境
    results["step"] = test_env_step(env, args.num_steps)

    # 测试4: 环境属性
    results["properties"] = test_env_properties(env)

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"{'='*60}")

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name.capitalize():15s}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
