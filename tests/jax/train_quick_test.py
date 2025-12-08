"""
快速训练测试脚本（10次更新）
验证所有组件正常工作
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jp
from rich.console import Console
from rich.panel import Panel
from rich import box
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.jiyuan_rl.envs.jiyuan_mjx_env import create_jiyuan_env
from src.jiyuan_rl.models.networks import ActorCriticNetwork, count_parameters
from src.jiyuan_rl.models.optimizer import create_ppo_optimizer
from src.jiyuan_rl.training.train_state import create_train_state
from src.jiyuan_rl.training.ppo_trainer import PPOTrainer, PPOConfig
from src.jiyuan_rl.training.logger import Logger

console = Console()


def main():
    """快速测试训练流程"""
    console.print(Panel.fit(
        "[bold green]快速训练测试[/bold green]\n"
        "[dim]验证所有组件正常工作（10次更新）[/dim]",
        border_style="green"
    ))
    
    # 配置（小规模）
    console.print("\n[bold cyan]配置[/bold cyan]")
    config = PPOConfig(
        num_envs=32,  # 小规模测试
        num_steps=20,
        num_epochs=2,
        num_minibatches=4,
        total_timesteps=100000,  # 不用完
        log_interval=5,
    )
    console.print(f"  并行环境: {config.num_envs}")
    console.print(f"  Rollout步数: {config.num_steps}")
    console.print(f"  批次大小: {config.batch_size}")
    
    # 设备
    console.print(f"\n[bold cyan]设备[/bold cyan]: {jax.devices()}")
    
    # 创建环境
    console.print("\n[bold cyan]创建环境[/bold cyan]")
    xml_path = "/home/chenpeel/Desktop/duck/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    env = create_jiyuan_env(xml_path=xml_path)
    console.print(f"  ✓ obs={env.observation_size}, act={env.action_size}")
    
    # 创建网络
    console.print("\n[bold cyan]创建网络[/bold cyan]")
    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=True,
        hidden_dims=(128, 128),  # 小模型
    )
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jp.zeros((1, env.observation_size)))
    console.print(f"  ✓ 参数: {count_parameters(params):,}")
    
    # 创建优化器
    console.print("\n[bold cyan]创建优化器[/bold cyan]")
    optimizer = create_ppo_optimizer(learning_rate=3e-4, max_grad_norm=0.5)
    console.print(f"  ✓ Adam, lr=3e-4")
    
    # 初始化训练状态
    console.print("\n[bold cyan]初始化训练状态[/bold cyan]")
    train_state = create_train_state(
        network=network,
        optimizer=optimizer,
        obs_shape=(env.observation_size,),
        rng=rng,
    )
    console.print(f"  ✓ step={train_state.step}")
    
    # 初始化环境
    console.print("\n[bold cyan]初始化环境[/bold cyan]")
    rng, reset_rng = jax.random.split(rng)
    env_state = env.batch_reset(reset_rng, config.num_envs)
    console.print(f"  ✓ {config.num_envs}个环境，obs shape={env_state.obs.shape}")
    
    # 创建训练器
    console.print("\n[bold cyan]创建训练器[/bold cyan]")
    trainer = PPOTrainer(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
    )
    console.print(f"  ✓ PPOTrainer")
    
    # 测试训练步
    console.print("\n[bold cyan]执行测试训练（10次更新）[/bold cyan]")
    
    try:
        for i in range(10):
            console.print(f"\n--- 更新 {i+1}/10 ---")
            
            train_state, env_state, info = trainer.train_step(train_state, env_state)
            
            console.print(f"  step={train_state.step}, env_steps={train_state.env_steps}")
            console.print(f"  mean_reward={info.get('mean_reward', 0):.4f}")
            console.print(f"  policy_loss={info.get('policy_loss', 0):.4f}")
            console.print(f"  value_loss={info.get('value_loss', 0):.4f}")
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]✓ 测试通过！所有组件正常工作[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            f"[bold red]✗ 测试失败[/bold red]\n{e}",
            border_style="red"
        ))
        import traceback
        console.print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
