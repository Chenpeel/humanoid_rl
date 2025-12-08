"""
简单环境测试 - 测试单个环境的reset和step
"""

import jax
import jax.numpy as jp
from rich.console import Console
from rich.panel import Panel

from jiyuan_rl.envs import create_jiyuan_env

console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]简单环境测试[/bold green]\n"
        "[dim]测试单个环境的reset和step[/dim]",
        border_style="green"
    ))
    
    # 创建环境
    console.print("\n[cyan]创建环境...[/cyan]")
    xml_path = "/home/chenpeel/Desktop/duck/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    env = create_jiyuan_env(xml_path=xml_path, verbose=False)
    console.print(f"  ✓ obs={env.observation_size}, act={env.action_size}")
    
    # 测试reset
    console.print("\n[cyan]测试reset...[/cyan]")
    rng = jax.random.PRNGKey(42)
    env_state = env.reset(rng)
    console.print(f"  ✓ obs shape: {env_state.obs.shape}")
    console.print(f"  ✓ reward: {env_state.reward}")
    console.print(f"  ✓ done: {env_state.done}")
    
    # 测试step
    console.print("\n[cyan]测试step (10步)...[/cyan]")
    for i in range(10):
        action = jp.zeros(env.action_size)  # 零动作
        env_state = env.step(env_state, action)
        console.print(f"  步{i+1}: reward={float(env_state.reward):.4f}, done={env_state.done}")
    
    console.print("\n[bold green]✓ 测试通过！[/bold green]")
    return 0

if __name__ == "__main__":
    exit(main())
