"""
测试渲染功能 - 简单示例
"""

import argparse
import jax
import jax.numpy as jp
from rich.console import Console
from rich.panel import Panel

from jiyuan_rl.envs import create_jiyuan_env
from jiyuan_rl.utils import setup_jiyuan_urdf, InteractiveViewer
import mujoco
import time

console = Console()


def test_renderer(xml_path: str, num_steps: int = 200):
    """测试渲染器"""
    console.print(Panel.fit(
        "[bold green]测试MuJoCo渲染器[/bold green]\n"
        f"[dim]运行{num_steps}步仿真并渲染[/dim]",
        border_style="green"
    ))
    
    console.print(f"\n[cyan]加载模型: {xml_path}[/cyan]")
    
    # 创建环境
    env = create_jiyuan_env(xml_path=xml_path, verbose=False)
    console.print(f"  ✓ 环境创建成功")
    
    # 创建MuJoCo模型和查看器
    console.print("\n[cyan]启动交互式查看器...[/cyan]")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    try:
        viewer = InteractiveViewer(mj_model, mj_data)
        console.print("[green]✓ 查看器已启动[/green]")
        
        # 重置环境
        rng = jax.random.PRNGKey(42)
        env_state = env.reset(rng)
        
        console.print(f"\n[cyan]开始仿真（{num_steps}步）...[/cyan]")
        console.print("[dim]提示: 关闭查看器窗口以停止[/dim]\n")
        
        step = 0
        while viewer.is_alive() and step < num_steps:
            # 随机动作
            action = jax.random.normal(rng, (env.action_size,)) * 0.1
            rng, _ = jax.random.split(rng)
            
            # 环境步进
            env_state = env.step(env_state, action)
            
            # 更新查看器
            mj_data.qpos[:] = env_state.pipeline_state.q
            mj_data.qvel[:] = env_state.pipeline_state.qd
            mujoco.mj_forward(mj_model, mj_data)
            viewer.update(mj_data)
            
            step += 1
            
            if step % 50 == 0:
                console.print(f"  步数: {step}/{num_steps}, 奖励: {float(env_state.reward):.4f}")
            
            time.sleep(0.02)  # ~50Hz
        
        console.print(f"\n[green]✓ 仿真完成！总步数: {step}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ 错误: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return 1
    finally:
        if 'viewer' in locals():
            viewer.close()
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='测试MuJoCo渲染功能')
    parser.add_argument('--xml-path', type=str, default=None,
                       help='MuJoCo XML路径')
    parser.add_argument('--use-local-urdf', action='store_true',
                       help='使用本地assets/urdf中的URDF模型')
    parser.add_argument('--num-steps', type=int, default=200,
                       help='仿真步数')
    
    args = parser.parse_args()
    
    # 处理XML路径
    xml_path = args.xml_path
    if args.use_local_urdf:
        console.print("[cyan]使用本地URDF模型...[/cyan]")
        xml_path = setup_jiyuan_urdf()
    elif xml_path is None:
        # 默认使用Open_Duck_Playground
        xml_path = "/home/chenpeel/Desktop/duck/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    
    return test_renderer(xml_path, args.num_steps)


if __name__ == "__main__":
    exit(main())
