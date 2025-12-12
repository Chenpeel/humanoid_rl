"""
评估脚本 - 加载训练好的模型并评估
支持3D渲染可视化
"""

import argparse
import os
import sys
import io

# 屏蔽 MuJoCo warp 警告
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

import jax
import jax.numpy as jp

# 恢复 stderr
sys.stderr = _original_stderr
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import time

from rl.envs import create_velocity_tracking_env
from rl.models import ActorCriticNetwork
from rl.utils import MujocoRenderer, InteractiveViewer, create_video_writer, save_frame_to_video

console = Console()


def load_checkpoint(checkpoint_path: str, network, rng):
    """加载检查点"""
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    params = restored['params']
    step = restored['step']

    console.print(f"[green]✓ 加载检查点: step={step}[/green]")
    return params, step


def evaluate_policy(
    env,
    network,
    params,
    num_episodes: int = 10,
    render: bool = False,
    render_interval: int = 1,
    save_video: bool = False,
    video_path: str = None,
    max_steps: int = 1000,
):
    """
    评估策略

    Args:
        env: 环境
        network: 网络
        params: 网络参数
        num_episodes: 评估episode数
        render: 是否渲染
        render_interval: 渲染间隔（每N步渲染一次）
        save_video: 是否保存视频
        video_path: 视频保存路径
        max_steps: 每个episode最大步数
    """
    console.print(Panel.fit(
        f"[bold cyan]评估策略[/bold cyan]\n"
        f"Episodes: {num_episodes}\n"
        f"渲染: {'是' if render else '否'}\n"
        f"保存视频: {'是' if save_video else '否'}",
        border_style="cyan"
    ))

    # 创建渲染器
    renderer = None
    viewer = None
    video_writer = None

    if render:
        try:
            # 尝试交互式查看器
            console.print("[cyan]启动交互式查看器...[/cyan]")
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(env.xml_path)
            mj_data = mujoco.MjData(mj_model)
            viewer = InteractiveViewer(mj_model, mj_data)
            console.print("[green]✓ 交互式查看器已启动[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ 无法启动交互式查看器: {e}[/yellow]")
            console.print("[cyan]使用离线渲染器...[/cyan]")
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(env.xml_path)
            renderer = MujocoRenderer(mj_model, width=1280, height=720)

            if save_video and video_path:
                video_writer = create_video_writer(video_path, fps=50)
                console.print(f"[green]✓ 视频写入器已创建: {video_path}[/green]")

    # 统计数据
    episode_returns = []
    episode_lengths = []

    rng = jax.random.PRNGKey(42)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]评估进度...", total=num_episodes)

            for episode in range(num_episodes):
                # 重置环境
                rng, reset_rng = jax.random.split(rng)
                env_state = env.reset(reset_rng)

                episode_return = 0.0
                episode_length = 0
                done = False
                step_count = 0

                while not done and step_count < max_steps:
                    # 获取动作（确定性策略，不采样）
                    mean, log_std, _ = network.apply(
                        params, env_state.obs[None, :])
                    action = mean[0]  # 使用均值，不加噪声

                    # 环境步进
                    env_state = env.step(env_state, action)

                    episode_return += float(env_state.reward)
                    episode_length += 1
                    done = bool(env_state.done)
                    step_count += 1

                    # 渲染
                    if render and step_count % render_interval == 0:
                        if viewer and viewer.is_alive():
                            # 更新交互式查看器
                            import mujoco
                            mj_data = mujoco.MjData(viewer.model)
                            # 从env_state.pipeline_state复制状态到mj_data
                            # 注意：这里需要从MJX状态转换回MuJoCo状态
                            # 简化处理：直接前向仿真
                            mj_data.qpos[:] = env_state.pipeline_state.q
                            mj_data.qvel[:] = env_state.pipeline_state.qd
                            mujoco.mj_forward(viewer.model, mj_data)
                            viewer.update(mj_data)
                            time.sleep(0.02)  # 控制帧率~50Hz

                        elif renderer:
                            # 离线渲染
                            import mujoco
                            renderer.data.qpos[:] = env_state.pipeline_state.q
                            renderer.data.qvel[:] = env_state.pipeline_state.qd
                            mujoco.mj_forward(renderer.model, renderer.data)
                            frame = renderer.render(renderer.data)

                            if video_writer:
                                save_frame_to_video(video_writer, frame)

                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)

                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Episode {episode+1}/{num_episodes} | Return: {episode_return:.2f} | Length: {episode_length}"
                )

    finally:
        # 清理
        if viewer:
            viewer.close()
        if renderer:
            renderer.close()
        if video_writer:
            video_writer.release()

    # 显示统计结果
    results_table = Table(title="评估结果", box=box.ROUNDED)
    results_table.add_column("指标", style="cyan")
    results_table.add_column("值", style="green", justify="right")

    results_table.add_row("Episodes", str(num_episodes))
    results_table.add_row("平均回报", f"{jp.mean(jp.array(episode_returns)):.2f}")
    results_table.add_row("回报标准差", f"{jp.std(jp.array(episode_returns)):.2f}")
    results_table.add_row("最大回报", f"{max(episode_returns):.2f}")
    results_table.add_row("最小回报", f"{min(episode_returns):.2f}")
    results_table.add_row("平均长度", f"{jp.mean(jp.array(episode_lengths)):.0f}")

    console.print(results_table)

    return {
        'mean_return': float(jp.mean(jp.array(episode_returns))),
        'std_return': float(jp.std(jp.array(episode_returns))),
        'mean_length': float(jp.mean(jp.array(episode_lengths))),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description='评估强化学习策略')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    parser.add_argument('--xml-path', type=str, default=None,
                        help='MuJoCo XML路径（默认使用Open_Duck_Playground）')
    parser.add_argument('--use-local-urdf', action='store_true',
                        help='使用本地assets/urdf中的URDF模型')
    parser.add_argument('--num-episodes', type=int,
                        default=10, help='评估episode数')
    parser.add_argument('--render', type=int, default=0,
                        help='渲染间隔（0=不渲染，N=每N步渲染一次）')
    parser.add_argument('--save-video', action='store_true', help='保存视频')
    parser.add_argument('--video-path', type=str, default='eval_video.mp4',
                        help='视频保存路径')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='每个episode最大步数')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256],
                        help='网络隐藏层维度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]策略评估[/bold green]\n"
        "[dim]加载检查点并评估策略性能[/dim]",
        border_style="green"
    ))

    # 设置随机种子
    rng = jax.random.PRNGKey(args.seed)

    # 处理XML路径
    xml_path = args.xml_path
    if args.use_local_urdf:
        console.print("[cyan]使用本地URDF模型...[/cyan]")
        from rl.utils import setup_urdf
        xml_path = setup_urdf()
    elif xml_path is None:
        # 默认使用场景文件
        xml_path = "../assets/xmls/scene.xml"

    console.print(f"[cyan]模型文件: {xml_path}[/cyan]")

    # 创建环境
    console.print("\n[bold cyan]创建环境[/bold cyan]")
    env = create_velocity_tracking_env(xml_path=xml_path, verbose=False)
    console.print(f"  ✓ obs={env.observation_size}, act={env.action_size}")

    # 创建网络
    console.print("\n[bold cyan]创建网络[/bold cyan]")
    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=True,
        hidden_dims=tuple(args.hidden_dims),
    )
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jp.zeros((1, env.observation_size)))
    console.print(f"  ✓ 网络参数初始化完成")

    # 加载检查点
    console.print("\n[bold cyan]加载检查点[/bold cyan]")
    params, step = load_checkpoint(args.checkpoint, network, rng)

    # 评估
    results = evaluate_policy(
        env=env,
        network=network,
        params=params,
        num_episodes=args.num_episodes,
        render=args.render > 0,
        render_interval=args.render,
        save_video=args.save_video,
        video_path=args.video_path,
        max_steps=args.max_steps,
    )

    console.print("\n[bold green]✓ 评估完成！[/bold green]")
    return 0


if __name__ == "__main__":
    exit(main())
