"""
评估脚本 - 加载训练好的模型并评估
支持3D渲染可视化
"""

import argparse
import os
import sys
import io
import numpy as np

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

from rl.envs import create_velocity_tracking_env, create_walking_env
from rl.models import ActorCriticNetwork
from rl.utils import MujocoRenderer, InteractiveViewer, create_video_writer, save_frame_to_video

console = Console()


def infer_network_config_from_checkpoint(checkpoint_path: str):
    """从检查点推断网络配置"""
    from flax import serialization

    with open(checkpoint_path, 'rb') as f:
        data = serialization.msgpack_restore(f.read())

    # 检查点可能有不同的结构
    if 'params' in data and isinstance(data['params'], dict) and 'params' in data['params']:
        # 嵌套结构: data['params']['params']
        params = data['params']['params']
    elif 'params' in data:
        params = data['params']
    else:
        params = data

    hidden_dims = []
    shared_backbone = False

    # 调试：打印参数顶层键
    console.print(f"  [dim]调试: 网络参数键 = {list(params.keys())}[/dim]")

    # 方法1: 检查是否有共享backbone
    if 'backbone' in params:
        shared_backbone = True
        backbone = params['backbone']
        console.print(f"  [dim]调试: 共享backbone层 = {list(backbone.keys())}[/dim]")
        layer_names = sorted([k for k in backbone.keys() if k.startswith('Dense_')])

        for layer_name in layer_names:
            if 'kernel' in backbone[layer_name]:
                kernel_shape = backbone[layer_name]['kernel'].shape
                hidden_dims.append(kernel_shape[1])  # 输出维度

    # 方法2: 非共享backbone，检查actor_backbone和critic_backbone
    elif 'actor_backbone' in params and 'critic_backbone' in params:
        shared_backbone = False
        # 使用actor_backbone推断hidden_dims
        actor_backbone = params['actor_backbone']
        console.print(f"  [dim]调试: actor_backbone层 = {list(actor_backbone.keys())}[/dim]")
        layer_names = sorted([k for k in actor_backbone.keys() if k.startswith('Dense_')])

        for layer_name in layer_names:
            if 'kernel' in actor_backbone[layer_name]:
                kernel_shape = actor_backbone[layer_name]['kernel'].shape
                hidden_dims.append(kernel_shape[1])  # 输出维度

    # 推断观测空间维度(从第一层输入)
    obs_dim = None
    if shared_backbone and 'backbone' in params and 'Dense_0' in params['backbone']:
        if 'kernel' in params['backbone']['Dense_0']:
            obs_dim = params['backbone']['Dense_0']['kernel'].shape[0]
    elif not shared_backbone and 'actor_backbone' in params and 'Dense_0' in params['actor_backbone']:
        if 'kernel' in params['actor_backbone']['Dense_0']:
            obs_dim = params['actor_backbone']['Dense_0']['kernel'].shape[0]

    # 推断动作空间维度(从actor_mean/actor_head层)
    action_dim = None
    if 'actor_mean' in params and 'kernel' in params['actor_mean']:
        action_dim = params['actor_mean']['kernel'].shape[1]
    elif 'actor_head' in params and 'kernel' in params['actor_head']:
        action_dim = params['actor_head']['kernel'].shape[1]

    return {
        'hidden_dims': hidden_dims,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'shared_backbone': shared_backbone,
        'step': data.get('step', data.get('params', {}).get('step', 0))
    }


def load_checkpoint(checkpoint_path: str, network, rng):
    """加载检查点"""
    from flax import serialization

    # 读取Flax序列化的检查点文件
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = serialization.msgpack_restore(f.read())

    params = checkpoint_data['params']
    step = checkpoint_data['step']

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
    mj_data_for_viewer = None  # 为交互式查看器保留一个MjData实例

    if render:
        # 如果要保存视频，使用离线渲染器
        # 否则优先使用交互式查看器
        if save_video:
            console.print("[cyan]使用离线渲染器（保存视频）...[/cyan]")
            import mujoco
            mj_model = env.mj_model
            renderer = MujocoRenderer(mj_model, width=1280, height=720, camera_name="track")
            console.print("[green]✓ 离线渲染器已创建（使用 track 相机）[/green]")

            if video_path:
                video_writer = create_video_writer(video_path, fps=50)
                console.print(f"[green]✓ 视频写入器已创建: {video_path}[/green]")
        else:
            # 不保存视频时，使用交互式查看器
            try:
                console.print("[cyan]启动交互式查看器...[/cyan]")
                import mujoco
                mj_model = env.mj_model
                mj_data_for_viewer = mujoco.MjData(mj_model)
                viewer = InteractiveViewer(mj_model, mj_data_for_viewer)
                console.print("[green]✓ 交互式查看器已启动[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ 无法启动交互式查看器: {e}[/yellow]")
                console.print("[cyan]使用离线渲染器...[/cyan]")
                mj_model = env.mj_model
                renderer = MujocoRenderer(mj_model, width=1280, height=720, camera_name="track")
                console.print("[green]✓ 离线渲染器已创建[/green]")

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

                # 获取并显示命令
                if "command" in env_state.info:
                    cmd = env_state.info["command"]
                    console.print(f"[yellow]Episode {episode+1} 命令: vx={cmd[0]:.3f}, vy={cmd[1]:.3f}, vyaw={cmd[2]:.3f}[/yellow]")

                episode_return = 0.0
                episode_length = 0
                done = False
                step_count = 0

                while not done and step_count < max_steps:
                    # 获取动作（确定性策略，不采样）
                    mean, log_std, _ = network.apply(
                        params, env_state.obs[None, :])
                    action = mean[0]  # 使用均值，不加噪声

                    # 调试：显示前几步的动作和奖励
                    if step_count < 10:  # 增加到10步，观察着陆过程
                        qpos_np = np.array(env_state.pipeline_state.qpos)
                        qvel_np = np.array(env_state.pipeline_state.qvel)
                        console.print(f"  Step {step_count}: action range=[{action.min():.3f}, {action.max():.3f}], reward={env_state.reward:.3f}")
                        console.print(f"    pos=[{qpos_np[0]:.3f}, {qpos_np[1]:.3f}, {qpos_np[2]:.3f}], quat=[{qpos_np[3]:.3f}, {qpos_np[4]:.3f}, {qpos_np[5]:.3f}, {qpos_np[6]:.3f}]")
                        console.print(f"    vel=[{qvel_np[0]:.3f}, {qvel_np[1]:.3f}, {qvel_np[2]:.3f}]")

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
                            qpos_np = np.array(env_state.pipeline_state.qpos)
                            qvel_np = np.array(env_state.pipeline_state.qvel)

                            mj_data_for_viewer.qpos[:] = qpos_np
                            mj_data_for_viewer.qvel[:] = qvel_np
                            mujoco.mj_forward(viewer.model, mj_data_for_viewer)
                            viewer.update(mj_data_for_viewer)
                            time.sleep(0.02)  # 控制帧率~50Hz

                        elif renderer:
                            # 离线渲染
                            import mujoco
                            qpos_np = np.array(env_state.pipeline_state.qpos)
                            qvel_np = np.array(env_state.pipeline_state.qvel)

                            renderer.data.qpos[:] = qpos_np
                            renderer.data.qvel[:] = qvel_np
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
    parser.add_argument('--use-local-mjcf', action='store_true',
                        help='使用本地assets/mjcf中的MJCF模型')
    parser.add_argument('--num-episodes', type=int,
                        default=10, help='评估episode数')
    parser.add_argument('--render', type=int, default=1,
                        help='渲染间隔（0=不渲染，N=每N步渲染一次，默认=1）')
    parser.add_argument('--save-video', action='store_true', default=True, help='保存视频（默认开启）')
    parser.add_argument('--no-save-video', dest='save_video', action='store_false', help='不保存视频')
    parser.add_argument('--video-path', type=str, default='eval_video.mp4',
                        help='视频保存路径')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='每个episode最大步数')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 512, 256],
                        help='网络隐藏层维度 (默认: [512, 512, 256]，自动从检查点推断)')
    parser.add_argument('--shared-backbone', action='store_true', default=True,
                        help='使用共享backbone (默认: True，自动从检查点推断)')
    parser.add_argument('--env-type', type=str, default='walking',
                        choices=['velocity', 'walking'],
                        help='环境类型: velocity=速度跟踪(61维), walking=行走任务(65维，默认)')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU运行（避免GPU冲突，速度较慢）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置设备
    if args.cpu:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'
        console.print("[yellow]使用 CPU 模式运行（速度较慢）[/yellow]")

    console.print(Panel.fit(
        "[bold green]策略评估[/bold green]\n"
        "[dim]加载检查点并评估策略性能[/dim]",
        border_style="green"
    ))

    # 🔍 自动推断检查点的网络配置
    console.print("\n[bold cyan]分析检查点[/bold cyan]")
    try:
        ckpt_config = infer_network_config_from_checkpoint(args.checkpoint)
        console.print(f"  ✓ 检查点 step: {ckpt_config['step']}")
        console.print(f"  ✓ 检测到网络结构: hidden_dims={ckpt_config['hidden_dims']}")
        console.print(f"  ✓ 共享backbone: {ckpt_config['shared_backbone']}")
        console.print(f"  ✓ 检测到观测维度: {ckpt_config['obs_dim']}")
        console.print(f"  ✓ 检测到动作维度: {ckpt_config['action_dim']}")

        # 自动使用检查点的配置（优先级高于命令行）
        console.print(f"  [yellow]→ 自动使用检查点配置:[/yellow]")
        args.hidden_dims = ckpt_config['hidden_dims']
        args.shared_backbone = ckpt_config['shared_backbone']
        console.print(f"    hidden_dims={args.hidden_dims}")
        console.print(f"    shared_backbone={args.shared_backbone}")

    except Exception as e:
        console.print(f"  [red]✗ 无法推断检查点配置: {e}[/red]")
        console.print(f"  [yellow]→ 使用命令行参数:[/yellow]")
        console.print(f"    hidden_dims={args.hidden_dims}")
        console.print(f"    shared_backbone={args.shared_backbone}")
        ckpt_config = None  # 标记为未能推断

    # 设置随机种子
    rng = jax.random.PRNGKey(args.seed)

    # 处理XML路径
    xml_path = args.xml_path
    if args.use_local_mjcf:
        console.print("[cyan]使用本地MJCF模型...[/cyan]")
        xml_path = "assets/xmls/scenes/flat_terrain.xml"
    elif xml_path is None:
        # 默认使用平地场景
        xml_path = "assets/xmls/scenes/flat_terrain.xml"
        console.print(f"[yellow]未指定场景文件，使用默认场景: {xml_path}[/yellow]")

    console.print(f"[cyan]模型文件: {xml_path}[/cyan]")

    # 创建环境
    console.print("\n[bold cyan]创建环境[/bold cyan]")
    if args.env_type == 'walking':
        env = create_walking_env(xml_path=xml_path, verbose=False)
        console.print(f"  ✓ 环境类型: 行走环境 (WalkingEnv)")
    else:
        env = create_velocity_tracking_env(xml_path=xml_path, verbose=False)
        console.print(f"  ✓ 环境类型: 速度跟踪环境 (VelocityTrackingEnv)")
    console.print(f"  ✓ obs={env.observation_size}, act={env.action_size}")

    # ⚠️ 检查环境和检查点是否匹配
    try:
        if ckpt_config['obs_dim'] and env.observation_size != ckpt_config['obs_dim']:
            console.print(f"  [red]✗ 观测空间不匹配！[/red]")
            console.print(f"    检查点需要: obs={ckpt_config['obs_dim']}")
            console.print(f"    当前环境: obs={env.observation_size}")
            console.print(f"  [red]→ 评估可能失败，请检查环境配置或使用正确的检查点[/red]")
        if ckpt_config['action_dim'] and env.action_size != ckpt_config['action_dim']:
            console.print(f"  [red]✗ 动作空间不匹配！[/red]")
            console.print(f"    检查点需要: act={ckpt_config['action_dim']}")
            console.print(f"    当前环境: act={env.action_size}")
            console.print(f"  [red]→ 评估可能失败，请检查环境配置或使用正确的检查点[/red]")
    except NameError:
        # ckpt_config 未定义(推断失败)
        pass

    # 创建网络
    console.print("\n[bold cyan]创建网络[/bold cyan]")
    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=args.shared_backbone,
        hidden_dims=tuple(args.hidden_dims),
    )
    console.print(f"  ✓ 网络创建完成")
    console.print(f"  ✓ shared_backbone={args.shared_backbone}, hidden_dims={args.hidden_dims}")

    # 加载检查点
    console.print("\n[bold cyan]加载检查点[/bold cyan]")
    params, step = load_checkpoint(args.checkpoint, network, rng)
    console.print(f"  ✓ 检查点参数已加载，覆盖默认初始化")

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
