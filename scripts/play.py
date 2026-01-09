"""Policy playback (visualization-only).

Loads a trained checkpoint and shows the robot acting in a MuJoCo viewer.
No evaluation metrics are computed; this is for qualitative visualization.
"""

import argparse
import contextlib
import io
import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化策略（仅播放，不评估）")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="MuJoCo XML路径（默认根据robot_name解析）",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="保存视频（离线渲染；可与viewer同时开启，但会更慢）",
    )
    parser.add_argument("--video-path", type=str,
                        default="play_video.mp4", help="视频保存路径")
    parser.add_argument("--video-fps", type=int, default=50, help="视频FPS")
    parser.add_argument("--render-width", type=int, default=1280, help="渲染宽度")
    parser.add_argument("--render-height", type=int, default=720, help="渲染高度")
    parser.add_argument("--camera-name", type=str,
                        default="track", help="MuJoCo相机名称")
    parser.add_argument(
        "--record-interval",
        type=int,
        default=None,
        help="录制间隔（每N步录一帧；默认=render或1）",
    )
    parser.add_argument(
        "--use-local-mjcf",
        action="store_true",
        help="使用本地assets/xmls/scenes/flat_terrain.xml",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="walking",
        choices=["velocity", "walking"],
        help="环境类型: velocity / walking",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="机器人名称 (可选: gaoda_jiyuan, unitree_h1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=1, help="播放episode数")
    parser.add_argument("--max-steps", type=int, default=2000, help="每个episode最大步数")
    parser.add_argument(
        "--render",
        type=int,
        default=1,
        help="渲染间隔（0=不渲染，N=每N步渲染一次，默认=1）",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=0.0,
        help="每帧sleep秒数（用于限速；默认0=尽快播放）",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按环境control_dt实时播放（会sleep限速；优先级高于--viewer-sleep）",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="使用策略均值动作（默认开启，更稳定）",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="按log_std采样动作（更随机）",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="禁用外层JIT（MJX仍可能编译；播放会显著变慢，通常仅用于调试）",
    )
    parser.add_argument(
        "--status-every",
        type=float,
        default=2.0,
        help="每隔N秒打印一次播放进度（0=不打印）",
    )
    parser.add_argument(
        "--no-jax-prealloc",
        action="store_true",
        help="禁用JAX预分配显存（降低显存占用，可能影响性能；需在导入jax前设置）",
    )
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="设置JAX显存占用比例(0~1)，例如0.5（需在导入jax前设置）",
    )
    parser.add_argument("--cpu", action="store_true", help="使用CPU运行（更慢）")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        best_model_file = ckpt_path / "best_model"
        if best_model_file.exists():
            ckpt_path = best_model_file
        else:
            files = sorted([p for p in ckpt_path.iterdir() if p.is_file()])
            if len(files) == 1:
                ckpt_path = files[0]
            elif files:
                ckpt_path = max(files, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(f"检查点目录为空: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")
    args.checkpoint = str(ckpt_path)

    # Enable persistent JAX compilation cache (reduces repeated first-step compile cost).
    # Respect user-provided env vars when set.
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    if args.no_jax_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.jax_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import numpy as np

    # MuJoCo may print optional warp-related warnings to stderr during import; keep playback logs clean.
    with contextlib.redirect_stderr(io.StringIO()):
        import mujoco

        from rl.envs import create_velocity_tracking_env, create_walking_env
        from rl.models import ActorCriticNetwork
        from rl.utils import (
            InteractiveViewer,
            MujocoRenderer,
            create_video_writer,
            save_frame_to_video,
        )

    eval_script_path = Path(__file__).with_name("eval.py")
    spec = spec_from_file_location("_jrl_eval_script", eval_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载评估脚本模块: {eval_script_path}")
    eval_script = module_from_spec(spec)
    spec.loader.exec_module(eval_script)
    infer_network_config_from_checkpoint = eval_script.infer_network_config_from_checkpoint
    load_checkpoint = eval_script.load_checkpoint

    console.print(
        Panel.fit(
            "[bold green]策略可视化播放[/bold green]\n[dim]加载检查点并在MuJoCo查看器中播放策略行为[/dim]",
            border_style="green",
        )
    )

    ckpt_config = infer_network_config_from_checkpoint(args.checkpoint)
    args_hidden_dims = tuple(ckpt_config["hidden_dims"]) if ckpt_config["hidden_dims"] else (512, 512, 256)
    args_shared_backbone = bool(ckpt_config["shared_backbone"])

    xml_path = args.xml_path
    if args.use_local_mjcf:
        xml_path = "assets/xmls/scenes/flat_terrain.xml"
    elif xml_path is None:
        robot_name = args.robot_name or "gaoda_jiyuan"
        from rl.utils.robot_config import resolve_scene_path

        xml_path = str(resolve_scene_path(robot_name))

    if args.env_type == "walking":
        env = create_walking_env(
            xml_path=xml_path, robot_name=args.robot_name or "gaoda_jiyuan", verbose=False
        )
    else:
        env = create_velocity_tracking_env(
            xml_path=xml_path, robot_name=args.robot_name or "gaoda_jiyuan", verbose=False
        )

    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=args_shared_backbone,
        hidden_dims=args_hidden_dims,
    )

    rng = jax.random.PRNGKey(args.seed)
    params, step = load_checkpoint(args.checkpoint, network, rng)

    console.print(
        f"[cyan]checkpoint step={step} | env={args.env_type} | render={args.render} | "
        f"deterministic={args.deterministic}[/cyan]"
    )

    if args.no_jit:
        console.print(
            "[yellow]提示: --no-jit 会显著降低MJX环境播放速度（每步会触发大量小的JAX调度/同步）。"
            "想流畅播放请去掉 NO_JIT=1，或提高 RENDER 间隔（如 RENDER=10）。[/yellow]"
        )

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    viewer = None
    if args.render > 0:
        viewer = InteractiveViewer(mj_model, mj_data)
        console.print("[green]✓ 交互式查看器已启动[/green]")

    renderer = None
    video_writer = None
    if args.save_video:
        renderer = MujocoRenderer(
            mj_model,
            width=args.render_width,
            height=args.render_height,
            camera_name=args.camera_name,
        )
        video_writer = create_video_writer(
            args.video_path,
            fps=args.video_fps,
            width=args.render_width,
            height=args.render_height,
        )

    def _policy_action(obs, rng_key):
        mean, log_std, _ = network.apply(params, obs)
        if args.deterministic:
            return mean, rng_key
        rng_key, noise_key = jax.random.split(rng_key)
        std = jax.numpy.exp(log_std)
        action = mean + std * jax.random.normal(noise_key, shape=mean.shape)
        return action, rng_key

    def _step_fn(state, rng_key):
        action, rng_key = _policy_action(state.obs, rng_key)
        action = jax.numpy.clip(action, -10.0, 10.0)
        new_state = env.step(state, action)
        return new_state, action, rng_key

    step_fn = _step_fn if args.no_jit else jax.jit(_step_fn)

    record_interval = args.record_interval
    if record_interval is None:
        record_interval = args.render if args.render and args.render > 0 else 1
    if args.save_video and record_interval <= 0:
        console.print("[yellow]record-interval<=0，自动改为1以便录制[/yellow]")
        record_interval = 1

    try:
        for ep in range(args.episodes):
            rng, reset_rng = jax.random.split(rng)
            env_state = env.reset(reset_rng)

            if not args.no_jit:
                console.print("[dim]编译JIT中（首次会较慢，请稍等一次）...[/dim]")
                t0 = time.perf_counter()
                env_state, _, rng = step_fn(env_state, rng)
                try:
                    env_state.reward.block_until_ready()
                except Exception:
                    jax.block_until_ready(env_state.reward)
                dt_compile = time.perf_counter() - t0
                console.print(f"[dim]JIT编译完成，用时 {dt_compile:.1f}s[/dim]")

            wall_start = time.perf_counter()
            steps_rendered = 0
            last_status_t = wall_start
            last_status_step = 0

            for step_idx in range(args.max_steps):
                if viewer and not viewer.is_alive():
                    return 0

                if ep == 0 and step_idx == 0:
                    console.print(
                        "[dim]开始执行第一步（如果是首次运行/首次GPU执行，可能会卡一会儿做编译）...[/dim]"
                    )

                t_step0 = time.perf_counter()
                env_state, _, rng = step_fn(env_state, rng)

                need_viewer = (
                    viewer is not None
                    and args.render > 0
                    and (step_idx + 1) % args.render == 0
                )
                need_record = (
                    renderer is not None
                    and video_writer is not None
                    and (step_idx + 1) % record_interval == 0
                )
                if not (need_viewer or need_record):
                    if args.status_every and args.status_every > 0:
                        now = time.perf_counter()
                        if now - last_status_t >= args.status_every:
                            try:
                                env_state.reward.block_until_ready()
                            except Exception:
                                jax.block_until_ready(env_state.reward)
                            sps = (step_idx + 1 - last_status_step) / (now - last_status_t)
                            console.print(
                                f"[dim]step={step_idx+1}/{args.max_steps} | "
                                f"sim={sps:.1f} steps/s | rendered={steps_rendered}[/dim]"
                            )
                            last_status_t = now
                            last_status_step = step_idx + 1
                    continue

                # Only sync device → host when we actually need to draw/record.
                qpos_np, qvel_np = jax.device_get(
                    (env_state.pipeline_state.qpos, env_state.pipeline_state.qvel)
                )
                mj_data.qpos[:] = qpos_np
                mj_data.qvel[:] = qvel_np
                mj_data.ctrl[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)

                if need_viewer:
                    viewer.update(mj_data)
                    steps_rendered += 1

                    if args.realtime:
                        target_t = steps_rendered * env.control_dt * args.render
                        elapsed = time.perf_counter() - wall_start
                        sleep_s = target_t - elapsed
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                    elif args.viewer_sleep > 0:
                        time.sleep(args.viewer_sleep)

                if need_record:
                    frame = renderer.render(mj_data)
                    save_frame_to_video(video_writer, frame)

                if ep == 0 and step_idx == 0:
                    dt_first = time.perf_counter() - t_step0
                    console.print(
                        f"[dim]第一帧完成，用时 {dt_first:.2f}s | base_z={mj_data.qpos[2]:.3f}[/dim]"
                    )

                if args.status_every and args.status_every > 0:
                    now = time.perf_counter()
                    if now - last_status_t >= args.status_every:
                        sps = (step_idx + 1 - last_status_step) / (now - last_status_t)
                        console.print(
                            f"[dim]step={step_idx+1}/{args.max_steps} | "
                            f"sim={sps:.1f} steps/s | rendered={steps_rendered}[/dim]"
                        )
                        last_status_t = now
                        last_status_step = step_idx + 1

            console.print(f"[dim]episode {ep+1}/{args.episodes} finished[/dim]")
    finally:
        if viewer:
            viewer.close()
        if renderer:
            renderer.close()
        if video_writer:
            video_writer.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
