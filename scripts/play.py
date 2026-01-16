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

from checkpoint_compat import (  # noqa: E402
    CheckpointFormat,
    resolve_checkpoint,
    run_xax_viewer_from_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化策略（仅播放，不评估）")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练/运行的YAML配置文件（用于复现env_config，如零速度命令/target_height/max_steps）",
    )
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
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="视频FPS（默认自动匹配control_dt/record_interval，避免视频加速/减速）",
    )
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
        default=None,
        choices=["velocity", "walking", "standing"],
        help="环境类型: velocity / walking",
    )
    parser.add_argument(
        "--cmd-x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="覆盖x方向速度命令范围（walking/velocity均适用）",
    )
    parser.add_argument(
        "--cmd-y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="覆盖y方向速度命令范围（walking/velocity均适用）",
    )
    parser.add_argument(
        "--cmd-yaw-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="覆盖yaw角速度命令范围（walking/velocity均适用）",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=None,
        help="覆盖target_height（仅walking适用）",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=None,
        help="覆盖环境episode最大步数max_steps（walking/velocity均适用）",
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

    config_data = {}
    if args.config:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError(
                f"无法导入yaml以读取配置文件: {e}. 请安装PyYAML或移除 --config"
            ) from e
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    env_type = args.env_type or config_data.get("env_type") or "walking"
    robot_name = args.robot_name or config_data.get("robot_name") or "gaoda_jiyuan"
    env_config = config_data.get("env_config") or {}

    resolved = resolve_checkpoint(args.checkpoint)
    args.checkpoint = str(resolved.path)

    # Enable persistent JAX compilation cache (reduces repeated first-step compile cost).
    # Respect user-provided env vars when set.
    project_root = Path(__file__).resolve().parent.parent
    src_root = project_root / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))

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

    if resolved.format == CheckpointFormat.XAX_TAR:
        console.print(
            Panel.fit(
                "[bold green]ksim/xax 策略回放[/bold green]\n"
                "[dim]检测到 xax checkpoint（tar.gz），将使用 ksim 内置 viewer 进行回放/录制。[/dim]",
                border_style="green",
            )
        )

        save_video = bool(args.save_video)
        if args.render > 0 and save_video:
            console.print("[yellow]提示: xax viewer 在 save_video 模式下为 offscreen，不会弹出交互窗口。[/yellow]")

        try:
            outputs = run_xax_viewer_from_checkpoint(
                resolved.path,
                num_steps=int(args.max_steps) if args.max_steps is not None else None,
                save_renders=save_video,
                save_video=save_video,
                render_width=int(args.render_width) if args.render_width else None,
                render_height=int(args.render_height) if args.render_height else None,
                camera_name=args.camera_name,
                deterministic=bool(args.deterministic),
                cpu=bool(args.cpu),
                no_jax_prealloc=bool(args.no_jax_prealloc),
                jax_mem_fraction=args.jax_mem_fraction,
            )
        except ModuleNotFoundError as e:
            console.print(f"[red]✗ 缺少依赖，无法回放 xax checkpoint: {e}[/red]")
            console.print("[yellow]建议使用包含 ksim/xax 的 Python（例如项目 .venv）运行 play.py[/yellow]")
            return 1
        except Exception as e:
            console.print(f"[red]✗ xax viewer 运行失败: {e}[/red]")
            msg = str(e)
            if "Failed to open display" in msg or "gladLoadGL" in msg:
                console.print("[yellow]提示: 离线渲染可尝试设置环境变量 `MUJOCO_GL=egl`（或 `MUJOCO_GL=osmesa`）[/yellow]")
            return 1

        console.print(f"[dim]xax exp_dir: {outputs.exp_dir}[/dim]")
        if save_video:
            if outputs.video_file is None:
                console.print("[red]✗ 未找到输出视频文件（可能渲染失败或提前退出）[/red]")
                return 1

            out_path = Path(args.video_path).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import shutil

                shutil.copy2(outputs.video_file, out_path)
            except Exception as e:
                console.print(f"[yellow]⚠ 复制视频失败: {e}[/yellow]")
                console.print(f"[yellow]视频仍保存在: {outputs.video_file}[/yellow]")
            else:
                console.print(f"[green]✓ 视频已保存: {out_path}[/green]")
        return 0

    import jax
    import numpy as np

    # MuJoCo may print optional warp-related warnings to stderr during import; keep playback logs clean.
    with contextlib.redirect_stderr(io.StringIO()):
        import mujoco

        from rl.envs import create_standing_env, create_velocity_tracking_env, create_walking_env
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

    try:
        import rl
        from rl.envs import robot_envs as _robot_envs

        console.print(f"[dim]rl: {rl.__file__}[/dim]")
        console.print(f"[dim]robot_envs: {_robot_envs.__file__}[/dim]")
    except Exception:
        pass

    ckpt_config = infer_network_config_from_checkpoint(args.checkpoint)
    args_hidden_dims = tuple(ckpt_config["hidden_dims"]) if ckpt_config["hidden_dims"] else (512, 512, 256)
    args_shared_backbone = bool(ckpt_config["shared_backbone"])

    xml_path = args.xml_path
    if args.use_local_mjcf:
        xml_path = "assets/xmls/scenes/flat_terrain.xml"
    elif xml_path is None:
        from rl.utils.robot_config import resolve_scene_path

        xml_path = str(resolve_scene_path(robot_name))

    cmd_x_range = args.cmd_x_range
    if cmd_x_range is None and "cmd_x_range" in env_config:
        cmd_x_range = env_config.get("cmd_x_range")
    cmd_y_range = args.cmd_y_range
    if cmd_y_range is None and "cmd_y_range" in env_config:
        cmd_y_range = env_config.get("cmd_y_range")
    cmd_yaw_range = args.cmd_yaw_range
    if cmd_yaw_range is None and "cmd_yaw_range" in env_config:
        cmd_yaw_range = env_config.get("cmd_yaw_range")

    env_max_steps = args.env_max_steps
    if env_max_steps is None and "max_steps" in env_config:
        env_max_steps = env_config.get("max_steps")

    env_kwargs = {}
    if cmd_x_range is not None:
        env_kwargs["cmd_x_range"] = tuple(cmd_x_range)
    if cmd_y_range is not None:
        env_kwargs["cmd_y_range"] = tuple(cmd_y_range)
    if cmd_yaw_range is not None:
        env_kwargs["cmd_yaw_range"] = tuple(cmd_yaw_range)
    if env_max_steps is not None:
        env_kwargs["max_steps"] = int(env_max_steps)

    if env_type in ("walking", "standing") and args.target_height is not None:
        env_kwargs["target_height"] = float(args.target_height)
    elif env_type in ("walking", "standing") and "target_height" in env_config:
        env_kwargs["target_height"] = float(env_config.get("target_height"))

    if env_type == "walking":
        env = create_walking_env(
            xml_path=xml_path, robot_name=robot_name, verbose=False, **env_kwargs
        )
    elif env_type == "standing":
        standing_kwargs = {}
        if "max_steps" in env_kwargs:
            standing_kwargs["max_steps"] = env_kwargs["max_steps"]
        if "target_height" in env_kwargs:
            standing_kwargs["target_height"] = env_kwargs["target_height"]
        env = create_standing_env(
            xml_path=xml_path, robot_name=robot_name, verbose=False, **standing_kwargs
        )
    else:
        env = create_velocity_tracking_env(
            xml_path=xml_path, robot_name=robot_name, verbose=False, **env_kwargs
        )

    # If user didn't override max steps, prefer env.max_steps to keep playback length aligned.
    if args.max_steps == 2000 and getattr(env, "max_steps", 2000) != 2000:
        args.max_steps = int(env.max_steps)

    network = ActorCriticNetwork(
        action_dim=env.action_size,
        shared_backbone=args_shared_backbone,
        hidden_dims=args_hidden_dims,
    )

    rng = jax.random.PRNGKey(args.seed)
    params, step = load_checkpoint(args.checkpoint, network, rng)

    console.print(
        f"[cyan]checkpoint step={step} | env={env_type} | robot={robot_name} | render={args.render} | "
        f"deterministic={args.deterministic}[/cyan]"
    )

    if args.no_jit:
        console.print(
            "[yellow]提示: --no-jit 会显著降低MJX环境播放速度（每步会触发大量小的JAX调度/同步）。"
            "想流畅播放请去掉 NO_JIT=1，或提高 RENDER 间隔（如 RENDER=10）。[/yellow]"
        )

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    record_interval = args.record_interval
    if record_interval is None:
        # When saving video, default to recording every control step to avoid time scaling surprises.
        record_interval = 1 if args.save_video else (args.render if args.render and args.render > 0 else 1)
    if args.save_video and record_interval <= 0:
        console.print("[yellow]record-interval<=0，自动改为1以便录制[/yellow]")
        record_interval = 1

    console.print(
        f"[dim]control_dt={getattr(env, 'control_dt', None)} dt={getattr(env, 'dt', None)} "
        f"frame_skip={getattr(env, 'frame_skip', None)} env.max_steps={getattr(env, 'max_steps', None)}[/dim]"
    )
    if hasattr(env, "cmd_x_range"):
        console.print(
            f"[dim]cmd ranges: x={getattr(env, 'cmd_x_range', None)} y={getattr(env, 'cmd_y_range', None)} "
            f"yaw={getattr(env, 'cmd_yaw_range', None)}[/dim]"
        )
    if hasattr(env, "target_height"):
        console.print(f"[dim]target_height={getattr(env, 'target_height', None)}[/dim]")

    viewer = None
    if args.render > 0:
        viewer = InteractiveViewer(mj_model, mj_data)
        console.print("[green]✓ 交互式查看器已启动[/green]")

    renderer = None
    video_writer = None
    if args.save_video:
        # Choose a video FPS that matches simulated time when recording every N control steps.
        # One environment step advances env.control_dt seconds.
        control_dt = float(getattr(env, "control_dt", 0.02))
        realtime_fps = max(1, int(round(1.0 / (control_dt * record_interval))))
        video_fps = int(args.video_fps) if args.video_fps is not None else realtime_fps
        speed_ratio = video_fps / float(realtime_fps) if realtime_fps > 0 else 1.0
        console.print(
            f"[dim]record_interval={record_interval} -> realtime_fps≈{realtime_fps} | "
            f"video_fps={video_fps} | speed≈{speed_ratio:.2f}x[/dim]"
        )

        renderer = MujocoRenderer(
            mj_model,
            width=args.render_width,
            height=args.render_height,
            camera_name=args.camera_name,
        )
        video_writer = create_video_writer(
            args.video_path,
            fps=video_fps,
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

    try:
        for ep in range(args.episodes):
            rng, reset_rng = jax.random.split(rng)
            env_state = env.reset(reset_rng)
            try:
                cmd0 = jax.device_get(env_state.info.get("command"))
                console.print(f"[dim]episode {ep+1}/{args.episodes} init command={cmd0}[/dim]")
            except Exception:
                pass

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
            min_base_z = float("inf")
            min_upright_z = float("inf")

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
                qpos_np, qvel_np, done_np = jax.device_get(
                    (
                        env_state.pipeline_state.qpos,
                        env_state.pipeline_state.qvel,
                        env_state.done,
                    )
                )
                done_host = bool(done_np)

                base_addr = getattr(env, "floating_base_qpos_addr", None)
                if base_addr is not None:
                    base_z = float(qpos_np[int(base_addr) + 2])
                    base_quat = qpos_np[int(base_addr) + 3: int(base_addr) + 7]
                else:
                    base_z = float(qpos_np[2])
                    base_quat = qpos_np[3:7]

                min_base_z = min(min_base_z, base_z)
                # upright_z: z-component of body z-axis in world frame
                # Apply the same fix_quat used in env termination/reward code.
                try:
                    qw, qx, qy, qz = [float(x) for x in base_quat]
                    fw, fx, fy, fz = 0.70710678, -0.70710678, 0.0, 0.0
                    rw = fw * qw - fx * qx - fy * qy - fz * qz
                    rx = fw * qx + fx * qw + fy * qz - fz * qy
                    ry = fw * qy - fx * qz + fy * qw + fz * qx
                    upright_z = 1.0 - 2.0 * (rx * rx + ry * ry)
                    min_upright_z = min(min_upright_z, float(upright_z))
                except Exception:
                    pass

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

                if done_host:
                    console.print(
                        f"[yellow]done=True at step={step_idx+1} | base_z={base_z:.3f} | upright_z≈{min_upright_z:.3f}[/yellow]"
                    )
                    break

            console.print(
                f"[dim]episode {ep+1}/{args.episodes} finished | "
                f"min_base_z={min_base_z:.3f} | min_upright_z≈{min_upright_z:.3f}[/dim]"
            )
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
