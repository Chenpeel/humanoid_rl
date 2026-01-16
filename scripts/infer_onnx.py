"""ONNX 推理/回放脚本。

用途：
- 使用 onnxruntime 加载导出的策略 ONNX（默认输入 observation -> 输出 action）
- 在 JRL 环境中 rollout，并可选交互式可视化/离线视频录制

注意：
- ONNX 推理在 Python 侧执行，无法被 JIT；每步会进行 host<->device 同步，速度会慢于纯 JAX policy。
- headless/offscreen 渲染可能需要：MUJOCO_GL=egl（或 MUJOCO_GL=osmesa）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _select_ort_providers(requested: str) -> list[str]:
    import onnxruntime as ort

    requested = (requested or "auto").lower().strip()
    available = list(ort.get_available_providers())
    available_set = set(available)

    if requested in ("cpu", "cpuexecutionprovider"):
        return ["CPUExecutionProvider"]

    if requested in ("cuda", "cudaexecutionprovider"):
        providers = []
        if "CUDAExecutionProvider" in available_set:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return [p for p in providers if p in available_set]

    # auto
    if "CUDAExecutionProvider" in available_set:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_io_names(
    session,
    input_name: str | None,
    output_name: str | None,
) -> tuple[str, str]:
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]

    def pick_name(preferred: str | None, candidates: list[str], fallback: str) -> str:
        if preferred:
            if preferred not in candidates:
                raise ValueError(f"ONNX {fallback} 不存在: {preferred}（可用: {candidates}）")
            return preferred
        if fallback in candidates:
            return fallback
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"ONNX {fallback} 名称不唯一，需显式指定（可用: {candidates}）")

    in_name = pick_name(input_name, inputs, "observation")
    out_name = pick_name(output_name, outputs, "action")
    return in_name, out_name


def _prepare_ort_input(session, input_name: str, obs_np):
    import numpy as np

    input_meta = None
    for meta in session.get_inputs():
        if meta.name == input_name:
            input_meta = meta
            break
    if input_meta is None:
        raise ValueError(f"找不到输入: {input_name}")

    x = np.asarray(obs_np, dtype=np.float32)
    shape = input_meta.shape
    rank = len(shape) if shape is not None else x.ndim

    if rank == 1:
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise ValueError(f"ONNX 输入期望 1D，但 obs 形状为 {x.shape}")
        return x

    if rank == 2:
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"当前仅支持单环境推理；obs 形状为 {x.shape}")
        return x

    raise ValueError(f"不支持的 ONNX 输入 rank={rank} shape={shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX 策略推理/回放（JRL 环境）")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练/运行的YAML配置文件（用于复现 env_config，如命令范围/target_height/max_steps）",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="MuJoCo XML路径（默认根据robot_name解析）",
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
        help="环境类型: velocity / walking / standing",
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
        help="覆盖target_height（walking/standing适用）",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=None,
        help="覆盖环境episode最大步数max_steps（walking/velocity适用）",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="机器人名称 (可选: gaoda_jiyuan, unitree_h1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=10, help="episode 数")
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
        help="每帧sleep秒数（用于限速；默认0=尽快）",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按环境control_dt实时播放（会sleep限速；优先级高于--viewer-sleep）",
    )
    parser.add_argument(
        "--status-every",
        type=float,
        default=2.0,
        help="每隔N秒打印一次播放进度（0=不打印）",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="保存视频（离线渲染；可与viewer同时开启，但会更慢）",
    )
    parser.add_argument("--video-path", type=str, default="infer_video.mp4", help="视频保存路径")
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="视频FPS（默认自动匹配control_dt/record_interval，避免视频加速/减速）",
    )
    parser.add_argument("--render-width", type=int, default=1280, help="渲染宽度")
    parser.add_argument("--render-height", type=int, default=720, help="渲染高度")
    parser.add_argument("--camera-name", type=str, default="track", help="MuJoCo相机名称")
    parser.add_argument(
        "--record-interval",
        type=int,
        default=None,
        help="录制间隔（每N步录一帧；默认=render或1）",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=10.0,
        help="动作裁剪范围 [-clip, clip]（默认10）",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default=None,
        help="ONNX 输入名（默认优先 observation，否则自动取单输入）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="ONNX 输出名（默认优先 action，否则自动取单输出）",
    )
    parser.add_argument(
        "--ort-provider",
        type=str,
        default="auto",
        help="onnxruntime provider: auto/cpu/cuda（默认auto）",
    )
    parser.add_argument(
        "--no-jax-prealloc",
        action="store_true",
        help="禁用JAX预分配显存（需在导入jax前设置）",
    )
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="设置JAX显存占用比例(0~1)，例如0.5（需在导入jax前设置）",
    )
    parser.add_argument("--cpu", action="store_true", help="使用CPU运行MJX环境（更慢）")

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        console.print(f"[red]✗ ONNX 模型不存在: {model_path}[/red]")
        return 1

    config_data = {}
    if args.config:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError(f"无法导入yaml以读取配置文件: {e}. 请安装PyYAML或移除 --config") from e
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    env_type = args.env_type or config_data.get("env_type") or "walking"
    robot_name = args.robot_name or config_data.get("robot_name") or "gaoda_jiyuan"
    env_config = config_data.get("env_config") or {}

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

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as e:
        console.print(f"[red]✗ 缺少依赖 onnxruntime: {e}[/red]")
        console.print("[yellow]安装建议: pip install onnxruntime  （或 onnxruntime-gpu）[/yellow]")
        return 1

    providers = _select_ort_providers(args.ort_provider)
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        console.print(f"[red]✗ 创建 onnxruntime Session 失败: {e}[/red]")
        return 1

    try:
        input_name, output_name = _resolve_io_names(session, args.input_name, args.output_name)
    except Exception as e:
        console.print(f"[red]✗ 解析 ONNX 输入/输出名失败: {e}[/red]")
        return 1

    console.print(
        Panel.fit(
            "[bold green]ONNX 策略推理[/bold green]\n"
            f"[dim]model={model_path} | providers={providers} | input={input_name} -> output={output_name}[/dim]",
            border_style="green",
        )
    )

    import numpy as np
    import jax
    import jax.numpy as jnp

    with contextlib.redirect_stderr(io.StringIO()):
        import mujoco

        from rl.envs import create_standing_env, create_velocity_tracking_env, create_walking_env
        from rl.utils import (
            InteractiveViewer,
            MujocoRenderer,
            create_video_writer,
            save_frame_to_video,
        )

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

    env_kwargs: dict[str, object] = {}
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
        env = create_walking_env(xml_path=xml_path, robot_name=robot_name, verbose=False, **env_kwargs)
    elif env_type == "standing":
        standing_kwargs = {}
        if "max_steps" in env_kwargs:
            standing_kwargs["max_steps"] = env_kwargs["max_steps"]
        if "target_height" in env_kwargs:
            standing_kwargs["target_height"] = env_kwargs["target_height"]
        env = create_standing_env(xml_path=xml_path, robot_name=robot_name, verbose=False, **standing_kwargs)
    else:
        env = create_velocity_tracking_env(xml_path=xml_path, robot_name=robot_name, verbose=False, **env_kwargs)

    if args.max_steps == 2000 and getattr(env, "max_steps", 2000) != 2000:
        args.max_steps = int(env.max_steps)

    console.print(
        f"[cyan]env={env_type} | robot={robot_name} | obs={getattr(env, 'observation_size', None)} | "
        f"act={getattr(env, 'action_size', None)} | render={args.render}[/cyan]"
    )
    console.print(
        f"[dim]control_dt={getattr(env, 'control_dt', None)} dt={getattr(env, 'dt', None)} "
        f"frame_skip={getattr(env, 'frame_skip', None)} env.max_steps={getattr(env, 'max_steps', None)}[/dim]"
    )

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    record_interval = args.record_interval
    if record_interval is None:
        record_interval = 1 if args.save_video else (args.render if args.render and args.render > 0 else 1)
    if args.save_video and record_interval <= 0:
        console.print("[yellow]record-interval<=0，自动改为1以便录制[/yellow]")
        record_interval = 1

    viewer = None
    if args.render > 0:
        viewer = InteractiveViewer(mj_model, mj_data)
        console.print("[green]✓ 交互式查看器已启动[/green]")

    renderer = None
    video_writer = None
    if args.save_video:
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

    step_fn = jax.jit(lambda s, a: env.step(s, a))

    def policy_action_np(obs_np):
        ort_in = _prepare_ort_input(session, input_name, obs_np)
        out = session.run([output_name], {input_name: ort_in})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out[0]
        if out.ndim != 1:
            raise ValueError(f"ONNX 输出形状不支持: {out.shape}")
        return out

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    rng = jax.random.PRNGKey(args.seed)
    compiled = False

    try:
        for ep in range(int(args.episodes)):
            rng, reset_rng = jax.random.split(rng)
            env_state = env.reset(reset_rng)
            obs_host = np.asarray(jax.device_get(env_state.obs), dtype=np.float32)

            wall_start = time.perf_counter()
            steps_rendered = 0
            last_status_t = wall_start
            last_status_step = 0
            ep_return = 0.0
            ep_len = 0

            for step_idx in range(int(args.max_steps)):
                if viewer and not viewer.is_alive():
                    return 0

                action_np = policy_action_np(obs_host)
                action = jnp.asarray(action_np)
                if args.action_clip is not None and float(args.action_clip) > 0:
                    clip = float(args.action_clip)
                    action = jnp.clip(action, -clip, clip)

                t0 = time.perf_counter()
                env_state = step_fn(env_state, action)
                if not compiled:
                    try:
                        env_state.reward.block_until_ready()
                    except Exception:
                        jax.block_until_ready(env_state.reward)
                    console.print(f"[dim]JIT 编译完成，用时 {time.perf_counter() - t0:.1f}s[/dim]")
                    compiled = True

                need_viewer = viewer is not None and args.render > 0 and (step_idx + 1) % int(args.render) == 0
                need_record = (
                    renderer is not None
                    and video_writer is not None
                    and (step_idx + 1) % int(record_interval) == 0
                )

                if need_viewer or need_record:
                    obs_host, reward_host, done_host, qpos_np, qvel_np = jax.device_get(
                        (
                            env_state.obs,
                            env_state.reward,
                            env_state.done,
                            env_state.pipeline_state.qpos,
                            env_state.pipeline_state.qvel,
                        )
                    )
                    obs_host = np.asarray(obs_host, dtype=np.float32)
                    ep_return += float(reward_host)
                    ep_len += 1

                    mj_data.qpos[:] = qpos_np
                    mj_data.qvel[:] = qvel_np
                    mj_data.ctrl[:] = 0.0
                    mujoco.mj_forward(mj_model, mj_data)

                    if need_viewer:
                        viewer.update(mj_data)
                        steps_rendered += 1

                        if args.realtime:
                            target_t = steps_rendered * float(env.control_dt) * int(args.render)
                            elapsed = time.perf_counter() - wall_start
                            sleep_s = target_t - elapsed
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                        elif args.viewer_sleep > 0:
                            time.sleep(float(args.viewer_sleep))

                    if need_record:
                        frame = renderer.render(mj_data)
                        save_frame_to_video(video_writer, frame)

                    if args.status_every and args.status_every > 0:
                        now = time.perf_counter()
                        if now - last_status_t >= float(args.status_every):
                            sps = (step_idx + 1 - last_status_step) / (now - last_status_t)
                            console.print(
                                f"[dim]ep={ep+1}/{args.episodes} step={step_idx+1}/{args.max_steps} | "
                                f"sim={sps:.1f} steps/s | rendered={steps_rendered}[/dim]"
                            )
                            last_status_t = now
                            last_status_step = step_idx + 1

                    if bool(done_host):
                        break
                else:
                    obs_host, reward_host, done_host = jax.device_get(
                        (env_state.obs, env_state.reward, env_state.done)
                    )
                    obs_host = np.asarray(obs_host, dtype=np.float32)
                    ep_return += float(reward_host)
                    ep_len += 1

                    if args.status_every and args.status_every > 0:
                        now = time.perf_counter()
                        if now - last_status_t >= float(args.status_every):
                            sps = (step_idx + 1 - last_status_step) / (now - last_status_t)
                            console.print(
                                f"[dim]ep={ep+1}/{args.episodes} step={step_idx+1}/{args.max_steps} | "
                                f"sim={sps:.1f} steps/s | rendered={steps_rendered}[/dim]"
                            )
                            last_status_t = now
                            last_status_step = step_idx + 1

                    if bool(done_host):
                        break

            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)
            console.print(f"[dim]episode {ep+1}/{args.episodes} return={ep_return:.2f} len={ep_len}[/dim]")

    except Exception as e:
        msg = str(e)
        console.print(f"[red]✗ 推理失败: {e}[/red]")
        if "Failed to open display" in msg or "gladLoadGL" in msg:
            console.print("[yellow]提示: headless 渲染可尝试设置 `MUJOCO_GL=egl`（或 `MUJOCO_GL=osmesa`）[/yellow]")
        return 1
    finally:
        if viewer:
            viewer.close()
        if renderer:
            renderer.close()
        if video_writer:
            video_writer.release()

    if episode_returns:
        mean_r = float(np.mean(np.asarray(episode_returns, dtype=np.float32)))
        std_r = float(np.std(np.asarray(episode_returns, dtype=np.float32)))
        mean_len = float(np.mean(np.asarray(episode_lengths, dtype=np.float32)))

        table = Table(title="ONNX 推理结果")
        table.add_column("指标")
        table.add_column("值", justify="right")
        table.add_row("Episodes", str(len(episode_returns)))
        table.add_row("平均回报", f"{mean_r:.2f}")
        table.add_row("回报标准差", f"{std_r:.2f}")
        table.add_row("平均长度", f"{mean_len:.0f}")
        console.print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

