"""ksim/xax ONNX 推理演示脚本。

目标：
- 加载 `scripts/export.py` 导出的 `policy_mean_xax.onnx`
- 使用 ksim 的 GaodaJiyuanTask 构建观测（与训练一致），用 onnxruntime 生成动作
- 在 ksim 物理引擎中 rollout，并保存视频（用于演示/验收）

注意：
- 由于 onnxruntime 在 Python 侧执行，每步会发生 host<->device 同步，速度会明显慢于纯 JAX policy。
- headless/offscreen 渲染常需要：`MUJOCO_GL=egl`（或 `MUJOCO_GL=osmesa`）。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from checkpoint_compat import CheckpointFormat, resolve_checkpoint

console = Console()


def _setup_jax_runtime() -> None:
    project_root = Path(__file__).resolve().parent.parent

    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    # 仅推理/录制时不强制占满显存
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")


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

    if "CUDAExecutionProvider" in available_set:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_io_names(session, input_name: str | None, output_name: str | None) -> tuple[str, str]:
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

    return pick_name(input_name, inputs, "observation"), pick_name(output_name, outputs, "action")


def _load_cfg_from_yaml(path: Path) -> Any:
    import yaml
    from omegaconf import OmegaConf

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 配置必须是 dict: {path}")
    return OmegaConf.create(data)


def _load_cfg_from_xax_ckpt(path: Path) -> Any:
    from xax.task.mixins.checkpointing import load_ckpt

    return load_ckpt(path, part="config")


def main() -> int:
    _setup_jax_runtime()

    parser = argparse.ArgumentParser(description="ksim/xax ONNX 推理回放（生成演示视频）")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径（policy_mean_xax.onnx）")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--checkpoint", type=str, default=None, help="xax checkpoint 路径（用于读取 config）")
    src_group.add_argument("--config", type=str, default=None, help="ksim YAML 配置（用于读取 config）")

    parser.add_argument("--task-class-name", type=str, default="GaodaJiyuanTask", help="task 类名（默认 GaodaJiyuanTask）")

    parser.add_argument("--num-envs", type=int, default=1, help="推理使用的环境数（强烈建议 1；避免显存/渲染问题）")
    parser.add_argument("--num-steps", type=int, default=None, help="rollout 步数（ctrl step）；默认使用 task.render_length_frames()")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（默认使用 task.prng_key()，与 ksim viewer 保持一致）",
    )

    parser.add_argument("--save-video", action="store_true", help="保存视频（默认不开启）")
    parser.add_argument("--video-path", type=str, default="plays/xax_onnx.mp4", help="输出视频路径")
    parser.add_argument("--target-fps", type=int, default=None, help="目标视频 FPS（默认由 task.render_trajectory_video 决定）")

    parser.add_argument("--render-width", type=int, default=1280, help="渲染宽度")
    parser.add_argument("--render-height", type=int, default=720, help="渲染高度")
    parser.add_argument("--camera-name", type=str, default=None, help="相机名称（覆盖 config.render_camera_name）")

    parser.add_argument("--input-name", type=str, default=None, help="ONNX 输入名（默认 observation）")
    parser.add_argument("--output-name", type=str, default=None, help="ONNX 输出名（默认 action）")
    parser.add_argument("--ort-provider", type=str, default="auto", help="onnxruntime provider: auto/cpu/cuda")

    parser.add_argument(
        "--compare-jax",
        action="store_true",
        help="对比 JAX actor(mean) 与 ONNX 输出（用于验证导出/推理是否一致）",
    )
    parser.add_argument("--compare-steps", type=int, default=3, help="对比输出前 N 次 sample_action")
    parser.add_argument("--compare-tol", type=float, default=1e-4, help="对比阈值（max_abs 超过则提示）")

    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（JAX_PLATFORMS=cpu）")
    parser.add_argument("--no-jax-prealloc", action="store_true", help="禁用 JAX 预分配显存（XLA_PYTHON_CLIENT_PREALLOCATE=false）")
    parser.add_argument("--jax-mem-fraction", type=float, default=None, help="设置 JAX 显存占用比例（XLA_PYTHON_CLIENT_MEM_FRACTION）")

    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        console.print(f"[red]✗ ONNX 模型不存在: {model_path}[/red]")
        return 1

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
        console.print("[yellow]请先执行: uv sync --extra onnx --extra ksim[/yellow]")
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

    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "logs" / "ksim_onnx" / f"view_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from checkpoint_compat import get_default_xax_task_cls
    except Exception as e:
        console.print(f"[red]✗ 无法加载 task 定义（scripts/train_ksim.py）: {e}[/red]")
        return 1

    task_cls = get_default_xax_task_cls(args.task_class_name)

    try:
        if args.checkpoint:
            resolved = resolve_checkpoint(args.checkpoint)
            if resolved.format != CheckpointFormat.XAX_TAR:
                console.print(f"[red]✗ 该 checkpoint 不是 xax/ksim(tar.gz) 格式: {resolved.path}[/red]")
                return 1
            cfg = _load_cfg_from_xax_ckpt(resolved.path)
        else:
            cfg = _load_cfg_from_yaml(Path(args.config).expanduser().resolve())
    except Exception as e:
        console.print(f"[red]✗ 读取 config 失败: {e}[/red]")
        return 1

    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {
        "disable_multiprocessing": True,
        "exp_dir": str(exp_dir),
        "compile": {"cache_dir": str(cache_dir)},
        "num_envs": int(max(1, args.num_envs)),
        # ksim RLTask 要求 num_envs % batch_size == 0；推理默认用单 batch（batch_size=num_envs）。
        "batch_size": int(max(1, args.num_envs)),
    }
    if args.camera_name:
        overrides["render_camera_name"] = args.camera_name
    if args.render_width:
        overrides["render_width"] = int(args.render_width)
    if args.render_height:
        overrides["render_height"] = int(args.render_height)

    # 默认不加载 checkpoint 的模型权重（ONNX 推理不需要）；仅在 compare-jax 时加载以便做一致性对比。
    if args.checkpoint and args.compare_jax:
        overrides["load_from_ckpt_path"] = str(resolved.path)

    try:
        config = task_cls.get_config(cfg, overrides, use_cli=False)
    except Exception as e:
        console.print(f"[red]✗ 构建 task config 失败: {e}[/red]")
        return 1

    console.print(
        Panel.fit(
            "[bold green]ksim ONNX 推理回放[/bold green]\n"
            f"[dim]model={model_path} | providers={providers} | input={input_name} -> output={output_name}[/dim]\n"
            f"[dim]exp_dir={exp_dir}[/dim]",
            border_style="green",
        )
    )

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import equinox as eqx
        import ksim
        import xax
        from ksim.task.rl import InitParams as KInitParams, get_default_viewer
    except ModuleNotFoundError as e:
        console.print(f"[red]✗ 缺少依赖 ksim/xax/jax: {e}[/red]")
        console.print("[yellow]请先执行: uv sync --extra ksim --extra onnx[/yellow]")
        return 1

    task = task_cls(config)
    mj_model = task.get_mujoco_model()

    # 绑定 ONNX 推理到 task.sample_action（必须在 jax.disable_jit() 下运行，否则会被 JIT trace）
    compare_remaining = int(max(0, args.compare_steps)) if args.compare_jax else 0
    compare_tol = float(args.compare_tol)

    def onnx_sample_action(
        *,
        model,
        model_carry,
        physics_model,
        physics_state,
        observations,
        commands,
        rng,
        argmax: bool,
    ):
        obs_n = task._build_obs(observations, commands["cmd"])  # noqa: SLF001
        obs_host = np.asarray(jax.device_get(obs_n), dtype=np.float32)
        if obs_host.ndim == 0:
            raise ValueError(f"观测维度错误：期望至少 1D(obs_dim)，实际 shape={obs_host.shape}")
        if obs_host.ndim == 1:
            obs_dim = int(obs_host.shape[0])
            lead_shape = ()
            obs_flat = obs_host.reshape((1, obs_dim))
        else:
            obs_dim = int(obs_host.shape[-1])
            lead_shape = tuple(int(x) for x in obs_host.shape[:-1])
            obs_flat = obs_host.reshape((-1, obs_dim))

        act_flat = session.run([output_name], {input_name: obs_flat})[0]
        act_flat = np.asarray(act_flat, dtype=np.float32)
        if act_flat.ndim == 1:
            act_flat = act_flat[None, :]
        if act_flat.shape[0] != obs_flat.shape[0]:
            raise ValueError(
                "ONNX 输出 batch 维度不匹配："
                f"obs_batch={obs_flat.shape[0]} vs act_batch={act_flat.shape[0]}"
            )
        act_dim = int(act_flat.shape[-1])
        act_host = act_flat.reshape(lead_shape + (act_dim,))
        action = jnp.asarray(act_host)

        nonlocal compare_remaining
        if compare_remaining > 0:
            try:
                dist = model.actor(obs_n)
                jax_action = dist.mode()
                diff = np.asarray(jax.device_get(action - jax_action), dtype=np.float32)
                max_abs = float(np.max(np.abs(diff)))
                mean_abs = float(np.mean(np.abs(diff)))
                console.print(f"[dim]compare-jax: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}[/dim]")
                if max_abs > compare_tol:
                    console.print(
                        f"[yellow]⚠ compare-jax 超过阈值: max_abs={max_abs:.3e} > tol={compare_tol:.3e}[/yellow]"
                    )
            except Exception as e:
                console.print(f"[yellow]⚠ compare-jax 失败: {e}[/yellow]")
            compare_remaining -= 1

        return ksim.Action(action=action, carry=None)

    task.sample_action = onnx_sample_action  # type: ignore[assignment]

    video_path = Path(args.video_path).expanduser().resolve()
    if args.save_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    with task, jax.disable_jit():
        rng = task.prng_key() if args.seed is None else jax.random.PRNGKey(int(args.seed))
        task.set_loggers()

        mj_model = task.set_mujoco_model_opts(mj_model)
        metadata = task.get_mujoco_model_metadata(mj_model)
        task.update_mj_model(mj_model, metadata)

        randomizers = xax.freeze_dict(task.get_physics_randomizers(mj_model))

        rng, model_rng = jax.random.split(rng)
        init_params = KInitParams(key=model_rng, physics_model=mj_model)
        models, _ = task.load_initial_state(init_params, load_optimizer=False)

        model_arrs, model_statics = (
            tuple(ms)
            for ms in zip(
                *(eqx.partition(model, task.model_partition_fn) for model in models),
                strict=True,
            )
        )

        constants = task._get_constants(  # noqa: SLF001
            metadata=metadata,
            physics_model=mj_model,
            model_statics=model_statics,
            argmax_action=True,
        )

        rng, env_rng = jax.random.split(rng)
        env_states = task._get_env_state(  # noqa: SLF001
            rng=env_rng,
            rollout_constants=constants,
            mj_model=mj_model,
            physics_model=mj_model,
            policy_model=models[0],
            randomizers=randomizers,
        )

        rng, shared_rng = jax.random.split(rng)
        shared_state = task._get_shared_state(  # noqa: SLF001
            rng=shared_rng,
            mj_model=mj_model,
            physics_model=mj_model,
            model_arrs=model_arrs,
        )

        if args.num_steps is not None:
            steps = int(args.num_steps)
        else:
            steps = int(task.render_length_frames())

        console.print(f"[dim]rollout steps={steps}[/dim]")

        transitions = []
        for _ in range(steps):
            new_commands = task.get_viewer_commands(
                commands=constants.commands,
                prev_command_inputs=env_states.commands,
            )
            env_states = replace(env_states, commands=new_commands)

            transition, env_states = task.step_engine(
                constants=constants,
                env_states=env_states,
                shared_state=shared_state,
            )
            transitions.append(transition)

            done = getattr(transition, "done", None)
            if done is not None:
                try:
                    if bool(jax.device_get(done).all()):
                        break
                except Exception:
                    pass

        if not transitions:
            console.print("[red]✗ rollout 为空（未产生任何 transition）[/red]")
            return 1

        trajectory = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)
        rng, traj_rng = jax.random.split(rng)
        trajectory = task.postprocess_trajectory(  # noqa: SLF001
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=traj_rng,
        )

        markers = task.get_markers(
            commands=constants.commands,
            observations=constants.observations,
            rewards=constants.rewards,
        )

        if not args.save_video:
            console.print("[green]✓ rollout 完成（未保存视频；添加 --save-video 输出 mp4）[/green]")
            return 0

        max_w = int(getattr(mj_model.vis.global_, "offwidth", 0))
        max_h = int(getattr(mj_model.vis.global_, "offheight", 0))
        width = int(args.render_width)
        height = int(args.render_height)
        if max_w > 0:
            width = min(width, max_w)
        if max_h > 0:
            height = min(height, max_h)

        viewer = get_default_viewer(mj_model=mj_model, config=task.config, width=width, height=height)
        try:
            frames, fps = task.render_trajectory_video(
                trajectory=trajectory,
                markers=markers,
                viewer=viewer,
                target_fps=int(args.target_fps) if args.target_fps is not None else None,
            )
        finally:
            try:
                viewer.close()
            except Exception:
                pass

        import mediapy as media

        media.write_video(str(video_path), frames, fps=fps)
        console.print(f"[green]✓ 视频已保存: {video_path} (fps={fps})[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
