"""Checkpoint 兼容层：自动识别并解析 JRL(Flax) / xax(ksim) 格式。

目的：
- 统一 scripts/eval.py、scripts/play.py、scripts/export.py 的 checkpoint 输入体验
- 支持输入文件或目录（run 目录 / checkpoints 目录 / best_model 目录等）

当前支持的格式：
- JRL(Flax): flax.serialization.msgpack_restore / from_bytes 保存的二进制文件
- xax(ksim): xax CheckpointingMixin 保存的 tar.gz（通常扩展名仍为 .bin）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location


class CheckpointFormat(str, Enum):
    JRL_FLAX = "jrl_flax"
    XAX_TAR = "xax_tar"


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    format: CheckpointFormat


@dataclass(frozen=True)
class XaxViewerOutputs:
    exp_dir: Path
    render_dir: Path | None
    video_file: Path | None


def _run_xax_offscreen_rollout_and_render(
    *,
    task,
    mj_model,
    num_steps: int | None,
    save_path: Path,
    deterministic: bool,
    target_fps: int | None,
    render_width: int | None,
    render_height: int | None,
) -> None:
    """不依赖 QtViewer(shared memory) 的离线回放/录制路径。

    直接复用 ksim 的 rollout/step_engine 生成 trajectory，再用 DefaultMujocoViewer 离线渲染。
    主要用于在容器/受限环境（无法创建 /dev/shm）中仍可录制视频。
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import xax
    from dataclasses import replace

    from ksim.task.rl import InitParams as KInitParams, get_default_viewer

    with task, jax.disable_jit():
        rng = task.prng_key()
        task.set_loggers()

        mj_model = task.set_mujoco_model_opts(mj_model)
        metadata = task.get_mujoco_model_metadata(mj_model)
        task.update_mj_model(mj_model, metadata)

        randomizers = xax.freeze_dict(task.get_physics_randomizers(mj_model))

        rng, model_rng = jax.random.split(rng)
        params = KInitParams(key=model_rng, physics_model=mj_model)
        models, _ = task.load_initial_state(params, load_optimizer=False)

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
            argmax_action=deterministic,
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

        transitions = []
        if num_steps is not None:
            steps = int(num_steps)
        else:
            try:
                steps = int(task.render_length_frames())
            except Exception:
                steps = 250
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

            # 单环境时可以提前退出；多环境时保持固定长度。
            done = getattr(transition, "done", None)
            if done is not None:
                try:
                    if bool(jax.device_get(done).all()):
                        break
                except Exception:
                    pass

        if not transitions:
            return

        trajectory = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)
        rng, traj_rng = jax.random.split(rng)
        trajectory = task.postprocess_trajectory(  # noqa: SLF001
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=traj_rng,
        )

        # markers 默认不启用；这里仍按 ksim 逻辑构造，便于后续开启 render_markers。
        markers = task.get_markers(
            commands=constants.commands,
            observations=constants.observations,
            rewards=constants.rewards,
        )

        # Default viewer 受 offwidth/offheight 限制；必要时自动裁剪。
        max_w = int(getattr(mj_model.vis.global_, "offwidth", 0))
        max_h = int(getattr(mj_model.vis.global_, "offheight", 0))
        width = int(render_width or task.config.render_width)
        height = int(render_height or task.config.render_height)
        if max_w > 0:
            width = min(width, max_w)
        if max_h > 0:
            height = min(height, max_h)

        viewer = get_default_viewer(mj_model=mj_model, config=task.config, width=width, height=height)
        try:
            frames, _ = task.render_trajectory_video(
                trajectory=trajectory,
                markers=markers,
                viewer=viewer,
                target_fps=target_fps,
            )
        finally:
            try:
                viewer.close()
            except Exception:
                pass

        task._save_viewer_video(list(frames), save_path)  # noqa: SLF001


def _is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _infer_format_from_file(path: Path) -> CheckpointFormat:
    # xax checkpoint 是 tar.gz，但常用 .bin 扩展名；用 gzip magic 最稳妥。
    return CheckpointFormat.XAX_TAR if _is_gzip(path) else CheckpointFormat.JRL_FLAX


def _iter_files(d: Path) -> Iterable[Path]:
    for p in d.iterdir():
        if p.is_file():
            yield p


def _pick_latest_file(files: Iterable[Path]) -> Path:
    candidates = list(files)
    if not candidates:
        raise FileNotFoundError("检查点目录为空")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(checkpoint: str | Path) -> ResolvedCheckpoint:
    """解析 checkpoint 输入（文件/目录），并返回最终 checkpoint 文件与格式。"""
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")

    if path.is_file():
        real_path = path.resolve()
        return ResolvedCheckpoint(path=real_path, format=_infer_format_from_file(real_path))

    # -------------------- 目录解析 --------------------
    # 1) xax run 目录：<run_dir>/checkpoints/ckpt.bin 或 ckpt.*.bin
    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        ckpt_link = checkpoints_dir / "ckpt.bin"
        if ckpt_link.exists():
            return ResolvedCheckpoint(path=ckpt_link.resolve(), format=CheckpointFormat.XAX_TAR)
        xax_bins = sorted(checkpoints_dir.glob("ckpt.*.bin"), key=lambda p: p.stat().st_mtime)
        if xax_bins:
            return ResolvedCheckpoint(path=xax_bins[-1].resolve(), format=CheckpointFormat.XAX_TAR)

        # JRL 训练日志有时会放在 checkpoints/best_model/best_model
        jrl_best = checkpoints_dir / "best_model" / "best_model"
        if jrl_best.exists():
            return ResolvedCheckpoint(path=jrl_best.resolve(), format=CheckpointFormat.JRL_FLAX)

    # 2) xax checkpoints 目录：<checkpoints_dir>/ckpt.bin
    ckpt_link = path / "ckpt.bin"
    if ckpt_link.exists():
        return ResolvedCheckpoint(path=ckpt_link.resolve(), format=CheckpointFormat.XAX_TAR)
    xax_bins = sorted(path.glob("ckpt.*.bin"), key=lambda p: p.stat().st_mtime)
    if xax_bins:
        return ResolvedCheckpoint(path=xax_bins[-1].resolve(), format=CheckpointFormat.XAX_TAR)

    # 3) JRL best_model 目录：<best_model_dir>/best_model
    jrl_best = path / "best_model"
    if jrl_best.is_file():
        return ResolvedCheckpoint(path=jrl_best.resolve(), format=CheckpointFormat.JRL_FLAX)

    # 4) 兜底：目录里挑最新文件，并按文件头推断格式
    latest = _pick_latest_file(_iter_files(path)).resolve()
    return ResolvedCheckpoint(path=latest, format=_infer_format_from_file(latest))


@lru_cache(maxsize=1)
def _load_train_ksim_module():
    # scripts/ 不是 Python package，使用 file-location 方式加载。
    train_ksim_path = Path(__file__).with_name("train_ksim.py").resolve()
    spec = spec_from_file_location("_jrl_train_ksim", train_ksim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {train_ksim_path}")
    mod = module_from_spec(spec)
    # dataclasses/typing 等机制会依赖 sys.modules[__name__]；手工加载时需提前注册。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_default_xax_task_cls(task_class_name: str = "GaodaJiyuanTask"):
    mod = _load_train_ksim_module()
    return getattr(mod, task_class_name)


def _pick_latest_render_dir(exp_dir: Path) -> Path | None:
    renders_dir = exp_dir / "renders"
    if not renders_dir.is_dir():
        return None
    render_dirs = [p for p in renders_dir.iterdir() if p.is_dir() and p.name.startswith("render_")]
    if not render_dirs:
        return None
    return max(render_dirs, key=lambda p: p.stat().st_mtime)


def run_xax_viewer_from_checkpoint(
    ckpt_path: str | Path,
    *,
    num_steps: int | None,
    save_renders: bool,
    save_video: bool,
    exp_dir: str | Path | None = None,
    render_width: int | None = None,
    render_height: int | None = None,
    camera_name: str | int | None = None,
    deterministic: bool = True,
    cpu: bool = False,
    no_jax_prealloc: bool = False,
    jax_mem_fraction: float | None = None,
    task_class_name: str = "GaodaJiyuanTask",
) -> XaxViewerOutputs:
    """用 xax(ksim) checkpoint 回放/录制（默认使用 scripts/train_ksim.py 的 task）。"""
    ckpt_path = Path(ckpt_path).expanduser().resolve()

    if no_jax_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if jax_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(jax_mem_fraction)
    if cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # 回放/录制时一般不需要启动 TensorBoard 服务（写 events 仍然保留）
    os.environ.setdefault("DISABLE_TENSORBOARD", "1")

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 若不指定 exp_dir，则统一写到 logs/ksim_compat 下，避免污染训练目录结构。
    if exp_dir is None:
        exp_dir = project_root / "logs" / "ksim_compat" / f"view_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(exp_dir).expanduser().resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1) 从 checkpoint 里读 config（OmegaConf DictConfig）
    from xax.task.mixins.checkpointing import load_ckpt

    cfg = load_ckpt(ckpt_path, part="config")

    # 2) 构建 task config（合并 overrides，避免 multiprocessing + 修正 cache_dir）
    mod = _load_train_ksim_module()
    task_cls = getattr(mod, task_class_name)
    overrides: dict[str, Any] = {
        "disable_multiprocessing": True,
        "load_from_ckpt_path": str(ckpt_path),
        "exp_dir": str(exp_dir),
        "viewer_save_video": bool(save_video),
        "compile": {"cache_dir": str(cache_dir)},
    }
    if render_width is not None:
        overrides["render_width"] = int(render_width)
    if render_height is not None:
        overrides["render_height"] = int(render_height)
    if camera_name is not None:
        overrides["render_camera_name"] = camera_name

    config = task_cls.get_config(cfg, overrides, use_cli=False)

    # 3) 运行 viewer（save_renders=True -> offscreen 模式，会输出 renders/render_*/render.mp4）
    task = task_cls(config)
    if save_renders:
        save_path = task.exp_dir / "renders" / f"render_{time.monotonic()}"
        save_path.mkdir(parents=True, exist_ok=True)
        _run_xax_offscreen_rollout_and_render(
            task=task,
            mj_model=task.get_mujoco_model(),
            num_steps=num_steps,
            save_path=save_path,
            deterministic=deterministic,
            target_fps=None,
            render_width=render_width,
            render_height=render_height,
        )
    else:
        task.run_model_viewer(
            num_steps=num_steps,
            save_renders=save_renders,
            argmax_action=deterministic,
        )

    render_dir = _pick_latest_render_dir(task.exp_dir) if save_renders else None
    video_file = None
    if render_dir is not None:
        video_file = render_dir / ("render.mp4" if save_video else "render.gif")
        if not video_file.exists():
            video_file = None

    return XaxViewerOutputs(exp_dir=task.exp_dir, render_dir=render_dir, video_file=video_file)
