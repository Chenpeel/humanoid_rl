"""
使用 ksim 训练 gaoda_jiyuan 的直立/行走策略（MJX + PPO）。

说明：
- 坐标系：gaoda_jiyuan 的 base 存在 90° 旋转偏置，必须做 Z-up 逆旋转校正。
- 优化器：使用统一的优化器配置，支持多种学习率调度策略。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar
from xml.etree import ElementTree as ET

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 dict: {path}")
    return data


def _find_config_path(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _ensure_venv_python() -> None:
    project_root = Path(__file__).resolve().parent.parent
    venv_bin = project_root / ".venv" / "bin"
    venv_python = venv_bin / "python"
    if venv_bin.is_dir():
        venv_path = str(venv_bin)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        path_parts = [p for p in path_parts if p and p != venv_path]
        os.environ["PATH"] = os.pathsep.join([venv_path] + path_parts)
    if not venv_python.is_file():
        return
    if sys.executable:
        try:
            if Path(sys.executable).resolve() == venv_python.resolve():
                return
        except OSError:
            pass
    if os.environ.get("JRL_VENV_REEXEC") == "1":
        return
    os.environ["JRL_VENV_REEXEC"] = "1"
    os.execv(venv_python.as_posix(), [venv_python.as_posix(), *sys.argv])


def _to_positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} 不能是布尔值")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是整数") from exc
    if parsed <= 0:
        raise ValueError(f"{key} 必须是正整数")
    return parsed


def _to_bool(value: Any, key: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{key} 必须是布尔值")


def _set_env_int(name: str, value: int | None) -> None:
    if value is None:
        return
    os.environ[name] = str(value)


def _upsert_xla_flag(name: str, value: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "").split()
    prefix = f"{name}="
    flags = [flag for flag in flags if not flag.startswith(prefix)]
    flags.append(f"{name}={value}")
    os.environ["XLA_FLAGS"] = " ".join(flags).strip()


def _apply_threading_env(cfg: Mapping[str, Any]) -> None:
    cpu_threads = _to_positive_int(cfg.get("cpu_threads"), "cpu_threads")
    xla_thread_count = _to_positive_int(
        cfg.get("xla_cpu_thread_count"), "xla_cpu_thread_count")
    xla_multi = _to_bool(cfg.get("xla_cpu_multi_thread_eigen"),
                         "xla_cpu_multi_thread_eigen")

    omp_threads = _to_positive_int(
        cfg.get("omp_num_threads"), "omp_num_threads") or cpu_threads
    mkl_threads = _to_positive_int(
        cfg.get("mkl_num_threads"), "mkl_num_threads") or cpu_threads
    openblas_threads = _to_positive_int(
        cfg.get("openblas_num_threads"), "openblas_num_threads") or cpu_threads
    numexpr_threads = _to_positive_int(
        cfg.get("numexpr_num_threads"), "numexpr_num_threads") or cpu_threads

    if xla_thread_count is not None:
        _upsert_xla_flag("--xla_cpu_thread_count", str(xla_thread_count))
    if xla_multi is not None:
        _upsert_xla_flag("--xla_cpu_multi_thread_eigen",
                         "true" if xla_multi else "false")

    _set_env_int("OMP_NUM_THREADS", omp_threads)
    _set_env_int("MKL_NUM_THREADS", mkl_threads)
    _set_env_int("OPENBLAS_NUM_THREADS", openblas_threads)
    _set_env_int("NUMEXPR_NUM_THREADS", numexpr_threads)


def _apply_threading_from_argv() -> None:
    config_path = _find_config_path(sys.argv)
    if not config_path:
        return
    try:
        cfg = _load_yaml(config_path)
    except (OSError, ValueError, yaml.YAMLError):
        return
    _apply_threading_env(cfg)


def _setup_jax_runtime() -> None:
    """在导入 jax/ksim 前设置环境变量（避免无效设置）。"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    project_root = Path(__file__).resolve().parent.parent

    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RUN_DIR", str(logs_dir / "ksim_train"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # import warnings
    # warnings.filterwarnings("ignore", category=Warning)


_ensure_venv_python()
_apply_threading_from_argv()
_setup_jax_runtime()

if True:
    import attrs
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import ksim
    import mujoco
    import optax
    import xax
    from jaxtyping import Array, PRNGKeyArray, PyTree

    # 导入统一优化器
    from rl.training.optimizer import (
        OptimizerConfig,
        ScheduleType,
        create_ksim_optimizer,
        create_optimizer_from_config,
    )


FIX_QUAT_ZUP = (0.70710678, -0.70710678, 0.0, 0.0)
ANKLE_JOINT_NAMES = (
    "right_ankle_cube_joint",
    "right_ankle_axle_joint",
    "right_foot_joint",
    "left_ankle_cube_joint",
    "left_ankle_axle_joint",
    "left_foot_joint",
)
FOOT_BODY_NAMES = ("right_foot_link", "left_foot_link")


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except UnicodeDecodeError:
        return False
    except OSError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def _collect_mujoco_include_files(xml_path: Path) -> list[Path]:
    """递归收集 MuJoCo XML 的 <include file="..."> 依赖。"""

    visited: set[Path] = set()
    queue: list[Path] = [xml_path]
    includes: list[Path] = []

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)

        if not current.exists():
            continue
        if _is_git_lfs_pointer(current):
            continue

        try:
            tree = ET.parse(current)
        except ET.ParseError:
            continue

        for include_elem in tree.findall(".//include"):
            include_file = include_elem.get("file")
            if not include_file:
                continue
            include_path = (current.parent / include_file).resolve()
            includes.append(include_path)
            queue.append(include_path)

    return includes


def _quat_normalize(quat: Array, eps: float = 1e-8) -> Array:
    norm = jnp.linalg.norm(quat, axis=-1, keepdims=True)
    return jnp.where(norm > eps, quat / norm, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype))


def _quat_to_zup(quat: Array) -> Array:
    """把 MJCF 的物理四元数校正为 Z-up 标准坐标系。"""
    fix = jnp.array(FIX_QUAT_ZUP, dtype=quat.dtype)
    return xax.quat_mul(fix, quat)


@attrs.define(frozen=True, kw_only=True)
class ZUpBaseQuaternionObservation(ksim.Observation):
    """返回 Z-up 校正后的 base quaternion（wxyz）。"""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        quat = _quat_normalize(state.physics_state.data.qpos[..., 3:7])
        return _quat_to_zup(quat)


@attrs.define(frozen=True, kw_only=True)
class ZUpProjectedGravityObservation(ksim.Observation):
    """返回 Z-up 校正后的 projected gravity（在机体坐标系，单位向量）。"""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        quat = _quat_normalize(state.physics_state.data.qpos[..., 3:7])
        quat_zup = _quat_to_zup(quat)
        z_world = jnp.array([0.0, 0.0, 1.0], dtype=quat.dtype)
        return xax.rotate_vector_by_quat(z_world, quat_zup, inverse=True)


@attrs.define(frozen=True, kw_only=True)
class ZUpUprightReward(ksim.Reward):
    """Z-up 坐标系下的直立奖励（用 projected gravity 的 z 分量）。"""

    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        quat = _quat_normalize(trajectory.qpos[..., 3:7])
        quat_zup = _quat_to_zup(quat)
        z_world = jnp.array([0.0, 0.0, 1.0], dtype=quat.dtype)
        z_in_body = xax.rotate_vector_by_quat(z_world, quat_zup, inverse=True)
        cos_tilt = jnp.clip(z_in_body[..., 2], -1.0, 1.0)
        # upright 时 cos_tilt≈1；倾倒后快速衰减
        return jnp.exp(-jnp.square(1.0 - cos_tilt) / (2.0 * self.kernel_scale**2))


@attrs.define(frozen=True, kw_only=True)
class WorldVelocityTrackingReward(ksim.Reward):
    """跟踪世界系速度命令 (vx, vy, wz)。"""

    cmd: str = attrs.field()
    lin_kernel_scale: float = attrs.field(default=0.25)
    yaw_kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Mapping[str, Array]:
        cmd = trajectory.command[self.cmd]  # (..., 3)
        lin_cmd = cmd[..., 0:2]
        yaw_cmd = cmd[..., 2]

        lin_vel = trajectory.qvel[..., 0:2]
        yaw_vel = trajectory.qvel[..., 5]

        lin_err = jnp.sum(jnp.square(lin_vel - lin_cmd), axis=-1)
        yaw_err = jnp.square(yaw_vel - yaw_cmd)

        lin_rew = jnp.exp(-lin_err / (2.0 * self.lin_kernel_scale**2))
        yaw_rew = jnp.exp(-yaw_err / (2.0 * self.yaw_kernel_scale**2))

        return {
            "lin": lin_rew,
            "yaw": yaw_rew,
        }


@attrs.define(frozen=True, kw_only=True)
class ZUpNotUprightTermination(ksim.Termination):
    """Z-up 坐标系下的倾倒终止。"""

    max_radians: float = attrs.field()

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        quat = _quat_normalize(state.qpos[..., 3:7])
        quat_zup = _quat_to_zup(quat)

        z_world = jnp.array([0.0, 0.0, 1.0], dtype=quat.dtype)
        z_in_body = xax.rotate_vector_by_quat(z_world, quat_zup, inverse=True)
        cos_tilt = jnp.clip(z_in_body[..., 2], -1.0, 1.0)
        tilt = jnp.arccos(cos_tilt)
        return jnp.where(tilt > self.max_radians, -1, 0)


@attrs.define(frozen=True, kw_only=True)
class ZeroObservation(ksim.Observation):
    """用于兜底的零观测（避免因为缺少 sensor 导致构建失败）。"""

    dim: int = attrs.field(validator=attrs.validators.ge(1))

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        dtype = state.physics_state.data.qpos.dtype
        return jnp.zeros((self.dim,), dtype=dtype)


@attrs.define(frozen=True, kw_only=True)
class RandomYawReset(ksim.Reset):
    """随机重置 base yaw 角（世界系）。"""

    yaw_range: tuple[float, float] = attrs.field(default=(-jnp.pi, jnp.pi))

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        min_yaw, max_yaw = self.yaw_range
        angle = jax.random.uniform(rng, data.qpos.shape[:-1], minval=min_yaw, maxval=max_yaw)
        euler = jnp.stack([jnp.zeros_like(angle), jnp.zeros_like(angle), angle], axis=-1)
        quat = xax.euler_to_quat(euler)
        # 左乘：在世界坐标系绕 Z 轴施加 yaw
        new_quat = xax.quat_mul(quat, data.qpos[..., 3:7])
        qpos = ksim.utils.mujoco.slice_update(data, "qpos", slice(3, 7), new_quat)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


class NormalizedTorqueActuators(ksim.Actuators):
    """把 [-1, 1] action 映射到 actuator ctrlrange。"""

    def __init__(self, physics_model: ksim.PhysicsModel) -> None:
        ctrl_min = jnp.array(physics_model.actuator_ctrlrange[..., 0])
        ctrl_max = jnp.array(physics_model.actuator_ctrlrange[..., 1])
        self._ctrl_center = (ctrl_min + ctrl_max) * 0.5
        self._ctrl_scale = jnp.maximum((ctrl_max - ctrl_min) * 0.5, 1e-6)
        self._ctrl_min = ctrl_min
        self._ctrl_max = ctrl_max

    def get_ctrl(
        self,
        action: Array,
        physics_data: ksim.PhysicsData,
        rng: PRNGKeyArray,
    ) -> Array:
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = action * self._ctrl_scale + self._ctrl_center
        return jnp.clip(ctrl, self._ctrl_min, self._ctrl_max)


@dataclass
class GaodaJiyuanConfig(ksim.PPOConfig):
    """gaoda_jiyuan + ksim 的 PPO 配置（可用 YAML 覆盖）。"""

    # 任务文件
    xml_path: str = xax.field(
        value="robots/gaoda_jiyuan/scene.xml",
        help="MuJoCo 场景 XML（建议使用 robots/gaoda_jiyuan/scene.xml）",
    )

    # 命令：世界系 (vx, vy, wz)
    cmd_x_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0), help="x 速度范围")
    cmd_y_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0), help="y 速度范围")
    cmd_yaw_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0), help="yaw 角速度范围")
    cmd_switch_prob: float = xax.field(value=0.02, help="命令重采样概率（每步）")
    cmd_zero_prob: float = xax.field(value=0.2, help="命令置零概率（每次采样）")

    # 站立/行走形态
    target_height: float = xax.field(value=0.92, help="目标 base 高度（m）")

    # reset 随机化
    reset_joint_pos_scale: float = xax.field(value=0.05, help="关节位置随机幅度（rad）")
    reset_joint_vel_scale: float = xax.field(value=0.5, help="关节速度随机幅度（rad/s）")
    reset_pitch_range: tuple[float, float] = xax.field(
        value=(-0.15, 0.15), help="pitch 随机范围（rad）")
    reset_roll_range: tuple[float, float] = xax.field(
        value=(-0.15, 0.15), help="roll 随机范围（rad）")
    reset_yaw_range: tuple[float, float] | None = xax.field(
        value=None, help="yaw 随机范围（rad），None 表示不启用")

    # 随机推力事件（物理步时间，单位：秒）
    enable_random_pushes: bool = xax.field(value=False, help="是否启用随机推力扰动")
    push_vel_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0), help="推力速度幅值范围（m/s）")
    push_interval_range: tuple[float, float] = xax.field(
        value=(0.5, 2.0), help="两次推力间隔范围（秒）")
    push_curriculum_range: tuple[float, float] = xax.field(
        value=(0.0, 1.0), help="推力强度随课程缩放范围")

    # 摩擦 / 质量随机化
    enable_random_friction: bool = xax.field(value=False, help="是否随机化摩擦参数")
    dof_friction_scale_range: tuple[float, float] = xax.field(
        value=(0.5, 2.0), help="关节摩擦缩放范围")
    floor_friction_range: tuple[float, float] = xax.field(
        value=(0.4, 1.0), help="地面摩擦范围（绝对值）")
    floor_geom_name: str = xax.field(value="floor", help="地面 geom 名称")
    enable_random_mass: bool = xax.field(value=False, help="是否随机化质量")
    mass_scale_range: tuple[float, float] = xax.field(
        value=(0.98, 1.02), help="全身质量缩放范围")

    # 观测噪声 / 延迟
    obs_noise_std: float = xax.field(value=0.0, help="观测高斯噪声标准差（0 关闭）")
    obs_noise_targets: tuple[str, ...] = xax.field(
        value=(
            "joint_position",
            "joint_velocity",
            "base_linear_velocity",
            "base_angular_velocity",
            "projected_gravity",
        ),
        help="需要添加噪声的观测键",
    )
    joint_pos_delay_steps: int = xax.field(value=1, help="关节位置观测延迟步数（>=1）")
    joint_vel_delay_steps: int = xax.field(value=1, help="关节速度观测延迟步数（>=1）")

    # 终止
    terminate_min_z: float = xax.field(value=0.65, help="过低高度终止阈值")
    terminate_max_z: float = xax.field(value=1.50, help="过高高度终止阈值（防止数值爆炸）")
    terminate_max_tilt_radians: float = xax.field(
        value=1.0, help="最大允许倾角（rad）")

    # 奖励权重
    w_alive: float = xax.field(value=100.0, help="存活奖励权重")
    alive_balance: float = xax.field(
        value=100.0,
        help="StayAliveReward 的 balance（越大则终止惩罚相对越重）",
    )
    w_upright: float = xax.field(value=5.0, help="直立奖励权重")
    w_height: float = xax.field(value=2.0, help="高度奖励权重")
    w_vel_track: float = xax.field(value=2.0, help="速度跟踪权重")
    w_action_rate: float = xax.field(value=0.02, help="动作变化率惩罚权重")
    w_torque: float = xax.field(value=0.002, help="扭矩惩罚权重")
    w_ankle_deviation: float = xax.field(value=0.0, help="踝关节偏转惩罚权重")
    w_foot_flat: float = xax.field(value=0.0, help="足底平行地面奖励权重")

    # reward kernel
    upright_kernel_scale: float = xax.field(value=0.25, help="直立核函数尺度")
    height_kernel_scale: float = xax.field(value=0.20, help="高度核函数尺度")
    vel_kernel_scale: float = xax.field(value=0.25, help="线速度核函数尺度")
    yaw_kernel_scale: float = xax.field(value=0.25, help="角速度核函数尺度")

    # 网络
    hidden_sizes: tuple[int, ...] = xax.field(
        value=(256, 256), help="MLP hidden sizes")
    init_log_std: float = xax.field(value=-0.5, help="初始 log_std")

    # 优化器
    learning_rate: float = xax.field(value=3e-4, help="Adam 学习率")
    grad_clip: float = xax.field(value=1.0, help="梯度裁剪")

    # 训练控制
    max_steps: int | None = xax.field(
        value=None,
        help="最大训练步数（None表示无限训练）"
    )
    warmup_steps: int = xax.field(
        value=0,
        help="学习率预热步数（仅当max_steps不为None时有效）"
    )

    # 线程/核心控制（可选）
    cpu_threads: int | None = xax.field(
        value=None,
        help="限制 CPU 线程/核心数（会同步设置 OMP/MKL/OPENBLAS/NUMEXPR + XLA CPU 线程）",
    )
    xla_cpu_thread_count: int | None = xax.field(
        value=None,
        help="XLA CPU 线程数（默认跟随 cpu_threads）",
    )
    xla_cpu_multi_thread_eigen: bool | None = xax.field(
        value=None,
        help="是否显式设置 XLA CPU 多线程（默认不设置）",
    )
    omp_num_threads: int | None = xax.field(value=None, help="OpenMP 线程数")
    mkl_num_threads: int | None = xax.field(value=None, help="MKL 线程数")
    openblas_num_threads: int | None = xax.field(
        value=None, help="OpenBLAS 线程数")
    numexpr_num_threads: int | None = xax.field(value=None, help="NumExpr 线程数")

    def __post_init__(self) -> None:
        self.cmd_x_range = tuple(self.cmd_x_range)
        self.cmd_y_range = tuple(self.cmd_y_range)
        self.cmd_yaw_range = tuple(self.cmd_yaw_range)
        self.reset_pitch_range = tuple(self.reset_pitch_range)
        self.reset_roll_range = tuple(self.reset_roll_range)
        if self.reset_yaw_range is not None:
            self.reset_yaw_range = tuple(self.reset_yaw_range)
        self.push_vel_range = tuple(self.push_vel_range)
        self.push_interval_range = tuple(self.push_interval_range)
        self.push_curriculum_range = tuple(self.push_curriculum_range)
        self.dof_friction_scale_range = tuple(self.dof_friction_scale_range)
        self.floor_friction_range = tuple(self.floor_friction_range)
        self.mass_scale_range = tuple(self.mass_scale_range)
        self.obs_noise_targets = tuple(self.obs_noise_targets)
        self.hidden_sizes = tuple(self.hidden_sizes)
        self.joint_pos_delay_steps = int(self.joint_pos_delay_steps)
        self.joint_vel_delay_steps = int(self.joint_vel_delay_steps)
        if self.joint_pos_delay_steps < 1:
            raise ValueError("joint_pos_delay_steps 必须 >= 1")
        if self.joint_vel_delay_steps < 1:
            raise ValueError("joint_vel_delay_steps 必须 >= 1")


class Actor(eqx.Module):
    mlp: eqx.nn.MLP
    log_std: Array

    def __call__(self, obs_n: Array) -> xax.Normal:
        mean_n = self.mlp(obs_n)
        std_n = jnp.exp(self.log_std).astype(mean_n.dtype)
        std_n = jnp.broadcast_to(std_n, mean_n.shape)
        return xax.Normal(loc_n=mean_n, scale_n=std_n)


class Critic(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, obs_n: Array) -> Array:
        return self.mlp(obs_n)  # (..., 1)


class Model(eqx.Module):
    actor: Actor
    critic: Critic


ConfigT = TypeVar("ConfigT", bound=GaodaJiyuanConfig)


class GaodaJiyuanTask(ksim.PPOTask[ConfigT]):
    def get_optimizer(self) -> optax.GradientTransformation:
        """使用统一优化器配置

        支持多种学习率调度策略，包括：
        - 常数学习率（默认）
        - 预热+余弦退火（推荐）
        - 预热+线性衰减
        - 指数衰减
        - 阶梯衰减
        """
        # 计算总优化步数（如果配置了max_steps）
        total_steps = getattr(self.config, 'max_steps', None)

        # 使用统一优化器
        optimizer = create_ksim_optimizer(
            learning_rate=self.config.learning_rate,
            grad_clip=self.config.grad_clip,
            total_steps=total_steps,
            warmup_steps=getattr(self.config, 'warmup_steps', 0),
        )

        return optimizer

    # pyright: ignore[reportAttributeAccessIssue]
    def get_mujoco_model(self) -> mujoco.MjModel:
        xml_path = Path(self.config.xml_path).resolve()
        if _is_git_lfs_pointer(xml_path):
            raise RuntimeError(
                "检测到 XML 是 Git LFS pointer（未拉取大文件）。\n"
                "请先在仓库根目录执行：\n"
                "  git lfs pull\n"
                f"当前文件：{xml_path}"
            )

        for include_path in _collect_mujoco_include_files(xml_path):
            if _is_git_lfs_pointer(include_path):
                raise RuntimeError(
                    "检测到 MuJoCo <include> 文件是 Git LFS pointer（未拉取大文件）。\n"
                    "请先在仓库根目录执行：\n"
                    "  git lfs pull\n"
                    f"当前文件：{include_path}"
                )
        # pyright: ignore[reportAttributeAccessIssue]
        return mujoco.MjModel.from_xml_path(xml_path.as_posix())

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        return NormalizedTorqueActuators(physics_model)

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.PhysicsRandomizer]:
        randomizers: dict[str, ksim.PhysicsRandomizer] = {}
        if self.config.enable_random_friction:
            scale_lo, scale_hi = self.config.dof_friction_scale_range
            randomizers["static_friction"] = ksim.StaticFrictionRandomizer(
                scale_lower=scale_lo,
                scale_upper=scale_hi,
            )
            floor_lo, floor_hi = self.config.floor_friction_range
            randomizers["floor_friction"] = ksim.FloorFrictionRandomizer.from_geom_name(
                physics_model,
                floor_geom_name=self.config.floor_geom_name,
                scale_lower=floor_lo,
                scale_upper=floor_hi,
            )
        if self.config.enable_random_mass:
            mass_lo, mass_hi = self.config.mass_scale_range
            randomizers["mass_scale"] = ksim.AllBodiesMassMultiplicationRandomizer(
                scale_lower=mass_lo,
                scale_upper=mass_hi,
            )
        return randomizers

    def get_events(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Event]:
        if not self.config.enable_random_pushes:
            return {}
        return {
            "random_push": ksim.LinearPushEvent(
                linvel=self.config.push_vel_range[1],
                vel_range=self.config.push_vel_range,
                interval_range=self.config.push_interval_range,
                curriculum_range=self.config.push_curriculum_range,
            )
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        resets = [
            ksim.RandomJointPositionReset.create(
                physics_model,
                scale=self.config.reset_joint_pos_scale,
                scale_by_curriculum=False,
            ),
            ksim.RandomJointVelocityReset(
                scale=self.config.reset_joint_vel_scale, scale_by_curriculum=False),
            ksim.RandomPitchRollReset(
                pitch_range=self.config.reset_pitch_range,
                roll_range=self.config.reset_roll_range,
            ),
        ]
        if self.config.reset_yaw_range is not None:
            resets.append(RandomYawReset(yaw_range=self.config.reset_yaw_range))
        return resets

    def get_observations(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Observation]:
        sensor_name_to_idx_range = ksim.utils.mujoco.get_sensor_data_idxs_by_name(
            physics_model)

        noise = None
        if self.config.obs_noise_std > 0.0:
            noise = ksim.AdditiveGaussianNoise(std=self.config.obs_noise_std)
        noise_targets = set(self.config.obs_noise_targets)

        def _noise_for(name: str) -> ksim.Noise | None:
            if noise is None or name not in noise_targets:
                return None
            return noise

        def _sensor_or_zero(name: str) -> ksim.Observation:
            if name not in sensor_name_to_idx_range:
                return ZeroObservation(dim=1, noise=_noise_for(name))
            return ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name=name,
                noise=_noise_for(name),
            )

        if self.config.joint_pos_delay_steps > 1:
            joint_pos_obs = ksim.DelayedJointPositionObservation(
                delay_steps=self.config.joint_pos_delay_steps,
                noise=_noise_for("joint_position"),
            )
        else:
            joint_pos_obs = ksim.JointPositionObservation(noise=_noise_for("joint_position"))

        if self.config.joint_vel_delay_steps > 1:
            joint_vel_obs = ksim.DelayedJointVelocityObservation(
                delay_steps=self.config.joint_vel_delay_steps,
                noise=_noise_for("joint_velocity"),
            )
        else:
            joint_vel_obs = ksim.JointVelocityObservation(noise=_noise_for("joint_velocity"))

        return {
            "joint_position": joint_pos_obs,
            "joint_velocity": joint_vel_obs,
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(
                noise=_noise_for("base_linear_velocity")),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(
                noise=_noise_for("base_angular_velocity")),
            "base_quat_zup": ZUpBaseQuaternionObservation(noise=_noise_for("base_quat_zup")),
            "projected_gravity": ZUpProjectedGravityObservation(
                noise=_noise_for("projected_gravity")),
            "right_foot_contact": _sensor_or_zero("right_foot_contact"),
            "right_toe_contact": _sensor_or_zero("right_toe_contact"),
            "left_foot_contact": _sensor_or_zero("left_foot_contact"),
            "left_toe_contact": _sensor_or_zero("left_toe_contact"),
        }

    def get_commands(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Command]:
        ranges = (
            tuple(self.config.cmd_x_range),
            tuple(self.config.cmd_y_range),
            tuple(self.config.cmd_yaw_range),
        )
        return {
            "cmd": ksim.FloatVectorCommand(
                ranges=ranges,
                switch_prob=self.config.cmd_switch_prob,
                zero_prob=self.config.cmd_zero_prob,
            )
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Reward]:
        rewards: dict[str, ksim.Reward] = {
            "alive": ksim.StayAliveReward(scale=self.config.w_alive, balance=self.config.alive_balance),
            "upright": ZUpUprightReward(scale=self.config.w_upright, kernel_scale=self.config.upright_kernel_scale),
            "height": ksim.BaseHeightReward(
                scale=self.config.w_height,
                height_target=self.config.target_height,
                norm="l2",
                monotonic_fn="exp",
                temp=2.0 * (self.config.height_kernel_scale**2),
            ),
            "vel_track": WorldVelocityTrackingReward(
                scale=self.config.w_vel_track,
                cmd="cmd",
                lin_kernel_scale=self.config.vel_kernel_scale,
                yaw_kernel_scale=self.config.yaw_kernel_scale,
            ),
            "action_rate": ksim.ActionVelocityPenalty(scale=self.config.w_action_rate),
            "torque": ksim.CtrlPenalty.create(physics_model, scale=self.config.w_torque),
        }

        if self.config.w_ankle_deviation != 0.0:
            rewards["ankle_deviation"] = ksim.JointDeviationPenalty.create(
                physics_model,
                joint_names=ANKLE_JOINT_NAMES,
                joint_targets=(0.0,) * len(ANKLE_JOINT_NAMES),
                scale=self.config.w_ankle_deviation,
            )

        if self.config.w_foot_flat != 0.0:
            rewards["foot_flat"] = ksim.FlatBodyReward.create(
                physics_model,
                body_names=FOOT_BODY_NAMES,
                scale=self.config.w_foot_flat,
            )

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Termination]:
        return {
            "bad_z": ksim.BadZTermination(min_z=self.config.terminate_min_z, max_z=self.config.terminate_max_z),
            "not_upright": ZUpNotUprightTermination(max_radians=self.config.terminate_max_tilt_radians),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.ConstantCurriculum(level=1.0)

    def get_model(self, params: ksim.InitParams) -> Model:
        nq = int(params.physics_model.nq)
        nv = int(params.physics_model.nv)
        num_act = int(params.physics_model.nu)

        num_joint_pos = nq - 7
        num_joint_vel = nv - 6
        num_contacts = 4
        num_cmd = 3
        num_base = 3 + 3  # linvel + angvel
        num_grav = 3
        num_quat = 4

        obs_dim = num_joint_pos + num_joint_vel + num_grav + \
            num_base + num_cmd + num_contacts + num_quat

        key1, key2, key3 = jax.random.split(params.key, 3)
        actor_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=num_act,
            width_size=self.config.hidden_sizes[0],
            depth=len(self.config.hidden_sizes),
            activation=jax.nn.tanh,
            key=key1,
        )
        critic_mlp = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=1,
            width_size=self.config.hidden_sizes[0],
            depth=len(self.config.hidden_sizes),
            activation=jax.nn.tanh,
            key=key2,
        )

        log_std = jnp.full(
            (num_act,), self.config.init_log_std, dtype=jnp.float32)

        return Model(
            actor=Actor(mlp=actor_mlp, log_std=log_std),
            critic=Critic(mlp=critic_mlp),
        )

    def get_initial_model_carry(self, model: Model, rng: PRNGKeyArray) -> PyTree | None:
        return None

    def _select_obs(self, obs: xax.FrozenDict[str, PyTree], name: str) -> Array:
        if self.config.obs_noise_std > 0.0 and name in self.config.obs_noise_targets:
            noisy_name = f"noisy_{name}"
            if noisy_name in obs:
                return obs[noisy_name]
        return obs[name]

    def _build_obs(self, obs: xax.FrozenDict[str, PyTree], cmd: Array) -> Array:
        jp = self._select_obs(obs, "joint_position")
        jv = self._select_obs(obs, "joint_velocity")
        grav = self._select_obs(obs, "projected_gravity")
        linvel = self._select_obs(obs, "base_linear_velocity")
        angvel = self._select_obs(obs, "base_angular_velocity")
        quat = self._select_obs(obs, "base_quat_zup")
        contacts = jnp.concatenate(
            [
                obs["right_foot_contact"],
                obs["right_toe_contact"],
                obs["left_foot_contact"],
                obs["left_toe_contact"],
            ],
            axis=-1,
        )

        return jnp.concatenate(
            [
                jp,
                jv / 10.0,
                grav,
                linvel,
                angvel,
                cmd,
                contacts,
                quat,
            ],
            axis=-1,
        )

    def sample_action(
        self,
        model: Model,
        model_carry: PyTree | None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        cmd = commands["cmd"]
        obs_n = self._build_obs(observations, cmd)
        dist = model.actor(obs_n)
        action = dist.mode() if argmax else dist.sample(rng)
        return ksim.Action(action=action, carry=None)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: PyTree | None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, PyTree | None]:
        def scan_fn(carry: None, xs: tuple[ksim.Trajectory, PRNGKeyArray]) -> tuple[None, ksim.PPOVariables]:
            transition, rng = xs
            cmd = transition.command["cmd"]
            obs_n = self._build_obs(transition.obs, cmd)
            dist = model.actor(obs_n)
            log_probs = dist.log_prob(transition.action)
            value = model.critic(obs_n).squeeze(-1)
            entropy = dist.entropy() if hasattr(dist, "entropy") else None
            variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value,
                entropy=entropy,
            )
            return None, variables

        rngs = jax.random.split(rng, trajectory.done.shape[0])
        _, ppo_vars = xax.scan(
            scan_fn,
            None,
            (trajectory, rngs),
            jit_level=ksim.JitLevel.RL_CORE,
        )
        return ppo_vars, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ksim 训练 gaoda_jiyuan（MJX + PPO）")
    parser.add_argument("--config", type=str,
                        required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default=None,
        help="从 checkpoint 初始化训练（不设置则从头训练）",
    )
    args = parser.parse_args()

    cfg_dict = _load_yaml(args.config)
    if args.load_ckpt:
        ckpt_path = Path(args.load_ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"检查点不存在: {ckpt_path}")
        cfg_dict = dict(cfg_dict)
        cfg_dict["load_from_ckpt_path"] = str(ckpt_path)
    config = GaodaJiyuanConfig(**cfg_dict)
    # xax 会默认把 sys.argv 当作 OmegaConf CLI override 解析；这里我们自己处理了 --config，
    # 所以必须关闭 xax 的 CLI 合并逻辑，避免把 --config 当作字段名导致报错。
    GaodaJiyuanTask.launch(config, use_cli=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
