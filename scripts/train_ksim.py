"""
使用 ksim 训练 gaoda_jiyuan 的直立/行走策略（MJX + PPO）。

说明：
- 坐标系：gaoda_jiyuan 的 base 存在 90° 旋转偏置，必须做 Z-up 逆旋转校正。
- 优化器：使用统一的优化器配置，支持多种学习率调度策略。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar
from xml.etree import ElementTree as ET


def _setup_jax_runtime() -> None:
    """在导入 jax/ksim 前设置环境变量（避免无效设置）。"""

    project_root = Path(__file__).resolve().parent.parent

    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RUN_DIR", str(logs_dir / "ksim_train"))

    # 1080Ti(11GB) 默认不要把显存吃满；用户可在外部覆盖。
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")


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
    import yaml
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

    def __post_init__(self) -> None:
        self.cmd_x_range = tuple(self.cmd_x_range)
        self.cmd_y_range = tuple(self.cmd_y_range)
        self.cmd_yaw_range = tuple(self.cmd_yaw_range)
        self.reset_pitch_range = tuple(self.reset_pitch_range)
        self.reset_roll_range = tuple(self.reset_roll_range)
        self.hidden_sizes = tuple(self.hidden_sizes)


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
        return {}

    def get_events(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Event]:
        return {}

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
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

    def get_observations(self, physics_model: ksim.PhysicsModel) -> Mapping[str, ksim.Observation]:
        sensor_name_to_idx_range = ksim.utils.mujoco.get_sensor_data_idxs_by_name(
            physics_model)

        def _sensor_or_zero(name: str) -> ksim.Observation:
            if name not in sensor_name_to_idx_range:
                return ZeroObservation(dim=1)
            return ksim.SensorObservation.create(physics_model=physics_model, sensor_name=name)

        return {
            "joint_position": ksim.JointPositionObservation(),
            "joint_velocity": ksim.JointVelocityObservation(),
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(),
            "base_quat_zup": ZUpBaseQuaternionObservation(),
            "projected_gravity": ZUpProjectedGravityObservation(),
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

    def _build_obs(self, obs: xax.FrozenDict[str, PyTree], cmd: Array) -> Array:
        jp = obs["joint_position"]
        jv = obs["joint_velocity"]
        grav = obs["projected_gravity"]
        linvel = obs["base_linear_velocity"]
        angvel = obs["base_angular_velocity"]
        quat = obs["base_quat_zup"]
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


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 dict: {path}")
    return data


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
