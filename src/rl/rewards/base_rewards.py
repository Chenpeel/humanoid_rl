"""基础奖励组合器与通用奖励项（VideoMimic/LeggedGym 风格）.

设计目标：
1) **动态配置驱动**：不硬编码奖励键列表，完全以 `reward_weights` 为准。
2) **可继承**：Standing/Walking 等任务奖励类继承本基类，复用通用 reward 项。
3) **条件返回安全**：当数据不可用/缺失时，reward 项必须返回 0.0（而不是缺键）。

说明：
- 本文件实现的是“reward 项函数 + 组合器”，不负责终止判定（_is_done 由 env 决定）。
- reward 的正负由 `reward_weights` 的符号控制：reward 项通常返回“正的量”（奖励或惩罚幅值）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jp

from . import math_funcs

RewardFn = Callable[[Any], jax.Array]


class BaseRewards:
    """基础奖励类：收集 `_reward_*` 方法并按 `reward_weights` 动态组合。"""

    def __init__(self) -> None:
        self._registry: Dict[str, RewardFn] = self._build_registry()

    # -----------------------------------------------------------------------------------------
    # Registry / composition
    # -----------------------------------------------------------------------------------------

    def _build_registry(self) -> Dict[str, RewardFn]:
        registry: Dict[str, RewardFn] = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_reward_"):
                continue
            key = attr_name[len("_reward_") :]
            fn = getattr(self, attr_name, None)
            if callable(fn):
                registry[key] = fn
        return registry

    @property
    def registry(self) -> Dict[str, RewardFn]:
        # 返回拷贝，避免外部修改内部 registry
        return dict(self._registry)

    def _ref(self, ctx: Any) -> jax.Array:
        ref = getattr(ctx, "torso_z", None)
        if ref is None:
            ref = getattr(ctx, "reward_ref", None)
        if ref is None:
            return jp.array(0.0)
        return jp.asarray(ref)

    def zeros_like(self, ctx: Any) -> jax.Array:
        return jp.zeros_like(self._ref(ctx))

    def ones_like(self, ctx: Any) -> jax.Array:
        return jp.ones_like(self._ref(ctx))

    def compute(
        self,
        ctx: Any,
        reward_weights: Mapping[str, float],
        *,
        exclude_keys: Iterable[str] = (),
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """按权重动态组合奖励。

        Args:
            ctx: 任意上下文对象（通常为 NamedTuple），包含奖励项需要的数据字段。
            reward_weights: 奖励权重（key -> float）。key 必须与 `_reward_<key>` 对应。
            exclude_keys: 不参与组合的 key（用于子类做“二选一/互斥项”）。

        Returns:
            (reward, reward_info)，其中 reward_info 的键集合严格等于 reward_weights.keys() 的集合。
        """
        weights = reward_weights or {}
        reward = jp.array(0.0)
        reward_info = {f"reward/{k}": jp.array(0.0) for k in weights.keys()}
        exclude = set(exclude_keys)

        for key, weight in weights.items():
            if key in exclude or weight == 0.0:
                continue
            fn = self._registry.get(key, None)
            if fn is None:
                term = self.zeros_like(ctx)
            else:
                term = fn(ctx)
            weighted = weight * term
            reward += weighted
            reward_info[f"reward/{key}"] = weighted

        return reward, reward_info

    # -----------------------------------------------------------------------------------------
    # VideoMimic / LeggedGym 通用 reward 项（按需被子类复用）
    # -----------------------------------------------------------------------------------------

    # 1) 基础运动控制
    def _reward_lin_vel_z(self, ctx: Any) -> jax.Array:
        base_linvel = getattr(ctx, "base_linvel", None)
        if base_linvel is None:
            return self.zeros_like(ctx)
        return jp.square(jp.asarray(base_linvel)[..., 2])

    def _reward_ang_vel_xy(self, ctx: Any) -> jax.Array:
        base_angvel = getattr(ctx, "base_angvel", None)
        if base_angvel is None:
            return self.zeros_like(ctx)
        return jp.sum(jp.square(jp.asarray(base_angvel)[..., :2]), axis=-1)

    def _reward_orientation(self, ctx: Any) -> jax.Array:
        projected_gravity = getattr(ctx, "projected_gravity", None)
        if projected_gravity is None:
            base_quat = getattr(ctx, "base_quat", None)
            if base_quat is None:
                return self.zeros_like(ctx)
            projected_gravity = math_funcs.projected_gravity_in_base(base_quat)
        return jp.sum(jp.square(jp.asarray(projected_gravity)[..., :2]), axis=-1)

    # 2) 位置和高度控制
    def _reward_base_height(self, ctx: Any) -> jax.Array:
        base_height = getattr(ctx, "torso_z", None)
        if base_height is None:
            return self.zeros_like(ctx)
        target = getattr(ctx, "base_height_target", None)
        if target is None:
            target = getattr(ctx, "target_height", 0.0)
        return jp.square(jp.asarray(base_height) - jp.asarray(target))

    # 3) 能耗/执行器控制
    def _reward_energy(self, ctx: Any) -> jax.Array:
        torques = getattr(ctx, "torques", None)
        dof_vel = getattr(ctx, "dof_vel", None)
        if dof_vel is None:
            dof_vel = getattr(ctx, "joint_vel", None)
        if torques is None or dof_vel is None:
            return self.zeros_like(ctx)
        torques = jp.asarray(torques)
        dof_vel = jp.asarray(dof_vel)
        n = min(torques.shape[-1], dof_vel.shape[-1])
        power = torques[..., :n] * dof_vel[..., :n]
        return jp.sum(jp.square(power), axis=-1)

    def _reward_torques(self, ctx: Any) -> jax.Array:
        torques = getattr(ctx, "torques", None)
        if torques is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_torque_penalty(jp.asarray(torques))

    def _reward_dof_vel(self, ctx: Any) -> jax.Array:
        dof_vel = getattr(ctx, "dof_vel", None)
        if dof_vel is None:
            dof_vel = getattr(ctx, "joint_vel", None)
        if dof_vel is None:
            return self.zeros_like(ctx)
        return jp.sum(jp.square(jp.asarray(dof_vel)), axis=-1)

    def _reward_dof_acc(self, ctx: Any) -> jax.Array:
        last_dof_vel = getattr(ctx, "last_dof_vel", None)
        dof_vel = getattr(ctx, "dof_vel", None)
        if dof_vel is None:
            dof_vel = getattr(ctx, "joint_vel", None)
        dt = getattr(ctx, "dt", None)
        if last_dof_vel is None or dof_vel is None or dt is None:
            return self.zeros_like(ctx)
        acc = (jp.asarray(last_dof_vel) - jp.asarray(dof_vel)) / jp.asarray(dt)
        return jp.sum(jp.square(acc), axis=-1)

    # 4) 动作变化控制
    def _reward_action_rate(self, ctx: Any) -> jax.Array:
        action = getattr(ctx, "action", None)
        last_action = getattr(ctx, "last_action", None)
        if action is None or last_action is None:
            return self.zeros_like(ctx)
        return math_funcs.compute_action_rate_penalty(jp.asarray(action), jp.asarray(last_action))

    def _reward_action_accel(self, ctx: Any) -> jax.Array:
        action = getattr(ctx, "action", None)
        last_action = getattr(ctx, "last_action", None)
        last_last_action = getattr(ctx, "last_last_action", None)
        if action is None or last_action is None or last_last_action is None:
            return self.zeros_like(ctx)
        accel = jp.asarray(action) - 2.0 * jp.asarray(last_action) + jp.asarray(last_last_action)
        return jp.sum(jp.square(accel), axis=-1)

    # 5) 碰撞/接触（需要更丰富的接触力信息；数据缺失时安全返回 0）
    def _reward_collision(self, ctx: Any) -> jax.Array:
        contact_forces = getattr(ctx, "contact_forces", None)
        penalised_contact_indices = getattr(ctx, "penalised_contact_indices", None)
        if contact_forces is None or penalised_contact_indices is None:
            return self.zeros_like(ctx)
        forces = jp.asarray(contact_forces)[..., penalised_contact_indices, :]
        collided = (jp.linalg.norm(forces, axis=-1) > 0.1).astype(jp.float32)
        return jp.sum(collided, axis=-1)

    def _reward_termination(self, ctx: Any) -> jax.Array:
        reset_buf = getattr(ctx, "reset_buf", None)
        time_out_buf = getattr(ctx, "time_out_buf", None)
        if reset_buf is None or time_out_buf is None:
            return self.zeros_like(ctx)
        return jp.asarray(reset_buf) * (1.0 - jp.asarray(time_out_buf))

    # 6) 关节限制
    def _reward_dof_pos_limits(self, ctx: Any) -> jax.Array:
        dof_pos = getattr(ctx, "dof_pos", None)
        if dof_pos is None:
            dof_pos = getattr(ctx, "joint_pos", None)
        dof_pos_limits = getattr(ctx, "dof_pos_limits", None)
        if dof_pos_limits is None:
            dof_pos_limits = getattr(ctx, "joint_limits", None)
        if dof_pos is None or dof_pos_limits is None:
            return self.zeros_like(ctx)

        dof_pos = jp.asarray(dof_pos)
        if isinstance(dof_pos_limits, tuple) and len(dof_pos_limits) == 2:
            lower, upper = dof_pos_limits
            lower = jp.asarray(lower)
            upper = jp.asarray(upper)
        else:
            limits = jp.asarray(dof_pos_limits)
            lower = limits[..., 0]
            upper = limits[..., 1]

        out_of_limits = -jp.clip(dof_pos - lower, a_max=0.0)
        out_of_limits = out_of_limits + jp.clip(dof_pos - upper, a_min=0.0)
        return jp.sum(out_of_limits, axis=-1)

    def _reward_dof_vel_limits(self, ctx: Any) -> jax.Array:
        dof_vel = getattr(ctx, "dof_vel", None)
        if dof_vel is None:
            dof_vel = getattr(ctx, "joint_vel", None)
        dof_vel_limits = getattr(ctx, "dof_vel_limits", None)
        soft = getattr(ctx, "soft_dof_vel_limit", 1.0)
        if dof_vel is None or dof_vel_limits is None:
            return self.zeros_like(ctx)
        dof_vel = jp.asarray(dof_vel)
        dof_vel_limits = jp.asarray(dof_vel_limits) * jp.asarray(soft)
        penalty = jp.clip(jp.abs(dof_vel) - dof_vel_limits, a_min=0.0, a_max=1.0)
        return jp.sum(penalty, axis=-1)

    def _reward_torque_limits(self, ctx: Any) -> jax.Array:
        torques = getattr(ctx, "torques", None)
        torque_limits = getattr(ctx, "torque_limits", None)
        soft = getattr(ctx, "soft_torque_limit", 1.0)
        if torques is None or torque_limits is None:
            return self.zeros_like(ctx)
        torques = jp.asarray(torques)
        torque_limits = jp.asarray(torque_limits) * jp.asarray(soft)
        penalty = jp.clip(jp.abs(torques) - torque_limits, a_min=0.0)
        return jp.sum(penalty, axis=-1)

    # 7) 速度跟踪（命令跟踪）
    def _reward_tracking_lin_vel(self, ctx: Any) -> jax.Array:
        command = getattr(ctx, "command", None)
        base_linvel = getattr(ctx, "base_linvel", None)
        tracking_sigma = getattr(ctx, "tracking_sigma", 0.25)
        if command is None or base_linvel is None:
            return self.zeros_like(ctx)
        command = jp.asarray(command)[..., :2]
        base_linvel = jp.asarray(base_linvel)[..., :2]
        err = jp.sum(jp.square(command - base_linvel), axis=-1)
        return jp.exp(-err / jp.asarray(tracking_sigma))

    def _reward_tracking_ang_vel(self, ctx: Any) -> jax.Array:
        command = getattr(ctx, "command", None)
        base_angvel = getattr(ctx, "base_angvel", None)
        tracking_sigma = getattr(ctx, "tracking_sigma", 0.25)
        if command is None or base_angvel is None:
            return self.zeros_like(ctx)
        err = jp.square(jp.asarray(command)[..., 2] - jp.asarray(base_angvel)[..., 2])
        return jp.exp(-err / jp.asarray(tracking_sigma))

    # 8) 步态相关
    def _reward_feet_air_time(self, ctx: Any) -> jax.Array:
        # 该项在原实现依赖 env 内部状态（feet_air_time/last_contacts），此处只在有 contact_history 时给一个近似实现。
        contact_history = getattr(ctx, "contact_history", None)
        if contact_history is None:
            return self.zeros_like(ctx)
        command = getattr(ctx, "command", None)
        rew = math_funcs.compute_feet_air_time_reward(jp.asarray(contact_history))
        if command is None:
            return rew
        speed = jp.linalg.norm(jp.asarray(command)[..., :2], axis=-1)
        return rew * (speed > 0.1).astype(jp.float32)

    def _reward_stumble(self, ctx: Any) -> jax.Array:
        # 需要 3D contact_forces；数据缺失则返回 0
        contact_forces = getattr(ctx, "contact_forces", None)
        feet_indices = getattr(ctx, "feet_indices", None)
        if contact_forces is None or feet_indices is None:
            return self.zeros_like(ctx)
        forces = jp.asarray(contact_forces)[..., feet_indices, :]
        horizontal = jp.linalg.norm(forces[..., :2], axis=-1)
        vertical = jp.abs(forces[..., 2])
        return jp.any(horizontal > 5.0 * vertical, axis=-1).astype(jp.float32)

    def _reward_stand_still(self, ctx: Any) -> jax.Array:
        dof_pos = getattr(ctx, "dof_pos", None)
        if dof_pos is None:
            dof_pos = getattr(ctx, "joint_pos", None)
        default = getattr(ctx, "default_dof_pos", None)
        if default is None:
            default = getattr(ctx, "joint_pos_default", None)
        command = getattr(ctx, "command", None)
        if dof_pos is None or default is None or command is None:
            return self.zeros_like(ctx)
        still = (jp.linalg.norm(jp.asarray(command)[..., :2], axis=-1) < 0.1).astype(jp.float32)
        return jp.sum(jp.abs(jp.asarray(dof_pos) - jp.asarray(default)), axis=-1) * still

    def _reward_feet_contact_forces(self, ctx: Any) -> jax.Array:
        # VideoMimic 原实现依赖 3D contact_forces；此处优先使用 ctx.contact_forces，否则回退到 contact_sensors(标量)。
        max_force = getattr(ctx, "max_contact_force", 100.0)
        contact_forces = getattr(ctx, "contact_forces", None)
        if contact_forces is not None:
            forces = jp.asarray(contact_forces)
            if forces.shape[-1] == 3:
                magnitudes = jp.linalg.norm(forces, axis=-1)
            else:
                magnitudes = jp.linalg.norm(forces, axis=-1)
            return jp.sum(jp.clip(magnitudes - max_force, a_min=0.0), axis=-1)

        contact_sensors = getattr(ctx, "contact_sensors", None)
        if contact_sensors is None:
            return self.zeros_like(ctx)
        magnitudes = jp.abs(jp.asarray(contact_sensors))
        return jp.sum(jp.clip(magnitudes - max_force, a_min=0.0), axis=-1)

    # -----------------------------------------------------------------------------------------
    # DeepMimic / imitation 风格（仅当上下文提供相关数据时生效）
    # -----------------------------------------------------------------------------------------

    def _reward_joint_pos_tracking(self, ctx: Any) -> jax.Array:
        dof_pos = getattr(ctx, "dof_pos", None)
        target = getattr(ctx, "target_motors", None)
        k = getattr(ctx, "joint_pos_tracking_k", None)
        if dof_pos is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(dof_pos) - jp.asarray(target)
        return jp.exp(-jp.sum(jp.square(err), axis=-1) * jp.asarray(k))

    def _reward_joint_vel_tracking(self, ctx: Any) -> jax.Array:
        dof_vel = getattr(ctx, "dof_vel", None)
        target = getattr(ctx, "target_motor_vels", None)
        k = getattr(ctx, "joint_vel_tracking_k", None)
        if dof_vel is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(dof_vel) - jp.asarray(target)
        return jp.exp(-jp.sum(jp.square(err), axis=-1) * jp.asarray(k))

    def _reward_root_pos_tracking(self, ctx: Any) -> jax.Array:
        root_pos = getattr(ctx, "env_root_pos", None)
        target = getattr(ctx, "target_root_pos", None)
        k = getattr(ctx, "root_pos_tracking_k", None)
        if root_pos is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(root_pos) - jp.asarray(target)
        return jp.exp(-jp.sum(jp.square(err), axis=-1) * jp.asarray(k))

    def _reward_root_orientation_tracking(self, ctx: Any) -> jax.Array:
        root_quat = getattr(ctx, "root_quat", None)
        target = getattr(ctx, "target_root_quat", None)
        k = getattr(ctx, "root_orientation_tracking_k", None)
        if root_quat is None or target is None or k is None:
            return self.zeros_like(ctx)
        quat_diff = math_funcs.quat_mul(jp.asarray(root_quat), math_funcs.quat_conjugate(jp.asarray(target)))
        angle = math_funcs.quat_angle(quat_diff)
        return jp.exp(-angle * jp.asarray(k))

    def _reward_torso_pos_tracking(self, ctx: Any) -> jax.Array:
        torso_pos = getattr(ctx, "torso_pos", None)
        target = getattr(ctx, "target_torso_pos", None)
        k = getattr(ctx, "torso_pos_tracking_k", None)
        if torso_pos is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(torso_pos) - jp.asarray(target)
        return jp.exp(-jp.sum(jp.square(err), axis=-1) * jp.asarray(k))

    def _reward_torso_orientation_tracking(self, ctx: Any) -> jax.Array:
        torso_quat = getattr(ctx, "torso_quat", None)
        target = getattr(ctx, "target_torso_quat", None)
        k = getattr(ctx, "torso_orientation_tracking_k", None)
        if torso_quat is None or target is None or k is None:
            return self.zeros_like(ctx)
        quat_diff = math_funcs.quat_mul(jp.asarray(torso_quat), math_funcs.quat_conjugate(jp.asarray(target)))
        angle = math_funcs.quat_angle(quat_diff)
        return jp.exp(-angle * jp.asarray(k))

    def _reward_link_pos_tracking(self, ctx: Any) -> jax.Array:
        link_pos_error = getattr(ctx, "link_pos_error", None)
        k = getattr(ctx, "link_pos_tracking_k", None)
        if link_pos_error is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(link_pos_error)
        return jp.exp(-jp.sum(jp.square(err), axis=(-2, -1)) * jp.asarray(k))

    def _reward_link_vel_tracking(self, ctx: Any) -> jax.Array:
        link_vel_error = getattr(ctx, "link_vel_error", None)
        k = getattr(ctx, "link_vel_tracking_k", None)
        if link_vel_error is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(link_vel_error)
        return jp.exp(-jp.sum(jp.square(err), axis=(-2, -1)) * jp.asarray(k))

    def _reward_root_vel_tracking(self, ctx: Any) -> jax.Array:
        root_vel = getattr(ctx, "root_vel", None)
        target = getattr(ctx, "target_root_vel", None)
        k = getattr(ctx, "root_vel_tracking_k", None)
        if root_vel is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(root_vel) - jp.asarray(target)
        return jp.exp(-jp.linalg.norm(err, axis=-1) * jp.asarray(k))

    def _reward_root_ang_vel_tracking(self, ctx: Any) -> jax.Array:
        root_ang_vel = getattr(ctx, "root_ang_vel", None)
        target = getattr(ctx, "target_root_ang_vel", None)
        k = getattr(ctx, "root_ang_vel_tracking_k", None)
        if root_ang_vel is None or target is None or k is None:
            return self.zeros_like(ctx)
        err = jp.asarray(root_ang_vel) - jp.asarray(target)
        return jp.exp(-jp.linalg.norm(err, axis=-1) * jp.asarray(k))

    def _reward_feet_contact_matching(self, ctx: Any) -> jax.Array:
        contacts = getattr(ctx, "contacts", None)
        desired = getattr(ctx, "target_contacts", None)
        if contacts is None or desired is None:
            return self.zeros_like(ctx)
        contacts = jp.asarray(contacts) > 0.5
        desired = jp.asarray(desired) > 0.5
        return jp.sum((contacts == desired).astype(jp.float32), axis=-1)

    def _reward_contact_smoothness(self, ctx: Any) -> jax.Array:
        invalid_changes = getattr(ctx, "invalid_changes", None)
        if invalid_changes is None:
            return self.zeros_like(ctx)
        return jp.sum(jp.asarray(invalid_changes).astype(jp.float32), axis=-1)

    def _reward_no_fly(self, ctx: Any) -> jax.Array:
        contacts = getattr(ctx, "contacts", None)
        desired = getattr(ctx, "target_contacts", None)
        if contacts is None or desired is None:
            return self.zeros_like(ctx)
        contacts = jp.asarray(contacts) > 0.5
        desired = jp.asarray(desired) > 0.5
        fly = (~contacts[..., 0]) & (~contacts[..., 1])
        should_fly = desired[..., 0] & desired[..., 1]
        return (fly & (~should_fly)).astype(jp.float32)

    def _reward_feet_swing_height(self, ctx: Any) -> jax.Array:
        swing_height_target = getattr(ctx, "swing_height_target", 0.08)
        feet_pos = getattr(ctx, "feet_pos", None)
        if feet_pos is None:
            feet_pos = getattr(ctx, "feet_positions", None)
        if feet_pos is None:
            return self.zeros_like(ctx)
        feet_pos = jp.asarray(feet_pos)

        target_contacts = getattr(ctx, "target_contacts", None)
        if target_contacts is not None:
            contact = jp.asarray(target_contacts).astype(jp.float32)
            pos_error = jp.square(feet_pos[..., 2] - swing_height_target) * (1.0 - contact)
            pos_error = pos_error + swing_height_target * contact
            return jp.sum(pos_error, axis=-1)

        contacts = getattr(ctx, "contacts", None)
        if contacts is None:
            return self.zeros_like(ctx)
        contact = (jp.asarray(contacts) > 0.5).astype(jp.float32)
        pos_error = jp.square(feet_pos[..., 2] - swing_height_target) * (1.0 - contact)
        return jp.sum(pos_error, axis=-1)

    def _reward_ankle_action(self, ctx: Any) -> jax.Array:
        ankle_indices = getattr(ctx, "ankle_indices", None)
        action = getattr(ctx, "action", None)
        if ankle_indices is None or action is None:
            return self.zeros_like(ctx)
        ankle = jp.asarray(action)[..., jp.asarray(ankle_indices)]
        return jp.sum(jp.square(ankle), axis=-1)

    def _reward_feet_orientation(self, ctx: Any) -> jax.Array:
        feet_quat = getattr(ctx, "feet_quat", None)
        body_quat = getattr(ctx, "body_quat", None)
        if body_quat is None:
            body_quat = getattr(ctx, "base_quat", None)
        if feet_quat is None or body_quat is None:
            return self.zeros_like(ctx)
        feet_quat = jp.asarray(feet_quat)
        body_quat = jp.asarray(body_quat)

        body_heading = math_funcs.quat_heading(body_quat)
        foot_heading = math_funcs.quat_heading(feet_quat)
        heading_error = jp.abs(math_funcs.wrap_to_pi(foot_heading - body_heading[..., None]))
        total_error = jp.sum(heading_error, axis=-1)
        k = jp.asarray(getattr(ctx, "feet_orientation_k", 6.0))
        return jp.exp(-total_error * k)

    def _reward_contact(self, ctx: Any) -> jax.Array:
        # phase/stance 由上层提供（例如 g1_env.py 的 leg_phase）
        leg_phase = getattr(ctx, "leg_phase", None)
        contacts = getattr(ctx, "contacts", None)
        if leg_phase is None or contacts is None:
            return self.zeros_like(ctx)
        is_stance = jp.asarray(leg_phase) < 0.55
        contact = jp.asarray(contacts) > 0.5
        return jp.sum((~(contact ^ is_stance)).astype(jp.float32), axis=-1)

    def _reward_alive(self, ctx: Any) -> jax.Array:
        return self.ones_like(ctx)

    def _reward_contact_no_vel(self, ctx: Any) -> jax.Array:
        contacts = getattr(ctx, "contacts", None)
        feet_vel = getattr(ctx, "feet_vel", None)
        if feet_vel is None:
            feet_vel = getattr(ctx, "feet_velocities", None)
        if contacts is None or feet_vel is None:
            return self.zeros_like(ctx)
        contacts = (jp.asarray(contacts) > 0.5).astype(jp.float32)
        feet_vel = jp.asarray(feet_vel)
        contact_vel = feet_vel[..., :3] * contacts[..., :, None]
        return jp.sum(jp.linalg.norm(contact_vel, axis=-1), axis=-1)

    def _reward_hip_pos(self, ctx: Any) -> jax.Array:
        dof_pos = getattr(ctx, "dof_pos", None)
        if dof_pos is None:
            dof_pos = getattr(ctx, "joint_pos", None)
        if dof_pos is None:
            return self.zeros_like(ctx)
        dof_pos = jp.asarray(dof_pos)
        hip_indices = jp.asarray(getattr(ctx, "hip_indices", jp.array([1, 2, 7, 8])))
        if dof_pos.shape[-1] <= int(jp.max(hip_indices)):
            return self.zeros_like(ctx)
        hips = dof_pos[..., hip_indices]
        return jp.sum(jp.square(hips), axis=-1)
