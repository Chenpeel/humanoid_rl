"""
Reward components (shared building blocks).

This module provides small, reusable reward terms that can be composed by task-specific
reward functions (e.g. standing, walking) via a registry pattern.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jp


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """Quaternion to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp)
                     * jp.pi / 2, jp.arcsin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)
    return jp.stack([roll, pitch, yaw], axis=-1)


def normalize_quaternion(quat: jax.Array) -> jax.Array:
    """Normalize quaternion."""
    norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    return jp.where(norm > 1e-8, quat / norm, jp.array([1.0, 0.0, 0.0, 0.0]))


def wrap_to_pi(angles: jax.Array) -> jax.Array:
    """Wrap angles to [-pi, pi]."""
    return jp.arctan2(jp.sin(angles), jp.cos(angles))


def compute_action_rate_penalty(action: jax.Array, last_action: jax.Array) -> jax.Array:
    """Penalty for action changes between steps."""
    return jp.sum(jp.square(action - last_action), axis=-1)


def compute_torque_penalty(torques: jax.Array) -> jax.Array:
    """Torque magnitude penalty."""
    return jp.sum(jp.square(torques), axis=-1)


def compute_joint_deviation_penalty(
    joint_pos: jax.Array,
    joint_pos_default: jax.Array,
    indices: Optional[jax.Array] = None,
) -> jax.Array:
    """Penalty for deviating from default/home pose."""
    joint_pos = jp.asarray(joint_pos)
    joint_pos_default = jp.asarray(joint_pos_default)
    if indices is not None:
        indices = jp.asarray(indices)
        joint_pos = joint_pos[..., indices]
        joint_pos_default = joint_pos_default[..., indices]
    diff = joint_pos - joint_pos_default
    return jp.mean(jp.square(diff), axis=-1)


def compute_lin_vel_penalty(base_linvel: jax.Array) -> jax.Array:
    """Squared linear velocity penalty (x,y,z)."""
    return jp.sum(jp.square(base_linvel), axis=-1)


def compute_lin_vel_xy_penalty(base_linvel: jax.Array) -> jax.Array:
    """Squared horizontal linear velocity penalty (x,y)."""
    return jp.sum(jp.square(base_linvel[..., :2]), axis=-1)


def compute_ang_vel_penalty(base_angvel: jax.Array) -> jax.Array:
    """Squared angular velocity penalty."""
    return jp.sum(jp.square(base_angvel), axis=-1)


def compute_knee_bend_reward(
    joint_pos: jax.Array,
    target_bend: float = 0.4,
    tolerance: float = 0.25,
    knee_indices: Tuple[int, int] = (3, 11),
) -> jax.Array:
    """Knee bend reward to avoid locked-knee standing.

    Notes:
    - |knee| to be robust to opposite sign conventions between legs.
    - `knee_indices` assumes 8 actuated joints per leg in the order:
      [hip_pitch, hip_yaw, hip_roll, knee, ankle_pitch, ankle_roll, ankle_yaw, toe] * 2.
    """
    joint_pos = jp.asarray(joint_pos)
    if joint_pos.shape[-1] <= max(knee_indices):
        return jp.zeros(joint_pos.shape[:-1], dtype=joint_pos.dtype)

    knees = joint_pos[..., jp.array(knee_indices)]
    bend = jp.abs(knees)
    bend_error = jp.mean(jp.abs(bend - target_bend), axis=-1)
    return jp.exp(-bend_error / tolerance)


def compute_toe_only_contact_penalty(
    contact_sensors: jax.Array,
    threshold: float = 1.0,
) -> jax.Array:
    """Penalty for toe-only support (toe contact without foot contact).

    Assumes 4 touch sensors ordered as:
    [right_foot, right_toe, left_foot, left_toe].
    Returns a value in [0, 1] (mean over both feet).
    """
    contact_sensors = jp.asarray(contact_sensors)
    if contact_sensors.shape[-1] != 4:
        return jp.zeros(contact_sensors.shape[:-1], dtype=contact_sensors.dtype)

    right_foot = contact_sensors[..., 0]
    right_toe = contact_sensors[..., 1]
    left_foot = contact_sensors[..., 2]
    left_toe = contact_sensors[..., 3]

    right_toe_only = (right_toe > threshold) & (right_foot <= threshold)
    left_toe_only = (left_toe > threshold) & (left_foot <= threshold)
    toe_only = jp.stack(
        [right_toe_only.astype(jp.float32), left_toe_only.astype(jp.float32)], axis=-1
    )
    return jp.mean(toe_only, axis=-1)
