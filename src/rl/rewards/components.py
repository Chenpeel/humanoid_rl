"""
Reward components (shared building blocks).

This module provides small, reusable reward terms that can be composed by task-specific
reward functions (e.g. standing, walking) via a registry pattern.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jp


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """Quaternion to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp) * jp.pi / 2, jp.arcsin(sinp))
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

