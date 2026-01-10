"""
数学工具函数

包含四元数、欧拉角等数学转换工具。
"""

import torch
from torch import Tensor


DEFAULT_BASE_QUAT_CORRECTION_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)


def make_identity_quaternion_like(quat: Tensor) -> Tensor:
    """创建与输入同 shape 的单位四元数（wxyz）。"""
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return identity


def quat_to_euler_xyz(quat: Tensor) -> Tensor:
    """将四元数转换为欧拉角（XYZ顺序）

    Args:
        quat: 四元数张量，形状 (N, 4)，格式 [w, x, y, z]

    Returns:
        欧拉角张量，形状 (N, 3)，格式 [roll, pitch, yaw]

    注意:
        - 输入四元数应该已经归一化
        - 输出欧拉角单位为弧度
    """
    # 提取四元数分量
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # 处理万向节死锁
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # 堆叠为 (N, 3) 张量
    return torch.stack([roll, pitch, yaw], dim=-1)


def normalize_quaternion(quat: Tensor) -> Tensor:
    """归一化四元数

    Args:
        quat: 四元数张量，形状 (N, 4)

    Returns:
        归一化后的四元数，形状 (N, 4)
    """
    norm = torch.norm(quat, dim=-1, keepdim=True)
    quat_normalized = quat / torch.clamp(norm, min=1e-8)
    return torch.where(norm > 1e-8, quat_normalized, make_identity_quaternion_like(quat))


def quat_conjugate(quat: Tensor) -> Tensor:
    """计算四元数共轭

    Args:
        quat: 四元数张量，形状 (N, 4)，格式 [w, x, y, z]

    Returns:
        共轭四元数，形状 (N, 4)，格式 [w, -x, -y, -z]
    """
    conj = quat.clone()
    conj[:, 1:] *= -1
    return conj


def quat_inverse(quat: Tensor) -> Tensor:
    """四元数求逆。

    Args:
        quat: 四元数张量，形状 (..., 4)，格式 [w, x, y, z]

    Returns:
        逆四元数，形状 (..., 4)
    """
    conj = quat_conjugate(quat)
    norm2 = torch.sum(quat * quat, dim=-1, keepdim=True)
    return conj / torch.clamp(norm2, min=1e-8)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """四元数乘法

    Args:
        q1: 第一个四元数，形状 (N, 4)
        q2: 第二个四元数，形状 (N, 4)

    Returns:
        乘积四元数，形状 (N, 4)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def rotate_vector_by_quat(vec: Tensor, quat: Tensor) -> Tensor:
    """使用四元数旋转向量

    Args:
        vec: 向量，形状 (N, 3)
        quat: 四元数，形状 (N, 4)

    Returns:
        旋转后的向量，形状 (N, 3)
    """
    # 将向量转换为纯四元数 [0, x, y, z]
    vec_quat = torch.cat([torch.zeros(vec.shape[0], 1, device=vec.device), vec], dim=-1)

    # 计算 q * v * q^-1
    quat_conj = quat_conjugate(quat)
    rotated = quat_multiply(quat_multiply(quat, vec_quat), quat_conj)

    # 返回向量部分
    return rotated[:, 1:]


def rotate_vector_by_quat_inverse(vec: Tensor, quat: Tensor) -> Tensor:
    """使用四元数的逆旋转向量（world -> body）

    Args:
        vec: 向量，形状 (N, 3)
        quat: 四元数，形状 (N, 4)，格式 [w, x, y, z]

    Returns:
        旋转后的向量，形状 (N, 3)
    """
    quat = normalize_quaternion(quat)
    quat_inv = quat_conjugate(quat)
    return rotate_vector_by_quat(vec, quat_inv)


def apply_quat_right_mul(quat: Tensor, right_quat_wxyz: tuple[float, float, float, float]) -> Tensor:
    """对每个 quat 做右乘（quat ⊗ right_quat）。

    用于应用/移除固定坐标系旋转（例如 Jiyuan base 的 90° 旋转）。
    """
    right = torch.tensor(right_quat_wxyz, device=quat.device, dtype=quat.dtype).expand_as(quat)
    return quat_multiply(quat, right)


def remove_fixed_quat_rotation(quat: Tensor, fixed_quat_wxyz: tuple[float, float, float, float]) -> Tensor:
    """移除固定旋转：quat_corrected = quat ⊗ fixed_quat^{-1}。"""
    fixed = torch.tensor(fixed_quat_wxyz, device=quat.device, dtype=quat.dtype).expand_as(quat)
    fixed_inv = quat_inverse(fixed)
    return normalize_quaternion(quat_multiply(quat, fixed_inv))


def apply_base_quat_correction_to_body_vec(
    vec_b: Tensor,
    base_quat_correction_wxyz: tuple[float, float, float, float] | None,
    default_base_quat_correction_wxyz: tuple[float, float, float, float] = DEFAULT_BASE_QUAT_CORRECTION_WXYZ,
) -> Tensor:
    """把 body-frame 向量校正到 Z-up 语义坐标系。

    适用场景：`root_lin_vel_b` / `root_ang_vel_b` / `projected_gravity_b` 等已经在 body frame 的量。

    说明：
    - 若 `root_quat_w` 含有固定 base 旋转（历史包袱），那么 body-frame 向量同样“错轴”。
    - 要得到语义一致的 body-frame 向量，需要左乘 fixed rotation：`v_corrected = R_fixed * v_b`。
    - 对默认的 +90° about X（wxyz=(√2/2, √2/2,0,0)）使用 fast-path：`(x,y,z)->(x,-z,y)`。
    """
    if base_quat_correction_wxyz is None:
        return vec_b

    if base_quat_correction_wxyz == default_base_quat_correction_wxyz:
        return torch.stack([vec_b[:, 0], -vec_b[:, 2], vec_b[:, 1]], dim=-1)

    fixed = torch.tensor(base_quat_correction_wxyz, device=vec_b.device, dtype=vec_b.dtype).expand(vec_b.shape[0], 4)
    return rotate_vector_by_quat(vec_b, fixed)


def wrap_to_pi(angles: Tensor) -> Tensor:
    """将角度包装到 [-π, π] 范围

    Args:
        angles: 角度张量（弧度）

    Returns:
        包装后的角度，范围 [-π, π]
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))
