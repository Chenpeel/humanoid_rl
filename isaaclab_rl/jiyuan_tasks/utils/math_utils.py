"""
数学工具函数

包含四元数、欧拉角等数学转换工具。
"""

import torch
from torch import Tensor


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
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp)
    )

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
    return quat / torch.norm(quat, dim=-1, keepdim=True)


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


def wrap_to_pi(angles: Tensor) -> Tensor:
    """将角度包装到 [-π, π] 范围

    Args:
        angles: 角度张量（弧度）

    Returns:
        包装后的角度，范围 [-π, π]
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))
