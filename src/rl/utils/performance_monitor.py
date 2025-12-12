"""
性能监控工具
用于监控训练过程中的GPU利用率、吞吐量等指标
"""

import time
from typing import Dict, Optional
import jax
import jax.numpy as jp


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.start_time = None
        self.last_step_time = None
        self.step_count = 0
        self.env_steps = 0
        self.compile_time = 0.0

    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.last_step_time = time.time()

    def step(self, batch_size: int) -> Dict[str, float]:
        """记录一步训练

        Args:
            batch_size: 当前步的批次大小

        Returns:
            性能指标字典
        """
        current_time = time.time()

        # 计算步时间
        step_time = current_time - self.last_step_time
        self.last_step_time = current_time

        # 更新计数
        self.step_count += 1
        self.env_steps += batch_size

        # 计算吞吐量
        steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
        env_steps_per_sec = batch_size / step_time if step_time > 0 else 0.0

        # 总时间
        total_time = current_time - self.start_time

        # 平均吞吐量
        avg_steps_per_sec = self.step_count / total_time if total_time > 0 else 0.0
        avg_env_steps_per_sec = self.env_steps / total_time if total_time > 0 else 0.0

        return {
            "perf/step_time": step_time,
            "perf/steps_per_sec": steps_per_sec,
            "perf/env_steps_per_sec": env_steps_per_sec,
            "perf/avg_steps_per_sec": avg_steps_per_sec,
            "perf/avg_env_steps_per_sec": avg_env_steps_per_sec,
            "perf/total_time": total_time,
            "perf/total_env_steps": self.env_steps,
        }

    def set_compile_time(self, compile_time: float):
        """设置编译时间"""
        self.compile_time = compile_time


def benchmark_train_step(train_step_fn, train_state, env_state, num_warmup=3, num_iterations=10):
    """基准测试训练步性能

    Args:
        train_step_fn: JIT编译的训练步函数
        train_state: 训练状态
        env_state: 环境状态
        num_warmup: 预热迭代次数
        num_iterations: 测试迭代次数

    Returns:
        性能统计信息
    """
    # 预热
    for _ in range(num_warmup):
        train_state, env_state, _ = train_step_fn(train_state, env_state)

    # 确保预热完成
    jax.block_until_ready(train_state)

    # 基准测试
    times = []
    for _ in range(num_iterations):
        start = time.time()
        train_state, env_state, _ = train_step_fn(train_state, env_state)
        jax.block_until_ready(train_state)  # 等待计算完成
        times.append(time.time() - start)

    # 计算统计
    times = jp.array(times)
    return {
        "mean_time": float(jp.mean(times)),
        "std_time": float(jp.std(times)),
        "min_time": float(jp.min(times)),
        "max_time": float(jp.max(times)),
        "median_time": float(jp.median(times)),
    }


def estimate_gpu_utilization(step_time: float, theoretical_min_time: Optional[float] = None) -> float:
    """估算GPU利用率

    Args:
        step_time: 实际步时间
        theoretical_min_time: 理论最小时间（如果已知）

    Returns:
        估算的GPU利用率（0-1之间）
    """
    if theoretical_min_time is None:
        # 如果没有理论最小时间，假设当前时间的倒数作为利用率指标
        # 这只是一个粗略的估计
        return 1.0 / (step_time + 1e-8)
    else:
        return theoretical_min_time / (step_time + 1e-8)
