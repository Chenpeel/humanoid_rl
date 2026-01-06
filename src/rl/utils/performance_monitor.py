"""
性能监控工具
用于监控训练过程中的GPU利用率、吞吐量等指标
"""

import time
from typing import Dict, Optional

import jax
import jax.numpy as jp

# ============================================================================================
# ======================================= 性能监控器 ===========================================
# ============================================================================================


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

        step_time = current_time - self.last_step_time
        self.last_step_time = current_time

        self.step_count += 1
        self.env_steps += batch_size

        steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
        env_steps_per_sec = batch_size / step_time if step_time > 0 else 0.0

        total_time = current_time - self.start_time

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


# ============================================================================================
# ===================================== END: 性能监控器 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 辅助函数 ============================================
# ============================================================================================


def benchmark_train_step(
    train_step_fn, train_state, env_state, num_warmup=3, num_iterations=10
):
    """基准测试训练步性能"""
    for _ in range(num_warmup):
        train_state, env_state, _ = train_step_fn(train_state, env_state)

    jax.block_until_ready(train_state)

    times = []
    for _ in range(num_iterations):
        start = time.time()
        train_state, env_state, _ = train_step_fn(train_state, env_state)
        jax.block_until_ready(train_state)
        times.append(time.time() - start)

    times = jp.array(times)
    return {
        "mean_time": float(jp.mean(times)),
        "std_time": float(jp.std(times)),
        "min_time": float(jp.min(times)),
        "max_time": float(jp.max(times)),
        "median_time": float(jp.median(times)),
    }


# --------------------------------------------------------------------------------------------


def estimate_gpu_utilization(
    step_time: float, theoretical_min_time: Optional[float] = None
) -> float:
    """估算GPU利用率"""
    if theoretical_min_time is None:
        return 1.0 / (step_time + 1e-8)
    else:
        return theoretical_min_time / (step_time + 1e-8)


# ============================================================================================
# ===================================== END: 辅助函数 ==========================================
# ============================================================================================
