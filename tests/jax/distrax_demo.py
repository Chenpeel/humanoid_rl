"""
验证 JAX 概率与JIT/vmap（纯JAX实现，无TensorFlow/TFP/distrax依赖）

测试内容：
1. JAX基础操作
2. 正态分布采样与对数概率（纯JAX）
3. 多维对角协方差正态分布（纯JAX）
4. JIT编译加速
5. vmap向量化
6. 设备信息
"""

import jax
import jax.numpy as jnp
import time

print("=" * 60)
print("JAX 概率/JIT/vmap 验证脚本（纯JAX）")
print("=" * 60)

# 基础工具：正态分布（标量/向量）
LOG2PI = jnp.log(2.0 * jnp.pi)

def normal_sample(key, mean, std, sample_shape=()):
    eps = jax.random.normal(key, shape=sample_shape + jnp.shape(jnp.asarray(mean)))
    return mean + eps * std

def normal_log_prob(x, mean, std):
    z = (x - mean) / std
    return -0.5 * (z * z + LOG2PI) - jnp.log(std)

def mvn_diag_sample(key, mean_vec, std_vec):
    eps = jax.random.normal(key, shape=jnp.shape(mean_vec))
    return mean_vec + eps * std_vec

def mvn_diag_log_prob(x, mean_vec, std_vec):
    return jnp.sum(normal_log_prob(x, mean_vec, std_vec))

# 1. 基础JAX操作
print("\n1. 测试基础JAX操作...")
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
z = x + y
print(f"   x + y = {z}")
print("   ✓ JAX数组操作正常")

# 2. 正态分布（标量）
print("\n2. 测试正态分布（标量）...")
mean = 0.0
std = 1.0
key = jax.random.PRNGKey(0)
samples = normal_sample(key, mean, std, sample_shape=(5,))
print(f"   正态分布样本: {samples}")
log_probs = normal_log_prob(samples, mean, std)
print(f"   对数概率: {log_probs}")
print("   ✓ 标量正态分布正常")

# 3. 多维正态分布（用于RL动作）
print("\n3. 测试多维正态分布（RL常用）...")
mean_vec = jnp.zeros(12)  # 假设12个关节
std_vec = jnp.ones(12) * 0.5
action_sample = mvn_diag_sample(key, mean_vec, std_vec)
print(f"   动作样本shape: {action_sample.shape}")
print(f"   动作样本: {action_sample[:4]}... (前4个)")
action_log_prob = mvn_diag_log_prob(action_sample, mean_vec, std_vec)
print(f"   动作对数概率: {action_log_prob}")
print("   ✓ 多维分布正常")

# 4. JIT编译
print("\n4. 测试JIT编译...")

def slow_function(x):
    result = x
    for _ in range(100):
        result = jnp.sin(result) + jnp.cos(result)
    return result

@jax.jit
def fast_function(x):
    result = x
    for _ in range(100):
        result = jnp.sin(result) + jnp.cos(result)
    return result

x_test = jnp.ones((1000,))
_ = fast_function(x_test)  # 预热JIT

start = time.time()
_ = slow_function(x_test)
time_slow = time.time() - start

start = time.time()
_ = fast_function(x_test)
time_fast = time.time() - start

print(f"   未JIT: {time_slow*1000:.2f}ms")
print(f"   JIT:   {time_fast*1000:.2f}ms")
print(f"   加速比: {time_slow/time_fast:.1f}x")
print("   ✓ JIT编译正常")

# 5. vmap向量化
print("\n5. 测试vmap向量化...")

def single_sample(key):
    return normal_sample(key, 0.0, 1.0)

batch_sample = jax.vmap(single_sample)
keys = jax.random.split(jax.random.PRNGKey(42), 10)
samples_batch = batch_sample(keys)
print(f"   批量样本 (10个): {samples_batch}")
print("   ✓ vmap向量化正常")

# 6. 设备信息
print("\n6. 设备信息...")
devices = jax.devices()
print(f"   可用设备: {devices}")
print(f"   设备类型: {devices[0].platform}")
print("   ✓ 设备信息获取正常")

print("\n" + "=" * 60)
print("✓ 所有测试通过！JAX环境配置正确（无TF/TFP/distrax）。")
print("=" * 60)
