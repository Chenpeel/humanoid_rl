"""
模型导出脚本
支持将训练好的JAX模型导出为ONNX、TensorFlow等格式
"""

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl.models.networks import ActorCriticNetwork
from rl.utils.checkpoint import CheckpointManager


def export_to_onnx(
    params,
    network,
    observation_size: int,
    output_path: str,
):
    """导出为ONNX格式

    Args:
        params: 模型参数
        network: 网络模型
        observation_size: 观测维度
        output_path: 输出路径

    Note:
        需要安装: pip install onnx jax2onnx (或 jax2torch + torch.onnx)
    """
    try:
        # 方法1: 使用JAX2ONNX (推荐)
        try:
            from jax.experimental import jax2onnx

            # 创建示例输入
            dummy_obs = jp.zeros((1, observation_size))

            # 定义推理函数（只输出均值，用于部署）
            def inference_fn(obs):
                mean, _, _ = network.apply(params, obs)
                return mean

            # 转换为ONNX
            onnx_model = jax2onnx.convert(
                inference_fn,
                dummy_obs,
            )

            # 保存
            import onnx

            onnx.save(onnx_model, output_path)
            print(f"✓ 模型已导出为ONNX: {output_path}")
            print("  使用JAX2ONNX转换")
            return True

        except ImportError:
            print("警告: jax.experimental.jax2onnx 不可用，尝试使用JAX2Torch...")

        # 方法2: 使用JAX2Torch + PyTorch ONNX
        try:
            import torch
            from jax2torch import jax2torch

            # 创建示例输入
            dummy_obs = jp.zeros((1, observation_size))

            # 定义推理函数
            def inference_fn(obs):
                mean, _, _ = network.apply(params, obs)
                return mean

            # 转换为PyTorch
            torch_fn = jax2torch(inference_fn, dummy_obs)

            # 转换输入为PyTorch
            torch_input = torch.from_numpy(np.array(dummy_obs))

            # 导出为ONNX
            torch.onnx.export(
                torch_fn,
                torch_input,
                output_path,
                input_names=["observation"],
                output_names=["action"],
                dynamic_axes={
                    "observation": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
                opset_version=14,
            )
            print(f"✓ 模型已导出为ONNX: {output_path}")
            print("  使用JAX2Torch + PyTorch ONNX转换")
            return True

        except ImportError as e:
            print(f"错误: 无法导出ONNX - {e}")
            print("请安装: pip install torch jax2torch onnx")
            return False

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def export_to_tensorflow(
    params,
    network,
    observation_size: int,
    output_path: str,
):
    """导出为TensorFlow SavedModel格式

    Args:
        params: 模型参数
        network: 网络模型
        observation_size: 观测维度
        output_path: 输出路径

    Note:
        需要安装: pip install tensorflow jax2tf
    """
    try:
        import tensorflow as tf
        from jax.experimental import jax2tf

        # 创建示例输入
        dummy_obs = jp.zeros((1, observation_size))

        # 定义推理函数
        def inference_fn(obs):
            mean, _, _ = network.apply(params, obs)
            return mean

        # 转换为TF函数
        tf_fn = jax2tf.convert(inference_fn, enable_xla=False)

        # 包装为TF模块
        class PolicyModule(tf.Module):
            def __init__(self):
                super().__init__()
                self.tf_fn = tf_fn

            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=(None, observation_size), dtype=tf.float32)
                ]
            )
            def __call__(self, obs):
                return self.tf_fn(obs)

        # 创建模块并保存
        policy_module = PolicyModule()
        tf.saved_model.save(
            policy_module,
            output_path,
            signatures={"serving_default": policy_module.__call__},
        )

        print(f"✓ 模型已导出为TensorFlow SavedModel: {output_path}")
        print("  使用JAX2TF转换")
        return True

    except ImportError as e:
        print(f"错误: 无法导出TensorFlow - {e}")
        print("请安装: pip install tensorflow jax2tf")
        return False

    except Exception as e:
        print(f"TensorFlow导出失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def export_to_msgpack(
    params,
    output_path: str,
):
    """导出为MessagePack格式（JAX原生参数）

    Args:
        params: 模型参数
        output_path: 输出路径

    Note:
        这是最简单的格式，保存JAX pytree参数，方便后续自定义转换
    """
    try:
        from flax import serialization

        # 序列化参数
        params_bytes = serialization.to_bytes(params)

        # 保存
        with open(output_path, "wb") as f:
            f.write(params_bytes)

        print(f"✓ 参数已导出为MessagePack: {output_path}")
        print(f"  文件大小: {len(params_bytes) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"MessagePack导出失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="导出训练好的模型")

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="检查点路径（或包含checkpoints的日志目录）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="exported_models", help="导出目录"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorflow", "msgpack", "all"],
        default="all",
        help="导出格式",
    )
    parser.add_argument("--use-best", action="store_true", help="使用最佳模型而非最新检查点")
    parser.add_argument(
        "--observation-size",
        type=int,
        default=54,  # 根据你的环境调整
        help="观测空间维度",
    )
    parser.add_argument(
        "--action-size", type=int, default=10, help="动作空间维度"  # 根据你的机器人调整
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[512, 512, 256], help="隐藏层维度"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("模型导出工具")
    print("=" * 60)

    # 1. 加载检查点
    print("\n[1/3] 加载检查点...")
    checkpoint_path = Path(args.checkpoint_path)

    # 判断是日志目录还是直接的检查点文件
    if checkpoint_path.is_dir():
        # 假设是日志目录，创建CheckpointManager
        ckpt_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_path / "checkpoints")
        )

        if args.use_best:
            print("  加载最佳模型...")
            checkpoint_data = ckpt_manager.load_best_model()
        else:
            print("  加载最新检查点...")
            checkpoint_data = ckpt_manager.load_checkpoint()
    else:
        # 直接加载检查点文件
        from flax import serialization

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = serialization.from_bytes(None, f.read())

    params = checkpoint_data["params"]
    step = checkpoint_data.get("step", 0)
    print(f"  ✓ 检查点加载完成 (step={step})")

    # 2. 重建网络结构
    print("\n[2/3] 重建网络结构...")
    network = ActorCriticNetwork(
        action_dim=args.action_size,
        shared_backbone=True,
        hidden_dims=tuple(args.hidden_dims),
    )
    print(f"  ✓ 网络创建完成")
    print(f"    观测维度: {args.observation_size}")
    print(f"    动作维度: {args.action_size}")
    print(f"    隐藏层: {args.hidden_dims}")

    # 3. 导出模型
    print("\n[3/3] 导出模型...")

    formats_to_export = []
    if args.format == "all":
        formats_to_export = ["onnx", "tensorflow", "msgpack"]
    else:
        formats_to_export = [args.format]

    success_count = 0

    for fmt in formats_to_export:
        print(f"\n  导出为 {fmt.upper()}...")

        if fmt == "onnx":
            output_path = output_dir / f"policy_step{step}.onnx"
            if export_to_onnx(params, network, args.observation_size, str(output_path)):
                success_count += 1

        elif fmt == "tensorflow":
            output_path = output_dir / f"policy_step{step}_tf"
            if export_to_tensorflow(
                params, network, args.observation_size, str(output_path)
            ):
                success_count += 1

        elif fmt == "msgpack":
            output_path = output_dir / f"policy_step{step}.msgpack"
            if export_to_msgpack(params, str(output_path)):
                success_count += 1

    # 总结
    print("\n" + "=" * 60)
    print(f"导出完成: {success_count}/{len(formats_to_export)} 成功")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

    # 提供使用建议
    if success_count > 0:
        print("\n使用建议:")
        if "onnx" in formats_to_export:
            print("  ONNX模型:")
            print("    - 可用于ONNX Runtime推理")
            print("    - 跨平台部署（C++, C#, Java等）")
        if "tensorflow" in formats_to_export:
            print("  TensorFlow SavedModel:")
            print("    - 可用于TensorFlow Serving")
            print("    - TensorFlow Lite转换")
        if "msgpack" in formats_to_export:
            print("  MessagePack参数:")
            print("    - 保存原始JAX参数")
            print("    - 自定义转换和部署")


if __name__ == "__main__":
    main()
