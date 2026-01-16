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

from checkpoint_compat import CheckpointFormat, get_default_xax_task_cls, resolve_checkpoint

# ============================================================================================
# ======================================= 导出逻辑 ============================================
# ============================================================================================


def _sorted_dense_layer_names(module_params) -> list[str]:
    dense_names = [name for name in module_params.keys() if name.startswith("Dense_")]

    def dense_index(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9

    dense_names.sort(key=lambda n: (dense_index(n), n))
    return dense_names


def _ensure_flax_variables(params):
    if hasattr(params, "keys") and "params" in params:
        return params
    return {"params": params}


def _unwrap_flax_params(variables):
    if hasattr(variables, "keys") and "params" in variables:
        maybe_params = variables["params"]
        if hasattr(maybe_params, "keys"):
            return maybe_params
    return variables


def _infer_network_config_from_params(params):
    module_params = _unwrap_flax_params(_ensure_flax_variables(params))

    if hasattr(module_params, "keys") and "backbone" in module_params:
        shared_backbone = True
        backbone_params = module_params.get("backbone")
    else:
        shared_backbone = False
        backbone_params = module_params.get("actor_backbone")

    head_params = module_params.get("actor_head") if hasattr(module_params, "get") else None

    if backbone_params is None or head_params is None:
        raise ValueError("参数结构不匹配：未找到 backbone/actor_backbone 或 actor_head 参数。")

    dense_layer_names = _sorted_dense_layer_names(backbone_params)
    if not dense_layer_names:
        raise ValueError("参数结构不匹配：backbone 内未找到 Dense_* 层。")

    observation_size = int(np.asarray(backbone_params[dense_layer_names[0]]["kernel"]).shape[0])
    hidden_dims = []
    prev_out = observation_size

    for layer_name in dense_layer_names:
        kernel = np.asarray(backbone_params[layer_name]["kernel"])
        in_dim, out_dim = int(kernel.shape[0]), int(kernel.shape[1])
        if in_dim != prev_out:
            raise ValueError(
                f"backbone 层维度不一致: 期望 in_dim={prev_out}, 实际 kernel={kernel.shape}"
            )
        hidden_dims.append(out_dim)
        prev_out = out_dim

    head_kernel = np.asarray(head_params["kernel"])
    if int(head_kernel.shape[0]) != prev_out:
        raise ValueError(
            f"actor_head 层维度不一致: 期望 in_dim={prev_out}, 实际 kernel={head_kernel.shape}"
        )

    action_size = int(head_kernel.shape[1])
    return {
        "shared_backbone": shared_backbone,
        "observation_size": observation_size,
        "action_size": action_size,
        "hidden_dims": tuple(hidden_dims),
    }


def export_to_onnx_manual_mlp(
    params,
    network,
    observation_size: int,
    output_path: str,
):
    """手工构建ONNX图（仅支持本项目的 MLP + Dense(mean) 策略头）。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    module_params = _unwrap_flax_params(_ensure_flax_variables(params))

    if getattr(network, "shared_backbone", True):
        backbone_params = module_params.get("backbone")
        head_params = module_params.get("actor_head")
    else:
        backbone_params = module_params.get("actor_backbone")
        head_params = module_params.get("actor_head")

    if backbone_params is None or head_params is None:
        raise ValueError("参数结构不匹配：未找到 backbone / actor_head 参数。")

    dense_layer_names = _sorted_dense_layer_names(backbone_params)
    if not dense_layer_names:
        raise ValueError("参数结构不匹配：backbone 内未找到 Dense_* 层。")

    initializers = []
    nodes = []

    x_name = "observation"
    inferred_obs_size = int(np.asarray(backbone_params[dense_layer_names[0]]["kernel"]).shape[0])
    current_in_dim = inferred_obs_size

    for i, layer_name in enumerate(dense_layer_names):
        layer_params = backbone_params[layer_name]
        kernel = np.asarray(layer_params["kernel"], dtype=np.float32)
        bias = np.asarray(layer_params["bias"], dtype=np.float32)

        if kernel.shape[0] != current_in_dim:
            raise ValueError(
                f"backbone 层维度不一致: 期望 in_dim={current_in_dim}, 实际 kernel={kernel.shape}"
            )

        w_name = f"backbone_W{i}"
        b_name = f"backbone_b{i}"
        mm_name = f"backbone_mm{i}"
        z_name = f"backbone_z{i}"
        a_name = f"backbone_a{i}"

        initializers.append(numpy_helper.from_array(kernel, name=w_name))
        initializers.append(numpy_helper.from_array(bias, name=b_name))

        nodes.append(
            helper.make_node("MatMul", inputs=[x_name, w_name], outputs=[mm_name])
        )
        nodes.append(helper.make_node("Add", inputs=[mm_name, b_name], outputs=[z_name]))

        is_last_backbone_layer = i == (len(dense_layer_names) - 1)
        if not is_last_backbone_layer:
            nodes.append(
                helper.make_node("Tanh", inputs=[z_name], outputs=[a_name])
            )
            x_name = a_name
        else:
            x_name = z_name

        current_in_dim = int(kernel.shape[1])

    head_kernel = np.asarray(head_params["kernel"], dtype=np.float32)
    head_bias = np.asarray(head_params["bias"], dtype=np.float32)

    if head_kernel.shape[0] != current_in_dim:
        raise ValueError(
            f"actor_head 层维度不一致: 期望 in_dim={current_in_dim}, 实际 kernel={head_kernel.shape}"
        )

    head_w_name = "actor_head_W"
    head_b_name = "actor_head_b"
    head_mm_name = "actor_head_mm"
    output_name = "action"

    initializers.append(numpy_helper.from_array(head_kernel, name=head_w_name))
    initializers.append(numpy_helper.from_array(head_bias, name=head_b_name))

    nodes.append(
        helper.make_node("MatMul", inputs=[x_name, head_w_name], outputs=[head_mm_name])
    )
    nodes.append(
        helper.make_node("Add", inputs=[head_mm_name, head_b_name], outputs=[output_name])
    )

    action_dim = int(head_kernel.shape[1])

    graph = helper.make_graph(
        nodes=nodes,
        name="policy_mlp",
        inputs=[
            helper.make_tensor_value_info(
                "observation", TensorProto.FLOAT, ["batch", inferred_obs_size]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "action", TensorProto.FLOAT, ["batch", action_dim]
            )
        ],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="jrl.export_model",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    # onnx==1.20+ 默认写入 IR_VERSION=13，但部分 onnxruntime 目前只支持到 11。
    if getattr(model, "ir_version", 0) > 11:
        model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✓ 模型已导出为ONNX: {output_path}")
    print("  使用手工ONNX图（MLP+Dense）导出（无需torch/tensorflow）")
    return True


def export_to_onnx(
    params,
    network,
    observation_size: int,
    output_path: str,
):
    """导出为ONNX格式"""
    try:
        onnx_available = True
        try:
            import onnx  # noqa: F401
        except ImportError:
            onnx_available = False

        # 方法1: 使用 JAX2ONNX（如果可用）
        if onnx_available:
            try:
                import onnx
                from jax.experimental import jax2onnx

                dummy_obs = jp.zeros((1, observation_size))

                def inference_fn(obs):
                    mean, _, _ = network.apply(_ensure_flax_variables(params), obs)
                    return mean

                onnx_model = jax2onnx.convert(
                    inference_fn,
                    dummy_obs,
                )

                onnx.save(onnx_model, output_path)
                print(f"✓ 模型已导出为ONNX: {output_path}")
                print("  使用JAX2ONNX转换")
                return True

            except ImportError:
                print("警告: 未找到可用的 JAX->ONNX 转换器（jax2onnx）。")

            # 方法2: 手工构建ONNX图（不依赖torch）
            try:
                return export_to_onnx_manual_mlp(
                    params, network, observation_size, output_path
                )
            except Exception as e:
                print(f"警告: 手工ONNX导出不可用 - {e}")
        else:
            print("警告: 未安装 onnx，无法进行手工ONNX导出。")
            print("请安装: pip install onnx")

        # 方法3: 使用JAX2Torch + PyTorch ONNX
        try:
            import torch
            from jax2torch import jax2torch

            dummy_obs = jp.zeros((1, observation_size))

            def inference_fn(obs):
                mean, _, _ = network.apply(_ensure_flax_variables(params), obs)
                return mean

            torch_fn = jax2torch(inference_fn, dummy_obs)
            torch_input = torch.from_numpy(np.array(dummy_obs))

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

        except (ImportError, OSError) as e:
            print(f"错误: 无法导出ONNX - {e}")
            if "libnvshmem_host.so" in str(e):
                print(
                    "提示: 你的 torch 可能是 CUDA 版本，但运行环境缺少 NVSHMEM 动态库。"
                )
                print(
                    "      若只需要导出/CPU 推理，建议安装 CPU 版 PyTorch（或在有完整 CUDA 环境的机器上导出）。"
                )
                print("      例如: pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu")
            print("请安装: pip install onnx jax2torch torch")
            return False

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        import traceback

        traceback.print_exc()
        return False


# --------------------------------------------------------------------------------------------


def export_to_tensorflow(
    params,
    network,
    observation_size: int,
    output_path: str,
):
    """导出为TensorFlow SavedModel格式"""
    try:
        import tensorflow as tf
        from jax.experimental import jax2tf

        dummy_obs = jp.zeros((1, observation_size))

        def inference_fn(obs):
            mean, _, _ = network.apply(_ensure_flax_variables(params), obs)
            return mean

        tf_fn = jax2tf.convert(inference_fn, enable_xla=False)

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
        print("请安装: pip install tensorflow")
        print("提示: `jax2tf` 属于 `jax.experimental`，不需要单独 pip 安装。")
        return False

    except Exception as e:
        print(f"TensorFlow导出失败: {e}")
        import traceback

        traceback.print_exc()
        return False


# --------------------------------------------------------------------------------------------


def export_to_msgpack(
    params,
    output_path: str,
):
    """导出为MessagePack格式（JAX原生参数）"""
    try:
        from flax import serialization

        params_bytes = serialization.to_bytes(params)

        with open(output_path, "wb") as f:
            f.write(params_bytes)

        print(f"✓ 参数已导出为MessagePack: {output_path}")
        print(f"  文件大小: {len(params_bytes) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"MessagePack导出失败: {e}")
        return False


# ============================================================================================
# =============================== xax(ksim) -> ONNX 导出 ======================================
# ============================================================================================


def _extract_eqx_mlp_layers(mlp) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = getattr(mlp, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("不支持的 Equinox MLP 结构：未找到 mlp.layers")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i, layer in enumerate(layers):
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        if weight is None or bias is None:
            raise ValueError(f"不支持的 layer[{i}]：缺少 weight/bias")
        out.append((np.asarray(weight), np.asarray(bias)))
    return out


def export_to_onnx_manual_eqx_mlp(
    *,
    layers: list[tuple[np.ndarray, np.ndarray]],
    output_path: str,
    input_name: str = "observation",
    output_name: str = "action",
) -> bool:
    """手工构建 ONNX 图：Equinox MLP(Linear+Tanh) -> action mean。

    约定：
    - 每层 Linear 采用 weight(out,in), bias(out,)（Equinox 默认）
    - ONNX 里使用 MatMul(input, W.T) + Add(bias)，隐藏层用 Tanh，最后一层不激活
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        print(f"错误: 导出 xax ONNX 需要 onnx - {e}")
        print("请安装: pip install onnx")
        return False

    if not layers:
        raise ValueError("layers 不能为空")

    in_dim = int(layers[0][0].shape[1])
    out_dim = int(layers[-1][0].shape[0])

    nodes = []
    initializers = []

    x_name = input_name
    current_in = in_dim

    for i, (w_out_in, b_out) in enumerate(layers):
        is_last = i == (len(layers) - 1)
        if w_out_in.ndim != 2:
            raise ValueError(f"layer[{i}] weight 维度错误: {w_out_in.shape}")
        if b_out.ndim != 1:
            raise ValueError(f"layer[{i}] bias 维度错误: {b_out.shape}")

        out_size, in_size = int(w_out_in.shape[0]), int(w_out_in.shape[1])
        if in_size != current_in:
            raise ValueError(f"layer[{i}] in_dim 不匹配: 期望 {current_in}, 实际 {in_size}")
        if int(b_out.shape[0]) != out_size:
            raise ValueError(f"layer[{i}] bias 不匹配: 期望 {out_size}, 实际 {b_out.shape[0]}")

        w_name = f"W{i}"
        b_name = f"b{i}"
        mm_name = f"mm{i}"
        z_name = output_name if is_last else f"z{i}"
        a_name = f"a{i}"

        # ONNX MatMul 需要 (in,out)；Equinox weight 是 (out,in)
        w_in_out = np.asarray(w_out_in.T, dtype=np.float32)
        b_out_f = np.asarray(b_out, dtype=np.float32)

        initializers.append(numpy_helper.from_array(w_in_out, name=w_name))
        initializers.append(numpy_helper.from_array(b_out_f, name=b_name))

        nodes.append(helper.make_node("MatMul", inputs=[x_name, w_name], outputs=[mm_name]))
        nodes.append(helper.make_node("Add", inputs=[mm_name, b_name], outputs=[z_name]))

        if not is_last:
            nodes.append(helper.make_node("Tanh", inputs=[z_name], outputs=[a_name]))
            x_name = a_name
        else:
            x_name = z_name

        current_in = out_size

    graph = helper.make_graph(
        nodes=nodes,
        name="xax_policy_mlp",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", in_dim])],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch", out_dim])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="jrl.export_model.xax",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    # onnx==1.20+ 默认写入 IR_VERSION=13，但部分 onnxruntime 目前只支持到 11。
    if getattr(model, "ir_version", 0) > 11:
        model.ir_version = 11
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✓ xax 模型已导出为ONNX: {output_path}")
    print(f"  输入: {input_name} shape=[batch,{in_dim}] -> 输出: {output_name} shape=[batch,{out_dim}]")
    return True


def export_xax_checkpoint_to_onnx(
    ckpt_path: Path,
    output_path: Path,
) -> bool:
    try:
        from xax.task.mixins.checkpointing import load_ckpt
        from ksim.task.rl import InitParams as KInitParams
    except ModuleNotFoundError as e:
        print(f"错误: 导出 xax checkpoint 需要 ksim/xax - {e}")
        print("建议使用包含 ksim/xax 的 Python（例如项目 .venv）运行 export.py")
        return False

    task_cls = get_default_xax_task_cls()

    cfg = load_ckpt(ckpt_path, part="config")
    state = load_ckpt(ckpt_path, part="state")

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 导出不需要 multiprocessing；同时把 cache_dir 固定到项目内，避免权限/环境差异。
    overrides = {
        "disable_multiprocessing": True,
        "compile": {"cache_dir": str(cache_dir)},
    }
    config = task_cls.get_config(cfg, overrides, use_cli=False)
    task = task_cls(config)

    mj_model = task.get_mujoco_model()

    rng = jax.random.PRNGKey(0)
    model_template = task.get_model(KInitParams(key=rng, physics_model=mj_model))
    models = load_ckpt(ckpt_path, part="model", model_templates=[model_template])

    model = models[0]
    actor = getattr(model, "actor", None)
    if actor is None or getattr(actor, "mlp", None) is None:
        raise ValueError("不支持的 xax 模型结构：未找到 model.actor.mlp")

    layers = _extract_eqx_mlp_layers(actor.mlp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"xax ckpt: {ckpt_path} | num_steps={int(state.num_steps)} | num_samples={int(state.num_samples)}")
    return export_to_onnx_manual_eqx_mlp(layers=layers, output_path=str(output_path))


# ============================================================================================
# ===================================== END: 导出逻辑 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 主函数 ==============================================
# ============================================================================================


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("模型导出工具")
    print("=" * 60)

    print("\n[1/3] 加载检查点...")
    resolved = resolve_checkpoint(args.checkpoint_path)
    checkpoint_path = resolved.path

    if resolved.format == CheckpointFormat.XAX_TAR:
        # xax/ksim checkpoint：仅支持导出 policy mean 为 ONNX（当前脚本网络/环境与 JRL 不同）
        if args.format not in ("onnx", "all"):
            print("提示: 检测到 xax checkpoint，目前仅支持 --format onnx（其他格式将跳过）")

        onnx_path = output_dir / "policy_mean_xax.onnx"
        ok = export_xax_checkpoint_to_onnx(checkpoint_path, onnx_path)
        print("=" * 60)
        print(f"导出完成: {'成功' if ok else '失败'}")
        print(f"输出目录: {output_dir.absolute()}")
        print("=" * 60)
        return 0 if ok else 1

    if checkpoint_path.is_dir():
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
        from flax import serialization

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = serialization.from_bytes(None, f.read())

    params = checkpoint_data["params"]
    step = checkpoint_data.get("step", 0)
    print(f"  ✓ 检查点加载完成 (step={step})")

    print("\n[2/3] 重建网络结构...")
    inferred = _infer_network_config_from_params(params)
    if args.observation_size != inferred["observation_size"]:
        print(
            f"  提示: 检测到检查点输入维度为 {inferred['observation_size']}，覆盖参数 --observation-size={args.observation_size}"
        )
    if args.action_size != inferred["action_size"]:
        print(
            f"  提示: 检测到检查点动作维度为 {inferred['action_size']}，覆盖参数 --action-size={args.action_size}"
        )
    if tuple(args.hidden_dims) != inferred["hidden_dims"]:
        print(
            f"  提示: 检测到检查点隐藏层为 {list(inferred['hidden_dims'])}，覆盖参数 --hidden-dims={args.hidden_dims}"
        )

    observation_size = inferred["observation_size"]
    action_size = inferred["action_size"]
    hidden_dims = inferred["hidden_dims"]
    shared_backbone = inferred["shared_backbone"]

    network = ActorCriticNetwork(
        action_dim=action_size,
        shared_backbone=shared_backbone,
        hidden_dims=hidden_dims,
    )

    print(f"  ✓ 网络创建完成")
    print(f"    观测维度: {observation_size}")
    print(f"    动作维度: {action_size}")
    print(f"    隐藏层: {list(hidden_dims)}")
    print(f"    共享Backbone: {shared_backbone}")

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
            if export_to_onnx(params, network, observation_size, str(output_path)):
                success_count += 1

        elif fmt == "tensorflow":
            output_path = output_dir / f"policy_step{step}_tf"
            if export_to_tensorflow(
                params, network, observation_size, str(output_path)
            ):
                success_count += 1

        elif fmt == "msgpack":
            output_path = output_dir / f"policy_step{step}.msgpack"
            if export_to_msgpack(params, str(output_path)):
                success_count += 1

    print("\n" + "=" * 60)
    print(f"导出完成: {success_count}/{len(formats_to_export)} 成功")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

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
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================================================
# ===================================== END: 主函数 ============================================
# ============================================================================================
