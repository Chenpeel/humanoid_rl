"""
模型检查点管理系统
支持保存/加载训练状态，方便导出为ONNX/TF等格式
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import msgpack
from flax import serialization
from flax.training import checkpoints

from ..training.train_state import TrainState

# ============================================================================================
# ======================================= 检查点管理器 ========================================
# ============================================================================================


class CheckpointManager:
    """检查点管理器

    功能：
    1. 定期保存训练检查点
    2. 保存最佳模型
    3. 支持恢复训练
    4. 提供导出接口（ONNX/TF）
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        keep_best: bool = True,
        metric_name: str = "mean_reward",
        metric_mode: str = "max",
    ):
        """初始化检查点管理器"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.best_step = None

        self.model_dir = self.checkpoint_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        self.best_model_dir = self.checkpoint_dir / "best_model"
        if keep_best:
            self.best_model_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------------------------------

    def save_checkpoint(
        self,
        train_state: TrainState,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> str:
        """保存检查点"""
        checkpoint_data = {
            "step": int(train_state.step),
            "env_steps": int(train_state.env_steps),
            "params": train_state.params,
            "opt_state": train_state.opt_state,
            "rng": train_state.rng,
        }

        if metrics is not None:
            checkpoint_data["metrics"] = metrics

        checkpoint_path = self.model_dir / f"checkpoint_{step}"

        with open(checkpoint_path, "wb") as f:
            f.write(serialization.to_bytes(checkpoint_data))

        if not force:
            self._cleanup_old_checkpoints()

        if self.keep_best and metrics is not None:
            self._maybe_save_best_model(train_state, step, metrics)

        return str(checkpoint_path)

    # --------------------------------------------------------------------------------------------

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最新的max_to_keep个"""
        checkpoints = sorted(
            self.model_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1])
        )

        if len(checkpoints) > self.max_to_keep:
            for old_ckpt in checkpoints[: -self.max_to_keep]:
                old_ckpt.unlink()

    # --------------------------------------------------------------------------------------------

    def _maybe_save_best_model(
        self, train_state: TrainState, step: int, metrics: Dict[str, float]
    ):
        """如果是最佳模型则保存"""
        if self.metric_name not in metrics:
            return

        current_metric = metrics[self.metric_name]
        is_best = False

        if self.metric_mode == "max":
            is_best = current_metric > self.best_metric
        else:
            is_best = current_metric < self.best_metric

        if is_best:
            self.best_metric = current_metric
            self.best_step = step

            best_model_data = {
                "step": int(train_state.step),
                "env_steps": int(train_state.env_steps),
                "params": train_state.params,
                "metrics": metrics,
                "metric_name": self.metric_name,
                "metric_value": float(current_metric),
            }

            best_model_path = self.best_model_dir / "best_model"
            with open(best_model_path, "wb") as f:
                f.write(serialization.to_bytes(best_model_data))

            meta_path = self.best_model_dir / "metadata.txt"
            with open(meta_path, "w") as f:
                f.write(f"Best {self.metric_name}: {current_metric:.6f}\n")
                f.write(f"Step: {step}\n")
                f.write(f"Env steps: {train_state.env_steps}\n")

    # --------------------------------------------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """加载检查点"""
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
        elif step is not None:
            ckpt_path = self.model_dir / f"checkpoint_{step}"
        else:
            checkpoints = sorted(
                self.model_dir.glob("checkpoint_*"),
                key=lambda p: int(p.name.split("_")[1]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"未找到检查点: {self.model_dir}")
            ckpt_path = checkpoints[-1]

        with open(ckpt_path, "rb") as f:
            checkpoint_data = serialization.from_bytes(None, f.read())

        return checkpoint_data

    # --------------------------------------------------------------------------------------------

    def load_best_model(self) -> Dict[str, Any]:
        """加载最佳模型"""
        best_model_path = self.best_model_dir / "best_model"

        if not best_model_path.exists():
            raise FileNotFoundError(f"未找到最佳模型: {best_model_path}")

        with open(best_model_path, "rb") as f:
            best_model_data = serialization.from_bytes(None, f.read())

        return best_model_data

    # --------------------------------------------------------------------------------------------

    def restore_train_state(
        self,
        checkpoint_data: Dict[str, Any],
        optimizer: Any,
    ) -> TrainState:
        """从检查点数据恢复训练状态"""
        return TrainState(
            step=checkpoint_data["step"],
            env_steps=checkpoint_data["env_steps"],
            params=checkpoint_data["params"],
            opt_state=checkpoint_data["opt_state"],
            rng=checkpoint_data["rng"],
        )

    # --------------------------------------------------------------------------------------------

    def export_for_inference(
        self,
        params: Any,
        export_path: str,
        format: str = "msgpack",
    ):
        """导出模型参数供推理使用"""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "msgpack":
            params_bytes = serialization.to_bytes(params)
            with open(export_path, "wb") as f:
                f.write(params_bytes)

        elif format == "pickle":
            with open(export_path, "wb") as f:
                pickle.dump(params, f)

        elif format == "flax":
            checkpoints.save_checkpoint(
                ckpt_dir=str(export_path.parent),
                target=params,
                step=0,
                prefix=export_path.stem,
            )

        else:
            raise ValueError(f"不支持的导出格式: {format}")

    # --------------------------------------------------------------------------------------------

    def list_checkpoints(self) -> list:
        """列出所有可用的检查点"""
        checkpoints = sorted(
            self.model_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1])
        )
        return [(int(ckpt.name.split("_")[1]), str(ckpt)) for ckpt in checkpoints]

    # --------------------------------------------------------------------------------------------

    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """获取最佳模型信息"""
        meta_path = self.best_model_dir / "metadata.txt"

        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            lines = f.readlines()

        info = {}
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                info[key.strip()] = value.strip()

        return info


# ============================================================================================
# ===================================== END: 检查点管理器 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 便捷函数 ============================================
# ============================================================================================


def create_checkpoint_manager(
    log_dir: str,
    max_to_keep: int = 5,
    keep_best: bool = True,
    metric_name: str = "mean_reward",
    metric_mode: str = "max",
) -> CheckpointManager:
    """创建检查点管理器（工厂函数）"""
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep,
        keep_best=keep_best,
        metric_name=metric_name,
        metric_mode=metric_mode,
    )


# ============================================================================================
# ===================================== END: 便捷函数 ==========================================
# ============================================================================================
