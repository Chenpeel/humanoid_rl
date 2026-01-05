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
        """初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量
            keep_best: 是否额外保留最佳模型
            metric_name: 用于判断最佳模型的指标名称
            metric_mode: "max"表示指标越大越好，"min"表示越小越好
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # 最佳模型追踪
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.best_step = None

        # 创建子目录
        self.model_dir = self.checkpoint_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        self.best_model_dir = self.checkpoint_dir / "best_model"
        if keep_best:
            self.best_model_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        train_state: TrainState,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> str:
        """保存检查点

        Args:
            train_state: 训练状态
            step: 当前训练步数
            metrics: 训练指标（用于判断最佳模型）
            force: 是否强制保存（忽略max_to_keep限制）

        Returns:
            检查点路径
        """
        # 创建检查点数据
        checkpoint_data = {
            "step": int(train_state.step),
            "env_steps": int(train_state.env_steps),
            "params": train_state.params,
            "opt_state": train_state.opt_state,
            "rng": train_state.rng,
        }

        if metrics is not None:
            checkpoint_data["metrics"] = metrics

        # 保存检查点
        checkpoint_path = self.model_dir / f"checkpoint_{step}"

        # 使用Flax的serialization保存（支持大模型）
        with open(checkpoint_path, "wb") as f:
            f.write(serialization.to_bytes(checkpoint_data))

        # 清理旧检查点（保留最新的max_to_keep个）
        if not force:
            self._cleanup_old_checkpoints()

        # 检查是否是最佳模型
        if self.keep_best and metrics is not None:
            self._maybe_save_best_model(train_state, step, metrics)

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最新的max_to_keep个"""
        checkpoints = sorted(
            self.model_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1])
        )

        # 删除多余的检查点
        if len(checkpoints) > self.max_to_keep:
            for old_ckpt in checkpoints[: -self.max_to_keep]:
                old_ckpt.unlink()

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

            # 保存最佳模型（仅保存参数，不保存优化器状态）
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

            # 保存元信息
            meta_path = self.best_model_dir / "metadata.txt"
            with open(meta_path, "w") as f:
                f.write(f"Best {self.metric_name}: {current_metric:.6f}\n")
                f.write(f"Step: {step}\n")
                f.write(f"Env steps: {train_state.env_steps}\n")

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """加载检查点

        Args:
            checkpoint_path: 检查点路径（如果指定，优先使用）
            step: 检查点步数（如果不指定checkpoint_path，则加载指定步数的检查点）

        Returns:
            检查点数据字典
        """
        # 确定要加载的检查点路径
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
        elif step is not None:
            ckpt_path = self.model_dir / f"checkpoint_{step}"
        else:
            # 加载最新的检查点
            checkpoints = sorted(
                self.model_dir.glob("checkpoint_*"),
                key=lambda p: int(p.name.split("_")[1]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"未找到检查点: {self.model_dir}")
            ckpt_path = checkpoints[-1]

        # 加载检查点
        with open(ckpt_path, "rb") as f:
            checkpoint_data = serialization.from_bytes(None, f.read())

        return checkpoint_data

    def load_best_model(self) -> Dict[str, Any]:
        """加载最佳模型

        Returns:
            最佳模型数据字典
        """
        best_model_path = self.best_model_dir / "best_model"

        if not best_model_path.exists():
            raise FileNotFoundError(f"未找到最佳模型: {best_model_path}")

        with open(best_model_path, "rb") as f:
            best_model_data = serialization.from_bytes(None, f.read())

        return best_model_data

    def restore_train_state(
        self,
        checkpoint_data: Dict[str, Any],
        optimizer: Any,
    ) -> TrainState:
        """从检查点数据恢复训练状态

        Args:
            checkpoint_data: 检查点数据
            optimizer: 优化器实例（用于重建opt_state）

        Returns:
            恢复的TrainState
        """
        return TrainState(
            step=checkpoint_data["step"],
            env_steps=checkpoint_data["env_steps"],
            params=checkpoint_data["params"],
            opt_state=checkpoint_data["opt_state"],
            rng=checkpoint_data["rng"],
        )

    def export_for_inference(
        self,
        params: Any,
        export_path: str,
        format: str = "msgpack",
    ):
        """导出模型参数供推理使用（方便转换为ONNX/TF）

        Args:
            params: 模型参数
            export_path: 导出路径
            format: 导出格式 ("msgpack", "pickle", "flax")
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "msgpack":
            # MessagePack格式（紧凑，跨语言）
            params_bytes = serialization.to_bytes(params)
            with open(export_path, "wb") as f:
                f.write(params_bytes)

        elif format == "pickle":
            # Pickle格式（Python专用）
            with open(export_path, "wb") as f:
                pickle.dump(params, f)

        elif format == "flax":
            # Flax原生格式
            checkpoints.save_checkpoint(
                ckpt_dir=str(export_path.parent),
                target=params,
                step=0,
                prefix=export_path.stem,
            )

        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def list_checkpoints(self) -> list:
        """列出所有可用的检查点

        Returns:
            检查点列表 [(step, path), ...]
        """
        checkpoints = sorted(
            self.model_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1])
        )

        return [(int(ckpt.name.split("_")[1]), str(ckpt)) for ckpt in checkpoints]

    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """获取最佳模型信息

        Returns:
            最佳模型元信息，如果不存在返回None
        """
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


def create_checkpoint_manager(
    log_dir: str,
    max_to_keep: int = 5,
    keep_best: bool = True,
    metric_name: str = "mean_reward",
    metric_mode: str = "max",
) -> CheckpointManager:
    """创建检查点管理器（工厂函数）

    Args:
        log_dir: 日志目录
        max_to_keep: 最多保留的检查点数量
        keep_best: 是否额外保留最佳模型
        metric_name: 用于判断最佳模型的指标名称
        metric_mode: "max"表示指标越大越好，"min"表示越小越好

    Returns:
        CheckpointManager实例
    """
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep,
        keep_best=keep_best,
        metric_name=metric_name,
        metric_mode=metric_mode,
    )
