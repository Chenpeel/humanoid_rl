"""
NaN检测工具
用于调试PPO训练中的数值稳定性问题
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
from rich.console import Console

console = Console()


# ============================================================================================
# ======================================= 检测函数 ============================================
# ============================================================================================


def check_for_nans(
    info: Dict[str, Any],
    update_idx: int,
    save_state: bool = True,
    save_dir: str = ".nan_debug",
    train_state: Optional[Any] = None,
    env_state: Optional[Any] = None,
) -> bool:
    """检测info字典中的NaN或Inf值

    Args:
        info: 包含训练指标的字典
        update_idx: 当前更新索引
        save_state: 是否保存状态用于事后分析
        save_dir: 保存目录
        train_state: 训练状态（可选）
        env_state: 环境状态（可选）

    Returns:
        是否检测到NaN/Inf
    """
    nan_detected = False
    inf_detected = False

    for key, value in info.items():
        if isinstance(value, (int, float)):
            if jp.isnan(value):
                console.print(
                    f"[red]❌ NaN detected at update {update_idx}, key={key}, value={value}[/red]"
                )
                nan_detected = True
            elif jp.isinf(value):
                console.print(
                    f"[yellow]⚠️  Inf detected at update {update_idx}, key={key}, value={value}[/yellow]"
                )
                inf_detected = True
        elif hasattr(value, "shape"):
            if jp.isnan(value).any():
                console.print(
                    f"[red]❌ NaN detected in array at update {update_idx}, key={key}[/red]"
                )
                console.print(
                    f"   Shape: {value.shape}, NaN count: {jp.isnan(value).sum()}"
                )
                nan_detected = True
            elif jp.isinf(value).any():
                console.print(
                    f"[yellow]⚠️  Inf detected in array at update {update_idx}, key={key}[/yellow]"
                )
                console.print(
                    f"   Shape: {value.shape}, Inf count: {jp.isinf(value).sum()}"
                )
                inf_detected = True

    if nan_detected or inf_detected:
        if save_state:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            state_file = save_path / f"nan_state_update_{update_idx}.pkl"

            try:
                with open(state_file, "wb") as f:
                    pickle.dump(
                        {
                            "update": update_idx,
                            "info": info,
                            "train_state": train_state,
                            "env_state": env_state,
                        },
                        f,
                    )
                console.print(f"[dim]State saved to {state_file}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to save state: {e}[/yellow]")

    return nan_detected or inf_detected


# ============================================================================================
# ===================================== END: 检测函数 ==========================================
# ============================================================================================


# ============================================================================================
# ===================================== 统计日志函数 ==========================================
# ============================================================================================


def log_rollout_stats(
    batch: Any,
    prefix: str = "Rollout",
) -> None:
    """记录rollout统计信息（用于监控数值范围）"""
    console.print(f"\n[cyan]{prefix} Statistics:[/cyan]")

    obs_min = float(batch.obs.min())
    obs_max = float(batch.obs.max())
    obs_mean = float(batch.obs.mean())
    console.print(
        f"  Observations: min={obs_min:.3f}, max={obs_max:.3f}, mean={obs_mean:.3f}"
    )

    action_min = float(batch.actions.min())
    action_max = float(batch.actions.max())
    action_mean = float(batch.actions.mean())
    console.print(
        f"  Actions: min={action_min:.3f}, max={action_max:.3f}, mean={action_mean:.3f}"
    )

    if hasattr(batch, "rewards"):
        reward_min = float(batch.rewards.min())
        reward_max = float(batch.rewards.max())
        reward_mean = float(batch.rewards.mean())
        console.print(
            f"  Rewards: min={reward_min:.3f}, max={reward_max:.3f}, mean={reward_mean:.3f}"
        )

    adv_min = float(batch.advantages.min())
    adv_max = float(batch.advantages.max())
    adv_mean = float(batch.advantages.mean())
    adv_std = float(batch.advantages.std())
    console.print(
        f"  Advantages: min={adv_min:.3f}, max={adv_max:.3f}, "
        f"mean={adv_mean:.3f}, std={adv_std:.3f}"
    )

    ret_min = float(batch.returns.min())
    ret_max = float(batch.returns.max())
    ret_mean = float(batch.returns.mean())
    console.print(
        f"  Returns: min={ret_min:.3f}, max={ret_max:.3f}, mean={ret_mean:.3f}"
    )

    logp_min = float(batch.old_log_probs.min())
    logp_max = float(batch.old_log_probs.max())
    logp_mean = float(batch.old_log_probs.mean())
    console.print(
        f"  Old log_probs: min={logp_min:.3f}, max={logp_max:.3f}, mean={logp_mean:.3f}"
    )

    warnings = []
    if abs(obs_max) > 100 or abs(obs_min) > 100:
        warnings.append("Observations range异常（绝对值 > 100）")
    if abs(action_max) > 10 or abs(action_min) > 10:
        warnings.append("Actions range异常（绝对值 > 10）")
    if adv_std < 1e-6:
        warnings.append("Advantages std过小（< 1e-6），可能导致标准化问题")
    if abs(adv_max) > 1000 or abs(adv_min) > 1000:
        warnings.append("Advantages range异常（绝对值 > 1000）")
    if abs(ret_max) > 1000 or abs(ret_min) > 1000:
        warnings.append("Returns range异常（绝对值 > 1000）")

    if warnings:
        console.print("[yellow]⚠️  Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")


# ============================================================================================
# ===================================== END: 统计日志函数 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 验证函数 ============================================
# ============================================================================================


def validate_ppo_loss_inputs(
    mean: jax.Array,
    log_std: jax.Array,
    values: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
) -> bool:
    """验证PPO损失函数的输入是否有效"""
    checks = {
        "mean": mean,
        "log_std": log_std,
        "values": values,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
    }

    all_valid = True
    for name, array in checks.items():
        if jp.isnan(array).any():
            console.print(f"[red]❌ NaN in {name}[/red]")
            all_valid = False
        if jp.isinf(array).any():
            console.print(f"[red]❌ Inf in {name}[/red]")
            all_valid = False

    return all_valid


# --------------------------------------------------------------------------------------------


def safe_clip_rewards(
    rewards: jax.Array,
    min_val: float = -10.0,
    max_val: float = 10.0,
) -> jax.Array:
    """安全裁剪奖励值"""
    clipped = jp.clip(rewards, min_val, max_val)

    clipped_count = jp.sum((rewards < min_val) | (rewards > max_val))
    if clipped_count > 0:
        console.print(
            f"[yellow]⚠️  {clipped_count} rewards clipped to [{min_val}, {max_val}][/yellow]"
        )
        console.print(
            f"   Original range: [{float(rewards.min()):.3f}, {float(rewards.max()):.3f}]"
        )

    return clipped


# --------------------------------------------------------------------------------------------


def monitor_training_health(
    info: Dict[str, Any],
    update_idx: int,
    thresholds: Optional[Dict[str, tuple]] = None,
) -> None:
    """监控训练健康度（检测异常指标）"""
    if thresholds is None:
        thresholds = {
            "policy_loss": (-10.0, 10.0),
            "value_loss": (0.0, 1000.0),
            "entropy": (0.0, 10.0),
            "approx_kl": (0.0, 1.0),
            "ratio_mean": (0.5, 2.0),
        }

    warnings = []
    for metric, (min_val, max_val) in thresholds.items():
        if metric in info:
            value = info[metric]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    warnings.append(
                        f"{metric}={value:.4f} 超出正常范围 [{min_val}, {max_val}]"
                    )

    if warnings:
        console.print(
            f"\n[yellow]⚠️  Training Health Warning (Update {update_idx}):[/yellow]"
        )
        for warning in warnings:
            console.print(f"  - {warning}")


# ============================================================================================
# ===================================== END: 验证函数 ==========================================
# ============================================================================================
