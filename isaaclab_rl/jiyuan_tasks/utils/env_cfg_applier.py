"""
环境配置应用器

将 YAML 训练配置（ConfigDict）中的可调参数安全地“落地”到 Isaac Lab 的 env_cfg 对象上。

设计目标:
- 配置驱动：奖励权重、终止阈值、随机化开关等应从 YAML 生效，而不是只停留在文件里。
- 动态键：不硬编码奖励键列表，按配置字典动态遍历并应用。
- 安全降级：配置缺失/键不存在时不报错；打印可读警告，方便排查。

注意:
- 本模块刻意不依赖 Isaac Sim/Isaac Lab 的具体类型，使用 duck-typing，便于做纯 Python 单测。
"""

from __future__ import annotations

from typing import Any, Mapping


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    return None


def _cfg_get(cfg: Any, *path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _reward_section_key(task: str) -> str:
    # YAML 里按“任务语义”分组，而训练脚本 task 名称更偏运行入口。
    if task in {"velocity", "flat", "rough", "curriculum"}:
        return "velocity_tracking"
    return task


def apply_reward_weights(env_cfg: Any, config: Any) -> None:
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return

    task = str(_cfg_get(config, "task", default=""))
    section_key = _reward_section_key(task)
    base_weights = _as_mapping(_cfg_get(config, "rewards", section_key)) or {}
    task_weights = _as_mapping(_cfg_get(config, "rewards", task)) or {}
    if not base_weights and not task_weights:
        return

    # 允许 task 级覆盖（例如 rewards.rough 覆盖 rewards.velocity_tracking 的少数项）
    weights: dict[str, Any] = dict(base_weights)
    weights.update(task_weights)

    for reward_name, weight in weights.items():
        term = getattr(rewards_cfg, reward_name, None)
        if term is None:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = None
            if w not in (0.0, None):
                print(f"[WARN] rewards: env_cfg.rewards 不存在项 `{reward_name}`（配置权重={w}），已跳过")
            continue
        try:
            term.weight = float(weight)
        except (TypeError, ValueError):
            print(f"[WARN] rewards: `{reward_name}` 权重非法: {weight!r}，已跳过")


def _set_path_value(root: Any, path: str, value: Any) -> bool:
    """给对象/字典按点号路径设置值。

    支持:
    - dict: key 存在则深入，不存在时在最后一层创建 key
    - object: getattr/setattr
    """
    keys = path.split(".")
    cur = root
    for key in keys[:-1]:
        if isinstance(cur, Mapping):
            if key not in cur:
                return False
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return False
            cur = getattr(cur, key)

    last = keys[-1]
    if isinstance(cur, Mapping):
        try:
            cur[last] = value
            return True
        except Exception:
            return False
    if hasattr(cur, last):
        try:
            setattr(cur, last, value)
            return True
        except Exception:
            return False
    return False


def apply_reward_params(env_cfg: Any, config: Any) -> None:
    """从 YAML 配置中应用 reward term 的 params 覆盖（动态键）。"""
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return

    task = str(_cfg_get(config, "task", default=""))
    section_key = _reward_section_key(task)
    base_params_cfg = _as_mapping(_cfg_get(config, "reward_params", section_key)) or {}
    task_params_cfg = _as_mapping(_cfg_get(config, "reward_params", task)) or {}
    if not base_params_cfg and not task_params_cfg:
        return

    def _apply_params_block(block: Mapping[str, Any]) -> None:
        for reward_name, param_map in block.items():
            term = getattr(rewards_cfg, reward_name, None)
            if term is None:
                print(f"[WARN] reward_params: env_cfg.rewards 不存在项 `{reward_name}`，已跳过")
                continue

            term_params = getattr(term, "params", None)
            if not isinstance(term_params, dict):
                print(f"[WARN] reward_params: `{reward_name}` 的 term.params 非 dict，已跳过")
                continue

            param_map = _as_mapping(param_map)
            if not param_map:
                continue

            for key, val in param_map.items():
                # 允许两种写法：
                # 1) key 直接是 params 的一级键（例如 threshold）
                # 2) key 为点号路径（例如 sensor_cfg.body_names）
                if "." in key:
                    ok = _set_path_value(term_params, key, val)
                else:
                    term_params[key] = val
                    ok = True
                if not ok:
                    print(f"[WARN] reward_params: `{reward_name}` 无法设置 `{key}`（路径不存在或类型不匹配），已跳过")

    # 叠加语义：先应用 section_key（例如 velocity_tracking），再用 task（例如 rough）覆盖。
    _apply_params_block(base_params_cfg)
    _apply_params_block(task_params_cfg)


def apply_termination_thresholds(env_cfg: Any, config: Any) -> None:
    term_cfg = getattr(env_cfg, "terminations", None)
    if term_cfg is None:
        return

    term_conf = _as_mapping(_cfg_get(config, "terminations"))
    if not term_conf:
        return

    # 常用：fallen 阈值
    fallen_term = getattr(term_cfg, "fallen", None)
    fallen_params = getattr(fallen_term, "params", None)
    if fallen_term is not None and isinstance(fallen_params, dict):
        if "min_height" in term_conf:
            fallen_params["min_height"] = float(term_conf["min_height"])
        if "max_tilt" in term_conf:
            max_tilt = float(term_conf["max_tilt"])
            fallen_params["max_roll"] = max_tilt
            fallen_params["max_pitch"] = max_tilt

    # 常用：速度上限
    vel_term = getattr(term_cfg, "velocity_out_of_bounds", None)
    vel_params = getattr(vel_term, "params", None)
    if vel_term is not None and isinstance(vel_params, dict) and "max_velocity" in term_conf:
        vel_params["max_velocity"] = float(term_conf["max_velocity"])


def apply_domain_randomization(env_cfg: Any, config: Any) -> None:
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is None:
        return

    dr = _as_mapping(_cfg_get(config, "domain_randomization"))
    if not dr:
        return

    # mass
    mass = _as_mapping(dr.get("randomize_mass"))
    if mass is not None:
        if not bool(mass.get("enable", True)):
            if hasattr(events_cfg, "randomize_robot_mass"):
                setattr(events_cfg, "randomize_robot_mass", None)
        else:
            term = getattr(events_cfg, "randomize_robot_mass", None)
            params = getattr(term, "params", None)
            mass_range = mass.get("range")
            operation = mass.get("operation")
            if isinstance(params, dict):
                if isinstance(mass_range, (list, tuple)) and len(mass_range) == 2:
                    params["mass_distribution_params"] = (float(mass_range[0]), float(mass_range[1]))
                if isinstance(operation, str):
                    params["operation"] = operation

    # actuator gains
    gains = _as_mapping(dr.get("randomize_actuator_gains"))
    if gains is not None:
        if not bool(gains.get("enable", True)):
            if hasattr(events_cfg, "randomize_actuator_gains"):
                setattr(events_cfg, "randomize_actuator_gains", None)
        else:
            term = getattr(events_cfg, "randomize_actuator_gains", None)
            params = getattr(term, "params", None)
            if isinstance(params, dict):
                k_range = gains.get("stiffness_range")
                d_range = gains.get("damping_range")
                if isinstance(k_range, (list, tuple)) and len(k_range) == 2:
                    params["stiffness_distribution_params"] = (float(k_range[0]), float(k_range[1]))
                if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
                    params["damping_distribution_params"] = (float(d_range[0]), float(d_range[1]))

    # friction
    friction = _as_mapping(dr.get("randomize_friction"))
    if friction is not None:
        if not bool(friction.get("enable", True)):
            if hasattr(events_cfg, "randomize_joint_friction"):
                setattr(events_cfg, "randomize_joint_friction", None)
        else:
            term = getattr(events_cfg, "randomize_joint_friction", None)
            params = getattr(term, "params", None)
            fr_range = friction.get("range")
            if isinstance(params, dict) and isinstance(fr_range, (list, tuple)) and len(fr_range) == 2:
                params["friction_distribution_params"] = (float(fr_range[0]), float(fr_range[1]))

    # external push
    push = _as_mapping(dr.get("external_push"))
    if push is not None:
        if not bool(push.get("enable", True)):
            if hasattr(events_cfg, "push_robot"):
                setattr(events_cfg, "push_robot", None)
        else:
            term = getattr(events_cfg, "push_robot", None)
            params = getattr(term, "params", None)
            interval_range = push.get("interval_range_s")
            velocity_range = _as_mapping(push.get("velocity_range"))
            if interval_range is not None and hasattr(term, "interval_range_s"):
                if isinstance(interval_range, (list, tuple)) and len(interval_range) == 2:
                    term.interval_range_s = (float(interval_range[0]), float(interval_range[1]))
            if isinstance(params, dict) and velocity_range:
                vx = velocity_range.get("x")
                vy = velocity_range.get("y")
                if isinstance(vx, (list, tuple)) and len(vx) == 2 and isinstance(vy, (list, tuple)) and len(vy) == 2:
                    params["velocity_range"] = {"x": (float(vx[0]), float(vx[1])), "y": (float(vy[0]), float(vy[1]))}


def apply_sim2real_overrides(env_cfg: Any, config: Any) -> None:
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return

    sim2real = _as_mapping(_cfg_get(config, "sim2real"))
    if not sim2real:
        return

    ankle_ws = _as_mapping(sim2real.get("ankle_workspace"))
    if ankle_ws is None:
        return

    term = getattr(rewards_cfg, "ankle_workspace", None)
    if term is None:
        return

    enable = ankle_ws.get("enable")
    if isinstance(enable, bool) and enable is False:
        setattr(rewards_cfg, "ankle_workspace", None)
        return

    margin = ankle_ws.get("margin")
    params = getattr(term, "params", None)
    if isinstance(params, dict) and isinstance(margin, (int, float)):
        params["margin"] = float(margin)


def apply_task_overrides(env_cfg: Any, config: Any) -> None:
    """应用任务相关但不属于 reward/termination/events 的配置项。"""
    task = str(_cfg_get(config, "task", default=""))
    if task not in {"velocity", "flat", "rough", "curriculum"}:
        return

    cmd_cfg = _cfg_get(config, "environment", "velocity_tracking", default=None)
    cmd_cfg = _as_mapping(cmd_cfg)
    if not cmd_cfg:
        return

    base_velocity = _cfg_get(env_cfg, "commands", "base_velocity", default=None)
    ranges = _cfg_get(base_velocity, "ranges", default=None)
    if ranges is None:
        return

    v_range = cmd_cfg.get("target_velocity_range")
    v_y_range = cmd_cfg.get("target_lateral_velocity_range")
    w_range = cmd_cfg.get("target_angular_velocity_range")
    if isinstance(v_range, (list, tuple)) and len(v_range) == 2:
        ranges.lin_vel_x = (float(v_range[0]), float(v_range[1]))
    if isinstance(v_y_range, (list, tuple)) and len(v_y_range) == 2:
        ranges.lin_vel_y = (float(v_y_range[0]), float(v_y_range[1]))
    if isinstance(w_range, (list, tuple)) and len(w_range) == 2:
        ranges.ang_vel_z = (float(w_range[0]), float(w_range[1]))

    # 命令重采样时间
    resample_time = cmd_cfg.get("command_resample_time")
    if isinstance(resample_time, (int, float)) and hasattr(base_velocity, "resampling_time_range"):
        base_velocity.resampling_time_range = (float(resample_time), float(resample_time))


def apply_standing_overrides(env_cfg: Any, config: Any) -> None:
    """站立任务的参数覆盖（主要是 reward 参数）。"""
    task = str(_cfg_get(config, "task", default=""))
    if task != "standing":
        return

    stand_cfg = _as_mapping(_cfg_get(config, "environment", "standing", default=None))
    if not stand_cfg:
        return

    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return

    height_term = getattr(rewards_cfg, "height", None)
    height_params = getattr(height_term, "params", None)
    if height_term is not None and isinstance(height_params, dict):
        if "target_height" in stand_cfg and isinstance(stand_cfg["target_height"], (int, float)):
            height_params["target_height"] = float(stand_cfg["target_height"])
        if "height_tolerance" in stand_cfg and isinstance(stand_cfg["height_tolerance"], (int, float)):
            height_params["tolerance"] = float(stand_cfg["height_tolerance"])


def apply_config_to_env_cfg(env_cfg: Any, config: Any) -> None:
    """一次性应用所有可配置项到 env_cfg。"""
    apply_task_overrides(env_cfg, config)
    apply_standing_overrides(env_cfg, config)
    apply_reward_weights(env_cfg, config)
    apply_reward_params(env_cfg, config)
    apply_termination_thresholds(env_cfg, config)
    apply_domain_randomization(env_cfg, config)
    apply_sim2real_overrides(env_cfg, config)
