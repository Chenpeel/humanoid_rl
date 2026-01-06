"""
MJX环境基类 - 纯JAX/MJX实现
采用纯函数式设计，支持JIT编译和vmap批量并行
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from flax import struct
from mujoco import mjx
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from rich.table import Table

console = Console()


# ============================================================================================
# ======================================= 数据结构定义 =========================================
# ============================================================================================


@struct.dataclass
class EnvState:
    """环境状态数据类 (不可变，JAX友好)

    所有字段都是JAX数组，可以被JIT编译和vmap
    """

    # MJX物理状态
    pipeline_state: Any  # mjx.Data

    # 环境信息
    obs: jax.Array  # 观测向量
    reward: jax.Array  # 标量奖励
    done: jax.Array  # 是否终止 (bool)
    step: jax.Array  # 当前步数

    # 随机数生成器
    rng: jax.Array  # PRNGKey

    # 历史信息 (用于计算action rate等)
    last_action: jax.Array  # 上一步动作

    # 额外信息字典 (可选)
    info: Dict[str, jax.Array] = struct.field(default_factory=dict)

    # 奖励权重 (用于课程学习动态调整)
    reward_weights: Dict[str, float] = struct.field(default_factory=dict)


# ============================================================================================
# ===================================== END: 数据结构定义 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= MJX环境基类 ==========================================
# ============================================================================================


class MJXBaseEnv:
    """MJX环境基类

    纯函数式设计：
    - reset()和step()返回新的EnvState，不修改self
    - 所有计算使用纯JAX函数，可JIT编译
    - 支持vmap进行批量并行
    """

    def __init__(
        self,
        xml_path: str,
        max_steps: int = 1000,
        dt: float = 0.002,
        frame_skip: int = 10,
        verbose: bool = True,
    ):
        """初始化MJX环境

        Args:
            xml_path: MuJoCo XML模型路径
            max_steps: 每个episode最大步数
            dt: 仿真时间步长 (秒)
            frame_skip: 动作重复次数 (控制频率 = 1/(dt*frame_skip))
            verbose: 是否显示加载进度
        """
        self.verbose = verbose
        self.max_steps = max_steps
        self.dt = dt
        self.frame_skip = frame_skip

        # 加载MuJoCo模型 (带进度条)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task("[cyan]加载MuJoCo模型...", total=4)

            # 1. 解析XML路径
            xml_path = self._resolve_xml_path(xml_path)
            progress.update(
                load_task, advance=1, description=f"[cyan]解析路径: {xml_path.name}"
            )

            # 2. 加载MuJoCo模型
            self.mj_model = self._load_mujoco_model(xml_path)
            self.mj_model.opt.timestep = self.dt
            progress.update(load_task, advance=1, description="[cyan]配置仿真参数")

            # 3. 转换为MJX模型
            self.mjx_model = mjx.put_model(self.mj_model)
            progress.update(load_task, advance=1, description="[cyan]转换为MJX模型")

            # 4. 提取模型信息
            self._extract_model_info()
            progress.update(load_task, advance=1, description="[green]✓ 模型加载完成")

        # 显示模型信息
        if self.verbose:
            self._display_model_info()

    # --------------------------------------------------------------------------------------------

    def _resolve_xml_path(self, xml_path: str) -> Path:
        """解析XML路径，支持相对路径和绝对路径"""
        path = Path(xml_path)

        if not path.is_absolute():
            # 尝试从多个位置查找
            candidates = [
                Path.cwd() / xml_path,  # 当前目录
                Path(__file__).parent.parent.parent.parent
                / "assets"
                / xml_path,  # 项目assets目录
                Path(__file__).parent.parent.parent.parent / xml_path,  # 项目根目录
            ]

            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"找不到模型文件: {xml_path}\n"
                    f"尝试过的位置:\n" + "\n".join(f"  - {c}" for c in candidates)
                )

        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        return path.resolve()

    # --------------------------------------------------------------------------------------------

    def _load_mujoco_model(self, xml_path: Path) -> mujoco.MjModel:
        """加载MuJoCo模型

        子类可以重写此方法以添加自定义资源加载逻辑
        """
        return mujoco.MjModel.from_xml_path(str(xml_path))

    # --------------------------------------------------------------------------------------------

    def _extract_model_info(self):
        """提取模型关键信息"""
        model = self.mj_model

        self.nq = model.nq  # 广义坐标数
        self.nv = model.nv  # 自由度数
        self.nu = model.nu  # 执行器数
        self.nbody = model.nbody  # 刚体数
        self.njnt = model.njnt  # 关节数
        self.nsensor = model.nsensor  # 传感器数

        # 提取关节名称
        self.joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(model.njnt)
        ]

        # 提取执行器名称
        self.actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            for i in range(model.nu)
        ]

        # 控制频率
        self.control_dt = self.dt * self.frame_skip
        self.control_freq = 1.0 / self.control_dt

    # --------------------------------------------------------------------------------------------

    def _display_model_info(self):
        """显示模型信息表格"""
        table = Table(title="MuJoCo模型信息", show_header=True, header_style="bold magenta")
        table.add_column("属性", style="cyan", width=20)
        table.add_column("值", style="green")

        table.add_row("广义坐标数 (nq)", str(self.nq))
        table.add_row("自由度数 (nv)", str(self.nv))
        table.add_row("执行器数 (nu)", str(self.nu))
        table.add_row("刚体数 (nbody)", str(self.nbody))
        table.add_row("关节数 (njnt)", str(self.njnt))
        table.add_row("传感器数 (nsensor)", str(self.nsensor))
        table.add_row("仿真时间步 (dt)", f"{self.dt:.4f} s")
        table.add_row("动作重复 (frame_skip)", str(self.frame_skip))
        table.add_row("控制频率", f"{self.control_freq:.1f} Hz")

        console.print(table)
        console.print(
            f"[dim]执行器: {', '.join(self.actuator_names[:5])}{'...' if len(self.actuator_names) > 5 else ''}[/dim]"
        )

    # --------------------------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> EnvState:
        """重置环境 (纯函数)

        Args:
            rng: JAX随机数生成器 PRNGKey

        Returns:
            初始的EnvState
        """
        # 创建初始物理状态
        rng, reset_rng = jax.random.split(rng)
        pipeline_state = self._reset_pipeline(reset_rng)

        # 计算初始观测
        obs = self._get_obs(pipeline_state, jp.zeros(self.nu))

        # 创建初始状态
        state = EnvState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(False),
            step=jp.array(0),
            rng=rng,
            last_action=jp.zeros(self.nu),
            info={},
            reward_weights={},  # 默认空字典，子类可以覆盖
        )

        return state

    # --------------------------------------------------------------------------------------------

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """执行一步 (纯函数)

        Args:
            state: 当前EnvState
            action: 动作向量 (shape: (nu,))

        Returns:
            新的EnvState
        """
        # 裁剪动作到合理范围
        action = jp.clip(action, -1.0, 1.0)

        # 执行物理仿真 (frame_skip步)
        pipeline_state = state.pipeline_state
        for _ in range(self.frame_skip):
            pipeline_state = self._step_pipeline(pipeline_state, action)

        # 计算观测
        obs = self._get_obs(pipeline_state, action)

        # 计算奖励 (支持标量或 (reward, reward_info) 元组)
        reward_result = self._compute_reward(state, action, pipeline_state)
        if isinstance(reward_result, tuple):
            reward, reward_info = reward_result
        else:
            reward = reward_result
            reward_info = {}

        # 检查终止条件
        done = self._is_done(state, pipeline_state)

        # 更新步数
        step = state.step + 1
        done = jp.logical_or(done, step >= self.max_steps)

        # 获取额外信息并合并奖励详情
        info = self._get_info(state, action, pipeline_state)
        if reward_info:
            info.update(reward_info)

        # 创建新状态
        new_state = EnvState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            step=step,
            rng=state.rng,
            last_action=action,
            info=info,
        )

        return new_state

    # --------------------------------------------------------------------------------------------
    # 子类需要实现的方法
    # --------------------------------------------------------------------------------------------

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理仿真状态

        默认实现：从默认qpos/qvel开始
        子类可以重写以添加随机化
        """
        data = mjx.make_data(self.mjx_model)
        return data

    def _step_pipeline(self, pipeline_state: Any, action: jax.Array) -> Any:
        """执行一步物理仿真

        Args:
            pipeline_state: 当前mjx.Data
            action: 动作向量

        Returns:
            新的mjx.Data
        """
        # 设置控制输入
        data = pipeline_state.replace(ctrl=action)

        # 执行仿真步
        data = mjx.step(self.mjx_model, data)

        return data

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测 (子类必须实现)"""
        raise NotImplementedError("子类必须实现_get_obs方法")

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """计算奖励 (子类必须实现)"""
        raise NotImplementedError("子类必须实现_compute_reward方法")

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否终止"""
        return jp.array(False)

    def _get_info(
        self,
        state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Dict[str, jax.Array]:
        """获取额外信息"""
        return {}

    # --------------------------------------------------------------------------------------------
    # 批量操作支持
    # --------------------------------------------------------------------------------------------

    def batch_reset(self, rng: jax.Array, batch_size: int) -> EnvState:
        """批量重置环境 (使用vmap)

        Args:
            rng: 主随机数生成器
            batch_size: 批量大小

        Returns:
            批量EnvState (每个字段的第一维是batch_size)
        """
        rngs = jax.random.split(rng, batch_size)
        return jax.vmap(self.reset)(rngs)

    def batch_step(self, states: EnvState, actions: jax.Array) -> EnvState:
        """批量执行步进 (使用vmap)

        Args:
            states: 批量EnvState
            actions: 批量动作 (shape: (batch_size, nu))

        Returns:
            批量新EnvState
        """
        return jax.vmap(self.step)(states, actions)

    # --------------------------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        """观测空间维度（子类需要设置）"""
        raise NotImplementedError("子类必须设置observation_size属性")

    @property
    def action_size(self) -> int:
        """动作空间维度"""
        return self.nu

    def get_state_dict(self, state: EnvState) -> Dict[str, Any]:
        """将EnvState转换为字典（用于调试）"""
        return {
            "step": int(state.step),
            "reward": float(state.reward),
            "done": bool(state.done),
            "obs_shape": state.obs.shape,
            "qpos": state.pipeline_state.qpos,
            "qvel": state.pipeline_state.qvel,
        }


# ============================================================================================
# ===================================== END: MJX环境基类 =======================================
# ============================================================================================
