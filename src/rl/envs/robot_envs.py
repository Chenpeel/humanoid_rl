"""
机器人MJX环境实现
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from mujoco import mjx
from rich.console import Console

from .mjx_base_env import EnvState, MJXBaseEnv

console = Console()


# ============================================================================================
# ======================================= 工具函数 ============================================
# ============================================================================================


def quaternion_multiply(q1, q2):
    """四元数乘法 (q1 * q2)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ============================================================================================
# ===================================== END: 工具函数 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 速度跟踪环境 ==========================================
# ============================================================================================


class VelocityTrackingEnv(MJXBaseEnv):
    """速度跟踪控制环境
    任务：跟踪速度命令（线速度x/y和角速度yaw）
    观测：机器人关节位置、速度、IMU数据等
    动作：关节位置控制目标
    奖励：速度跟踪 + 存活 + 动作平滑
    """

    def __init__(
        self,
        xml_path: str = "../assets/xmls/scene.xml",
        max_steps: int = 1000,
        dt: float = 0.002,
        frame_skip: int = 10,
        verbose: bool = True,
        # 任务参数
        cmd_x_range: tuple = (-0.15, 0.15),
        cmd_y_range: tuple = (-0.2, 0.2),
        cmd_yaw_range: tuple = (-1.0, 1.0),
        # 奖励权重
        reward_weights: Dict[str, float] = None,
    ):
        """初始化机器人环境"""
        # 调用父类初始化（加载模型）
        super().__init__(xml_path, max_steps, dt, frame_skip, verbose)

        # 命令范围
        self.cmd_x_range = cmd_x_range
        self.cmd_y_range = cmd_y_range
        self.cmd_yaw_range = cmd_yaw_range

        # 奖励权重（默认值）
        if reward_weights is None:
            reward_weights = {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.5,
                "alive": 0.1,
                "action_rate": -0.01,
                "torques": -0.0001,
            }
        self.reward_weights = reward_weights

        # 提取关键身体和传感器ID
        self._extract_indices()

        # 设置观测维度
        self._setup_observation_space()

        if verbose:
            console.print(f"[green]✓ 速度跟踪环境初始化完成[/green]")
            console.print(f"  观测维度: {self.observation_size}")
            console.print(f"  动作维度: {self.action_size}")
            console.print(
                f"  命令范围: x={cmd_x_range}, y={cmd_y_range}, yaw={cmd_yaw_range}"
            )

    # --------------------------------------------------------------------------------------------

    def _load_mujoco_model(self, xml_path: Path) -> mujoco.MjModel:
        """加载MuJoCo模型"""
        try:
            return mujoco.MjModel.from_xml_path(str(xml_path))
        except ImportError:
            console.print(
                "[red]错误：无法加载MuJoCo模型。请确保已正确安装MuJoCo和mujoco-python,且路径正确[/red]"
            )
            raise

    # --------------------------------------------------------------------------------------------

    def _extract_indices(self):
        """提取关键索引"""
        model = self.mj_model

        # 找到躯干body（通常是第一个freejoint连接的body）
        self.torso_body_id = 1  # 假设torso是body[1]

        # 找到传感器
        sensor_names = [
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
            for i in range(model.nsensor)
        ]

        # 查找各类传感器（真实硬件对应）
        self.gyro_idx = None
        self.accelerometer_idx = None
        self.magnetometer_idx = None
        self.right_foot_contact_idx = None
        self.left_foot_contact_idx = None
        self.right_foot_force_idx = None
        self.left_foot_force_idx = None

        # 用于调试的仿真专用传感器
        self.base_pos_sim_idx = None
        self.base_quat_sim_idx = None
        self.base_linvel_sim_idx = None
        self.base_angvel_sim_idx = None

        for i, name in enumerate(sensor_names):
            if "imu_gyro" in name.lower():
                self.gyro_idx = i
            elif "imu_accel" in name.lower():
                self.accelerometer_idx = i
            elif "imu_mag" in name.lower():
                self.magnetometer_idx = i
            elif "right_foot_contact" in name.lower() and "force" not in name.lower():
                self.right_foot_contact_idx = i
            elif "left_foot_contact" in name.lower() and "force" not in name.lower():
                self.left_foot_contact_idx = i
            elif "right_foot_force" in name.lower():
                self.right_foot_force_idx = i
            elif "left_foot_force" in name.lower():
                self.left_foot_force_idx = i
            elif "base_pos_sim" in name.lower():
                self.base_pos_sim_idx = i
            elif "base_quat_sim" in name.lower():
                self.base_quat_sim_idx = i
            elif "base_linvel_sim" in name.lower():
                self.base_linvel_sim_idx = i
            elif "base_angvel_sim" in name.lower():
                self.base_angvel_sim_idx = i

        # 找到浮动基座的地址（freejoint）
        self.floating_base_qpos_addr = None
        self.floating_base_qvel_addr = None
        for i in range(model.njnt):
            if model.jnt_type[i] == 0:  # 0 = freejoint
                self.floating_base_qpos_addr = model.jnt_qposadr[i]
                self.floating_base_qvel_addr = model.jnt_dofadr[i]
                break

        # 执行器关节地址
        self.actuator_joint_ids = []
        self.actuator_qpos_indices = []
        self.actuator_qvel_indices = []

        for i in range(model.nu):
            trnid = model.actuator_trnid[i, 0]
            if trnid >= 0:
                joint_id = trnid
                self.actuator_joint_ids.append(joint_id)
                qpos_addr = model.jnt_qposadr[joint_id]
                dof_addr = model.jnt_dofadr[joint_id]
                self.actuator_qpos_indices.append(qpos_addr)
                self.actuator_qvel_indices.append(dof_addr)

        # 默认关节位置
        self.default_qpos = jp.array(model.qpos0)
        try:
            home_key_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key_id >= 0:
                self.default_qpos = jp.array(
                    model.key_qpos[
                        home_key_id * model.nq: (home_key_id + 1) * model.nq
                    ]
                )
        except:
            pass

    # --------------------------------------------------------------------------------------------

    def _setup_observation_space(self):
        """设置观测空间维度"""
        # 观测包括：
        # - 浮动基座姿态: 4 (quaternion)
        # - 浮动基座线速度: 3
        # - 浮动基座角速度: 3
        # - 关节位置: nu
        # - 关节速度: nu
        # - 上一个动作: nu
        # - 命令: 3 (vx, vy, vyaw)
        self._observation_size = 4 + 3 + 3 + self.nu + self.nu + self.nu + 3

    @property
    def observation_size(self) -> int:
        return self._observation_size

    # --------------------------------------------------------------------------------------------

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理状态（带随机化）"""
        data = mjx.make_data(self.mjx_model)
        qpos = jp.array(self.default_qpos).flatten()

        # 随机化浮动基座位置和朝向
        if self.floating_base_qpos_addr is not None:
            rng, key1, key2, key3 = jax.random.split(rng, 4)

            # 随机化xy位置: ±0.05m
            dxy = jax.random.uniform(key1, (2,), minval=-0.05, maxval=0.05)
            base_xy = qpos[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ]
            qpos = qpos.at[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ].set(base_xy + dxy)

            # 随机化yaw角度: ±π
            yaw = jax.random.uniform(key2, minval=-jp.pi, maxval=jp.pi)
            quat_standard = jp.array(
                [jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])

            # Jiyuan 逆校正
            jiyuan_base_rotation = jp.array([0.70710678, 0.70710678, 0.0, 0.0])
            quat_physical = quaternion_multiply(
                jiyuan_base_rotation, quat_standard)

            quat_norm = jp.linalg.norm(quat_physical)
            quat_physical = jp.where(
                quat_norm > 1e-8,
                quat_physical / quat_norm,
                jp.array([0.70710678, 0.70710678, 0.0, 0.0]),
            )
            qpos = qpos.at[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ].set(quat_physical)

        # 随机化关节位置: ±0.1 rad
        rng, key4 = jax.random.split(rng)
        if self.nu > 0:
            joint_noise = jax.random.uniform(
                key4, (self.nu,), minval=-0.1, maxval=0.1)
            indices = jp.array(self.actuator_qpos_indices)
            joint_pos = qpos[indices]
            qpos = qpos.at[indices].set(joint_pos + joint_noise)

        qvel = jp.zeros(self.nv)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        return data

    # --------------------------------------------------------------------------------------------

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测（使用真实硬件可用的传感器）"""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        sensordata = pipeline_state.sensordata

        # 1. 浮动基座姿态
        if self.floating_base_qpos_addr is not None:
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            quat_norm = jp.linalg.norm(base_quat)
            base_quat = jp.where(
                quat_norm > 1e-8,
                base_quat / quat_norm,
                jp.array([1.0, 0.0, 0.0, 0.0]),
            )
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
        else:
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        # 2. 浮动基座角速度
        if self.gyro_idx is not None:
            base_angvel = sensordata[self.gyro_idx: self.gyro_idx + 3]
        elif self.floating_base_qvel_addr is not None:
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_angvel = jp.zeros(3)

        # 3. 浮动基座线速度
        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
        else:
            base_linvel = jp.zeros(3)

        # 4. 关节位置和速度
        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        # 5. 命令
        command = jp.array([0.0, 0.0, 0.0])

        obs = jp.concatenate(
            [
                base_quat,
                base_linvel,
                base_angvel,
                joint_pos,
                joint_vel,
                action,
                command,
            ]
        )
        obs = jp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    # --------------------------------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """计算奖励"""
        command = prev_state.info.get("command", jp.zeros(3))
        qvel = pipeline_state.qvel

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        # 1. 线速度跟踪
        lin_vel_error = jp.sum(jp.square(base_linvel[:2] - command[:2]))
        reward_lin_vel = jp.exp(-lin_vel_error / 0.25)

        # 2. 角速度跟踪
        ang_vel_error = jp.square(base_angvel[2] - command[2])
        reward_ang_vel = jp.exp(-ang_vel_error / 0.25)

        # 3. 存活奖励
        reward_alive = 1.0

        # 4. 动作变化率
        action_rate = jp.sum(jp.square(action - prev_state.last_action))
        cost_action_rate = jp.clip(action_rate, 0.0, 10.0)

        # 5. 扭矩惩罚
        torques = pipeline_state.qfrc_actuator
        torques_squared = jp.sum(jp.square(torques))
        cost_torques = jp.clip(torques_squared, 0.0, 1000.0)

        # 组合
        weighted_lin_vel = self.reward_weights["tracking_lin_vel"] * \
            reward_lin_vel
        weighted_ang_vel = self.reward_weights["tracking_ang_vel"] * \
            reward_ang_vel
        weighted_alive = self.reward_weights["alive"] * reward_alive
        weighted_action_rate = self.reward_weights["action_rate"] * \
            cost_action_rate
        weighted_torques = self.reward_weights["torques"] * cost_torques

        reward = (
            weighted_lin_vel
            + weighted_ang_vel
            + weighted_alive
            + weighted_action_rate
            + weighted_torques
        )

        reward = jp.clip(reward, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

        reward_info = {
            "reward/tracking_lin_vel": weighted_lin_vel,
            "reward/tracking_ang_vel": weighted_ang_vel,
            "reward/alive": weighted_alive,
            "reward/action_rate": weighted_action_rate,
            "reward/torques": weighted_torques,
        }
        return reward, reward_info

    # --------------------------------------------------------------------------------------------

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """执行一步"""
        action = jp.clip(action, -1.0, 1.0)
        pipeline_state = state.pipeline_state
        for _ in range(self.frame_skip):
            pipeline_state = self._step_pipeline(pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)
        reward, reward_info = self._compute_reward(
            state, action, pipeline_state)
        done = self._is_done(state, pipeline_state)
        step = state.step + 1
        done = jp.logical_or(done, step >= self.max_steps)

        info = self._get_info(state, action, pipeline_state)
        info.update(reward_info)

        return EnvState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            step=step,
            rng=state.rng,
            last_action=action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否终止（摔倒检测）"""
        qpos = pipeline_state.qpos
        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            return jp.array(torso_z < 0.15)
        return jp.array(False)

    # --------------------------------------------------------------------------------------------

    def _get_info(
        self,
        state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Dict[str, jax.Array]:
        """获取额外信息"""
        info = {}
        if "command" in state.info:
            info["command"] = state.info["command"]

        base_lin_vel = pipeline_state.qvel[:3]
        base_ang_vel = pipeline_state.qvel[3:6]
        info["actual_velocity"] = jp.array(
            [base_lin_vel[0], base_lin_vel[1], base_ang_vel[2]]
        )
        return info

    # --------------------------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> EnvState:
        """重置环境，初始化完整的info字典以保持pytree结构一致

        关键修复：确保reset返回的info字典结构与step返回的完全一致，
        避免在jax.lax.scan中出现pytree结构不匹配错误。
        """
        state = super().reset(rng)
        rng, cmd_rng = jax.random.split(state.rng)
        command = self._sample_command(cmd_rng)

        # 初始化环境信息
        info = {
            "command": command,
            "actual_velocity": jp.zeros(3),
        }

        # 动态初始化所有reward_weights中定义的奖励键为0.0
        # 这确保了无论配置文件中定义了哪些奖励项，pytree结构都保持一致
        for weight_key in self.reward_weights.keys():
            reward_key = f"reward/{weight_key}"
            info[reward_key] = jp.array(0.0)

        obs = state.obs.at[-3:].set(command)

        return EnvState(
            pipeline_state=state.pipeline_state,
            obs=obs,
            reward=state.reward,
            done=state.done,
            step=state.step,
            rng=rng,
            last_action=state.last_action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """采样速度命令"""
        rng, key1, key2, key3 = jax.random.split(rng, 4)
        cmd_x = jax.random.uniform(
            key1, minval=self.cmd_x_range[0], maxval=self.cmd_x_range[1]
        )
        cmd_y = jax.random.uniform(
            key2, minval=self.cmd_y_range[0], maxval=self.cmd_y_range[1]
        )
        cmd_yaw = jax.random.uniform(
            key3, minval=self.cmd_yaw_range[0], maxval=self.cmd_yaw_range[1]
        )
        return jp.array([cmd_x, cmd_y, cmd_yaw])


def create_velocity_tracking_env(
    xml_path: str = "assets/xmls/scene.xml", **kwargs
) -> VelocityTrackingEnv:
    """创建速度跟踪环境（便捷函数）"""
    return VelocityTrackingEnv(xml_path=xml_path, **kwargs)


# ============================================================================================
# ===================================== END: 速度跟踪环境 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 站立环境 ============================================
# ============================================================================================


class StandingEnv(MJXBaseEnv):
    """站立保持环境

    任务：保持站立平衡，抵抗扰动
    观测：机器人姿态、速度、关节状态
    动作：关节位置/力矩控制
    奖励：使用rewards.standing_rewards模块中的奖励函数
    """

    def __init__(
        self,
        xml_path: str = "assets/xmls/scene.xml",
        max_steps: int = 2000,
        dt: float = 0.002,
        frame_skip: int = 10,
        verbose: bool = True,
        target_height: float = 0.3,
        reward_weights: Dict[str, float] = None,
    ):
        """初始化站立环境"""
        from ..rewards.standing_rewards import (
            DEFAULT_STANDING_REWARD_WEIGHTS, compute_standing_reward)

        self._compute_reward_fn = compute_standing_reward
        default_weights = DEFAULT_STANDING_REWARD_WEIGHTS

        super().__init__(xml_path, max_steps, dt, frame_skip, verbose)

        self.target_height = target_height
        if reward_weights is None:
            reward_weights = default_weights.copy()
        self.reward_weights = reward_weights

        self._extract_indices()
        self._setup_observation_space()

        if verbose:
            console.print(f"[green]✓ 站立环境初始化完成[/green]")
            console.print(f"  观测维度: {self.observation_size}")
            console.print(f"  动作维度: {self.action_size}")
            console.print(f"  目标高度: {target_height}m")

    # --------------------------------------------------------------------------------------------

    def _extract_indices(self) -> None:
        """提取关键索引"""
        model = self.mj_model
        self.floating_base_qpos_addr = None
        self.floating_base_qvel_addr = None
        for i in range(model.njnt):
            if model.jnt_type[i] == 0:
                self.floating_base_qpos_addr = model.jnt_qposadr[i]
                self.floating_base_qvel_addr = model.jnt_dofadr[i]
                break

        self.actuator_qpos_indices = []
        self.actuator_qvel_indices = []
        for i in range(model.nu):
            trnid = model.actuator_trnid[i, 0]
            if trnid >= 0:
                joint_id = trnid
                self.actuator_qpos_indices.append(model.jnt_qposadr[joint_id])
                self.actuator_qvel_indices.append(model.jnt_dofadr[joint_id])

        self.default_qpos = jp.array(model.qpos0)
        try:
            home_key_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key_id >= 0:
                self.default_qpos = jp.array(
                    model.key_qpos[
                        home_key_id * model.nq: (home_key_id + 1) * model.nq
                    ]
                )
        except Exception:
            pass

    # --------------------------------------------------------------------------------------------

    def _setup_observation_space(self) -> None:
        """设置观测空间维度"""
        self._observation_size = 4 + 3 + 3 + self.nu + self.nu + self.nu

    @property
    def observation_size(self) -> int:
        return self._observation_size

    # --------------------------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> EnvState:
        """重置环境，初始化完整的info字典以保持pytree结构一致

        关键修复：确保reset返回的info字典结构与step返回的完全一致，
        避免在jax.lax.scan中出现pytree结构不匹配错误。
        """
        # 调用基类reset方法
        state = super().reset(rng)

        # 初始化info字典（StandingEnv的_get_info返回空字典，所以这里也初始化为空）
        info = {}

        # 动态初始化所有reward_weights中定义的奖励键为0.0
        # 这确保了无论配置文件中定义了哪些奖励项，pytree结构都保持一致
        for weight_key in self.reward_weights.keys():
            reward_key = f"reward/{weight_key}"
            info[reward_key] = jp.array(0.0)

        # 返回新的EnvState
        return EnvState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            step=state.step,
            rng=state.rng,
            last_action=state.last_action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理状态"""
        data = mjx.make_data(self.mjx_model)
        qpos = jp.array(self.default_qpos).flatten()

        if self.floating_base_qpos_addr is not None:
            rng, key1, key2 = jax.random.split(rng, 3)
            dxy = jax.random.uniform(key1, (2,), minval=-0.02, maxval=0.02)
            base_xy = qpos[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ]
            qpos = qpos.at[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ].set(base_xy + dxy)

            yaw = jax.random.uniform(key2, minval=-0.1, maxval=0.1)
            quat_standard = jp.array(
                [jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
            jiyuan_base_rotation = jp.array([0.70710678, 0.70710678, 0.0, 0.0])
            quat_physical = quaternion_multiply(
                jiyuan_base_rotation, quat_standard)
            quat_norm = jp.linalg.norm(quat_physical)
            quat_physical = jp.where(
                quat_norm > 1e-8,
                quat_physical / quat_norm,
                jp.array([0.70710678, 0.70710678, 0.0, 0.0]),
            )
            qpos = qpos.at[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ].set(quat_physical)

        rng, key3 = jax.random.split(rng)
        if self.nu > 0:
            joint_noise = jax.random.uniform(
                key3, (self.nu,), minval=-0.05, maxval=0.05
            )
            indices = jp.array(self.actuator_qpos_indices)
            joint_pos = qpos[indices]
            qpos = qpos.at[indices].set(joint_pos + joint_noise)

        qvel = jp.zeros(self.nv)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        return data

    # --------------------------------------------------------------------------------------------

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测"""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel

        if self.floating_base_qpos_addr is not None:
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            quat_norm = jp.linalg.norm(base_quat)
            base_quat = jp.where(
                quat_norm > 1e-8, base_quat /
                quat_norm, jp.array([1.0, 0.0, 0.0, 0.0])
            )
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
        else:
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        obs = jp.concatenate(
            [base_quat, base_linvel, base_angvel, joint_pos, joint_vel, action]
        )
        obs = jp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    # --------------------------------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """计算站立奖励"""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel

        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
        else:
            torso_z = self.target_height
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        torques = pipeline_state.qfrc_actuator

        reward_params = {
            "torso_z": torso_z,
            "base_quat": base_quat,
            "base_linvel": base_linvel,
            "base_angvel": base_angvel,
            "action": action,
            "last_action": prev_state.last_action,
            "torques": torques,
            "target_height": self.target_height,
            "reward_weights": self.reward_weights,
        }
        return self._compute_reward_fn(**reward_params)

    # --------------------------------------------------------------------------------------------

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """执行一步"""
        action = jp.clip(action, -1.0, 1.0)
        pipeline_state = state.pipeline_state
        for _ in range(self.frame_skip):
            pipeline_state = self._step_pipeline(pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)
        reward, reward_info = self._compute_reward(
            state, action, pipeline_state)
        done = self._is_done(state, pipeline_state)
        step = state.step + 1
        done = jp.logical_or(done, step >= self.max_steps)

        info = self._get_info(state, action, pipeline_state)
        info.update(reward_info)

        return EnvState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            step=step,
            rng=state.rng,
            last_action=action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否摔倒"""
        from ..rewards.standing_rewards import check_standing_termination

        qpos = pipeline_state.qpos
        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
            return check_standing_termination(torso_z, base_quat)
        return jp.array(False)

    # --------------------------------------------------------------------------------------------

    def _get_info(
        self, state: EnvState, action: jax.Array, pipeline_state: Any
    ) -> Dict[str, jax.Array]:
        return {}


def create_standing_env(
    xml_path: str = "assets/xmls/scene.xml", **kwargs
) -> StandingEnv:
    """创建站立环境（便捷函数）"""
    return StandingEnv(xml_path=xml_path, **kwargs)


# ============================================================================================
# ===================================== END: 站立环境 =========================================
# ============================================================================================


# ============================================================================================
# ======================================= 行走环境 ============================================
# ============================================================================================


class WalkingEnv(MJXBaseEnv):
    """行走环境

    任务：向前行走，跟踪速度命令，保持稳定步态
    观测：机器人姿态、速度、关节状态、接触传感器、命令
    动作：关节位置控制
    奖励：使用 walking_rewards 模块
    """

    def __init__(
        self,
        xml_path: str = "assets/xmls/scenes/flat_terrain.xml",
        max_steps: int = 2000,
        dt: float = 0.002,
        frame_skip: int = 10,
        verbose: bool = True,
        # 任务参数
        target_velocity: float = 0.5,
        cmd_x_range: tuple = (-0.2, 0.8),
        cmd_y_range: tuple = (-0.3, 0.3),
        cmd_yaw_range: tuple = (-1.0, 1.0),
        target_height: float = 0.78,
        # 奖励权重
        reward_weights: Dict[str, float] = None,
    ):
        """初始化行走环境"""
        from ..rewards.walking_rewards import DEFAULT_WALKING_REWARD_WEIGHTS

        super().__init__(xml_path, max_steps, dt, frame_skip, verbose)

        self.target_velocity = target_velocity
        self.cmd_x_range = cmd_x_range
        self.cmd_y_range = cmd_y_range
        self.cmd_yaw_range = cmd_yaw_range
        self.target_height = target_height

        if reward_weights is None:
            reward_weights = DEFAULT_WALKING_REWARD_WEIGHTS.copy()
        self.reward_weights = reward_weights

        self._extract_indices()
        self._setup_observation_space()

        if verbose:
            console.print(f"[green]✓ 行走环境初始化完成[/green]")
            console.print(f"  观测维度: {self.observation_size}")
            console.print(f"  动作维度: {self.action_size}")
            console.print(f"  目标速度: {target_velocity}m/s")
            console.print(f"  目标高度: {target_height}m")
            console.print(
                f"  命令范围: x={cmd_x_range}, y={cmd_y_range}, yaw={cmd_yaw_range}"
            )

    # --------------------------------------------------------------------------------------------

    def _extract_indices(self) -> None:
        """提取关键索引"""
        model = self.mj_model
        self.floating_base_qpos_addr = None
        self.floating_base_qvel_addr = None
        for i in range(model.njnt):
            if model.jnt_type[i] == 0:
                self.floating_base_qpos_addr = model.jnt_qposadr[i]
                self.floating_base_qvel_addr = model.jnt_dofadr[i]
                break

        self.actuator_qpos_indices = []
        self.actuator_qvel_indices = []
        self.actuator_joint_ids = []
        for i in range(model.nu):
            trnid = model.actuator_trnid[i, 0]
            if trnid >= 0:
                joint_id = trnid
                self.actuator_joint_ids.append(joint_id)
                self.actuator_qpos_indices.append(model.jnt_qposadr[joint_id])
                self.actuator_qvel_indices.append(model.jnt_dofadr[joint_id])

        sensor_names = [
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
            for i in range(model.nsensor)
        ]

        self.contact_sensor_indices = []
        for i, name in enumerate(sensor_names):
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ["touch", "contact", "force"]):
                if any(foot in name_lower for foot in ["foot", "toe", "feet"]):
                    self.contact_sensor_indices.append(i)

        if not self.contact_sensor_indices and model.nsensor >= 4:
            self.contact_sensor_indices = [0, 1, 2, 3]

        self.right_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link"
        )
        self.left_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link"
        )

        self.default_qpos = jp.array(model.qpos0)
        try:
            home_key_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key_id >= 0:
                self.default_qpos = jp.array(
                    model.key_qpos[
                        home_key_id * model.nq: (home_key_id + 1) * model.nq
                    ]
                )
        except Exception:
            pass

    # --------------------------------------------------------------------------------------------

    def _setup_observation_space(self) -> None:
        """设置观测空间维度"""
        num_contacts = len(self.contact_sensor_indices)
        if num_contacts == 0:
            num_contacts = 4
        self._observation_size = (
            4 + 3 + 3 + self.nu + self.nu + self.nu + 3 + num_contacts
        )

    @property
    def observation_size(self) -> int:
        return self._observation_size

    # --------------------------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> EnvState:
        """重置环境,初始化完整的info字典以保持pytree结构一致

        """
        # 调用基类reset方法
        state = super().reset(rng)

        # 采样速度命令
        rng, cmd_rng = jax.random.split(state.rng)
        command = self._sample_command(cmd_rng)

        # 初始化info字典,包含环境信息
        info = {
            "command": command,
            "actual_velocity": jp.zeros(3),
            "contact_history": jp.zeros((10, 4)),
        }

        # 确保pytree结构保持一致
        for weight_key in self.reward_weights.keys():
            reward_key = f"reward/{weight_key}"
            info[reward_key] = jp.array(0.0)

        # 更新观测中的命令部分
        obs = state.obs.at[
            -3 - len(self.contact_sensor_indices): -len(self.contact_sensor_indices)
        ].set(command)

        # 返回新的EnvState
        return EnvState(
            pipeline_state=state.pipeline_state,
            obs=obs,
            reward=state.reward,
            done=state.done,
            step=state.step,
            rng=rng,
            last_action=state.last_action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理状态"""
        data = mjx.make_data(self.mjx_model)
        qpos = jp.array(self.default_qpos).flatten()

        if self.floating_base_qpos_addr is not None:
            rng, key1, key2 = jax.random.split(rng, 3)
            dxy = jax.random.uniform(key1, (2,), minval=-0.02, maxval=0.02)
            base_xy = qpos[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ]
            qpos = qpos.at[
                self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2
            ].set(base_xy + dxy)

            yaw = jax.random.uniform(key2, minval=-0.1, maxval=0.1)
            quat_standard = jp.array(
                [jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
            jiyuan_base_rotation = jp.array([0.70710678, 0.70710678, 0.0, 0.0])
            quat_physical = quaternion_multiply(
                jiyuan_base_rotation, quat_standard)
            quat_norm = jp.linalg.norm(quat_physical)
            quat_physical = jp.where(
                quat_norm > 1e-8,
                quat_physical / quat_norm,
                jp.array([0.70710678, 0.70710678, 0.0, 0.0]),
            )
            qpos = qpos.at[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ].set(quat_physical)

        rng, key3 = jax.random.split(rng)
        if self.nu > 0:
            joint_noise = jax.random.uniform(
                key3, (self.nu,), minval=-0.05, maxval=0.05
            )
            indices = jp.array(self.actuator_qpos_indices)
            joint_pos = qpos[indices]
            qpos = qpos.at[indices].set(joint_pos + joint_noise)

        qvel = jp.zeros(self.nv)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        return data

    # --------------------------------------------------------------------------------------------

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测"""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        sensordata = pipeline_state.sensordata

        if self.floating_base_qpos_addr is not None:
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            quat_norm = jp.linalg.norm(base_quat)
            base_quat = jp.where(
                quat_norm > 1e-8, base_quat /
                quat_norm, jp.array([1.0, 0.0, 0.0, 0.0])
            )
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
        else:
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        num_contacts = len(self.contact_sensor_indices)
        if num_contacts > 0:
            contact_indices = jp.array(self.contact_sensor_indices)
            contact_data = sensordata[contact_indices]
        else:
            contact_data = jp.zeros(num_contacts if num_contacts > 0 else 4)

        command = jp.array([self.target_velocity, 0.0, 0.0])

        obs = jp.concatenate(
            [
                base_quat,
                base_linvel,
                base_angvel,
                joint_pos,
                joint_vel,
                action,
                command,
                contact_data,
            ]
        )
        obs = jp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    # --------------------------------------------------------------------------------------------

    def _get_feet_positions(self, pipeline_state: Any) -> jax.Array:
        """获取脚部位置"""
        right_foot_pos = pipeline_state.xpos[self.right_foot_body_id]
        left_foot_pos = pipeline_state.xpos[self.left_foot_body_id]
        return jp.stack([right_foot_pos, left_foot_pos], axis=0)

    def _get_feet_velocities(self, pipeline_state: Any) -> jax.Array:
        """获取脚部速度"""
        right_foot_vel = pipeline_state.cvel[self.right_foot_body_id, :3]
        left_foot_vel = pipeline_state.cvel[self.left_foot_body_id, :3]
        return jp.stack([right_foot_vel, left_foot_vel], axis=0)

    # --------------------------------------------------------------------------------------------

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """采样速度命令"""
        rng, key1, key2, key3 = jax.random.split(rng, 4)
        cmd_x = jax.random.uniform(
            key1, minval=self.cmd_x_range[0], maxval=self.cmd_x_range[1]
        )
        cmd_y = jax.random.uniform(
            key2, minval=self.cmd_y_range[0], maxval=self.cmd_y_range[1]
        )
        cmd_yaw = jax.random.uniform(
            key3, minval=self.cmd_yaw_range[0], maxval=self.cmd_yaw_range[1]
        )
        return jp.array([cmd_x, cmd_y, cmd_yaw])

    # --------------------------------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """计算行走奖励"""
        from ..rewards.walking_rewards import compute_walking_reward

        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        sensordata = pipeline_state.sensordata

        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
        else:
            torso_z = self.target_height
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        if self.contact_sensor_indices:
            contact_indices = jp.array(self.contact_sensor_indices)
            contact_sensors = sensordata[contact_indices]
        else:
            contact_sensors = jp.zeros(4)

        try:
            feet_positions = self._get_feet_positions(pipeline_state)
        except Exception:
            feet_positions = None

        try:
            feet_velocities = self._get_feet_velocities(pipeline_state)
        except Exception:
            feet_velocities = None

        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        try:
            if hasattr(self, "actuator_joint_ids"):
                joint_limits = (
                    jp.array(
                        [
                            self.mj_model.jnt_range[jid, 0]
                            for jid in self.actuator_joint_ids
                        ]
                    ),
                    jp.array(
                        [
                            self.mj_model.jnt_range[jid, 1]
                            for jid in self.actuator_joint_ids
                        ]
                    ),
                )
            else:
                joint_limits = None
        except Exception:
            joint_limits = None

        contact_history = prev_state.info.get("contact_history", None)
        torques = pipeline_state.qfrc_actuator
        command = prev_state.info.get(
            "command", jp.array([self.target_velocity, 0.0, 0.0])
        )
        target_vel = command[0]

        return compute_walking_reward(
            torso_z=torso_z,
            base_quat=base_quat,
            base_linvel=base_linvel,
            base_angvel=base_angvel,
            contact_sensors=contact_sensors,
            feet_positions=feet_positions,
            action=action,
            last_action=prev_state.last_action,
            torques=torques,
            feet_velocities=feet_velocities,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_limits=joint_limits,
            contact_history=contact_history,
            target_velocity=target_vel,
            target_height=self.target_height,
            reward_weights=self.reward_weights,
        )

    # --------------------------------------------------------------------------------------------

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否摔倒"""
        from ..rewards.walking_rewards import check_walking_termination

        qpos = pipeline_state.qpos
        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            fix_quat = jp.array([0.70710678, -0.70710678, 0.0, 0.0])
            base_quat = quaternion_multiply(fix_quat, base_quat)
            return check_walking_termination(torso_z, base_quat)
        return jp.array(False)

    # --------------------------------------------------------------------------------------------

    def _get_info(
        self, state: EnvState, action: jax.Array, pipeline_state: Any
    ) -> Dict[str, jax.Array]:
        """获取额外信息"""
        info = {
            "command": state.info.get("command", jp.zeros(3)),
            "actual_velocity": jp.zeros(3),
            "contact_history": jp.zeros((10, 4)),
        }

        if self.floating_base_qvel_addr is not None:
            base_lin_vel = pipeline_state.qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
            base_ang_vel = pipeline_state.qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
            info["actual_velocity"] = jp.array(
                [base_lin_vel[0], base_lin_vel[1], base_ang_vel[2]]
            )

        if self.contact_sensor_indices:
            contact_indices = jp.array(self.contact_sensor_indices)
            current_contacts = (
                pipeline_state.sensordata[contact_indices] > 1.0
            ).astype(jp.float32)
            prev_history = state.info.get("contact_history", jp.zeros((10, 4)))
            new_history = jp.concatenate(
                [prev_history[1:, :], current_contacts[jp.newaxis, :]], axis=0
            )
            info["contact_history"] = new_history

        return info

    # --------------------------------------------------------------------------------------------

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        """执行一步"""
        action = jp.clip(action, -1.0, 1.0)
        pipeline_state = state.pipeline_state
        for _ in range(self.frame_skip):
            pipeline_state = self._step_pipeline(pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)
        reward, reward_info = self._compute_reward(
            state, action, pipeline_state)
        done = self._is_done(state, pipeline_state)
        termination_penalty = self.reward_weights.get("termination", 0.0)

        # 记录终止惩罚（仅在摔倒时应用）
        reward_info["reward/termination"] = jp.where(
            done, termination_penalty, 0.0)

        reward = jp.where(
            done,
            termination_penalty,
            jp.clip(reward, -10.0, 10.0),
        )
        reward = jp.nan_to_num(
            reward, nan=0.0, posinf=10.0, neginf=termination_penalty)

        step = state.step + 1
        done = jp.logical_or(done, step >= self.max_steps)

        info = self._get_info(state, action, pipeline_state)
        info.update(reward_info)

        return EnvState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            step=step,
            rng=state.rng,
            last_action=action,
            info=info,
        )

    # --------------------------------------------------------------------------------------------


def create_walking_env(
    xml_path: str = "assets/xmls/scenes/flat_terrain.xml", **kwargs
) -> WalkingEnv:
    """创建行走环境（便捷函数）"""
    return WalkingEnv(xml_path=xml_path, **kwargs)


# ============================================================================================
# ===================================== END: 行走环境 =========================================
# ============================================================================================
