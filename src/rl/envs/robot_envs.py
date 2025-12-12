"""
机器人MJX环境实现
"""

from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from mujoco import mjx
from rich.console import Console

from .mjx_base_env import EnvState, MJXBaseEnv

console = Console()


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
        """初始化机器人环境

        Args:
            xml_path: MuJoCo XML模型路径
            max_steps: 最大步数
            dt: 仿真时间步
            frame_skip: 动作重复次数
            verbose: 是否显示信息
            cmd_x_range: 前向速度命令范围 (m/s)
            cmd_y_range: 侧向速度命令范围 (m/s)
            cmd_yaw_range: 偏航角速度命令范围 (rad/s)
            reward_weights: 奖励权重字典
        """
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

    def _load_mujoco_model(self, xml_path: Path) -> mujoco.MjModel:
        """加载MuJoCo模型"""
        try:
            return mujoco.MjModel.from_xml_path(str(xml_path))
        except ImportError:
            console.print(
                "[red]错误：无法加载MuJoCo模型。请确保已正确安装MuJoCo和mujoco-python,且路径正确[/red]"
            )
            raise

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
            # 真实硬件传感器
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
            # 仿真专用传感器（仅用于调试）
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
            # 获取执行器对应的transmission
            trnid = model.actuator_trnid[i, 0]
            if trnid >= 0:
                # 获取transmission对应的joint
                # 注意：这里假设transmission直接连接到joint
                joint_id = trnid
                self.actuator_joint_ids.append(joint_id)

                # 获取qpos和qvel的索引
                qpos_addr = model.jnt_qposadr[joint_id]
                dof_addr = model.jnt_dofadr[joint_id]

                self.actuator_qpos_indices.append(qpos_addr)
                self.actuator_qvel_indices.append(dof_addr)

        self.actuator_qpos_indices = self.actuator_qpos_indices  # 保持为Python列表
        self.actuator_qvel_indices = self.actuator_qvel_indices  # 保持为Python列表

        # 默认关节位置（从"home" keyframe获取，如果存在）
        self.default_qpos = jp.array(model.qpos0)

        # 尝试获取home keyframe
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
            pass  # 没有home keyframe，使用qpos0

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

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理状态（带随机化）"""
        # 创建初始数据
        data = mjx.make_data(self.mjx_model)

        # 设置初始qpos（从default_qpos开始）
        # 确保qpos是1D数组
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
            # 将yaw转换为quaternion (qw, qx, qy, qz)
            quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
            # 归一化四元数
            quat = quat / jp.linalg.norm(quat)
            # freejoint的四元数从索引3开始 (位置: 0-2, 四元数: 3-6)
            qpos = qpos.at[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ].set(quat)

        # 随机化关节位置: ±0.1 rad
        rng, key4 = jax.random.split(rng)
        if self.nu > 0:
            joint_noise = jax.random.uniform(
                key4, (self.nu,), minval=-0.1, maxval=0.1)

            # 使用actuator索引更新关节位置
            # 将Python列表转换为JAX数组进行索引
            indices = jp.array(self.actuator_qpos_indices)
            joint_pos = qpos[indices]
            qpos = qpos.at[indices].set(joint_pos + joint_noise)

        # 设置qvel为0
        qvel = jp.zeros(self.nv)

        # 更新data
        data = data.replace(qpos=qpos, qvel=qvel)

        # 前向运动学
        data = mjx.forward(self.mjx_model, data)

        return data

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测（使用真实硬件可用的传感器）"""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        sensordata = pipeline_state.sensordata

        # 1. 浮动基座姿态 (quaternion)
        # 注意：真实硬件需要通过IMU融合算法（EKF/互补滤波）估计姿态
        # 这里简化处理：仿真中直接使用qpos，实际部署时需要替换为IMU融合输出
        if self.floating_base_qpos_addr is not None:
            base_quat = qpos[
                self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7
            ]
            # 确保四元数归一化
            base_quat = base_quat / jp.linalg.norm(base_quat)
        else:
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        # 2. 浮动基座角速度 - 从IMU陀螺仪读取
        if self.gyro_idx is not None:
            base_angvel = sensordata[self.gyro_idx:self.gyro_idx + 3]
        elif self.floating_base_qvel_addr is not None:
            # 降级方案：使用qvel（仅用于仿真，真实硬件必须有IMU）
            base_angvel = qvel[
                self.floating_base_qvel_addr + 3: self.floating_base_qvel_addr + 6
            ]
        else:
            base_angvel = jp.zeros(3)

        # 3. 浮动基座线速度 - 从IMU加速度计积分（简化处理）
        # 真实硬件中，线速度通常通过以下方式估计：
        # - IMU加速度积分（短期精确，长期漂移）
        # - 足底里程计（接地时积分）
        # - 光流传感器（可选）
        # 这里仍使用qvel作为ground truth，实际部署时需要替换
        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[
                self.floating_base_qvel_addr: self.floating_base_qvel_addr + 3
            ]
        else:
            base_linvel = jp.zeros(3)

        # 3. 关节位置和速度（执行器对应的关节）
        # 将Python列表转换为JAX数组进行索引
        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        # 4. 命令（此处为随机命令，后续可以从state.info中获取）
        command = jp.array([0.0, 0.0, 0.0])  # 将在reset时设置

        # 组合观测
        obs = jp.concatenate(
            [
                base_quat,  # 4
                base_linvel,  # 3
                base_angvel,  # 3
                joint_pos,  # nu
                joint_vel,  # nu
                action,  # nu
                command,  # 3
            ]
        )

        return obs

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> jax.Array:
        """计算奖励"""
        # 从info中获取命令（如果存在）
        command = prev_state.info.get("command", jp.zeros(3))

        # 获取当前速度
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

        # 1. 线速度跟踪奖励 (x, y)
        lin_vel_error = jp.sum(jp.square(base_linvel[:2] - command[:2]))
        reward_lin_vel = jp.exp(-lin_vel_error / 0.25)

        # 2. 角速度跟踪奖励 (yaw)
        ang_vel_error = jp.square(base_angvel[2] - command[2])
        reward_ang_vel = jp.exp(-ang_vel_error / 0.25)

        # 3. 存活奖励
        reward_alive = 1.0

        # 4. 动作变化率惩罚
        action_rate = jp.sum(jp.square(action - prev_state.last_action))
        cost_action_rate = action_rate

        # 5. 扭矩惩罚
        torques = pipeline_state.qfrc_actuator
        cost_torques = jp.sum(jp.square(torques))

        # 组合奖励
        reward = (
            self.reward_weights["tracking_lin_vel"] * reward_lin_vel
            + self.reward_weights["tracking_ang_vel"] * reward_ang_vel
            + self.reward_weights["alive"] * reward_alive
            + self.reward_weights["action_rate"] * cost_action_rate
            + self.reward_weights["torques"] * cost_torques
        )

        return reward

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否终止（摔倒检测）"""
        qpos = pipeline_state.qpos

        # 检查躯干高度（如果太低则认为摔倒）
        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            fall_threshold = 0.15  # 低于15cm认为摔倒
            is_fallen = torso_z < fall_threshold
        else:
            is_fallen = False

        return jp.array(is_fallen)

    def _get_info(
        self,
        state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Dict[str, jax.Array]:
        """获取额外信息"""
        # 保留命令信息（如果存在）
        info = {}
        if "command" in state.info:
            info["command"] = state.info["command"]

        return info

    def reset(self, rng: jax.Array) -> EnvState:
        """重置环境（带命令采样）"""
        # 调用父类reset
        state = super().reset(rng)

        # 采样新的速度命令
        rng, cmd_rng = jax.random.split(state.rng)
        command = self._sample_command(cmd_rng)

        # 更新info
        info = state.info.copy()
        info["command"] = command

        # 重新计算obs（包含命令）
        obs = state.obs.at[-3:].set(command)

        # 更新state
        state = state.replace(
            rng=rng,
            obs=obs,
            info=info,
        )

        return state

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


# ==================== 便捷的创建函数 ====================


def create_velocity_tracking_env(
    xml_path: str = "assets/xmls/scene.xml", **kwargs
) -> VelocityTrackingEnv:
    """创建速度跟踪环境（便捷函数）

    Args:
        xml_path: MuJoCo XML模型路径
        **kwargs: 传递给VelocityTrackingEnv的其他参数

    Returns:
        VelocityTrackingEnv实例
    """
    return VelocityTrackingEnv(xml_path=xml_path, **kwargs)


class StandingEnv(MJXBaseEnv):
    """站立保持环境

    任务：保持站立平衡，抵抗扰动
    观测：机器人姿态、速度、关节状态
    动作：关节位置/力矩控制
    奖励：使用rewards.standing_rewards模块中的奖励函数

    Attributes:
        target_height: 目标躯干高度 (m)
        reward_weights: 奖励权重字典
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
        """初始化站立环境

        Args:
            xml_path: MuJoCo XML模型路径
            max_steps: 最大步数
            dt: 仿真时间步
            frame_skip: 动作重复次数
            verbose: 是否显示信息
            target_height: 目标躯干高度 (m)
            reward_weights: 奖励权重字典
        """
        # 延迟导入避免循环依赖
        from ..rewards.standing_rewards import DEFAULT_STANDING_REWARD_WEIGHTS

        super().__init__(xml_path, max_steps, dt, frame_skip, verbose)

        self.target_height = target_height

        # 站立任务的奖励权重
        if reward_weights is None:
            reward_weights = DEFAULT_STANDING_REWARD_WEIGHTS.copy()
        self.reward_weights = reward_weights

        # 提取关键索引
        self._extract_indices()

        # 设置观测维度
        self._setup_observation_space()

        if verbose:
            console.print(f"[green]✓ 站立环境初始化完成[/green]")
            console.print(f"  观测维度: {self.observation_size}")
            console.print(f"  动作维度: {self.action_size}")
            console.print(f"  目标高度: {target_height}m")

    def _extract_indices(self) -> None:
        """提取关键索引"""
        model = self.mj_model

        # 找到浮动基座的地址（freejoint）
        self.floating_base_qpos_addr = None
        self.floating_base_qvel_addr = None
        for i in range(model.njnt):
            if model.jnt_type[i] == 0:  # 0 = freejoint
                self.floating_base_qpos_addr = model.jnt_qposadr[i]
                self.floating_base_qvel_addr = model.jnt_dofadr[i]
                break

        # 执行器关节地址
        self.actuator_qpos_indices = []
        self.actuator_qvel_indices = []
        for i in range(model.nu):
            trnid = model.actuator_trnid[i, 0]
            if trnid >= 0:
                joint_id = trnid
                self.actuator_qpos_indices.append(model.jnt_qposadr[joint_id])
                self.actuator_qvel_indices.append(model.jnt_dofadr[joint_id])

        # 默认关节位置
        self.default_qpos = jp.array(model.qpos0)

        # 尝试获取home keyframe
        try:
            home_key_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key_id >= 0:
                self.default_qpos = jp.array(
                    model.key_qpos[home_key_id *
                                   model.nq: (home_key_id + 1) * model.nq]
                )
        except Exception:
            pass

    def _setup_observation_space(self) -> None:
        """设置观测空间维度

        观测：四元数(4) + 线速度(3) + 角速度(3) + 关节位置(nu) + 关节速度(nu) + 上一动作(nu)
        """
        self._observation_size = 4 + 3 + 3 + self.nu + self.nu + self.nu

    @property
    def observation_size(self) -> int:
        """观测空间维度"""
        return self._observation_size

    def _reset_pipeline(self, rng: jax.Array) -> Any:
        """重置物理状态

        Args:
            rng: JAX随机数生成器

        Returns:
            初始化的MJX数据
        """
        data = mjx.make_data(self.mjx_model)
        qpos = jp.array(self.default_qpos).flatten()

        # 随机化初始姿态（站立任务使用更小的随机化范围）
        if self.floating_base_qpos_addr is not None:
            rng, key1, key2 = jax.random.split(rng, 3)

            # 随机化xy位置: ±0.02m
            dxy = jax.random.uniform(key1, (2,), minval=-0.02, maxval=0.02)
            base_xy = qpos[self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2]
            qpos = qpos.at[self.floating_base_qpos_addr: self.floating_base_qpos_addr + 2].set(
                base_xy + dxy
            )

            # 随机化yaw: ±0.1 rad
            yaw = jax.random.uniform(key2, minval=-0.1, maxval=0.1)
            quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
            quat = quat / jp.linalg.norm(quat)
            qpos = qpos.at[self.floating_base_qpos_addr + 3: self.floating_base_qpos_addr + 7].set(
                quat
            )

        # 随机化关节位置: ±0.05 rad
        rng, key3 = jax.random.split(rng)
        if self.nu > 0:
            joint_noise = jax.random.uniform(
                key3, (self.nu,), minval=-0.05, maxval=0.05)
            indices = jp.array(self.actuator_qpos_indices)
            joint_pos = qpos[indices]
            qpos = qpos.at[indices].set(joint_pos + joint_noise)

        qvel = jp.zeros(self.nv)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        return data

    def _get_obs(self, pipeline_state: Any, action: jax.Array) -> jax.Array:
        """计算观测

        Args:
            pipeline_state: MJX pipeline状态
            action: 当前动作

        Returns:
            观测向量
        """
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel

        # 浮动基座姿态和速度
        if self.floating_base_qpos_addr is not None:
            base_quat = qpos[self.floating_base_qpos_addr +
                             3: self.floating_base_qpos_addr + 7]
            base_quat = base_quat / jp.linalg.norm(base_quat)
        else:
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[self.floating_base_qvel_addr:
                               self.floating_base_qvel_addr + 3]
            base_angvel = qvel[self.floating_base_qvel_addr +
                               3: self.floating_base_qvel_addr + 6]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        # 关节状态
        qpos_indices = jp.array(self.actuator_qpos_indices)
        qvel_indices = jp.array(self.actuator_qvel_indices)
        joint_pos = qpos[qpos_indices]
        joint_vel = qvel[qvel_indices]

        obs = jp.concatenate([
            base_quat,      # 4
            base_linvel,    # 3
            base_angvel,    # 3
            joint_pos,      # nu
            joint_vel,      # nu
            action,         # nu
        ])

        return obs

    def _compute_reward(
        self,
        prev_state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> jax.Array:
        """计算站立奖励

        使用rewards.standing_rewards模块中的compute_standing_reward函数。

        Args:
            prev_state: 上一步的环境状态
            action: 当前动作
            pipeline_state: 当前MJX pipeline状态

        Returns:
            奖励值
        """
        from ..rewards.standing_rewards import compute_standing_reward

        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel

        # 获取躯干状态
        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[self.floating_base_qpos_addr +
                             3: self.floating_base_qpos_addr + 7]
        else:
            torso_z = self.target_height
            base_quat = jp.array([1.0, 0.0, 0.0, 0.0])

        if self.floating_base_qvel_addr is not None:
            base_linvel = qvel[self.floating_base_qvel_addr:
                               self.floating_base_qvel_addr + 3]
            base_angvel = qvel[self.floating_base_qvel_addr +
                               3: self.floating_base_qvel_addr + 6]
        else:
            base_linvel = jp.zeros(3)
            base_angvel = jp.zeros(3)

        torques = pipeline_state.qfrc_actuator

        return compute_standing_reward(
            torso_z=torso_z,
            base_quat=base_quat,
            base_linvel=base_linvel,
            base_angvel=base_angvel,
            action=action,
            last_action=prev_state.last_action,
            torques=torques,
            target_height=self.target_height,
            reward_weights=self.reward_weights,
        )

    def _is_done(self, state: EnvState, pipeline_state: Any) -> jax.Array:
        """检查是否摔倒

        使用rewards.standing_rewards模块中的check_standing_termination函数。

        Args:
            state: 当前环境状态
            pipeline_state: 当前MJX pipeline状态

        Returns:
            是否终止的布尔值
        """
        from ..rewards.standing_rewards import check_standing_termination

        qpos = pipeline_state.qpos

        if self.floating_base_qpos_addr is not None:
            torso_z = qpos[self.floating_base_qpos_addr + 2]
            base_quat = qpos[self.floating_base_qpos_addr +
                             3: self.floating_base_qpos_addr + 7]
            return check_standing_termination(torso_z, base_quat)
        else:
            return jp.array(False)

    def _get_info(
        self,
        state: EnvState,
        action: jax.Array,
        pipeline_state: Any,
    ) -> Dict[str, jax.Array]:
        """获取额外信息

        Args:
            state: 当前环境状态
            action: 当前动作
            pipeline_state: 当前MJX pipeline状态

        Returns:
            额外信息字典
        """
        return {}


def create_standing_env(
    xml_path: str = "assets/xmls/scene.xml", **kwargs
) -> StandingEnv:
    """创建站立环境（便捷函数）

    Args:
        xml_path: MuJoCo XML模型路径
        **kwargs: 传递给StandingEnv的其他参数

    Returns:
        StandingEnv实例
    """
    return StandingEnv(xml_path=xml_path, **kwargs)
