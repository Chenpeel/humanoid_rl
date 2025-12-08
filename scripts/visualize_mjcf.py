"""
可视化MJCF模型脚本
用于查看和测试机器人的MJCF模型
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time
import os


def apply_stabilization(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """增加数值稳定性，避免自由度速度发散。"""
    # 适当增加阻尼，防止自由度漂移
    if model.dof_damping is not None:
        min_damping = 0.5
        model.dof_damping[:] = np.maximum(model.dof_damping, min_damping)

    # 使用隐式积分器增强稳定性
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

    # 降低步长，避免过大dt导致的不稳定
    model.opt.timestep = min(model.opt.timestep, 0.01)

    # 重置数据，确保修改生效
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def visualize_mjcf(xml_path: str, interactive: bool = True):
    """
    可视化MJCF模型

    Args:
        xml_path: MJCF文件路径
        interactive: 是否启用交互式查看器
    """
    # 加载模型
    print(f"正在加载模型: {xml_path}")
    # MuJoCo会相对于工作目录解析mesh路径
    # XML中使用 assets_jiyuan_right/meshes/ 格式
    # 需要切换到 rl 目录，并使用相对路径传递XML
    xml_path_obj = Path(xml_path).resolve()

    # 从 mjcf 目录往上三级到达 rl 目录
    rl_dir = xml_path_obj.parent.parent.parent
    original_dir = os.getcwd()
    os.chdir(rl_dir)

    # 计算XML文件相对于rl目录的路径
    xml_relative = xml_path_obj.relative_to(rl_dir)

    try:
        print(f"工作目录: {os.getcwd()}")
        # 使用相对路径加载，这样MuJoCo会从工作目录解析mesh路径
        model = mujoco.MjModel.from_xml_path(str(xml_relative))
        data = mujoco.MjData(model)

        # 打印模型信息
        print("\n" + "=" * 60)
        print("模型信息")
        print("=" * 60)
        print(f"模型名称: jiyuan")
        print(f"自由度数 (nv): {model.nv}")
        print(f"位置维度 (nq): {model.nq}")
        print(f"执行器数量: {model.nu}")
        print(f"传感器数量: {model.nsensor}")
        print(f"身体数量: {model.nbody}")
        print(f"关节数量: {model.njnt}")
        print(f"几何体数量: {model.ngeom}")

        # 打印关节信息
        print("\n" + "=" * 60)
        print("关节列表")
        print("=" * 60)
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = model.jnt_type[i]
            type_name = {0: 'free', 1: 'ball',
                         2: 'slide', 3: 'hinge'}[joint_type]
            print(f"{i:2d}. {joint_name:30s} ({type_name})")

        # 打印执行器信息
        print("\n" + "=" * 60)
        print("执行器列表")
        print("=" * 60)
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"{i:2d}. {actuator_name}")

        # 打印传感器信息
        print("\n" + "=" * 60)
        print("传感器列表")
        print("=" * 60)
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_type = model.sensor_type[i]
            print(f"{i:2d}. {sensor_name:30s} (type={sensor_type})")

        print("\n" + "=" * 60)

        # 设置初始姿态（站立姿态）
        mujoco.mj_resetData(model, data)

        # 可选稳定性处理
        apply_stabilization(model, data)

        # 设置一些关节到初始位置（可选）
        try:
            right_knee_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
            left_knee_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
            if right_knee_id >= 0:
                data.qpos[model.jnt_qposadr[right_knee_id]] = 0.3  # 弯曲30度左右
            if left_knee_id >= 0:
                data.qpos[model.jnt_qposadr[left_knee_id]] = 0.3
        except:
            pass

        # 前向运动学
        mujoco.mj_forward(model, data)

        if interactive:
            print("\n启动交互式查看器...")
            print("提示:")
            print("  - 鼠标左键拖动: 旋转视角")
            print("  - 鼠标右键拖动: 平移视角")
            print("  - 鼠标滚轮: 缩放")
            print("  - 双击: 选择物体")
            print("  - Ctrl+右键: 施加力")
            print("  - 空格: 暂停/继续")
            print("  - Backspace: 重置")
            print("  - ESC: 退出")
            print("\n按Ctrl+C退出...\n")

            # 启动交互式查看器
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # 设置相机
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                start_time = time.time()

                try:
                    while viewer.is_running():
                        step_start = time.time()

                        # 仿真步进
                        if np.any(~np.isfinite(data.qvel)):
                            print("检测到速度NaN/Inf，重新重置状态以避免发散...")
                            mujoco.mj_resetData(model, data)
                            mujoco.mj_forward(model, data)
                            continue

                        mujoco.mj_step(model, data)

                        # 更新查看器
                        viewer.sync()

                        # 控制帧率
                        time_until_next_step = model.opt.timestep - \
                            (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)

                except KeyboardInterrupt:
                    print("\n用户中断，退出...")
        else:
            print("\n非交互模式 - 仅显示模型信息")

    finally:
        # 恢复原工作目录
        os.chdir(original_dir)


def test_forward_kinematics(xml_path: str):
    """
    测试正向运动学
    设置不同的关节角度，查看机器人姿态
    """
    print("测试正向运动学...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 测试配置
    test_configs = [
        {
            "name": "站立姿态",
            "joints": {}
        },
        {
            "name": "深蹲姿态",
            "joints": {
                "right_knee_joint": 1.5,
                "left_knee_joint": 1.5,
                "right_hip_pitch_joint": -0.5,
                "left_hip_pitch_joint": -0.5,
            }
        },
        {
            "name": "单腿抬起",
            "joints": {
                "right_hip_pitch_joint": 1.0,
                "right_knee_joint": -0.5,
            }
        }
    ]

    for config in test_configs:
        print(f"\n配置: {config['name']}")
        mujoco.mj_resetData(model, data)

        # 设置关节角度
        for joint_name, angle in config['joints'].items():
            try:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    data.qpos[model.jnt_qposadr[joint_id]] = angle
                    print(
                        f"  {joint_name}: {angle:.2f} rad ({np.degrees(angle):.1f}°)")
            except:
                print(f"  警告: 找不到关节 {joint_name}")

        # 计算正向运动学
        mujoco.mj_forward(model, data)

        # 打印基座位置
        print(
            f"  基座位置: [{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")


def check_model_validity(xml_path: str):
    """
    检查模型有效性
    """
    print("\n检查模型有效性...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # 尝试运行几步仿真
        for _ in range(100):
            mujoco.mj_step(model, data)

        print("✓ 模型有效，仿真正常")

        # 检查是否有无限值或NaN
        if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
            print("✗ 警告: 位置数据包含NaN或无限值")
        else:
            print("✓ 位置数据正常")

        if np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel)):
            print("✗ 警告: 速度数据包含NaN或无限值")
        else:
            print("✓ 速度数据正常")

        return True

    except Exception as e:
        print(f"✗ 模型无效: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="可视化机器人MJCF模型")
    parser.add_argument(
        "--xml",
        type=str,
        default="../assets_jiyuan_right/mjcf/jiyuan.xml",
        help="MJCF文件路径（相对于scripts目录）"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="禁用交互式查看器，仅显示模型信息"
    )
    parser.add_argument(
        "--test-fk",
        action="store_true",
        help="测试正向运动学"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查模型有效性"
    )

    args = parser.parse_args()

    # 解析路径
    script_dir = Path(__file__).parent
    xml_path = (script_dir / args.xml).resolve()

    if not xml_path.exists():
        print(f"错误: 找不到文件 {xml_path}")
        return

    print("=" * 60)
    print("机器人MJCF模型可视化工具")
    print("=" * 60)
    print(f"文件路径: {xml_path}")

    # 检查模型
    if args.check:
        check_model_validity(str(xml_path))

    # 测试正向运动学
    if args.test_fk:
        test_forward_kinematics(str(xml_path))

    # 可视化
    if not args.no_interactive:
        visualize_mjcf(str(xml_path), interactive=True)
    else:
        visualize_mjcf(str(xml_path), interactive=False)


if __name__ == "__main__":
    main()
