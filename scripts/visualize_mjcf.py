"""
可视化MJCF模型脚本
用于查看和测试机器人的MJCF模型
支持热刷新: 修改并保存XML文件后自动重新加载
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time
import os
import re


def apply_stabilization(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """增加数值稳定性，避免自由度速度发散。"""
    if model.dof_damping is not None:
        min_damping = 0.5
        model.dof_damping[:] = np.maximum(model.dof_damping, min_damping)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    model.opt.timestep = min(model.opt.timestep, 0.01)

    # 将重力设为0，方便静止观察和微调
    model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def get_include_files(xml_path: Path) -> list[Path]:
    """递归获取所有被 <include> 标签引用的文件路径"""
    includes = [xml_path]
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 简单的正则匹配 <include file="..."/>
            matches = re.findall(r'<include\s+file="([^"]+)"', content)
            for m in matches:
                include_path = (xml_path.parent / m).resolve()
                if include_path.exists() and include_path not in includes:
                    includes.extend(get_include_files(include_path))
    except Exception:
        pass
    return includes


def get_last_modified_time(files: list[Path]) -> float:
    """获取一组文件中最新的修改时间"""
    return max(os.path.getmtime(f) for f in files if f.exists())

def visualize_mjcf(xml_path: str, interactive: bool = True):
    """可视化MJCF模型并支持热刷新"""
    xml_path_obj = Path(xml_path).resolve()
    if not xml_path_obj.exists():
        # 尝试相对于项目根目录解析
        current = Path.cwd()
        project_root = None
        while current != current.parent:
            if (current / "assets").exists():
                project_root = current
                break
            current = current.parent
        if project_root:
            xml_path_obj = (project_root / xml_path).resolve()

    if not xml_path_obj.exists():
        print(f"❌ 错误: 文件不存在: {xml_path}")
        return

    print(f"📂 正在监视模型: {xml_path_obj}")
    
    # 获取初始监控文件列表
    watched_files = get_include_files(xml_path_obj)
    last_mtime = get_last_modified_time(watched_files)
    
    original_dir = os.getcwd()

    while True:
        print(f"\n🔄 正在加载/刷新模型...")
        try:
            # 切换到 XML 目录以解析相对路径
            os.chdir(xml_path_obj.parent)
            
            model = mujoco.MjModel.from_xml_path(xml_path_obj.name)
            data = mujoco.MjData(model)

            # 初始化和稳定化
            mujoco.mj_resetData(model, data)
            apply_stabilization(model, data)
            
            # 设置初始姿态
            try:
                for side in ["right", "left"]:
                    knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee_joint")
                    if knee_id >= 0:
                        data.qpos[model.jnt_qposadr[knee_id]] = 0.3
            except:
                pass

            mujoco.mj_forward(model, data)

            if not interactive:
                print("模型加载成功（非交互模式）。")
                break

            print(f"✓ 模型加载成功。正在运行... (修改 XML 可自动刷新)")
            
            # 启动被动查看器
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                should_reload = False
                while viewer.is_running():
                    step_start = time.time()
                    
                    # 仿真步进
                    mujoco.mj_step(model, data)
                    viewer.sync()

                    # 检查热刷新
                    # 每隔一段时间检查一次，避免过度占用IO
                    if int(time.time() * 2) % 2 == 0: 
                        watched_files = get_include_files(xml_path_obj)
                        current_mtime = get_last_modified_time(watched_files)
                        if current_mtime > last_mtime:
                            last_mtime = current_mtime
                            should_reload = True
                            print(f"\n⚡ 检测到文件修改，正在热刷新...")
                            break

                    # 控制帧率
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                
                if not should_reload:
                    # 如果不是因为要刷新而跳出，说明是用户关掉了窗口
                    return

        except Exception as e:
            print(f"\n❌ 加载失败: {e}")
            print("请检查 XML 语法。将在 2 秒后尝试重新监视...")
            time.sleep(2)
            # 持续监控，直到用户修正错误
            while True:
                watched_files = get_include_files(xml_path_obj)
                if get_last_modified_time(watched_files) > last_mtime:
                    last_mtime = get_last_modified_time(watched_files)
                    break
                time.sleep(1)
        finally:
            os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="可视化机器人MJCF模型（支持热刷新）")
    parser.add_argument("--xml", type=str, default="assets/xmls/models/jiyuan_fit.xml", help="MJCF文件路径")
    parser.add_argument("--no-interactive", action="store_true", help="禁用交互式查看器")
    args = parser.parse_args()

    visualize_mjcf(args.xml, interactive=not args.no_interactive)


if __name__ == "__main__":
    main()