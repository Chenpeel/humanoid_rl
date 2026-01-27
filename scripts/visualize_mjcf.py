
import argparse
import os
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================================================
# ======================================= 工具函数 ============================================
# ============================================================================================

def apply_stabilization(model: mujoco.MjModel, data: mujoco.MjData, gravity_enabled: bool) -> None:
    """增加数值稳定性，避免自由度速度发散。"""
    if model.dof_damping is not None:
        min_damping = 0.5
        model.dof_damping[:] = np.maximum(model.dof_damping, min_damping)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    model.opt.timestep = min(model.opt.timestep, 0.01)
    
    if not gravity_enabled:
        model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

# --------------------------------------------------------------------------------------------


def get_include_files(xml_path: Path) -> list[Path]:
    """递归获取所有被 <include> 标签引用的文件路径"""
    includes = [xml_path]
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            content = f.read()
            matches = re.findall(r'<include\s+file="([^"]+)"', content)
            for m in matches:
                include_path = (xml_path.parent / m).resolve()
                if include_path.exists() and include_path not in includes:
                    includes.extend(get_include_files(include_path))
    except Exception:
        pass
    return includes

# --------------------------------------------------------------------------------------------


def get_last_modified_time(files: list[Path]) -> float:
    """获取一组文件中最新的修改时间"""
    return max(os.path.getmtime(f) for f in files if f.exists())

# --------------------------------------------------------------------------------------------


def load_model_data(
    xml_path_obj: Path, gravity_enabled: bool
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载模型并应用基础稳定化设置。"""
    model = mujoco.MjModel.from_xml_path(xml_path_obj.name)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    apply_stabilization(model, data, gravity_enabled=gravity_enabled)

    try:
        for side in ["right", "left"]:
            knee_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee_joint"
            )
            if knee_id >= 0:
                data.qpos[model.jnt_qposadr[knee_id]] = 0.3
    except Exception:
        pass

    mujoco.mj_forward(model, data)
    return model, data


# ============================================================================================
# ===================================== END: 工具函数 ==========================================
# ============================================================================================


# ============================================================================================
# ======================================= 可视化逻辑 ==========================================
# ============================================================================================

def visualize_mjcf(
    xml_path: str,
    interactive: bool = True,
    gravity_enabled: bool = False,
    autoreload: bool = True,
    mode: str = "sim",
):
    """可视化MJCF模型并支持热刷新"""
    xml_path_obj = Path(xml_path).resolve()
    if not xml_path_obj.exists():
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

    if autoreload:
        print(f"📂 正在监视模型: {xml_path_obj}")
    else:
        print(f"📂 正在加载模型: {xml_path_obj}")

    original_dir = os.getcwd()

    if not interactive:
        try:
            os.chdir(xml_path_obj.parent)
            model, data = load_model_data(
                xml_path_obj, gravity_enabled=gravity_enabled
            )
            print("模型加载成功（非交互模式）。")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
        finally:
            os.chdir(original_dir)
        return

    if autoreload and mode != "sim":
        print("⚠️ autoreload=1 时仅支持 mode=sim，已强制切换。")
        mode = "sim"

    if not autoreload and mode == "launch":
        try:
            os.chdir(xml_path_obj.parent)

            def loader():
                model, data = load_model_data(
                    xml_path_obj, gravity_enabled=gravity_enabled
                )
                return model, data, str(xml_path_obj)

            mujoco.viewer.launch(loader=loader)
        except Exception as e:
            print(f"❌ 加载失败: {e}")
        finally:
            os.chdir(original_dir)
        return

    if not autoreload and mode != "sim":
        print(f"❌ 错误: 不支持的 mode: {mode}")
        return

    watched_files = get_include_files(xml_path_obj) if autoreload else []
    last_mtime = get_last_modified_time(watched_files) if autoreload else 0.0

    while True:
        print(f"\n🔄 正在加载/刷新模型...")
        try:
            os.chdir(xml_path_obj.parent)
            model, data = load_model_data(
                xml_path_obj, gravity_enabled=gravity_enabled
            )

            if autoreload:
                print("✓ 模型加载成功。正在运行... (修改 XML 可自动刷新)")
            else:
                print("✓ 模型加载成功。正在运行...")

            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                should_reload = False
                while viewer.is_running():
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    viewer.sync()

                    if autoreload and int(time.time() * 2) % 2 == 0:
                        watched_files = get_include_files(xml_path_obj)
                        current_mtime = get_last_modified_time(watched_files)
                        if current_mtime > last_mtime:
                            last_mtime = current_mtime
                            should_reload = True
                            print(f"\n⚡ 检测到文件修改，正在热刷新...")
                            break

                    time_until_next_step = model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                if not should_reload:
                    return

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            if not autoreload:
                return
            print("请检查 XML 语法。将在 2 秒后尝试重新监视...")
            time.sleep(2)
            while True:
                watched_files = get_include_files(xml_path_obj)
                if get_last_modified_time(watched_files) > last_mtime:
                    last_mtime = get_last_modified_time(watched_files)
                    break
                time.sleep(1)
        finally:
            os.chdir(original_dir)

# ============================================================================================
# ===================================== END: 可视化逻辑 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 主函数 ==============================================
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description="可视化机器人MJCF模型（支持热刷新）")
    parser.add_argument(
        "--xml",
        type=str,
        default="assets/xmls/models/jiyuan_fit.xml",
        help="MJCF文件路径",
    )
    parser.add_argument("--no-interactive",
                        action="store_true", help="禁用交互式查看器")
    parser.add_argument(
        "--autoreload",
        type=int,
        choices=[0, 1],
        default=1,
        help="是否启用热重载 (1=启用, 0=禁用). 默认为1 (启用).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sim", "launch"],
        default="sim",
        help="非热重载时的查看模式: sim=手动步进, launch=Simulate GUI。",
    )
    parser.add_argument(
        "--gravity",
        type=int,
        choices=[0, 1],
        default=0,
        help="是否启用重力 (1=启用, 0=禁用). 默认为0 (禁用).",
    )
    args = parser.parse_args()

    visualize_mjcf(
        args.xml,
        interactive=not args.no_interactive,
        gravity_enabled=bool(args.gravity),
        autoreload=bool(args.autoreload),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

# ============================================================================================
# ===================================== END: 主函数 ============================================
# ============================================================================================
