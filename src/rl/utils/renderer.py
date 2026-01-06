"""
MuJoCo渲染器 - 支持3D可视化
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from mujoco import mjx
from PIL import Image, ImageDraw, ImageFont

# ============================================================================================
# ======================================= 渲染器 =============================================
# ============================================================================================


class MujocoRenderer:
    """MuJoCo可视化渲染器"""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 1280,
        height: int = 720,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """初始化渲染器"""
        self.model = model
        self.data = mujoco.MjData(model)
        self.width = width
        self.height = height

        self.renderer = mujoco.Renderer(model, height=height, width=width)

        if camera_id is not None:
            self.renderer.camera_id = camera_id
        elif camera_name is not None:
            self.renderer.camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )

        self.last_render_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def render(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """渲染当前状态"""
        if data is not None:
            self.data = data

        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()

        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_render_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_render_time = current_time

        return pixels

    def close(self):
        """关闭渲染器"""
        self.renderer.close()


# ============================================================================================
# ===================================== END: 渲染器 ===========================================
# ============================================================================================


# ============================================================================================
# ======================================= 交互式查看器 =========================================
# ============================================================================================


class InteractiveViewer:
    """交互式MuJoCo查看器"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """初始化查看器"""
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(model, data)
            self.model = model
            self.data = data
            self.is_running = True
        except ImportError:
            raise ImportError("交互式查看器需要mujoco.viewer模块。请确保安装了完整的mujoco包。")

    def update(self, data: Optional[mujoco.MjData] = None):
        """更新显示"""
        if data is not None:
            self.data.qpos[:] = data.qpos
            self.data.qvel[:] = data.qvel
            self.data.ctrl[:] = data.ctrl
            mujoco.mj_forward(self.model, self.data)

        self.viewer.sync()

    def is_alive(self) -> bool:
        """检查查看器是否还在运行"""
        return self.viewer.is_running()

    def close(self):
        """关闭查看器"""
        self.viewer.close()
        self.is_running = False


# ============================================================================================
# ===================================== END: 交互式查看器 ======================================
# ============================================================================================


# ============================================================================================
# ======================================= 叠加渲染器 ==========================================
# ============================================================================================


class OverlayRenderer:
    """在图像上叠加文本信息"""

    def __init__(
        self,
        font_size: int = 24,
        text_color: tuple = (255, 255, 255),
        bg_color: tuple = (0, 0, 0, 180),
        margin: int = 10,
    ):
        """初始化叠加渲染器"""
        self.font_size = font_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.margin = margin

        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
            )
        except:
            try:
                self.font = ImageFont.truetype(
                    "/System/Library/Fonts/Menlo.ttc", font_size
                )
            except:
                self.font = ImageFont.load_default()

    def render(self, frame: np.ndarray, text_lines: List[str]) -> np.ndarray:
        """在帧上叠加文本"""
        if not text_lines:
            return frame

        img = Image.fromarray(frame)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        line_height = self.font_size + 5
        text_block_height = len(text_lines) * line_height + 2 * self.margin

        try:
            max_text_width = max(
                draw.textlength(line, font=self.font) for line in text_lines
            )
        except AttributeError:
            max_text_width = max(
                draw.textsize(line, font=self.font)[0] for line in text_lines
            )

        text_block_width = int(max_text_width) + 2 * self.margin

        bg_rect = [
            self.margin,
            self.margin,
            self.margin + text_block_width,
            self.margin + text_block_height,
        ]
        draw.rectangle(bg_rect, fill=self.bg_color)

        y_offset = self.margin * 2
        for line in text_lines:
            draw.text(
                (self.margin * 2, y_offset),
                line,
                fill=self.text_color + (255,),
                font=self.font,
            )
            y_offset += line_height

        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)

        return np.array(img.convert("RGB"))


# ============================================================================================
# ===================================== END: 叠加渲染器 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 视频录制器 ==========================================
# ============================================================================================


class VideoRecorder:
    """视频录制器 - 集成 MJX 数据渲染和视频保存"""

    def __init__(
        self,
        mujoco_model: mujoco.MjModel,
        output_dir: str = "videos",
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = "track",
    ):
        """初始化视频录制器"""
        self.mj_model = mujoco_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self.renderer = MujocoRenderer(
            model=mujoco_model,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
        )

        self.overlay = OverlayRenderer(font_size=28)
        self.mj_data = mujoco.MjData(mujoco_model)
        self.frames: List[np.ndarray] = []
        self.is_recording = False

    def start_recording(self):
        """开始录制"""
        self.frames = []
        self.is_recording = True

    def add_frame_from_mjx(
        self,
        mjx_data: Any,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """从 MJX 数据添加一帧"""
        if not self.is_recording:
            return

        try:
            mjx.put_data(self.mj_data, mjx_data)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            frame = self.renderer.render(self.mj_data)

            if metrics:
                text_lines = self._format_metrics(metrics)
                frame = self.overlay.render(frame, text_lines)

            self.frames.append(frame)

        except Exception as e:
            print(f"警告：录制帧失败，跳过该帧: {e}")

    def save_video(self, filename: str) -> Path:
        """保存视频文件"""
        if not self.frames:
            raise ValueError("没有录制任何帧，无法保存视频")

        output_path = self.output_dir / f"{filename}.mp4"

        try:
            mediapy.write_video(
                str(output_path),
                self.frames,
                fps=self.fps,
                codec="h264",
                qp=20,
            )
        except Exception as e:
            print(f"错误：视频保存失败: {e}")
            raise
        finally:
            self.is_recording = False

        return output_path

    def _format_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """格式化指标为文本行"""
        lines = []

        if "episode_reward" in metrics:
            lines.append(f"Reward: {metrics['episode_reward']:>7.2f}")

        if "update" in metrics:
            lines.append(f"Update: {int(metrics['update']):>6d}")

        if "env_steps" in metrics:
            lines.append(f"Steps:  {int(metrics['env_steps']):>8,}")

        if lines:
            lines.append("─" * 25)

        if "cmd_vx" in metrics and "actual_vx" in metrics:
            lines.append(
                f"Vx: {metrics['cmd_vx']:>5.2f} → {metrics['actual_vx']:>5.2f} m/s"
            )

        if "cmd_vy" in metrics and "actual_vy" in metrics:
            lines.append(
                f"Vy: {metrics['cmd_vy']:>5.2f} → {metrics['actual_vy']:>5.2f} m/s"
            )

        if "cmd_vyaw" in metrics and "actual_vyaw" in metrics:
            lines.append(
                f"Wz: {metrics['cmd_vyaw']:>5.2f} → {metrics['actual_vyaw']:>5.2f} rad/s"
            )

        return lines

    def close(self):
        """关闭录制器"""
        self.renderer.close()


# ============================================================================================
# ===================================== END: 视频录制器 ========================================
# ============================================================================================


# ============================================================================================
# ======================================= 工具函数 ============================================
# ============================================================================================


def create_video_writer(
    output_path: str,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
):
    """创建视频写入器"""
    try:
        import cv2
    except ImportError:
        raise ImportError("保存视频需要opencv-python: pip install opencv-python")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


# --------------------------------------------------------------------------------------------


def save_frame_to_video(writer, frame: np.ndarray):
    """保存帧到视频"""
    try:
        import cv2
    except ImportError:
        raise ImportError("保存视频需要opencv-python: pip install opencv-python")

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)


# ============================================================================================
# ===================================== END: 工具函数 ==========================================
# ============================================================================================
