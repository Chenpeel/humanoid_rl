"""
MuJoCo渲染器 - 支持3D可视化
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# JAX 和 MJX 支持
import jax
import jax.numpy as jp
# 视频编码
import mediapy
import mujoco
import numpy as np
from mujoco import mjx
# 图像处理
from PIL import Image, ImageDraw, ImageFont


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
        """
        Args:
            model: MuJoCo模型
            width: 窗口宽度
            height: 窗口高度
            camera_id: 相机ID（可选）
            camera_name: 相机名称（可选）
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.width = width
        self.height = height

        # 创建渲染器
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # 设置相机
        if camera_id is not None:
            self.renderer.camera_id = camera_id
        elif camera_name is not None:
            self.renderer.camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )

        # 用于FPS计算
        self.last_render_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def render(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """
        渲染当前状态

        Args:
            data: MuJoCo数据（如果为None则使用内部data）

        Returns:
            RGB图像数组 (height, width, 3)
        """
        if data is not None:
            self.data = data

        # 更新渲染器
        self.renderer.update_scene(self.data)

        # 渲染
        pixels = self.renderer.render()

        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_render_time
        if elapsed > 1.0:  # 每秒更新一次FPS
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_render_time = current_time

        return pixels

    def close(self):
        """关闭渲染器"""
        self.renderer.close()


class InteractiveViewer:
    """交互式MuJoCo查看器"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
        """
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(model, data)
            self.model = model
            self.data = data
            self.is_running = True
        except ImportError:
            raise ImportError("交互式查看器需要mujoco.viewer模块。" "请确保安装了完整的mujoco包。")

    def update(self, data: Optional[mujoco.MjData] = None):
        """
        更新显示

        Args:
            data: 新的MuJoCo数据
        """
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


def create_video_writer(
    output_path: str,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
):
    """
    创建视频写入器

    Args:
        output_path: 输出视频路径
        fps: 帧率
        width: 视频宽度
        height: 视频高度

    Returns:
        VideoWriter对象
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("保存视频需要opencv-python: pip install opencv-python")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


def save_frame_to_video(writer, frame: np.ndarray):
    """
    保存帧到视频

    Args:
        writer: VideoWriter对象
        frame: RGB图像 (height, width, 3)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("保存视频需要opencv-python: pip install opencv-python")

    # MuJoCo渲染是RGB，OpenCV需要BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)


class OverlayRenderer:
    """在图像上叠加文本信息

    用于在训练视频帧上绘制实时指标（奖励、速度等）
    """

    def __init__(
        self,
        font_size: int = 24,
        text_color: tuple = (255, 255, 255),  # 白色
        bg_color: tuple = (0, 0, 0, 180),  # 半透明黑色（RGBA）
        margin: int = 10,
    ):
        """初始化叠加渲染器

        Args:
            font_size: 字体大小
            text_color: 文本颜色 (R, G, B)
            bg_color: 背景颜色 (R, G, B, A)，A=透明度
            margin: 边距
        """
        self.font_size = font_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.margin = margin

        # 尝试加载等宽字体（降级到默认）
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
            )
        except:
            try:
                # macOS 备用字体
                self.font = ImageFont.truetype(
                    "/System/Library/Fonts/Menlo.ttc", font_size
                )
            except:
                # 降级到默认字体
                self.font = ImageFont.load_default()

    def render(self, frame: np.ndarray, text_lines: List[str]) -> np.ndarray:
        """在帧上叠加文本

        Args:
            frame: RGB 图像 (H, W, 3)，uint8
            text_lines: 文本行列表

        Returns:
            叠加后的 RGB 图像
        """
        if not text_lines:
            return frame

        # 转换为 PIL Image
        img = Image.fromarray(frame)

        # 创建可绘制对象（RGBA 模式用于半透明）
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 计算文本块尺寸
        line_height = self.font_size + 5
        text_block_height = len(text_lines) * line_height + 2 * self.margin

        # 计算最大文本宽度
        try:
            # PIL 10.0+ 使用 textlength
            max_text_width = max(
                draw.textlength(line, font=self.font) for line in text_lines
            )
        except AttributeError:
            # 旧版本 PIL 降级方案
            max_text_width = max(
                draw.textsize(line, font=self.font)[0] for line in text_lines
            )

        text_block_width = int(max_text_width) + 2 * self.margin

        # 绘制半透明背景矩形
        bg_rect = [
            self.margin,
            self.margin,
            self.margin + text_block_width,
            self.margin + text_block_height,
        ]
        draw.rectangle(bg_rect, fill=self.bg_color)

        # 绘制文本
        y_offset = self.margin * 2
        for line in text_lines:
            draw.text(
                (self.margin * 2, y_offset),
                line,
                fill=self.text_color + (255,),  # 添加 alpha 通道
                font=self.font,
            )
            y_offset += line_height

        # 合并图层
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)

        # 转换回 RGB
        return np.array(img.convert("RGB"))


class VideoRecorder:
    """视频录制器 - 集成 MJX 数据渲染和视频保存

    核心功能：
    1. 从 MJX 数据（GPU）转换为 MuJoCo 数据（CPU）
    2. 渲染 RGB 帧
    3. 叠加训练指标
    4. 保存为 MP4 视频
    """

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
        """初始化视频录制器

        Args:
            mujoco_model: MuJoCo 模型（用于创建 mj_data 容器）
            output_dir: 视频输出目录
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            camera_id: 相机 ID
            camera_name: 相机名称（推荐使用）
        """
        self.mj_model = mujoco_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        # 创建 MuJoCo 渲染器
        self.renderer = MujocoRenderer(
            model=mujoco_model,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
        )

        # 创建文本叠加渲染器
        self.overlay = OverlayRenderer(font_size=28)

        # 创建 CPU 端的 mj_data 容器（接收 MJX 数据）
        self.mj_data = mujoco.MjData(mujoco_model)

        # 录制状态
        self.frames: List[np.ndarray] = []
        self.is_recording = False

    def start_recording(self):
        """开始录制（清空帧缓冲）"""
        self.frames = []
        self.is_recording = True

    def add_frame_from_mjx(
        self,
        mjx_data: Any,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """从 MJX 数据添加一帧

        Args:
            mjx_data: MJX 数据（单个环境，在 GPU 上）
            metrics: 训练指标字典（可选，用于叠加显示）
        """
        if not self.is_recording:
            return

        try:
            # 步骤 1: MJX → MuJoCo 转换（GPU → CPU）
            # 使用官方推荐的 mjx.put_data()
            mjx.put_data(self.mj_data, mjx_data)

            # 步骤 2: 前向运动学（更新导出的量）
            mujoco.mj_forward(self.mj_model, self.mj_data)

            # 步骤 3: 渲染 RGB 帧
            frame = self.renderer.render(self.mj_data)

            # 步骤 4: 叠加训练指标
            if metrics:
                text_lines = self._format_metrics(metrics)
                frame = self.overlay.render(frame, text_lines)

            # 步骤 5: 添加到帧缓冲
            self.frames.append(frame)

        except Exception as e:
            # 错误处理：跳过异常帧，不中断训练
            print(f"警告：录制帧失败，跳过该帧: {e}")

    def save_video(self, filename: str) -> Path:
        """保存视频文件

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            保存的视频路径

        Raises:
            ValueError: 如果没有录制任何帧
        """
        if not self.frames:
            raise ValueError("没有录制任何帧，无法保存视频")

        output_path = self.output_dir / f"{filename}.mp4"

        try:
            # 使用 mediapy 保存（H.264 编码）
            mediapy.write_video(
                str(output_path),
                self.frames,
                fps=self.fps,
                codec="h264",
                qp=20,  # 质量参数（0-51，越小质量越高，推荐 18-23）
            )
        except Exception as e:
            print(f"错误：视频保存失败: {e}")
            raise
        finally:
            self.is_recording = False

        return output_path

    def _format_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """格式化指标为文本行

        Args:
            metrics: 指标字典

        Returns:
            格式化的文本行列表
        """
        lines = []

        # 第一行：奖励
        if "episode_reward" in metrics:
            lines.append(f"Reward: {metrics['episode_reward']:>7.2f}")

        # 第二行：训练进度
        if "update" in metrics:
            lines.append(f"Update: {int(metrics['update']):>6d}")

        if "env_steps" in metrics:
            lines.append(f"Steps:  {int(metrics['env_steps']):>8,}")

        # 分隔线
        if lines:
            lines.append("─" * 25)

        # 速度命令 vs 实际速度
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
        """关闭录制器（释放资源）"""
        self.renderer.close()
