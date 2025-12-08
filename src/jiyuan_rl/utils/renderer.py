"""
MuJoCo渲染器 - 支持3D可视化
"""

import mujoco
import numpy as np
from typing import Optional
import time


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
            raise ImportError(
                "交互式查看器需要mujoco.viewer模块。"
                "请确保安装了完整的mujoco包。"
            )
    
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
