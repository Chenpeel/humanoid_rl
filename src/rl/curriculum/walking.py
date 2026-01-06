"""
Jiyuan 行走任务的课程学习（标准版）

这是 WalkingCurriculum 的标准实现，它现在是一个包装器，
负责加载标准的 configs/train/curriculum.yaml 配置文件。
"""

import os
import yaml
from .base import ConfigurableCurriculum

class WalkingCurriculum(ConfigurableCurriculum):
    """Jiyuan 标准行走课程学习

    自动加载 configs/train/curriculum.yaml。
    """

    def __init__(self):
        # 确定配置文件路径
        # 假设当前工作目录是项目根目录
        config_path = os.path.join("configs", "train", "curriculum.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"找不到标准课程配置文件: {config_path}\\n"
                "请确保从项目根目录运行，并检查 configs/train/curriculum.yaml 是否存在。"
            )

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 初始化基类
        super().__init__(
            config=config,
            env_config={},  # 标准课程在yaml中定义了各自的环境配置
            default_stage_name="StandardWalking"
        )