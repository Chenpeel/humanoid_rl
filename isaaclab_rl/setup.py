"""
Isaac Lab RL - Jiyuan 双足机器人强化学习项目

本包提供使用 Isaac Lab + PyTorch + RSL_RL 训练 Jiyuan 双足机器人的完整实现。

安装方式:
    # 开发模式（推荐）
    pip install -e .

    # 带额外依赖
    pip install -e ".[dev]"

依赖:
    - Isaac Lab >= 1.2.0
    - RSL_RL >= 1.0.2
    - PyTorch >= 2.0.0
"""

from setuptools import setup, find_packages
import os

# 读取 README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 版本信息
VERSION = "0.1.0"

setup(
    name="isaaclab_rl",
    version=VERSION,
    author="Jiyuan Robotics Team",
    author_email="",
    description="Jiyuan 双足机器人强化学习（Isaac Lab + RSL_RL）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",  # Isaac Lab 要求
    install_requires=[
        # 核心依赖（由 Isaac Lab 提供）
        # "torch>=2.0.0",
        # "numpy>=1.20.0",
        # "gymnasium>=0.29.0",

        # Isaac Lab 相关（需要先安装 Isaac Lab）
        # "omni-isaac-lab",

        # RSL_RL（需要单独安装）
        # "rsl-rl>=1.0.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "tensorboard>=2.11.0",
            "wandb>=0.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    zip_safe=False,
)
