#!/usr/bin/env python3
"""
双足机器人强化学习训练框架

基于 Isaac Lab + PyTorch + RSL_RL

安装方式:
    # 开发模式安装（推荐，代码修改立即生效）
    pip install -e .

    # 标准安装
    pip install .

    # 安装额外依赖
    pip install -e ".[dev]"        # 开发工具
    pip install -e ".[vis]"        # 可视化工具
    pip install -e ".[imitation]"  # 模仿学习（BVH/FBX）
    pip install -e ".[all]"        # 所有依赖

依赖:
    - Isaac Lab >= 1.2.0（需要先安装 Isaac Sim 2024.1.1+）
    - RSL_RL >= 1.0.2（从 GitHub 安装）
    - PyTorch >= 2.0.0
    - Python >= 3.11, < 3.12
"""

from pathlib import Path
from setuptools import setup, find_packages

# 读取 README
ROOT_DIR = Path(__file__).parent
README_PATH = ROOT_DIR / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "双足机器人强化学习训练框架（基于 Isaac Lab + RSL_RL）"

# 版本信息
VERSION = "0.3.0"

# 核心依赖
# 注意：Isaac Lab 提供的依赖（torch, numpy, gymnasium）不在这里列出
# 这些依赖会在 Isaac Lab 安装时自动安装
INSTALL_REQUIRES = [
    # 配置和日志
    "pyyaml>=6.0",
    "tensorboard>=2.11.0",

    # 科学计算（Isaac Lab 已提供，但明确版本）
    # "torch>=2.0.0",  # 由 Isaac Lab 提供
    # "numpy>=1.23.0",  # 由 Isaac Lab 提供
    # "gymnasium>=0.29.0",  # 由 Isaac Lab 提供
]

# 开发工具依赖
DEV_REQUIRES = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# 可视化和分析工具
VIS_REQUIRES = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pandas>=2.0.0",
    "wandb>=0.15.0",  # Weights & Biases（可选）
]

# 模仿学习（BVH/FBX 支持）
IMITATION_REQUIRES = [
    "bvh>=0.3",  # PyPI 上最新版本是 0.3
    # "pyfbx",  # FBX SDK 需要单独安装
]

# 所有可选依赖
ALL_REQUIRES = DEV_REQUIRES + VIS_REQUIRES + IMITATION_REQUIRES

setup(
    name="isaaclab-biped-rl",
    version=VERSION,
    author="Chenpeel",
    author_email="chenpeel@foxmail.com",
    description="双足机器人强化学习训练框架（基于 Isaac Lab + RSL_RL）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenpeel/biped-rl",

    # 包配置
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs", "logs", "configs", "assets"]),
    python_requires=">=3.11,<3.12",  # Isaac Lab 包发行版要求 Python 3.11

    # 依赖
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "vis": VIS_REQUIRES,
        "imitation": IMITATION_REQUIRES,
        "all": ALL_REQUIRES,
    },

    # 入口点（命令行工具，可选）
    entry_points={
        "console_scripts": [
            # "biped-train=scripts.train:main",
            # "biped-play=scripts.play:main",
        ],
    },

    # 包数据
    package_data={
        "": ["*.yaml", "*.xml", "*.mjcf"],
    },
    include_package_data=True,

    # 分类信息
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],

    # 许可证
    license="MIT",

    # 关键词
    keywords="robotics reinforcement-learning isaac-lab rsl-rl biped locomotion sim2real",

    # 其他
    zip_safe=False,
)
