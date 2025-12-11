"""
Setup脚本用于编译Cython扩展模块
编译命令: python setup_tdpm.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="tdpm_fast",
        sources=["tdpm_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",           # 最高优化级别
            "-march=native", # 针对当前CPU优化
            "-ffast-math",   # 快速数学运算
        ],
        extra_link_args=["-fopenmp"],  # 可选：启用OpenMP并行
        language="c++",
    )
]

setup(
    name="tdpm_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
        },
        annotate=True,  # 生成.html文件查看优化效果
    ),
    zip_safe=False,
)
