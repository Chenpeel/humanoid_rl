# Makefile for JRL - JAX Reinforcement Learning Training
# 用于管理机器人训练（自动课程学习）

.PHONY: help sync sync-ksim sync-onnx lock install install-dev submodule-update train train-vis train-long train-long-vis train-test train-test-vis train-stand train-stand-vis train-custom train-custom-vis train-ksim train-ksim-stand train-ksim-walk eval play export infer infer-ksim-onnx visualize-mjcf clean clean-cache clean-logs clean-train clean-makelog clean-all
.DEFAULT_GOAL := help
SHELL := /bin/bash

# ==================== 配置变量 ====================

PROJECT_ROOT := $(shell pwd)

# uv：用于依赖管理/锁文件（替代 pip install）
# 允许用户通过 `make ... UV=/path/to/uv` 覆盖；若 UV 为空则自动探测。
UV ?=
ifeq ($(strip $(UV)),)
UV := $(shell \
	if command -v uv >/dev/null 2>&1; then command -v uv; \
	elif [ -x "$$HOME/.local/bin/uv" ]; then echo "$$HOME/.local/bin/uv"; \
	else echo uv; fi)
endif
UV_DEFAULT_INDEX ?= https://pypi.tuna.tsinghua.edu.cn/simple
UV_INDEX_STRATEGY ?= unsafe-first-match
# 可选：额外 index（逗号分隔 URL）
UV_INDEX ?=
UV_ENV ?= UV_DEFAULT_INDEX=$(UV_DEFAULT_INDEX) UV_INDEX_STRATEGY=$(UV_INDEX_STRATEGY) $(if $(UV_INDEX),UV_INDEX=$(UV_INDEX),)

# Python 解释器：优先使用绝对路径（减少对环境激活的依赖）
# 1) 项目 uv venv：$(PROJECT_ROOT)/.venv/bin/python
# 2) 当前激活环境：$CONDA_PREFIX/bin/python
# 3) 默认环境：~/.miniconda3/envs/jrl/bin/python
# 4) 回退：which python3 / python
PYTHON ?= $(shell \
	if [ -x "$(PROJECT_ROOT)/.venv/bin/python" ]; then echo "$(PROJECT_ROOT)/.venv/bin/python"; \
	elif [ -n "$$CONDA_PREFIX" ] && [ -x "$$CONDA_PREFIX/bin/python" ]; then echo "$$CONDA_PREFIX/bin/python"; \
	elif [ -x "$$HOME/.miniconda3/envs/jrl/bin/python" ]; then echo "$$HOME/.miniconda3/envs/jrl/bin/python"; \
	elif command -v python3 >/dev/null 2>&1; then command -v python3; \
	else command -v python; fi)

MUJOCO_GL_IS_CMDLINE := $(filter command line,$(origin MUJOCO_GL))
LOG_DIR := $(PROJECT_ROOT)/logs
# Make命令执行日志固定写入项目根目录下的logs/makelog，
# 避免用户覆写LOG_DIR（如指定某次训练run目录/某个checkpoint文件）导致日志目录解析失败。
MAKELOG_DIR := $(PROJECT_ROOT)/logs/makelog
CACHE_DIR := $(PROJECT_ROOT)/.jax_cache
TRAIN_SCRIPT := scripts/train.py
TRAIN_KSIM_SCRIPT := scripts/train_ksim.py
EVAL_SCRIPT := scripts/eval.py
EXPORT_SCRIPT := scripts/export.py
INFER_SCRIPT := scripts/infer_onnx.py
INFER_KSIM_SCRIPT := scripts/infer_onnx_ksim.py
VISUALIZE_MJCF_SCRIPT := scripts/visualize_mjcf.py

# 配置文件路径
CONFIG_TRAIN := configs/train/train.yaml
CONFIG_LONG := configs/train-10h/train.yaml
CONFIG_QUICK := configs/quick_test/train.yaml
CONFIG_STAND := configs/train-stand/train.yaml
CONFIG_KSIM_STAND := configs/ksim/gaoda_jiyuan_stand.yaml
CONFIG_KSIM_WALK := configs/ksim/gaoda_jiyuan_walk.yaml
LOAD_CKPT ?= $(load_ckpt)

# 日志时间戳生成函数
TIMESTAMP := $(shell date '+%Y%m%d_%H%M%S')
MAKELOG_FILE = $(MAKELOG_DIR)/$(1)_$(TIMESTAMP).log

# ==================== 帮助信息 ====================

help:
	@echo "JRL 训练系统 - Makefile 命令（自动课程学习）"
	@echo ""
	@echo "环境配置："
	@echo "  make sync                 使用 uv 同步基础依赖"
	@echo "  make sync-ksim            使用 uv 同步依赖（含 ksim extra）"
	@echo "  make sync-onnx            使用 uv 同步依赖（onnx/onnxruntime extra）"
	@echo "  make lock                 生成/更新 uv.lock（可复现）"
	@echo "  make install              = make sync"
	@echo "  make install-dev          = make sync + 可选依赖（dev/tensorboard）"
	@echo "  make check-env            检查 JAX/CUDA/ksim 环境"
	@echo "  make submodule-update     初始化/更新子模块（models）"
	@echo ""
	@echo "训练命令（自动课程学习）："
	@echo "  make train                标准训练（2048 envs，200M steps，约8-12小时）"
	@echo "  make train-long           长时间训练（4096 envs，500M steps，约24-48小时）"
	@echo "  make train-test           测试训练（256 envs，16M steps，约3-5小时）"
	@echo "  make train-stand          站立专训（walking env + 站立奖励，20M steps）"
	@echo "  make train-vis            标准训练 + MuJoCo窗口实时可视化（更慢）"
	@echo "  make train-long-vis       长时间训练 + MuJoCo窗口实时可视化（更慢）"
	@echo "  make train-test-vis       测试训练 + MuJoCo窗口实时可视化（更慢）"
	@echo "  make train-stand-vis      站立专训 + MuJoCo窗口实时可视化（更慢）"
	@echo "  make train-test ENV_TYPE=standing          用站立环境跑快速训练"
	@echo "  make train-custom CONFIG=... RESUME_FROM=...  从检查点继续训练"
	@echo "  make train-custom CONFIG=path/to/config.yaml  自定义配置训练"
	@echo "  make train RENDER=N       训练时开启MuJoCo窗口（0=不显示，N=每N步更新一次窗口）"
	@echo "  make train RENDER=10 VIEWER_SLEEP=0.01     限速渲染（减少CPU占用/更平滑）"
	@echo "  make train RENDER=10 RENDER_CPU=1          强制软件OpenGL渲染窗口（更慢，尽量减少GPU图形占用）"
	@echo "  make train ENABLE_VIDEO=1 VIDEO_INTERVAL=500  周期性录制训练视频到 logs/diy_train/.../videos"
	@echo ""
	@echo "ksim 训练（gaoda_jiyuan）："
	@echo "  make train-ksim-stand      ksim 站立专训（自动执行 sync-ksim）"
	@echo "  make train-ksim-stand load_ckpt=...ckpt.bin  从 checkpoint 开始站立训练"
	@echo "  make train-ksim-walk       ksim 行走训练（自动执行 sync-ksim）"
	@echo "  make train-ksim-walk load_ckpt=...ckpt.bin  从 checkpoint 开始行走训练"
	@echo ""
	@echo "课程学习机制："
	@echo "  - 阶段1 (0-20M env steps):    站立平衡"
	@echo "  - 阶段2 (20M-60M env steps): 低速行走"
	@echo "  - 阶段3 (60M+ env steps):    全速行走"
	@echo "  - 奖励权重和环境参数自动切换，无需手动干预"
	@echo ""
	@echo "评估命令："
	@echo "  make eval CKPT=path/to/checkpoint         评估模型（默认渲染，不保存视频）"
	@echo "  make eval CKPT=... CPU=1                   使用CPU评估"
	@echo "  make eval CKPT=... RENDER=0                不渲染（最快）"
	@echo "  make eval CKPT=... RENDER=10               每10步渲染（加速）"
	@echo "  make eval CKPT=... SAVE_VIDEO=1            保存视频（较慢）"
	@echo "  make eval CKPT=... NO_VIDEO=1              明确不保存视频"
	@echo "  make eval CKPT=... ENV_TYPE=walking        指定环境类型"
	@echo "  make eval CKPT=... ROBOT_NAME=unitree_h1   指定机器人模型"
	@echo "  make play CKPT=...                         仅可视化播放（不评估）"
	@echo "  make play CKPT=... PLAY_CONFIG=...         使用YAML配置复现env_config（如零速度命令）"
	@echo "  make export CKPT=... FORMAT=onnx OUT_DIR=exported_models  导出模型（ONNX/TF/msgpack）"
	@echo "  make infer MODEL=... ENV_TYPE=walking      ONNX 推理/回放（可 SAVE_VIDEO=1 录制）"
	@echo "  make infer-ksim-onnx MODEL=... CKPT=...    ksim/xax ONNX 推理回放（可 SAVE_VIDEO=1 录制）"
	@echo ""
	@echo "开发工具："
	@echo "  make tensorboard          启动TensorBoard"
	@echo "  make format               格式化代码"
	@echo "  make test                 运行单元测试"
	@echo "  make validate-config      验证配置文件语法"
	@echo "  make visualize-mjcf       MJCF 模型可视化（支持热刷新）"
	@echo "  make visualize-mjcf XML=robots/gaoda_jiyuan/jiyuan.xml AUTORELOAD=0 MODE=launch"
	@echo "  make visualize-mjcf XML=robots/gaoda_jiyuan/jiyuan.xml NO_INTERACTIVE=1 GRAVITY=1"
	@echo ""
	@echo "清理命令："
	@echo "  make clean                清理临时文件"
	@echo "  make clean-cache          清理JAX缓存"
	@echo "  make clean-train          清理训练日志（慎用！会删除所有模型检查点）"
	@echo "  make clean-makelog        清理make执行日志"
	@echo "  make clean-logs           清理所有日志（训练日志+make日志）"
	@echo "  make clean-all            清理所有（保留训练日志，仅删除缓存+make日志+临时文件）"
	@echo ""
	@echo "日志说明："
	@echo "  训练日志: logs/diy_train/ppo_[时间戳]/ (包含TensorBoard、检查点、视频等)"
	@echo "  Make日志: logs/makelog/[命令]_[时间戳].log (make命令执行记录)"
	@echo "  实时终端输出和日志文件内容完全一致（使用tee实现）"

# ==================== 安装相关 ====================

sync:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	$(UV_ENV) $(UV) sync

sync-ksim:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	$(UV_ENV) $(UV) sync --extra ksim --inexact

sync-onnx:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	$(UV_ENV) $(UV) sync --extra onnx --inexact

lock:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	$(UV_ENV) $(UV) lock

install: sync
	@echo "✓ 依赖已通过 uv 同步完成"

install-dev:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	@echo "=== 安装开发依赖（dev/tensorboard）==="
	$(UV_ENV) $(UV) sync --extra tensorboard --extra dev
	@echo "✓ 开发依赖已通过 uv 同步完成"

check-env:
	@echo "=== 检查环境 ==="
	@echo "Python版本: $$($(PYTHON) --version)"
	@echo "JAX版本: $$($(PYTHON) -c 'import jax; print(jax.__version__)')"
	@echo "JAX后端: $$($(PYTHON) -c 'import jax; print(jax.default_backend())')"
	@$(PYTHON) -c 'import ksim; print("ksim版本:", getattr(ksim, "__version__", "unknown"))' 2>/dev/null || echo "ksim: 未安装（请先 make sync-ksim）"
	@echo "可用设备:"
	@$(PYTHON) -c 'import jax; [print(f"  - {d}") for d in jax.devices()]'

# ==================== Git 子模块 ====================

submodule-update:
	@echo "=== 更新子模块 ==="
	git submodule sync --recursive
	git submodule update --init --recursive
	@echo "✓ 子模块更新完成"

# ==================== 训练相关 ====================

train:
	@LOGFILE="$(call MAKELOG_FILE,train)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 标准训练（自动课程学习）==="; \
	echo "配置: $(CONFIG_TRAIN)"; \
	echo "环境数: 2048，总步数: 200M"; \
	echo "预计时间: 8-12小时"; \
	echo "开始时间: $$(date)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 训练日志 ==="; \
		echo "命令: make train"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_TRAIN)"; \
		echo ""; \
			CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_TRAIN)"; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(RESUME_FROM)" ]; then CMD="$$CMD --resume-from $(RESUME_FROM)"; echo "恢复: $(RESUME_FROM)"; fi; \
			if [ -n "$(ENABLE_VIDEO)" ]; then CMD="$$CMD --enable-video"; echo "训练视频: 开启"; \
			elif [ -n "$(DISABLE_VIDEO)" ]; then CMD="$$CMD --disable-video"; echo "训练视频: 关闭"; fi; \
			if [ -n "$(VIDEO_INTERVAL)" ]; then CMD="$$CMD --video-interval $(VIDEO_INTERVAL)"; echo "video_interval: $(VIDEO_INTERVAL)"; fi; \
			if [ -n "$(VIDEO_FRAMES)" ]; then CMD="$$CMD --video-frames $(VIDEO_FRAMES)"; echo "video_frames: $(VIDEO_FRAMES)"; fi; \
			if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "video_fps: $(VIDEO_FPS)"; fi; \
			if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --video-width $(RENDER_WIDTH)"; echo "video_width: $(RENDER_WIDTH)"; \
			elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --video-width $(WIDTH)"; echo "video_width: $(WIDTH)"; \
			elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --video-width $(WEIGHT)"; echo "video_width: $(WEIGHT)"; fi; \
			if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --video-height $(RENDER_HEIGHT)"; echo "video_height: $(RENDER_HEIGHT)"; \
			elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --video-height $(HEIGHT)"; echo "video_height: $(HEIGHT)"; fi; \
			if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --video-camera $(CAMERA_NAME)"; echo "video_camera: $(CAMERA_NAME)"; fi; \
			if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "render: $(RENDER)"; fi; \
			if [ -n "$(RENDER_STEPS)" ]; then CMD="$$CMD --render-steps $(RENDER_STEPS)"; echo "render_steps: $(RENDER_STEPS)"; fi; \
			if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
			if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
			if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; \
			elif [ -n "$(RENDER)" ] && [ "$(RENDER)" != "0" ]; then CMD="MUJOCO_GL=glfw $$CMD"; echo "MUJOCO_GL: glfw"; fi; \
			eval $$CMD 2>&1; \
			EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="; \
	echo "结束时间: $$(date)"

train-long:
	@LOGFILE="$(call MAKELOG_FILE,train-long)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 长时间训练（自动课程学习）==="; \
	echo "配置: $(CONFIG_LONG)"; \
	echo "环境数: 4096，总步数: 500M"; \
	echo "预计时间: 24-48小时"; \
	echo "开始时间: $$(date)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 训练日志 ==="; \
		echo "命令: make train-long"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_LONG)"; \
		echo ""; \
			CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_LONG)"; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(RESUME_FROM)" ]; then CMD="$$CMD --resume-from $(RESUME_FROM)"; echo "恢复: $(RESUME_FROM)"; fi; \
			if [ -n "$(ENABLE_VIDEO)" ]; then CMD="$$CMD --enable-video"; echo "训练视频: 开启"; \
			elif [ -n "$(DISABLE_VIDEO)" ]; then CMD="$$CMD --disable-video"; echo "训练视频: 关闭"; fi; \
			if [ -n "$(VIDEO_INTERVAL)" ]; then CMD="$$CMD --video-interval $(VIDEO_INTERVAL)"; echo "video_interval: $(VIDEO_INTERVAL)"; fi; \
			if [ -n "$(VIDEO_FRAMES)" ]; then CMD="$$CMD --video-frames $(VIDEO_FRAMES)"; echo "video_frames: $(VIDEO_FRAMES)"; fi; \
			if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "video_fps: $(VIDEO_FPS)"; fi; \
			if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --video-width $(RENDER_WIDTH)"; echo "video_width: $(RENDER_WIDTH)"; \
			elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --video-width $(WIDTH)"; echo "video_width: $(WIDTH)"; \
			elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --video-width $(WEIGHT)"; echo "video_width: $(WEIGHT)"; fi; \
			if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --video-height $(RENDER_HEIGHT)"; echo "video_height: $(RENDER_HEIGHT)"; \
			elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --video-height $(HEIGHT)"; echo "video_height: $(HEIGHT)"; fi; \
			if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --video-camera $(CAMERA_NAME)"; echo "video_camera: $(CAMERA_NAME)"; fi; \
			if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "render: $(RENDER)"; fi; \
			if [ -n "$(RENDER_STEPS)" ]; then CMD="$$CMD --render-steps $(RENDER_STEPS)"; echo "render_steps: $(RENDER_STEPS)"; fi; \
			if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
			if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
			if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; \
			elif [ -n "$(RENDER)" ] && [ "$(RENDER)" != "0" ]; then CMD="MUJOCO_GL=glfw $$CMD"; echo "MUJOCO_GL: glfw"; fi; \
			eval $$CMD 2>&1; \
			EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="; \
	echo "日志文件: $$LOGFILE"; \
	echo "结束时间: $$(date)"

train-test:
		@LOGFILE="$(call MAKELOG_FILE,train-test)"; \
		mkdir -p "$$(dirname "$$LOGFILE")"; \
		echo "=== 快速测试（自动课程学习）==="; \
		echo "配置: $(CONFIG_QUICK)"; \
		echo "环境数: 256，总步数: 16M"; \
		echo "预计时间: 3-5小时（含首次JIT编译开销）"; \
		echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 测试日志 ==="; \
		echo "命令: make train-test"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_QUICK)"; \
		echo ""; \
			CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_QUICK)"; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(RESUME_FROM)" ]; then CMD="$$CMD --resume-from $(RESUME_FROM)"; echo "恢复: $(RESUME_FROM)"; fi; \
			if [ -n "$(ENABLE_VIDEO)" ]; then CMD="$$CMD --enable-video"; echo "训练视频: 开启"; \
			elif [ -n "$(DISABLE_VIDEO)" ]; then CMD="$$CMD --disable-video"; echo "训练视频: 关闭"; fi; \
			if [ -n "$(VIDEO_INTERVAL)" ]; then CMD="$$CMD --video-interval $(VIDEO_INTERVAL)"; echo "video_interval: $(VIDEO_INTERVAL)"; fi; \
			if [ -n "$(VIDEO_FRAMES)" ]; then CMD="$$CMD --video-frames $(VIDEO_FRAMES)"; echo "video_frames: $(VIDEO_FRAMES)"; fi; \
			if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "video_fps: $(VIDEO_FPS)"; fi; \
			if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --video-width $(RENDER_WIDTH)"; echo "video_width: $(RENDER_WIDTH)"; \
			elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --video-width $(WIDTH)"; echo "video_width: $(WIDTH)"; \
			elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --video-width $(WEIGHT)"; echo "video_width: $(WEIGHT)"; fi; \
			if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --video-height $(RENDER_HEIGHT)"; echo "video_height: $(RENDER_HEIGHT)"; \
			elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --video-height $(HEIGHT)"; echo "video_height: $(HEIGHT)"; fi; \
			if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --video-camera $(CAMERA_NAME)"; echo "video_camera: $(CAMERA_NAME)"; fi; \
			if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "render: $(RENDER)"; fi; \
			if [ -n "$(RENDER_STEPS)" ]; then CMD="$$CMD --render-steps $(RENDER_STEPS)"; echo "render_steps: $(RENDER_STEPS)"; fi; \
			if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
			if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
			if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; \
			elif [ -n "$(RENDER)" ] && [ "$(RENDER)" != "0" ]; then CMD="MUJOCO_GL=glfw $$CMD"; echo "MUJOCO_GL: glfw"; fi; \
			eval $$CMD 2>&1; \
			EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 测试完成 ==="

train-stand:
	@LOGFILE="$(call MAKELOG_FILE,train-stand)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 站立专训（walking env + 单阶段站立课程）==="; \
	echo "配置: $(CONFIG_STAND)"; \
	echo "总步数: 20M env steps"; \
	echo "开始时间: $$(date)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 训练日志 ==="; \
		echo "命令: make train-stand"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_STAND)"; \
		echo ""; \
			CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_STAND)"; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(RESUME_FROM)" ]; then CMD="$$CMD --resume-from $(RESUME_FROM)"; echo "恢复: $(RESUME_FROM)"; fi; \
			if [ -n "$(ENABLE_VIDEO)" ]; then CMD="$$CMD --enable-video"; echo "训练视频: 开启"; \
			elif [ -n "$(DISABLE_VIDEO)" ]; then CMD="$$CMD --disable-video"; echo "训练视频: 关闭"; fi; \
			if [ -n "$(VIDEO_INTERVAL)" ]; then CMD="$$CMD --video-interval $(VIDEO_INTERVAL)"; echo "video_interval: $(VIDEO_INTERVAL)"; fi; \
			if [ -n "$(VIDEO_FRAMES)" ]; then CMD="$$CMD --video-frames $(VIDEO_FRAMES)"; echo "video_frames: $(VIDEO_FRAMES)"; fi; \
			if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "video_fps: $(VIDEO_FPS)"; fi; \
			if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --video-width $(RENDER_WIDTH)"; echo "video_width: $(RENDER_WIDTH)"; \
			elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --video-width $(WIDTH)"; echo "video_width: $(WIDTH)"; \
			elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --video-width $(WEIGHT)"; echo "video_width: $(WEIGHT)"; fi; \
			if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --video-height $(RENDER_HEIGHT)"; echo "video_height: $(RENDER_HEIGHT)"; \
			elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --video-height $(HEIGHT)"; echo "video_height: $(HEIGHT)"; fi; \
			if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --video-camera $(CAMERA_NAME)"; echo "video_camera: $(CAMERA_NAME)"; fi; \
			if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "render: $(RENDER)"; fi; \
			if [ -n "$(RENDER_STEPS)" ]; then CMD="$$CMD --render-steps $(RENDER_STEPS)"; echo "render_steps: $(RENDER_STEPS)"; fi; \
			if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
			if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
			if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; \
			elif [ -n "$(RENDER)" ] && [ "$(RENDER)" != "0" ]; then CMD="MUJOCO_GL=glfw $$CMD"; echo "MUJOCO_GL: glfw"; fi; \
			# 使用 script 分配 pseudo-TTY，确保 Rich Live 进度条在日志模式(tee)下也能实时刷新 \
			script -q -e -c "$$CMD" /dev/null 2>&1; \
			EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="; \
	echo "结束时间: $$(date)"

train-custom:
	@if [ -z "$(CONFIG)" ]; then \
		echo "错误: 请指定配置文件 CONFIG=path/to/config.yaml"; \
		echo "示例: make train-custom CONFIG=configs/train/train_default.yaml"; \
		exit 1; \
	fi
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "错误: 配置文件不存在: $(CONFIG)"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,train-custom)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 自定义配置训练 ==="; \
	echo "配置: $(CONFIG)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 自定义训练日志 ==="; \
		echo "命令: make train-custom CONFIG=$(CONFIG)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG)"; \
		echo ""; \
			CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG)"; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(RESUME_FROM)" ]; then CMD="$$CMD --resume-from $(RESUME_FROM)"; echo "恢复: $(RESUME_FROM)"; fi; \
			if [ -n "$(ENABLE_VIDEO)" ]; then CMD="$$CMD --enable-video"; echo "训练视频: 开启"; \
			elif [ -n "$(DISABLE_VIDEO)" ]; then CMD="$$CMD --disable-video"; echo "训练视频: 关闭"; fi; \
			if [ -n "$(VIDEO_INTERVAL)" ]; then CMD="$$CMD --video-interval $(VIDEO_INTERVAL)"; echo "video_interval: $(VIDEO_INTERVAL)"; fi; \
			if [ -n "$(VIDEO_FRAMES)" ]; then CMD="$$CMD --video-frames $(VIDEO_FRAMES)"; echo "video_frames: $(VIDEO_FRAMES)"; fi; \
			if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "video_fps: $(VIDEO_FPS)"; fi; \
			if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --video-width $(RENDER_WIDTH)"; echo "video_width: $(RENDER_WIDTH)"; \
			elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --video-width $(WIDTH)"; echo "video_width: $(WIDTH)"; \
			elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --video-width $(WEIGHT)"; echo "video_width: $(WEIGHT)"; fi; \
			if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --video-height $(RENDER_HEIGHT)"; echo "video_height: $(RENDER_HEIGHT)"; \
			elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --video-height $(HEIGHT)"; echo "video_height: $(HEIGHT)"; fi; \
			if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --video-camera $(CAMERA_NAME)"; echo "video_camera: $(CAMERA_NAME)"; fi; \
			if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "render: $(RENDER)"; fi; \
			if [ -n "$(RENDER_STEPS)" ]; then CMD="$$CMD --render-steps $(RENDER_STEPS)"; echo "render_steps: $(RENDER_STEPS)"; fi; \
			if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
			if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
			if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; \
			elif [ -n "$(RENDER)" ] && [ "$(RENDER)" != "0" ]; then CMD="MUJOCO_GL=glfw $$CMD"; echo "MUJOCO_GL: glfw"; fi; \
			eval $$CMD 2>&1; \
			EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="

train-vis:
	@$(MAKE) train RENDER=1

train-long-vis:
	@$(MAKE) train-long RENDER=1

train-test-vis:
	@$(MAKE) train-test RENDER=1

train-stand-vis:
	@$(MAKE) train-stand RENDER=1

train-custom-vis:
	@$(MAKE) train-custom RENDER=1

# ==================== ksim 训练相关 ====================

train-ksim-stand: sync-ksim
	@LOGFILE="$(call MAKELOG_FILE,train-ksim-stand)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== ksim 站立专训 ==="; \
	echo "配置: $(CONFIG_KSIM_STAND)"; \
	echo "开始时间: $$(date)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 训练日志 ==="; \
		echo "命令: make train-ksim-stand"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_KSIM_STAND)"; \
		echo ""; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_KSIM_SCRIPT) --config $(CONFIG_KSIM_STAND)"; \
		if [ -n "$(LOAD_CKPT)" ]; then CMD="$$CMD --load-ckpt $(LOAD_CKPT)"; echo "load_ckpt: $(LOAD_CKPT)"; fi; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="

train-ksim-walk: sync-ksim
	@LOGFILE="$(call MAKELOG_FILE,train-ksim-walk)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== ksim 行走训练 ==="; \
	echo "配置: $(CONFIG_KSIM_WALK)"; \
	echo "开始时间: $$(date)"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 训练日志 ==="; \
		echo "命令: make train-ksim-walk"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_KSIM_WALK)"; \
		echo ""; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(TRAIN_KSIM_SCRIPT) --config $(CONFIG_KSIM_WALK)"; \
		if [ -n "$(LOAD_CKPT)" ]; then CMD="$$CMD --load-ckpt $(LOAD_CKPT)"; echo "load_ckpt: $(LOAD_CKPT)"; fi; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="

train-ksim: train-ksim-walk

validate-config:
	@echo "=== 验证配置文件 ==="
	@$(PYTHON) -c "import yaml, sys; configs = ['$(CONFIG_TRAIN)', '$(CONFIG_LONG)', '$(CONFIG_QUICK)', '$(CONFIG_STAND)', '$(CONFIG_KSIM_STAND)', '$(CONFIG_KSIM_WALK)']; [yaml.safe_load(open(c)) or print(f'✓ {c}') for c in configs]; print('✓ 所有配置文件有效')"

# ==================== 评估相关 ====================

eval:
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make eval CKPT=logs/ppo_*/checkpoints/best_model"; \
		echo "示例: make eval CKPT=models/xxx ENV_TYPE=walking"; \
		echo "示例: make eval CKPT=models/xxx CPU=1  # 使用CPU避免GPU冲突"; \
		echo "示例: make eval CKPT=models/xxx RENDER=0  # 不渲染（最快）"; \
		echo "示例: make eval CKPT=models/xxx RENDER=10  # 每10步渲染（加速）"; \
		echo "示例: make eval CKPT=models/xxx SAVE_VIDEO=1 VIDEO_PATH=eval.mp4  # 保存视频（较慢）"; \
		echo "示例: make eval CKPT=models/xxx ROBOT_NAME=unitree_h1  # 指定机器人模型"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,eval)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 评估模型 ==="; \
	echo "检查点: $(CKPT)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 评估日志 ==="; \
		echo "命令: make eval CKPT=$(CKPT)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "检查点: $(CKPT)"; \
		CKPT_FORMAT="$$($(PYTHON) -c 'import sys; sys.path.insert(0, "scripts"); from checkpoint_compat import resolve_checkpoint; print(resolve_checkpoint(sys.argv[1]).format.value)' "$(CKPT)" 2>/dev/null || true)"; \
		if [ "$$CKPT_FORMAT" = "xax_tar" ]; then \
			echo "[deps] 检测到 xax/ksim checkpoint，自动同步 ksim extra ..."; \
			$(UV_ENV) $(UV) sync --extra ksim --inexact || exit $$?; \
		fi; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(EVAL_SCRIPT) --checkpoint $(CKPT)"; \
		if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
		if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; echo "设备: CPU"; fi; \
		if [ -n "$(NO_JAX_PREALLOC)" ]; then CMD="$$CMD --no-jax-prealloc"; echo "JAX预分配: 关闭"; fi; \
		if [ -n "$(JAX_MEM_FRACTION)" ]; then CMD="$$CMD --jax-mem-fraction $(JAX_MEM_FRACTION)"; echo "JAX显存比例: $(JAX_MEM_FRACTION)"; fi; \
		if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "渲染间隔: $(RENDER)"; fi; \
		if [ -n "$(NUM_EPISODES)" ]; then CMD="$$CMD --num-episodes $(NUM_EPISODES)"; echo "num-episodes: $(NUM_EPISODES)"; fi; \
		if [ -n "$(MAX_STEPS)" ]; then CMD="$$CMD --max-steps $(MAX_STEPS)"; echo "max-steps: $(MAX_STEPS)"; fi; \
		if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
		if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --render-width $(RENDER_WIDTH)"; echo "渲染宽度: $(RENDER_WIDTH)"; \
		elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --render-width $(WIDTH)"; echo "渲染宽度: $(WIDTH)"; \
		elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --render-width $(WEIGHT)"; echo "渲染宽度: $(WEIGHT)"; fi; \
		if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --render-height $(RENDER_HEIGHT)"; echo "渲染高度: $(RENDER_HEIGHT)"; \
		elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --render-height $(HEIGHT)"; echo "渲染高度: $(HEIGHT)"; fi; \
		if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --camera-name $(CAMERA_NAME)"; echo "相机: $(CAMERA_NAME)"; fi; \
		if [ -n "$(DEBUG_STEPS)" ]; then CMD="$$CMD --debug-first-n-steps $(DEBUG_STEPS)"; echo "调试步数: $(DEBUG_STEPS)"; fi; \
		if [ -n "$(NO_VIDEO)" ]; then CMD="$$CMD --no-save-video"; echo "保存视频: 否"; \
		elif [ -n "$(SAVE_VIDEO)" ]; then CMD="$$CMD --save-video"; echo "保存视频: 是"; fi; \
		if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; echo "视频路径: $(VIDEO_PATH)"; fi; \
		if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "视频FPS: $(VIDEO_FPS)"; fi; \
		if [ -n "$(ROBOT_NAME)" ]; then CMD="$$CMD --robot-name $(ROBOT_NAME)"; echo "机器人名称: $(ROBOT_NAME)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

play:
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make play CKPT=logs/ppo_*/checkpoints/best_model"; \
		echo "示例: make play CKPT=models/xxx RENDER=1 REALTIME=1"; \
		echo "示例: make play CKPT=models/xxx RENDER=10  # 每10步渲染（加速）"; \
		echo "示例: make play CKPT=models/xxx SAVE_VIDEO=1 VIDEO_PATH=play.mp4  # 保存视频"; \
		echo "示例: make play CKPT=models/xxx SAVE_VIDEO=1 RENDER_WIDTH=1280 RENDER_HEIGHT=720  # 指定分辨率"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,play)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 播放策略（仅可视化）==="; \
	echo "检查点: $(CKPT)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 播放日志 ==="; \
		echo "命令: make play CKPT=$(CKPT)"; \
			echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
			CKPT_FORMAT="$$($(PYTHON) -c 'import sys; sys.path.insert(0, "scripts"); from checkpoint_compat import resolve_checkpoint; print(resolve_checkpoint(sys.argv[1]).format.value)' "$(CKPT)" 2>/dev/null || true)"; \
			if [ "$$CKPT_FORMAT" = "xax_tar" ]; then \
				echo "[deps] 检测到 xax/ksim checkpoint，自动同步 ksim extra ..."; \
				$(UV_ENV) $(UV) sync --extra ksim --inexact || exit $$?; \
			fi; \
			CMD="FORCE_COLOR=1 $(PYTHON) scripts/play.py --checkpoint $(CKPT)"; \
			if [ -n "$(PLAY_CONFIG)" ]; then CMD="$$CMD --config $(PLAY_CONFIG)"; echo "配置: $(PLAY_CONFIG)"; fi; \
			if [ -n "$(XML_PATH)" ]; then CMD="$$CMD --xml-path $(XML_PATH)"; echo "模型: $(XML_PATH)"; fi; \
			if [ -n "$(USE_LOCAL_MJCF)" ]; then CMD="$$CMD --use-local-mjcf"; echo "模型: local mjcf"; fi; \
			if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
			if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; echo "设备: CPU"; fi; \
		if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "渲染间隔: $(RENDER)"; fi; \
		if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
		if [ -n "$(STATUS_EVERY)" ]; then CMD="$$CMD --status-every $(STATUS_EVERY)"; echo "status-every: $(STATUS_EVERY)s"; fi; \
		if [ -n "$(REALTIME)" ]; then CMD="$$CMD --realtime"; echo "播放: realtime"; fi; \
		if [ -n "$(EPISODES)" ]; then CMD="$$CMD --episodes $(EPISODES)"; echo "episodes: $(EPISODES)"; fi; \
		if [ -n "$(MAX_STEPS)" ]; then CMD="$$CMD --max-steps $(MAX_STEPS)"; echo "max-steps: $(MAX_STEPS)"; fi; \
		if [ -n "$(STOCHASTIC)" ]; then CMD="$$CMD --stochastic"; echo "动作: stochastic"; fi; \
		if [ -n "$(NO_JIT)" ]; then CMD="$$CMD --no-jit"; echo "JIT: 关闭"; fi; \
		if [ -n "$(SAVE_VIDEO)" ]; then CMD="$$CMD --save-video"; echo "保存视频: 是"; fi; \
		if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; echo "视频路径: $(VIDEO_PATH)"; fi; \
		if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "视频FPS: $(VIDEO_FPS)"; fi; \
		if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --render-width $(RENDER_WIDTH)"; echo "渲染宽度: $(RENDER_WIDTH)"; \
		elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --render-width $(WIDTH)"; echo "渲染宽度: $(WIDTH)"; \
		elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --render-width $(WEIGHT)"; echo "渲染宽度: $(WEIGHT)"; fi; \
		if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --render-height $(RENDER_HEIGHT)"; echo "渲染高度: $(RENDER_HEIGHT)"; \
		elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --render-height $(HEIGHT)"; echo "渲染高度: $(HEIGHT)"; fi; \
		if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --camera-name $(CAMERA_NAME)"; echo "相机: $(CAMERA_NAME)"; fi; \
		if [ -n "$(RECORD_INTERVAL)" ]; then CMD="$$CMD --record-interval $(RECORD_INTERVAL)"; echo "录制间隔: $(RECORD_INTERVAL)"; fi; \
		if [ -n "$(NO_JAX_PREALLOC)" ]; then CMD="$$CMD --no-jax-prealloc"; echo "JAX预分配: 关闭"; fi; \
		if [ -n "$(JAX_MEM_FRACTION)" ]; then CMD="$$CMD --jax-mem-fraction $(JAX_MEM_FRACTION)"; echo "JAX显存比例: $(JAX_MEM_FRACTION)"; fi; \
		if [ -n "$(ROBOT_NAME)" ]; then CMD="$$CMD --robot-name $(ROBOT_NAME)"; echo "机器人名称: $(ROBOT_NAME)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

export: sync-onnx
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make export CKPT=logs/ppo_*/checkpoints/best_model FORMAT=onnx"; \
		echo "示例: make export CKPT=logs/ksim_train/gaoda_jiyuan_task/run_000 FORMAT=onnx  # xax/ksim"; \
		echo "示例: make export CKPT=models/xxx FORMAT=all OUT_DIR=exported_models"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,export)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== 导出模型 ==="; \
	echo "检查点: $(CKPT)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 导出日志 ==="; \
		echo "命令: make export CKPT=$(CKPT)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "检查点: $(CKPT)"; \
		CKPT_FORMAT="$$($(PYTHON) -c 'import sys; sys.path.insert(0, "scripts"); from checkpoint_compat import resolve_checkpoint; print(resolve_checkpoint(sys.argv[1]).format.value)' "$(CKPT)" 2>/dev/null || true)"; \
		if [ "$$CKPT_FORMAT" = "xax_tar" ]; then \
			echo "[deps] 检测到 xax/ksim checkpoint，自动同步 ksim+onnx extras ..."; \
			$(UV_ENV) $(UV) sync --extra ksim --extra onnx --inexact || exit $$?; \
		fi; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(EXPORT_SCRIPT) --checkpoint-path $(CKPT)"; \
		if [ -n "$(OUT_DIR)" ]; then CMD="$$CMD --output-dir $(OUT_DIR)"; echo "输出目录: $(OUT_DIR)"; fi; \
		if [ -n "$(FORMAT)" ]; then CMD="$$CMD --format $(FORMAT)"; echo "导出格式: $(FORMAT)"; fi; \
		if [ -n "$(USE_BEST)" ]; then CMD="$$CMD --use-best"; echo "use-best: true"; fi; \
		if [ -n "$(CPU)" ]; then CMD="JAX_PLATFORMS=cpu $$CMD"; echo "设备: CPU (JAX_PLATFORMS=cpu)"; fi; \
		if [ -n "$(NO_JAX_PREALLOC)" ]; then CMD="XLA_PYTHON_CLIENT_PREALLOCATE=false $$CMD"; echo "JAX预分配: 关闭"; fi; \
		if [ -n "$(JAX_MEM_FRACTION)" ]; then CMD="XLA_PYTHON_CLIENT_MEM_FRACTION=$(JAX_MEM_FRACTION) $$CMD"; echo "JAX显存比例: $(JAX_MEM_FRACTION)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

infer: sync-onnx
	@if [ -z "$(MODEL)" ]; then \
		echo "错误: 请指定 ONNX 模型路径 MODEL=..."; \
		echo "示例: make infer MODEL=exported_models/policy_step123.onnx"; \
		echo "示例: make infer MODEL=exported_models/policy_step123.onnx ENV_TYPE=standing RENDER=1"; \
		echo "示例: make infer MODEL=... SAVE_VIDEO=1 VIDEO_PATH=infer.mp4  # 录制视频"; \
		echo "示例: make infer MODEL=... CPU=1  # 使用CPU运行MJX环境"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,infer)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== ONNX 推理/回放 ==="; \
	echo "模型: $(MODEL)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 推理日志 ==="; \
		echo "命令: make infer MODEL=$(MODEL)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "模型: $(MODEL)"; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(INFER_SCRIPT) --model $(MODEL)"; \
		if [ -n "$(CONFIG)" ]; then CMD="$$CMD --config $(CONFIG)"; echo "配置: $(CONFIG)"; fi; \
		if [ -n "$(XML_PATH)" ]; then CMD="$$CMD --xml-path $(XML_PATH)"; echo "模型: $(XML_PATH)"; fi; \
		if [ -n "$(USE_LOCAL_MJCF)" ]; then CMD="$$CMD --use-local-mjcf"; echo "模型: local mjcf"; fi; \
		if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
		if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; echo "设备: CPU"; fi; \
		if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "渲染间隔: $(RENDER)"; fi; \
		if [ -n "$(VIEWER_SLEEP)" ]; then CMD="$$CMD --viewer-sleep $(VIEWER_SLEEP)"; echo "viewer sleep: $(VIEWER_SLEEP)s"; fi; \
		if [ -n "$(STATUS_EVERY)" ]; then CMD="$$CMD --status-every $(STATUS_EVERY)"; echo "status-every: $(STATUS_EVERY)s"; fi; \
		if [ -n "$(REALTIME)" ]; then CMD="$$CMD --realtime"; echo "播放: realtime"; fi; \
		if [ -n "$(EPISODES)" ]; then CMD="$$CMD --episodes $(EPISODES)"; echo "episodes: $(EPISODES)"; fi; \
		if [ -n "$(MAX_STEPS)" ]; then CMD="$$CMD --max-steps $(MAX_STEPS)"; echo "max-steps: $(MAX_STEPS)"; fi; \
		if [ -n "$(ACTION_CLIP)" ]; then CMD="$$CMD --action-clip $(ACTION_CLIP)"; echo "action-clip: $(ACTION_CLIP)"; fi; \
		if [ -n "$(SAVE_VIDEO)" ]; then CMD="$$CMD --save-video"; echo "保存视频: 是"; fi; \
		if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; echo "视频路径: $(VIDEO_PATH)"; fi; \
		if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --video-fps $(VIDEO_FPS)"; echo "视频FPS: $(VIDEO_FPS)"; fi; \
		if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --render-width $(RENDER_WIDTH)"; echo "渲染宽度: $(RENDER_WIDTH)"; \
		elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --render-width $(WIDTH)"; echo "渲染宽度: $(WIDTH)"; \
		elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --render-width $(WEIGHT)"; echo "渲染宽度: $(WEIGHT)"; fi; \
		if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --render-height $(RENDER_HEIGHT)"; echo "渲染高度: $(RENDER_HEIGHT)"; \
		elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --render-height $(HEIGHT)"; echo "渲染高度: $(HEIGHT)"; fi; \
		if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --camera-name $(CAMERA_NAME)"; echo "相机: $(CAMERA_NAME)"; fi; \
		if [ -n "$(RECORD_INTERVAL)" ]; then CMD="$$CMD --record-interval $(RECORD_INTERVAL)"; echo "录制间隔: $(RECORD_INTERVAL)"; fi; \
		if [ -n "$(NO_JAX_PREALLOC)" ]; then CMD="$$CMD --no-jax-prealloc"; echo "JAX预分配: 关闭"; fi; \
		if [ -n "$(JAX_MEM_FRACTION)" ]; then CMD="$$CMD --jax-mem-fraction $(JAX_MEM_FRACTION)"; echo "JAX显存比例: $(JAX_MEM_FRACTION)"; fi; \
		if [ -n "$(ROBOT_NAME)" ]; then CMD="$$CMD --robot-name $(ROBOT_NAME)"; echo "机器人名称: $(ROBOT_NAME)"; fi; \
		if [ -n "$(ORT_PROVIDER)" ]; then CMD="$$CMD --ort-provider $(ORT_PROVIDER)"; echo "ort-provider: $(ORT_PROVIDER)"; fi; \
		if [ -n "$(INPUT_NAME)" ]; then CMD="$$CMD --input-name $(INPUT_NAME)"; echo "input-name: $(INPUT_NAME)"; fi; \
		if [ -n "$(OUTPUT_NAME)" ]; then CMD="$$CMD --output-name $(OUTPUT_NAME)"; echo "output-name: $(OUTPUT_NAME)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

infer-ksim-onnx:
	@$(UV) --version >/dev/null 2>&1 || (echo "❌ 未找到 uv，请先安装 uv"; exit 1)
	@if [ -z "$(MODEL)" ]; then \
		echo "错误: 请指定 ONNX 模型路径 MODEL=..."; \
		echo "示例: MUJOCO_GL=egl make infer-ksim-onnx MODEL=exported_models/policy_mean_xax.onnx CKPT=logs/ksim_train/.../checkpoints/ckpt.XXXX.bin SAVE_VIDEO=1 VIDEO_PATH=plays/xax_onnx.mp4"; \
		exit 1; \
	fi
	@if [ -z "$(CKPT)" ] && [ -z "$(CONFIG)" ]; then \
		echo "错误: 请指定配置来源：CKPT=... 或 CONFIG=..."; \
		echo "示例: make infer-ksim-onnx MODEL=... CKPT=logs/ksim_train/.../checkpoints/ckpt.XXXX.bin"; \
		echo "示例: make infer-ksim-onnx MODEL=... CONFIG=configs/ksim/gaoda_jiyuan_stand.yaml"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,infer_ksim_onnx)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== ksim ONNX 推理/回放 ==="; \
	echo "模型: $(MODEL)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 推理日志 ==="; \
		echo "命令: make infer-ksim-onnx MODEL=$(MODEL)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "模型: $(MODEL)"; \
		echo "[deps] 同步 ksim+onnx extras ..."; \
		$(UV_ENV) $(UV) sync --extra ksim --extra onnx --inexact || exit $$?; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(INFER_KSIM_SCRIPT) --model $(MODEL)"; \
		if [ -n "$(CKPT)" ]; then CMD="$$CMD --checkpoint $(CKPT)"; echo "checkpoint: $(CKPT)"; fi; \
		if [ -n "$(CONFIG)" ]; then CMD="$$CMD --config $(CONFIG)"; echo "config: $(CONFIG)"; fi; \
		if [ -n "$(SEED)" ]; then CMD="$$CMD --seed $(SEED)"; echo "seed: $(SEED)"; fi; \
		if [ -n "$(COMPARE_JAX)" ]; then CMD="$$CMD --compare-jax"; echo "compare-jax: true"; fi; \
		if [ -n "$(COMPARE_STEPS)" ]; then CMD="$$CMD --compare-steps $(COMPARE_STEPS)"; echo "compare-steps: $(COMPARE_STEPS)"; fi; \
		if [ -n "$(COMPARE_TOL)" ]; then CMD="$$CMD --compare-tol $(COMPARE_TOL)"; echo "compare-tol: $(COMPARE_TOL)"; fi; \
		if [ -n "$(NUM_ENVS)" ]; then CMD="$$CMD --num-envs $(NUM_ENVS)"; echo "num-envs: $(NUM_ENVS)"; fi; \
		if [ -n "$(MAX_STEPS)" ]; then CMD="$$CMD --num-steps $(MAX_STEPS)"; echo "num-steps: $(MAX_STEPS)"; fi; \
		if [ -n "$(SAVE_VIDEO)" ]; then CMD="$$CMD --save-video"; echo "保存视频: 是"; fi; \
		if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; echo "视频路径: $(VIDEO_PATH)"; fi; \
		if [ -n "$(VIDEO_FPS)" ]; then CMD="$$CMD --target-fps $(VIDEO_FPS)"; echo "视频FPS: $(VIDEO_FPS)"; fi; \
		if [ -n "$(RENDER_WIDTH)" ]; then CMD="$$CMD --render-width $(RENDER_WIDTH)"; echo "渲染宽度: $(RENDER_WIDTH)"; \
		elif [ -n "$(WIDTH)" ]; then CMD="$$CMD --render-width $(WIDTH)"; echo "渲染宽度: $(WIDTH)"; \
		elif [ -n "$(WEIGHT)" ]; then CMD="$$CMD --render-width $(WEIGHT)"; echo "渲染宽度: $(WEIGHT)"; fi; \
		if [ -n "$(RENDER_HEIGHT)" ]; then CMD="$$CMD --render-height $(RENDER_HEIGHT)"; echo "渲染高度: $(RENDER_HEIGHT)"; \
		elif [ -n "$(HEIGHT)" ]; then CMD="$$CMD --render-height $(HEIGHT)"; echo "渲染高度: $(HEIGHT)"; fi; \
		if [ -n "$(CAMERA_NAME)" ]; then CMD="$$CMD --camera-name $(CAMERA_NAME)"; echo "相机: $(CAMERA_NAME)"; fi; \
		if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; echo "设备: CPU"; fi; \
		if [ -n "$(NO_JAX_PREALLOC)" ]; then CMD="$$CMD --no-jax-prealloc"; echo "JAX预分配: 关闭"; fi; \
		if [ -n "$(JAX_MEM_FRACTION)" ]; then CMD="$$CMD --jax-mem-fraction $(JAX_MEM_FRACTION)"; echo "JAX显存比例: $(JAX_MEM_FRACTION)"; fi; \
		if [ -n "$(ORT_PROVIDER)" ]; then CMD="$$CMD --ort-provider $(ORT_PROVIDER)"; echo "ort-provider: $(ORT_PROVIDER)"; fi; \
		if [ -n "$(INPUT_NAME)" ]; then CMD="$$CMD --input-name $(INPUT_NAME)"; echo "input-name: $(INPUT_NAME)"; fi; \
		if [ -n "$(OUTPUT_NAME)" ]; then CMD="$$CMD --output-name $(OUTPUT_NAME)"; echo "output-name: $(OUTPUT_NAME)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

visualize-mjcf:
	@LOGFILE="$(call MAKELOG_FILE,visualize_mjcf)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	echo "=== MJCF 可视化 ==="; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== MJCF 可视化日志 ==="; \
		echo "命令: make visualize-mjcf"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(VISUALIZE_MJCF_SCRIPT)"; \
		if [ -n "$(XML)" ]; then CMD="$$CMD --xml $(XML)"; echo "模型: $(XML)"; \
		elif [ -n "$(XML_PATH)" ]; then CMD="$$CMD --xml $(XML_PATH)"; echo "模型: $(XML_PATH)"; fi; \
		if [ -n "$(NO_INTERACTIVE)" ]; then CMD="$$CMD --no-interactive"; echo "交互: 关闭"; fi; \
		if [ -n "$(AUTORELOAD)" ]; then CMD="$$CMD --autoreload $(AUTORELOAD)"; echo "autoreload: $(AUTORELOAD)"; fi; \
		if [ -n "$(MODE)" ]; then CMD="$$CMD --mode $(MODE)"; echo "mode: $(MODE)"; fi; \
		if [ -n "$(GRAVITY)" ]; then CMD="$$CMD --gravity $(GRAVITY)"; echo "gravity: $(GRAVITY)"; fi; \
		if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; echo "LIBGL_ALWAYS_SOFTWARE: 1"; fi; \
		if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; echo "MUJOCO_GL: $(MUJOCO_GL)"; fi; \
		echo ""; \
		eval $$CMD 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

tensorboard:
	@LOGFILE="$(call MAKELOG_FILE,tensorboard)"; \
	mkdir -p "$$(dirname "$$LOGFILE")"; \
	TB_INPUT="$(LOG_DIR)"; \
	TB_SEARCH="$$TB_INPUT"; \
	if [ ! -e "$$TB_SEARCH" ]; then \
		echo "错误: 指定的LOG_DIR不存在: $$TB_INPUT"; \
		echo "示例: make tensorboard LOG_DIR=logs  # 监控全部训练"; \
		echo "示例: make tensorboard LOG_DIR=logs/diy_train/ppo_YYYYMMDD_HHMMSS  # 监控单次run"; \
		exit 1; \
	fi; \
	if [ -f "$$TB_SEARCH" ]; then TB_SEARCH="$$(dirname "$$TB_SEARCH")"; fi; \
	TB_LOGDIR=""; \
	TB_UP="$$TB_SEARCH"; \
	while [ "$$TB_UP" != "/" ] && [ -n "$$TB_UP" ]; do \
		if [ -d "$$TB_UP/checkpoints" ]; then TB_LOGDIR="$$TB_UP"; break; fi; \
		TB_UP="$$(dirname "$$TB_UP")"; \
	done; \
	if [ -z "$$TB_LOGDIR" ]; then TB_LOGDIR="$$TB_SEARCH"; fi; \
	echo "=== 启动TensorBoard ==="; \
	echo "访问: http://localhost:6006"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== TensorBoard日志 ==="; \
		echo "命令: make tensorboard"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "监控目录: $$TB_LOGDIR/"; \
		echo ""; \
		$(PYTHON) -m tensorboard.main --version >/dev/null 2>&1 || (echo "❌ tensorboard 未安装，请先运行: make install-dev"; exit 1); \
		FORCE_COLOR=1 $(PYTHON) -m tensorboard.main --logdir="$$TB_LOGDIR" --host=127.0.0.1 --port=6006 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"

# ==================== 清理相关 ====================

clean:
	@echo "=== 清理临时文件 ==="
	find $(PROJECT_ROOT) -type f -name "*.pyc" -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ 清理完成"

clean-cache:
	@echo "=== 清理JAX缓存 ==="
	rm -rf $(CACHE_DIR)/*
	@echo "✓ 缓存已清空"

clean-train:
	@echo "=== 清理训练日志 ==="
	@echo "⚠️  警告: 将删除所有训练日志、模型检查点和视频！"
	@read -p "确认删除? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
			rm -rf $(LOG_DIR)/diy_train/*; \
		echo "✓ 训练日志已清空"; \
	else \
		echo "✗ 已取消"; \
	fi

clean-makelog:
	@echo "=== 清理Make执行日志 ==="
	rm -rf $(MAKELOG_DIR)
	mkdir -p $(MAKELOG_DIR)
	@echo "✓ Make日志已清空"

clean-logs:
	@echo "=== 清理所有日志 ==="
	@echo "⚠️  警告: 将删除所有训练日志（含检查点）和make执行日志！"
	@read -p "确认删除? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(LOG_DIR)/*; \
		echo "✓ 所有日志已清空"; \
	else \
		echo "✗ 已取消"; \
	fi

clean-all:
	@echo "=== 清理所有（保留训练日志）==="
	$(MAKE) clean
	$(MAKE) clean-cache
	$(MAKE) clean-makelog
	@echo "✓ 全部清理完成（训练日志已保留）"
	@echo ""
	@echo "提示: 如需删除训练日志，请运行: make clean-train"

# ==================== 开发相关 ====================

format:
	@echo "=== 格式化代码（与 VSCode 保持一致）==="
	@echo "  使用 black (line-length=100) + isort (profile=black)"
	@$(PYTHON) -m black --version >/dev/null 2>&1 || (echo "❌ black 未安装，请先运行: make install-dev"; exit 1)
	@$(PYTHON) -m isort --version >/dev/null 2>&1 || (echo "❌ isort 未安装，请先运行: make install-dev"; exit 1)
	$(PYTHON) -m black --line-length=100 src/ scripts/ tests/ utils/ 2>/dev/null || true
	$(PYTHON) -m isort --profile=black --line-length=100 src/ scripts/ tests/ utils/ 2>/dev/null || true
	@echo "✓ 格式化完成"
	@echo ""
	@echo "格式化配置来源: pyproject.toml"
	@echo "VSCode 配置: .vscode/settings.json"
	@echo "EditorConfig: .editorconfig"

test:
	@echo "=== 运行测试 ==="
	@$(PYTHON) -m pytest --version >/dev/null 2>&1 || (echo "❌ pytest 未安装，请先运行: make install-dev"; exit 1)
	$(PYTHON) -m pytest tests/ -v

# ==================== 信息相关 ====================

info:
	@echo "JRL 项目信息（自动课程学习）"
	@echo ""
	@echo "项目根目录: $(PROJECT_ROOT)"
	@echo "日志根目录: $(LOG_DIR)"
	@echo "  训练日志: $(LOG_DIR)/diy_train/"
	@echo "  Make日志: $(MAKELOG_DIR)"
	@echo "缓存目录: $(CACHE_DIR)"
	@echo ""
	@echo "配置文件:"
	@echo "  标准训练: $(CONFIG_TRAIN)"
	@echo "  长时间训练: $(CONFIG_LONG)"
	@echo "  快速测试: $(CONFIG_QUICK)"
	@echo ""
	@echo "课程学习阶段:"
	@echo "  阶段1 (0-50k):    站立平衡（学习保持直立）"
	@echo "  阶段2 (50k-150k): 低速行走（学习基本步态）"
	@echo "  阶段3 (150k+):    全速行走（跟踪任意速度）"
	@echo ""
	@echo "训练日志字段:"
	@echo "  - curriculum_stage: 当前阶段名称"
	@echo "  - curriculum_stage_index: 阶段索引 (0/1/2)"
	@echo "  - curriculum_progress: 当前阶段进度 (0-1)"
