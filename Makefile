# ==============================================================================
# Makefile - 双足机器人 RL 项目快速命令
# ==============================================================================
# 推荐使用方式（流水线训练）:
#   make train              - 运行完整的手动分阶段流水线（站立->平坦->行走->粗糙）
#   make train-curriculum   - 运行全自动课程学习（单任务自动化）
#   make train-standing     - 仅进行站立平衡训练
#   make play               - 评估最新训练的策略
#   make play-video         - 录制策略视频
#
# 环境配置:
#   make install            - 自动同步环境（Linux: dev + training；其他平台: dev）
# ==============================================================================

# 运行模式
# - RUNNER=uv: 使用根目录 uv workspace 的 .venv（唯一支持的运行方式）
RUNNER ?= uv
UV ?= uv
UV_PROJECT ?= .
UV_SYNC_FLAGS ?= --project $(UV_PROJECT) --all-packages
UV_RUN_GROUPS ?=
UV_TRAINING_GROUPS = --group training
HOST_OS := $(shell uname -s)
HOST_ARCH := $(shell uname -m)
TRAINING_PLATFORM_SUPPORTED := 0
ifeq ($(HOST_OS),Linux)
ifeq ($(HOST_ARCH),x86_64)
TRAINING_PLATFORM_SUPPORTED := 1
endif
ifeq ($(HOST_ARCH),aarch64)
TRAINING_PLATFORM_SUPPORTED := 1
endif
endif
ifeq ($(TRAINING_PLATFORM_SUPPORTED),1)
INSTALL_GROUPS := --group dev --group training
INSTALL_LABEL := 完整训练环境（dev + training）
INSTALL_NOTE :=
else
INSTALL_GROUPS := --group dev
INSTALL_LABEL := 开发环境（dev）
INSTALL_NOTE := 当前平台 $(HOST_OS)/$(HOST_ARCH) 不提供 Isaac Lab 官方 wheel，install 自动跳过 training 组。
endif

# Isaac Sim AppLauncher 常用参数
# HEADLESS=1 时自动追加 --headless（云服务器/无显示环境推荐）
HEADLESS ?= 1
DEVICE ?= cuda:0
APP_ARGS :=
ifeq ($(HEADLESS),1)
APP_ARGS += --headless
endif
APP_ARGS += --device $(DEVICE)
PYTHON_RUN = $(UV) run --project $(UV_PROJECT) --all-packages $(UV_RUN_GROUPS) python

# AutoDL 云服务器平台专用配置
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CARB_LOG_LEVEL=ERROR
export TERM=xterm

.PHONY: help install clean clean-logs verify
.PHONY: check-env check-uv check-runner check-train-runner check-train-platform convert-usd
.PHONY: train train-curriculum train-standing train-flat train-walking train-rough
.PHONY: train-smoke
.PHONY: play play-velocity play-standing play-walking play-video play-video-velocity play-video-standing
.PHONY: uv-run

convert-usd train-smoke train-curriculum train-standing train-flat train-walking train-rough play-velocity play-standing play-walking play-video-velocity play-video-standing play-video-walking: UV_RUN_GROUPS = $(UV_TRAINING_GROUPS)

# 默认目标
help:
	@echo "双足机器人 RL 项目 - 可用命令:"
	@echo ""
	@echo "==================================================================="
	@echo "🚀 训练流水线 (Pipeline & Curriculum)"
	@echo "==================================================================="
	@echo ""
	@echo "流水线训练（分阶段恢复）:"
	@echo "  make train                   - 启动完整分阶段流水线（推荐！）"
	@echo "  make train-curriculum        - 全自动课程学习（单任务内部动态难度）"
	@echo ""
	@echo "单步任务训练:"
	@echo "  make train-standing          - 站立平衡训练 (2000 iters)"
	@echo "  make train-flat              - 平坦地形行走训练 (5000 iters)"
	@echo "  make train-walking           - 正常速度行走训练 (10000 iters)"
	@echo "  make train-rough             - 粗糙地形适应训练 (30000 iters)"
	@echo ""
	@echo "==================================================================="
	@echo "🎮 评估与录制"
	@echo "==================================================================="
	@echo ""
	@echo "策略评估:"
	@echo "  make play                    - 评估最新的速度跟踪模型"
	@echo "  make play-standing           - 评估最新的站立模型"
	@echo "  make play-video              - 录制评估视频"
	@echo ""
	@echo "==================================================================="
	@echo "🔧 环境与安装"
	@echo "==================================================================="
	@echo ""
	@echo "  make install                 - 自动同步环境（Linux: dev + training；其他平台: dev）"
	@echo "  make convert-usd XML=<path>  - 转换模型并在同级生成 USD"
	@echo "  make visualize-mjcf          - 可视化 MJCF"
	@echo "                                参数: XML=<路径> AUTORELOAD=0/1 MODE=sim/launch GRAVITY=0/1 NO_INTERACTIVE=1 RENDER_CPU=1"
	@echo "                                MODE=sim 为手动步进，MODE=launch 为 Simulate GUI"
	@echo ""

# ==============================================================================
# 安装与检查
# ==============================================================================

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || (echo "Error: 未找到 uv（请先安装 uv 或设置 UV=...）"; exit 1)

check-runner:
ifeq ($(RUNNER),uv)
	@$(MAKE) check-uv
else
	@echo "Error: dep/ 已移除，当前仓库仅支持 RUNNER=uv"
	@exit 1
endif

check-train-platform:
ifeq ($(TRAINING_PLATFORM_SUPPORTED),1)
	@true
else
	@echo "Error: 当前平台 $(HOST_OS)/$(HOST_ARCH) 不支持 Isaac Lab 训练运行时。"
	@echo "       make install 已自动退化为 dev 环境；训练/评估/convert-usd 需在 Linux x86_64 或 Linux aarch64 上运行。"
	@exit 1
endif

check-train-runner: check-runner check-train-platform

install: check-uv
	@echo "使用 uv 同步 $(INSTALL_LABEL)..."
	@if [ -n "$(INSTALL_NOTE)" ]; then echo "$(INSTALL_NOTE)"; fi
	$(UV) sync $(UV_SYNC_FLAGS) $(INSTALL_GROUPS)
	@echo "✓ 安装完成"

uv-run:
	@echo "示例：make uv-run CMD=\"python -m black isaaclab_rl\""
	@$(MAKE) check-uv
	@if [ -z "$(CMD)" ]; then echo "Error: 需要提供 CMD=..."; exit 1; fi
	$(UV) run --project $(UV_PROJECT) --all-packages $(UV_RUN_GROUPS) $(CMD)

# ==============================================================================
# MJCF 可视化
# ==============================================================================
VISUALIZE_MJCF_SCRIPT ?= scripts/visualize_mjcf.py

visualize-mjcf:
	@{ \
		test -f "$(VISUALIZE_MJCF_SCRIPT)" || (echo "Error: 找不到 $(VISUALIZE_MJCF_SCRIPT)"; exit 1); \
		CMD="FORCE_COLOR=1 $(PYTHON_RUN) $(VISUALIZE_MJCF_SCRIPT)"; \
		if [ -n "$(XML)" ]; then CMD="$$CMD --xml $(XML)"; \
		elif [ -n "$(XML_PATH)" ]; then CMD="$$CMD --xml $(XML_PATH)"; fi; \
		if [ -n "$(NO_INTERACTIVE)" ]; then CMD="$$CMD --no-interactive"; fi; \
		if [ -n "$(AUTORELOAD)" ]; then CMD="$$CMD --autoreload $(AUTORELOAD)"; fi; \
		if [ -n "$(MODE)" ]; then CMD="$$CMD --mode $(MODE)"; fi; \
		if [ -n "$(GRAVITY)" ]; then CMD="$$CMD --gravity $(GRAVITY)"; fi; \
		if [ -n "$(RENDER_CPU)" ]; then CMD="LIBGL_ALWAYS_SOFTWARE=1 $$CMD"; fi; \
		if [ -n "$(MUJOCO_GL_IS_CMDLINE)" ]; then CMD="MUJOCO_GL=$(MUJOCO_GL) $$CMD"; fi; \
		eval $$CMD; \
	}



# ==============================================================================
# 资产转换
# ==============================================================================

XML ?=
CONVERT_MJCF_SCRIPT ?= scripts/convert_mjcf.py

convert-usd: check-train-runner
	@{ \
		set -e; \
		test -n "$(XML)" || (echo "Error: 请提供 XML 路径，例如: make convert-usd XML=robots/gaoda_jiyuan/scene.xml"; exit 1); \
		test -f "$(XML)" || (echo "Error: 输入文件不存在: $(XML)"; exit 1); \
		USD_OUT="$$(dirname "$(XML)")/$$(basename "$(XML)" .xml).usd"; \
		echo "转换 MJCF 到 USD..."; \
		echo "  输入: $(XML)"; \
		echo "  输出: $$USD_OUT"; \
		mkdir -p "$$(dirname "$$USD_OUT")"; \
		$(PYTHON_RUN) $(CONVERT_MJCF_SCRIPT) "$(XML)" "$$USD_OUT" --import-sites; \
		echo "✓ 转换完成: $$USD_OUT"; \
	}

# ==============================================================================
# 训练目标 (Pipeline & Curriculum)
# ==============================================================================

# 一键启动流水线流程
train: train-standing train-flat train-walking train-rough
	@echo "✓ 流水线全流程训练完成！"

# 低资源 smoke：用于验证 reward/termination/坐标系是否一致（1080 Ti 友好）
# 用法：
#   make train-smoke                     # 默认 128 envs, 50 iters
#   make train-smoke ARGS="--task rough" # 指定任务
#   make train-smoke NUM_ENVS=256 ITERS=200
NUM_ENVS_SMOKE ?= 128
ITERS_SMOKE ?= 50
train-smoke: check-train-runner
	@echo ">>> [SMOKE] 低资源快速验证 (NUM_ENVS=$(NUM_ENVS_SMOKE), ITERS=$(ITERS_SMOKE))..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--num_envs $(NUM_ENVS_SMOKE) \
		--max_iterations $(ITERS_SMOKE) $(ARGS)

# 全自动课程学习
train-curriculum: check-train-runner
	@echo ">>> [全自动课程学习] 开始训练..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task curriculum \
		--max_iterations 30000 $(ARGS)

# -- 各阶段任务 --

train-standing: check-train-runner
	@echo ">>> [Stage 1: 站立平衡] 开始训练 (2000 iterations)..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task standing \
		--max_iterations 2000 $(ARGS)

train-flat: check-train-runner
	@echo ">>> [Stage 2: 平坦地形] 加载站立权重并训练 (5000 iterations)..."
	@LATEST_STANDING=$$(ls -td logs/jiyuan_standing/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_STANDING" ]; then echo "Error: 未找到站立训练记录"; exit 1; fi; \
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task flat \
		--max_iterations 5000 \
		--resume --load_run jiyuan_standing/$$LATEST_STANDING $(ARGS)

train-walking: check-train-runner
	@echo ">>> [Stage 3: 正常行走] 加载平坦地形权重并训练 (10000 iterations)..."
	@LATEST_VEL=$$(ls -td logs/jiyuan_velocity_tracking/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_VEL" ]; then echo "Error: 未找到平坦地形训练记录"; exit 1; fi; \
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task velocity \
		--max_iterations 10000 \
		--resume --load_run jiyuan_velocity_tracking/$$LATEST_VEL $(ARGS)

train-rough: check-train-runner
	@echo ">>> [Stage 4: 粗糙地形] 加载行走权重并训练 (30000 iterations)..."
	@LATEST_VEL=$$(ls -td logs/jiyuan_velocity_tracking/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_VEL" ]; then echo "Error: 未找到行走训练记录"; exit 1; fi; \
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task rough \
		--max_iterations 30000 \
		--resume --load_run jiyuan_velocity_tracking/$$LATEST_VEL $(ARGS)

# ==============================================================================
# 策略评估和视频录制
# ==============================================================================

# 可选参数
CHECKPOINT ?=
VIDEO_LENGTH ?= 500
NUM_ENVS ?= 1

# 默认 play：评估速度跟踪策略（GUI 可视化）
play: play-velocity

# 评估速度跟踪策略
play-velocity: check-train-runner
	@echo "评估速度跟踪策略（GUI 可视化）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task velocity \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task velocity \
			--num_envs $(NUM_ENVS) $(ARGS); \
	fi

# 评估站立平衡策略
play-standing: check-train-runner
	@echo "评估站立平衡策略（GUI 可视化）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task standing \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task standing \
			--num_envs $(NUM_ENVS) $(ARGS); \
	fi

# 评估行走步态策略
play-walking: check-train-runner
	@echo "评估行走步态策略（GUI 可视化）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task walking \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task walking \
			--num_envs $(NUM_ENVS) $(ARGS); \
	fi

# 默认视频录制：速度跟踪任务
play-video: play-video-velocity

# 录制速度跟踪视频
play-video-velocity: check-train-runner
	@echo "录制速度跟踪视频（$(VIDEO_LENGTH) 步）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task velocity \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task velocity \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	fi
	@echo "✓ 视频录制完成！查看 logs/<实验名>/<运行ID>/videos/play/"

# 录制站立平衡视频
play-video-standing: check-train-runner
	@echo "录制站立平衡视频（$(VIDEO_LENGTH) 步）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task standing \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task standing \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	fi
	@echo "✓ 视频录制完成！查看 logs/<实验名>/<运行ID>/videos/play/"

# 录制行走步态视频
play-video-walking: check-train-runner
	@echo "录制行走步态视频（$(VIDEO_LENGTH) 步）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task walking \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(PYTHON_RUN) isaaclab_rl/scripts/play.py $(APP_ARGS) \
			--task walking \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	fi
	@echo "✓ 视频录制完成！查看 logs/<实验名>/<运行ID>/videos/play/"


# ==============================================================================
# 验证目标
# ==============================================================================

verify: check-runner
	@echo "验证安装..."
	@$(PYTHON_RUN) -c "import yaml; print('✓ PyYAML 可用')" || echo "✗ PyYAML 未安装"
	@$(PYTHON_RUN) -c "import tensorboard; print('✓ TensorBoard 可用')" || echo "✗ TensorBoard 未安装"
ifeq ($(TRAINING_PLATFORM_SUPPORTED),1)
	@$(UV) run --project $(UV_PROJECT) --all-packages $(UV_TRAINING_GROUPS) python -c "import isaaclab; print('✓ Isaac Lab 可用')" || echo "✗ Isaac Lab 未安装"
	@$(UV) run --project $(UV_PROJECT) --all-packages $(UV_TRAINING_GROUPS) python -c "import rsl_rl; print('✓ RSL_RL 可用')" || echo "✗ RSL_RL 未安装"
	@$(UV) run --project $(UV_PROJECT) --all-packages $(UV_TRAINING_GROUPS) python -c "from jiyuan_tasks import *; print('✓ jiyuan_tasks 模块可用')" || echo "✗ jiyuan_tasks 模块未安装"
else
	@echo "ℹ 当前平台 $(HOST_OS)/$(HOST_ARCH) 仅验证 dev 环境；training 运行时未安装，这是预期行为。"
endif
	@echo ""
	@echo "如果所有检查都通过，安装成功！"

# ==============================================================================
# 清理目标
# ==============================================================================

clean:
	@echo "清理构建文件..."
	rm -rf isaaclab_rl/build/ isaaclab_rl/dist/ isaaclab_rl/*.egg-info
	find isaaclab_rl -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find isaaclab_rl -type f -name "*.pyc" -delete
	find isaaclab_rl -type f -name "*.pyo" -delete
	find utils -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find utils -type f -name "*.pyc" -delete
	find utils -type f -name "*.pyo" -delete
	@echo "✓ 清理完成！"

clean-logs:
	@echo "清理日志文件..."
	rm -rf logs/* 2>/dev/null || true
	@echo "✓ 日志清理完成！"

# ==============================================================================
# 开发辅助目标
# ==============================================================================

# 格式化代码（需要安装 dev 依赖）
format: check-runner
	@echo "格式化代码..."
	$(PYTHON_RUN) -m black --config isaaclab_rl/pyproject.toml isaaclab_rl/jiyuan_tasks/ isaaclab_rl/scripts/
	@echo "✓ 代码格式化完成！"

# 运行测试（需要安装 dev 依赖）
test: check-runner
	@echo "运行测试..."
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON_RUN) -m pytest -c isaaclab_rl/pyproject.toml -p pytest_cov isaaclab_rl/tests/ -v
	@echo "✓ 测试完成！"
