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
#   make submodule-update   - 初始化并更新git子模块
#   make install            - 安装项目（开发模式）
# ==============================================================================

# Isaac Lab 路径配置
ISAACLAB_PATH ?= dep/IsaacLab
ISAACLAB_PYTHON = $(ISAACLAB_PATH)/isaaclab.sh -p

# 运行模式
# - RUNNER=isaaclab: 使用 Isaac Lab 管理的 Python（默认，训练/评估推荐）
# - RUNNER=uv: 使用 uv 的虚拟环境（适合 format/lint/纯 Python 单测；训练需你自行安装 Isaac 依赖）
RUNNER ?= isaaclab
UV ?= uv
UV_PROJECT ?= isaaclab_rl

# Isaac Sim AppLauncher 常用参数
# HEADLESS=1 时自动追加 --headless（云服务器/无显示环境推荐）
HEADLESS ?= 1
DEVICE ?= cuda:0
APP_ARGS :=
ifeq ($(HEADLESS),1)
APP_ARGS += --headless
endif
APP_ARGS += --device $(DEVICE)

ifeq ($(RUNNER),uv)
PYTHON_RUN = $(UV) run --project $(UV_PROJECT) python
else
PYTHON_RUN = $(ISAACLAB_PYTHON)
endif

# AutoDL 云服务器平台专用配置
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CARB_LOG_LEVEL=ERROR
export TERM=xterm

.PHONY: help install install-dev install-vis install-all clean clean-logs verify
.PHONY: submodule-init submodule-update submodule-status submodule-update-remote
.PHONY: check-env check-isaaclab check-uv check-runner convert-usd
.PHONY: train train-curriculum train-standing train-flat train-walking train-rough
.PHONY: train-smoke
.PHONY: play play-velocity play-standing play-walking play-video play-video-velocity play-video-standing
.PHONY: uv-sync uv-run

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
	@echo "  make submodule-update        - 更新所有子模块"
	@echo "  make install                 - 安装 RL 环境"
	@echo "  make convert-usd             - 转换模型 (MJCF -> USD)"
	@echo ""

# ==============================================================================
# 子模块与安装
# ==============================================================================

check-isaaclab:
	@test -x $(ISAACLAB_PATH)/isaaclab.sh || (echo "Error: 找不到 $(ISAACLAB_PATH)/isaaclab.sh，请先初始化子模块或修正 ISAACLAB_PATH"; exit 1)

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || (echo "Error: 未找到 uv（请先安装 uv 或设置 UV=...）"; exit 1)

check-runner:
ifeq ($(RUNNER),uv)
	@$(MAKE) check-uv
else
	@$(MAKE) check-isaaclab
endif

submodule-init:
	@git submodule update --init --recursive

submodule-update: submodule-init
	@echo "✓ 子模块已更新"

submodule-status:
	@git submodule status --recursive

submodule-update-remote:
	@git submodule update --remote --merge --recursive

install: check-isaaclab
	@echo "安装项目（开发模式，安装到 Isaac Lab Python）..."
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl"
	@echo "✓ 安装完成"

install-dev: check-isaaclab
	@echo "安装项目（含 dev 依赖，安装到 Isaac Lab Python）..."
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl[dev]"
	@echo "✓ 安装完成"

install-vis: check-isaaclab
	@echo "安装项目（含 vis 依赖，安装到 Isaac Lab Python）..."
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl[vis]"
	@echo "✓ 安装完成"

install-all: check-isaaclab
	@echo "安装项目（含 all 依赖，安装到 Isaac Lab Python）..."
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl[all]"
	@echo "✓ 安装完成"

uv-sync:
	@echo "使用 uv 同步 $(UV_PROJECT) 依赖（仅管理纯 Python 依赖/工具）..."
	@$(MAKE) check-uv
	$(UV) sync --project $(UV_PROJECT) --all-extras
	@echo "✓ uv 同步完成"

uv-run:
	@echo "示例：make uv-run CMD=\"python -m black isaaclab_rl\""
	@$(MAKE) check-uv
	@if [ -z "$(CMD)" ]; then echo "Error: 需要提供 CMD=..."; exit 1; fi
	$(UV) run --project $(UV_PROJECT) $(CMD)

# ==============================================================================
# 资产转换
# ==============================================================================

MJCF ?= assets/xmls/models/jiyuan_fit.xml
USD_OUT ?= assets/usd/jiyuan_fit/jiyuan_fit.usd

convert-usd: check-isaaclab
	@echo "转换 MJCF 到 USD..."
	@mkdir -p $(dir $(USD_OUT))
	$(ISAACLAB_PYTHON) dep/IsaacLab/scripts/tools/convert_mjcf.py $(MJCF) $(USD_OUT) --import-sites
	@echo "✓ 转换完成"

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
train-smoke: check-isaaclab
	@echo ">>> [SMOKE] 低资源快速验证 (NUM_ENVS=$(NUM_ENVS_SMOKE), ITERS=$(ITERS_SMOKE))..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--num_envs $(NUM_ENVS_SMOKE) \
		--max_iterations $(ITERS_SMOKE) $(ARGS)

# 全自动课程学习
train-curriculum: check-isaaclab
	@echo ">>> [全自动课程学习] 开始训练..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task curriculum \
		--max_iterations 30000 $(ARGS)

# -- 各阶段任务 --

train-standing: check-isaaclab
	@echo ">>> [Stage 1: 站立平衡] 开始训练 (2000 iterations)..."
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task standing \
		--max_iterations 2000 $(ARGS)

train-flat: check-isaaclab
	@echo ">>> [Stage 2: 平坦地形] 加载站立权重并训练 (5000 iterations)..."
	@LATEST_STANDING=$$(ls -td logs/jiyuan_standing/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_STANDING" ]; then echo "Error: 未找到站立训练记录"; exit 1; fi; \
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task flat \
		--max_iterations 5000 \
		--resume --load_run jiyuan_standing/$$LATEST_STANDING $(ARGS)

train-walking: check-isaaclab
	@echo ">>> [Stage 3: 正常行走] 加载平坦地形权重并训练 (10000 iterations)..."
	@LATEST_VEL=$$(ls -td logs/jiyuan_velocity_tracking/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_VEL" ]; then echo "Error: 未找到平坦地形训练记录"; exit 1; fi; \
	$(PYTHON_RUN) isaaclab_rl/scripts/train.py $(APP_ARGS) \
		--config isaaclab_rl/configs/train_config.yaml \
		--task velocity \
		--max_iterations 10000 \
		--resume --load_run jiyuan_velocity_tracking/$$LATEST_VEL $(ARGS)

train-rough: check-isaaclab
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
play-velocity: check-isaaclab
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
play-standing: check-isaaclab
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
play-walking: check-isaaclab
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
play-video-velocity: check-isaaclab
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
play-video-standing: check-isaaclab
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
play-video-walking: check-isaaclab
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

verify: check-isaaclab
	@echo "验证安装..."
	@$(ISAACLAB_PYTHON) -c "import yaml; print('✓ PyYAML 可用')" || echo "✗ PyYAML 未安装"
	@$(ISAACLAB_PYTHON) -c "import tensorboard; print('✓ TensorBoard 可用')" || echo "✗ TensorBoard 未安装"
	@$(ISAACLAB_PYTHON) -c "import omni.isaac.lab; print('✓ Isaac Lab 可用')" || echo "✗ Isaac Lab 未安装"
	@$(ISAACLAB_PYTHON) -c "import rsl_rl; print('✓ RSL_RL 可用')" || echo "✗ RSL_RL 未安装"
	@$(ISAACLAB_PYTHON) -c "from jiyuan_tasks import *; print('✓ jiyuan_tasks 模块可用')" || echo "✗ jiyuan_tasks 模块未安装"
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
	$(PYTHON_RUN) -m black isaaclab_rl/jiyuan_tasks/ isaaclab_rl/scripts/ --line-length 120
	@echo "✓ 代码格式化完成！"

# 运行测试（需要安装 dev 依赖）
test: check-runner
	@echo "运行测试..."
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON_RUN) -m pytest -p pytest_cov isaaclab_rl/tests/ -v
	@echo "✓ 测试完成！"
