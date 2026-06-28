# ==============================================================================
# Makefile - 双足机器人 RL 项目快速命令
# ==============================================================================

.PHONY: help install verify train play clean train-smoke

# Python 版本（与 .python-version 保持一致）
PYTHON := python3.10

# ==============================================================================
# 帮助
# ==============================================================================
help:
	@echo "================================================================================"
	@echo "🤖 双足机器人 RL 训练框架"
	@echo "================================================================================"
	@echo ""
	@echo "环境配置:"
	@echo "  make install                 - 安装依赖（uv sync）"
	@echo "  make verify                  - 验证环境"
	@echo ""
	@echo "训练:"
	@echo "  make train ROBOT=h1 TASK=standing   - 训练指定机器人和任务"
	@echo "  make train-smoke                    - 冒烟测试（快速验证）"
	@echo ""
	@echo "评估:"
	@echo "  make play ROBOT=h1                   - 评估最新策略"
	@echo ""
	@echo "清理:"
	@echo "  make clean                  - 清理训练日志"
	@echo ""
	@echo "================================================================================"

# ==============================================================================
# 环境
# ==============================================================================
install:
	@echo "📦 安装依赖..."
	uv sync --all-extras
	@echo "✅ 安装完成"

verify:
	@echo "🔍 验证环境..."
	@$(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import isaaclab; print(f'  Isaac Lab: ready')" 2>/dev/null || echo "  ⚠️ Isaac Lab 未检测到（仅 Linux + GPU 环境）"
	@$(PYTHON) -c "import rsl_rl; print(f'  RSL-RL: ready')" 2>/dev/null || echo "  ⚠️ RSL-RL 未检测到（仅 Linux + GPU 环境）"
	@echo "✅ 验证完成"

# ==============================================================================
# 训练
# ==============================================================================
ROBOT ?= h1
TASK ?= standing
NUM_ENVS ?= 4096
MAX_ITERATIONS ?= 10000
HEADLESS ?= true

train:
	@echo "🚀 开始训练: Robot=$(ROBOT) Task=$(TASK)"
	$(PYTHON) scripts/train.py \
		robot=$(ROBOT) \
		task=$(TASK) \
		num_envs=$(NUM_ENVS) \
		max_iterations=$(MAX_ITERATIONS) \
		headless=$(HEADLESS)

train-smoke:
	@echo "🧪 冒烟测试..."
	$(PYTHON) scripts/train.py robot=h1 task=standing num_envs=64 max_iterations=5 headless=true

# ==============================================================================
# 评估
# ==============================================================================
play:
	@echo "🎮 评估策略: Robot=$(ROBOT)"
	$(PYTHON) scripts/play.py robot=$(ROBOT) checkpoint=latest

# ==============================================================================
# 清理
# ==============================================================================
clean:
	@echo "🧹 清理日志..."
	rm -rf logs/ runs/
	@echo "✅ 清理完成"
