# Makefile for JRL - JAX Reinforcement Learning Training
# 用于管理四足机器人分阶段训练

.PHONY: help install train train-stage train-all eval clean clean-cache clean-all
.DEFAULT_GOAL := help

# ==================== 配置变量 ====================

# 优先使用当前激活的conda环境中的python
# 如果没有激活conda环境，则使用系统的python
PYTHON := python
PROJECT_ROOT := $(shell pwd)
LOG_DIR := $(PROJECT_ROOT)/logs
CACHE_DIR := $(PROJECT_ROOT)/.jax_cache
TRAIN_SCRIPT := scripts/train.py
EVAL_SCRIPT := scripts/eval.py
CONFIG_DIR := configs

# ==================== 帮助信息 ====================

help:
	@echo "JRL - JAX强化学习训练管理"
	@echo ""
	@echo "可用命令:"
	@echo "  make install              - 安装依赖"
	@echo "  make install-dev          - 安装开发依赖"
	@echo "  make check-env            - 检查环境"
	@echo "  make train                - 快速开始训练（阶段0）"
	@echo "  make train-stage STAGE=N  - 训练指定阶段（0-6）"
	@echo "  make train-all            - 训练所有阶段（自动化）"
	@echo "  make eval CKPT=...        - 评估模型"
	@echo "  make tensorboard          - 启动TensorBoard"
	@echo "  make clean                - 清理日志"
	@echo "  make clean-all            - 清理所有"

# ==================== 安装相关 ====================

install:
	@echo "=== 安装JAX和依赖 ==="
	$(PYTHON) -m pip install -e . --upgrade
	@echo "✓ 安装完成"

install-dev:
	@echo "=== 安装开发依赖 ==="
	$(PYTHON) -m pip install -e ".[tensorboard]" --upgrade
	$(PYTHON) -m pip install pytest pytest-cov black isort mypy --upgrade
	@echo "✓ 开发依赖安装完成"

check-env:
	@echo "=== 检查环境 ==="
	@echo "Python版本: $$($(PYTHON) --version)"
	@echo "JAX版本: $$($(PYTHON) -c 'import jax; print(jax.__version__)')"
	@echo "JAX后端: $$($(PYTHON) -c 'import jax; print(jax.default_backend())')"
	@echo "可用设备:"
	@$(PYTHON) -c 'import jax; [print(f"  - {d}") for d in jax.devices()]'

# ==================== 训练相关 ====================

train:
	@echo "=== 开始训练（阶段0：站立平衡） ==="
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_DIR)/stage0_standing.yaml

train-stage:
	@if [ -z "$(STAGE)" ]; then \
		echo "错误: 请指定阶段号 STAGE=0-6"; \
		echo "示例: make train-stage STAGE=0"; \
		exit 1; \
	fi
	@echo "=== 训练阶段$(STAGE) ==="
	@CONFIG_FILE="$(CONFIG_DIR)/stage$(STAGE)_*.yaml"; \
	if [ ! -f $$CONFIG_FILE ]; then \
		echo "错误: 配置文件不存在: $$CONFIG_FILE"; \
		exit 1; \
	fi; \
	echo "配置文件: $$CONFIG_FILE"; \
	$(PYTHON) $(TRAIN_SCRIPT) --config $$CONFIG_FILE

train-all:
	@echo "=== 开始分阶段训练（自动化） ==="
	$(PYTHON) scripts/train_staged.py
	@echo "=== 分阶段训练完成 ==="

train-from:
	@if [ -z "$(FROM_STAGE)" ]; then \
		echo "错误: 请指定起始阶段 FROM_STAGE=0-6"; \
		echo "示例: make train-from FROM_STAGE=0"; \
		exit 1; \
	fi
	@echo "=== 从阶段$(FROM_STAGE)开始训练 ==="
	@for STAGE in $$(seq $(FROM_STAGE) 6); do \
		echo ">>> 训练阶段$$STAGE <<<"; \
		$(MAKE) train-stage STAGE=$$STAGE || exit 1; \
		echo "✓ 阶段$$STAGE完成"; \
	done
	@echo "=== 所有阶段训练完成 ==="

resume:
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make resume CKPT=logs/ppo_*/checkpoints/best_model"; \
		exit 1; \
	fi
	@echo "=== 恢复训练 ==="
	@echo "检查点: $(CKPT)"
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_DIR)/stage0_standing.yaml --resume $(CKPT)

# ==================== 评估相关 ====================

eval:
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make eval CKPT=logs/ppo_*/checkpoints/best_model"; \
		exit 1; \
	fi
	@echo "=== 评估模型 ==="
	@echo "检查点: $(CKPT)"
	$(PYTHON) $(EVAL_SCRIPT) --checkpoint $(CKPT)

tensorboard:
	@echo "=== 启动TensorBoard ==="
	tensorboard --logdir=$(LOG_DIR) --host=0.0.0.0 --port=6006

# ==================== 清理相关 ====================

clean:
	@echo "=== 清理日志 ==="
	find $(LOG_DIR) -type f -name "*.log" -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name "*.pyc" -delete
	find $(PROJECT_ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ 清理完成"

clean-cache:
	@echo "=== 清理JAX缓存 ==="
	rm -rf $(CACHE_DIR)/*
	@echo "✓ 缓存已清空"

clean-logs:
	@echo "=== 清理所有日志 ==="
	rm -rf $(LOG_DIR)/*
	@echo "✓ 日志已清空"

clean-all:
	@echo "=== 清理所有 ==="
	$(MAKE) clean
	$(MAKE) clean-cache
	$(MAKE) clean-logs
	@echo "✓ 全部清理完成"

# ==================== 开发相关 ====================

format:
	@echo "=== 格式化代码 ==="
	black src/ scripts/ tests/ 2>/dev/null || true
	isort src/ scripts/ tests/ 2>/dev/null || true
	@echo "✓ 格式化完成"

test:
	@echo "=== 运行测试 ==="
	pytest tests/ -v 2>/dev/null || python -m pytest tests/ -v

# ==================== 信息相关 ====================

info:
	@echo "JRL 项目信息"
	@echo ""
	@echo "项目根目录: $(PROJECT_ROOT)"
	@echo "日志目录: $(LOG_DIR)"
	@echo "缓存目录: $(CACHE_DIR)"
	@echo ""
	@echo "训练阶段配置:"
	@ls -1 $(CONFIG_DIR)/stage*.yaml 2>/dev/null || echo "  无阶段配置文件"
