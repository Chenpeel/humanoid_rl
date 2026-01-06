# Makefile for JRL - JAX Reinforcement Learning Training
# 用于管理机器人训练（自动课程学习）

.PHONY: help install train train-long train-test train-quick eval clean clean-cache clean-logs clean-train clean-makelog clean-all
.DEFAULT_GOAL := help
SHELL := /bin/bash

# ==================== 配置变量 ====================

# 优先使用当前激活的conda环境中的python
# 如果没有激活conda环境，则使用系统的python
PYTHON := python
PROJECT_ROOT := $(shell pwd)
LOG_DIR := $(PROJECT_ROOT)/logs
MAKELOG_DIR := $(LOG_DIR)/makelog
CACHE_DIR := $(PROJECT_ROOT)/.jax_cache
TRAIN_SCRIPT := scripts/train.py
EVAL_SCRIPT := scripts/eval.py

# 配置文件路径
CONFIG_TRAIN := configs/train/train.yaml
CONFIG_LONG := configs/train-10h/train.yaml
CONFIG_QUICK := configs/quick_test/train.yaml

# 日志时间戳生成函数
TIMESTAMP := $(shell date '+%Y%m%d_%H%M%S')
MAKELOG_FILE = $(MAKELOG_DIR)/$(1)_$(TIMESTAMP).log

# 创建日志目录
$(shell mkdir -p $(MAKELOG_DIR))

# ==================== 帮助信息 ====================

help:
	@echo "JRL 训练系统 - Makefile 命令（自动课程学习）"
	@echo ""
	@echo "环境配置："
	@echo "  make install              安装JAX依赖"
	@echo "  make install-dev          安装开发依赖"
	@echo "  make check-env            检查JAX/CUDA环境"
	@echo ""
	@echo "训练命令（自动课程学习）："
	@echo "  make train                标准训练（2048 envs，200M steps，约8-12小时）"
	@echo "  make train-long           长时间训练（4096 envs，500M steps，约24-48小时）"
	@echo "  make train-test           快速测试（128 envs，1M steps，约5-10分钟）"
	@echo "  make train-custom CONFIG=path/to/config.yaml  自定义配置训练"
	@echo ""
	@echo "课程学习机制："
	@echo "  - 阶段1 (0-50k steps):    站立平衡"
	@echo "  - 阶段2 (50k-150k steps): 低速行走"
	@echo "  - 阶段3 (150k+ steps):    全速行走"
	@echo "  - 奖励权重和环境参数自动切换，无需手动干预"
	@echo ""
	@echo "评估命令："
	@echo "  make eval CKPT=path/to/checkpoint         评估模型（默认渲染+保存视频）"
	@echo "  make eval CKPT=... CPU=1                   使用CPU评估"
	@echo "  make eval CKPT=... NO_VIDEO=1              不保存视频"
	@echo "  make eval CKPT=... ENV_TYPE=walking        指定环境类型"
	@echo ""
	@echo "开发工具："
	@echo "  make tensorboard          启动TensorBoard"
	@echo "  make format               格式化代码"
	@echo "  make test                 运行单元测试"
	@echo "  make validate-config      验证配置文件语法"
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
	@echo "  训练日志: logs/train/ppo_[时间戳]/ (包含TensorBoard、检查点、视频等)"
	@echo "  Make日志: logs/makelog/[命令]_[时间戳].log (make命令执行记录)"
	@echo "  实时终端输出和日志文件内容完全一致（使用tee实现）"

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
	@LOGFILE="$(call MAKELOG_FILE,train)"; \
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
		FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_TRAIN) 2>&1; \
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
		FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_LONG) 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="; \
	echo "结束时间: $$(date)"

train-test:
	@LOGFILE="$(call MAKELOG_FILE,train-test)"; \
	echo "=== 快速测试（自动课程学习）==="; \
	echo "配置: $(CONFIG_QUICK)"; \
	echo "环境数: 128，总步数: 1M"; \
	echo "预计时间: 5-10分钟"; \
	echo "日志文件: $$LOGFILE"; \
	echo ""; \
	{ \
		echo "=== 测试日志 ==="; \
		echo "命令: make train-test"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "配置文件: $(CONFIG_QUICK)"; \
		echo ""; \
		FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_QUICK) 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 测试完成 ==="

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
		FORCE_COLOR=1 $(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG) 2>&1; \
		EXIT_CODE=$$?; \
		echo ""; \
		echo "结束时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "退出码: $$EXIT_CODE"; \
		exit $$EXIT_CODE; \
	} 2>&1 | tee "$$LOGFILE"; \
	echo ""; \
	echo "=== 训练完成 ==="

validate-config:
	@echo "=== 验证配置文件 ==="
	@$(PYTHON) -c "import yaml, sys; configs = ['$(CONFIG_TRAIN)', '$(CONFIG_LONG)', '$(CONFIG_QUICK)']; [yaml.safe_load(open(c)) or print(f'✓ {c}') for c in configs]; print('✓ 所有配置文件有效')"

# ==================== 评估相关 ====================

eval:
	@if [ -z "$(CKPT)" ]; then \
		echo "错误: 请指定检查点路径 CKPT=..."; \
		echo "示例: make eval CKPT=logs/ppo_*/checkpoints/best_model"; \
		echo "示例: make eval CKPT=models/xxx ENV_TYPE=walking"; \
		echo "示例: make eval CKPT=models/xxx CPU=1  # 使用CPU避免GPU冲突"; \
		echo "示例: make eval CKPT=models/xxx NO_VIDEO=1  # 实时查看器（不保存视频）"; \
		echo "示例: make eval CKPT=models/xxx RENDER=10  # 每10步渲染"; \
		exit 1; \
	fi
	@LOGFILE="$(call MAKELOG_FILE,eval)"; \
	echo "=== 评估模型 ==="; \
	echo "检查点: $(CKPT)"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== 评估日志 ==="; \
		echo "命令: make eval CKPT=$(CKPT)"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "检查点: $(CKPT)"; \
		CMD="FORCE_COLOR=1 $(PYTHON) $(EVAL_SCRIPT) --checkpoint $(CKPT)"; \
		if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; echo "环境类型: $(ENV_TYPE)"; fi; \
		if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; echo "设备: CPU"; fi; \
		if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; echo "渲染间隔: $(RENDER)"; fi; \
		if [ -n "$(NO_VIDEO)" ]; then CMD="$$CMD --no-save-video"; echo "保存视频: 否"; fi; \
		if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; echo "视频路径: $(VIDEO_PATH)"; fi; \
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
	echo "=== 启动TensorBoard ==="; \
	echo "访问: http://localhost:6006"; \
	echo "日志文件: $$LOGFILE"; \
	{ \
		echo "=== TensorBoard日志 ==="; \
		echo "命令: make tensorboard"; \
		echo "开始时间: $$(date '+%Y-%m-%d %H:%M:%S')"; \
		echo "监控目录: $(LOG_DIR)/train/"; \
		echo ""; \
		FORCE_COLOR=1 tensorboard --logdir=$(LOG_DIR)/train --host=0.0.0.0 --port=6006 2>&1; \
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
		rm -rf $(LOG_DIR)/train/*; \
		echo "✓ 训练日志已清空"; \
	else \
		echo "✗ 已取消"; \
	fi

clean-makelog:
	@echo "=== 清理Make执行日志 ==="
	rm -rf $(MAKELOG_DIR)/*
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
	@which black > /dev/null 2>&1 || (echo "❌ black 未安装，请运行: pip install black"; exit 1)
	@which isort > /dev/null 2>&1 || (echo "❌ isort 未安装，请运行: pip install isort"; exit 1)
	black --line-length=100 src/ scripts/ tests/ utils/ 2>/dev/null || true
	isort --profile=black --line-length=100 src/ scripts/ tests/ utils/ 2>/dev/null || true
	@echo "✓ 格式化完成"
	@echo ""
	@echo "格式化配置来源: pyproject.toml"
	@echo "VSCode 配置: .vscode/settings.json"
	@echo "EditorConfig: .editorconfig"

test:
	@echo "=== 运行测试 ==="
	pytest tests/ -v 2>/dev/null || python -m pytest tests/ -v

# ==================== 信息相关 ====================

info:
	@echo "JRL 项目信息（自动课程学习）"
	@echo ""
	@echo "项目根目录: $(PROJECT_ROOT)"
	@echo "日志根目录: $(LOG_DIR)"
	@echo "  训练日志: $(LOG_DIR)/train/"
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
