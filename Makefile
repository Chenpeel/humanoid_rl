# Makefile for JRL - JAX Reinforcement Learning Training
# 用于管理四足机器人训练（自动课程学习）

.PHONY: help install train train-long train-test train-quick eval clean clean-cache clean-all
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

# 配置文件路径
CONFIG_TRAIN := configs/train/train_default.yaml
CONFIG_LONG := configs/train-10h/train_long.yaml
CONFIG_QUICK := configs/quick_test/quick_test.yaml

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
	@echo "  make clean-logs           清理所有日志"
	@echo "  make clean-all            清理所有（缓存+日志+临时文件）"

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
	@echo "=== 标准训练（自动课程学习）==="
	@echo "配置: $(CONFIG_TRAIN)"
	@echo "环境数: 2048，总步数: 200M"
	@echo "预计时间: 8-12小时"
	@echo "开始时间: $$(date)"
	@echo ""
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_TRAIN)
	@echo ""
	@echo "=== 训练完成 ==="
	@echo "结束时间: $$(date)"

train-long:
	@echo "=== 长时间训练（自动课程学习）==="
	@echo "配置: $(CONFIG_LONG)"
	@echo "环境数: 4096，总步数: 500M"
	@echo "预计时间: 24-48小时"
	@echo "开始时间: $$(date)"
	@echo ""
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_LONG)
	@echo ""
	@echo "=== 训练完成 ==="
	@echo "结束时间: $$(date)"

train-test:
	@echo "=== 快速测试（自动课程学习）==="
	@echo "配置: $(CONFIG_QUICK)"
	@echo "环境数: 128，总步数: 1M"
	@echo "预计时间: 5-10分钟"
	@echo ""
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG_QUICK)
	@echo ""
	@echo "=== 测试完成 ==="

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
	@echo "=== 自定义配置训练 ==="
	@echo "配置: $(CONFIG)"
	@echo ""
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG)
	@echo ""
	@echo "=== 训练完成 ==="

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
	@echo "=== 评估模型 ==="
	@echo "检查点: $(CKPT)"
	@CMD="$(PYTHON) $(EVAL_SCRIPT) --checkpoint $(CKPT)"; \
	if [ -n "$(ENV_TYPE)" ]; then CMD="$$CMD --env-type $(ENV_TYPE)"; fi; \
	if [ -n "$(CPU)" ]; then CMD="$$CMD --cpu"; fi; \
	if [ -n "$(RENDER)" ]; then CMD="$$CMD --render $(RENDER)"; fi; \
	if [ -n "$(NO_VIDEO)" ]; then CMD="$$CMD --no-save-video"; fi; \
	if [ -n "$(VIDEO_PATH)" ]; then CMD="$$CMD --video-path $(VIDEO_PATH)"; fi; \
	eval $$CMD

tensorboard:
	@echo "=== 启动TensorBoard ==="
	@echo "访问: http://localhost:6006"
	tensorboard --logdir=$(LOG_DIR) --host=0.0.0.0 --port=6006

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
	@echo "JRL 项目信息（自动课程学习）"
	@echo ""
	@echo "项目根目录: $(PROJECT_ROOT)"
	@echo "日志目录: $(LOG_DIR)"
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
