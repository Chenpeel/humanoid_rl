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
TRAIN_CONFIG_DIR := $(CONFIG_DIR)/train
TEST_CONFIG_DIR := $(CONFIG_DIR)/test

# ==================== 帮助信息 ====================

help:
	@echo "JRL 训练系统 - Makefile 命令"
	@echo ""
	@echo "环境配置："
	@echo "  make install              安装JAX依赖"
	@echo "  make install-dev          安装开发依赖"
	@echo "  make check-env            检查JAX/CUDA环境"
	@echo ""
	@echo "训练命令："
	@echo "  make train                训练阶段0（站立）"
	@echo "  make train-stage STAGE=N  训练指定阶段(0-4)"
	@echo "  make train-all            流水线训练所有阶段"
	@echo "  make train-range FROM=N TO=M  训练阶段N到M"
	@echo ""
	@echo "测试命令："
	@echo "  make quick-test           极速测试（32 envs，Eager模式，无需编译）"
	@echo "  make test-pipeline        标准测试（128 envs，JIT模式，<4分钟）"
	@echo "  make validate-config      验证配置文件语法"
	@echo ""
	@echo "说明："
	@echo "  - quick-test: 使用Eager模式（禁用JIT），无需编译但较慢"
	@echo "  - test-pipeline: 使用JIT模式，首次编译后缓存"
	@echo "  - train: 所有阶段统一网络结构，首次编译后全程复用"
	@echo ""
	@echo "其他："
	@echo "  make eval CKPT=...              评估模型（默认渲染+保存视频）"
	@echo "  make eval CKPT=... RENDER=0     评估模型（不渲染）"
	@echo "  make eval CKPT=... CPU=1        评估模型（使用CPU）"
	@echo "  make tensorboard                启动TensorBoard"
	@echo "  make clean-all                  清理所有缓存和日志"

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
	$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage0_standing.yaml

train-stage:
	@if [ -z "$(STAGE)" ]; then \
		echo "错误: 请指定阶段号 STAGE=0-4"; \
		echo "示例: make train-stage STAGE=0"; \
		exit 1; \
	fi
	@echo "=== 训练阶段$(STAGE) ==="
	@if [ "$(STAGE)" = "0" ]; then \
		$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage0_standing.yaml; \
	elif [ "$(STAGE)" = "1" ]; then \
		$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage1_stepping.yaml; \
	elif [ "$(STAGE)" = "2" ]; then \
		$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage2_slow_walk.yaml; \
	elif [ "$(STAGE)" = "3" ]; then \
		$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage3_normal_walk.yaml; \
	elif [ "$(STAGE)" = "4" ]; then \
		$(PYTHON) $(TRAIN_SCRIPT) --config $(TRAIN_CONFIG_DIR)/stage4_fast_walk.yaml; \
	else \
		echo "错误: STAGE必须是0-4之间的数字"; exit 1; \
	fi

train-all:
	@echo "=== 开始分阶段训练（自动化） ==="
	$(PYTHON) scripts/train_staged.py --start-stage 0 --end-stage 4
	@echo "=== 分阶段训练完成 ==="

train-range:
	@if [ -z "$(FROM)" ] || [ -z "$(TO)" ]; then \
		echo "错误: 请指定起止阶段 FROM=N TO=M"; \
		echo "示例: make train-range FROM=2 TO=4"; \
		exit 1; \
	fi
	@echo "=== 训练阶段$(FROM)到$(TO) ==="
	$(PYTHON) scripts/train_staged.py --start-stage $(FROM) --end-stage $(TO)
	@echo "=== 训练完成 ==="

# ==================== 测试相关 ====================

quick-test:
	@echo "=== 极速测试（32 envs，3阶段，Eager模式）==="
	@echo "策略：禁用JIT编译，直接执行（无需等待编译）"
	$(PYTHON) scripts/train_staged.py --quick-test --no-jit --start-stage 0 --end-stage 2

test-pipeline:
	@echo "=== 标准流水线测试（128 envs，3阶段）==="
	$(PYTHON) scripts/train_staged.py --test-mode --start-stage 0 --end-stage 2

validate-config:
	@echo "=== 验证配置文件 ==="
	@$(PYTHON) -c "import yaml, sys; \
		configs = ['$(TRAIN_CONFIG_DIR)/stage0_standing.yaml', \
		           '$(TRAIN_CONFIG_DIR)/stage1_stepping.yaml', \
		           'configs/quick_test/stage0_standing.yaml', \
		           '$(TEST_CONFIG_DIR)/pipeline_stage0_standing.yaml']; \
		[yaml.safe_load(open(c)) for c in configs]; \
		print('✓ 所有配置文件有效')"

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
	@ls -1 $(TRAIN_CONFIG_DIR)/stage*.yaml 2>/dev/null || echo "  无阶段配置文件"
	@echo ""
	@echo "测试配置:"
	@ls -1 $(TEST_CONFIG_DIR)/*.yaml 2>/dev/null || echo "  无测试配置文件"
