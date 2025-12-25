# ==============================================================================
# Makefile - 双足机器人 RL 项目快速命令
# ==============================================================================
# 使用方式:
#   make submodule-update   - 初始化并更新git子模块
#   make install            - 安装项目（开发模式）
#   make install-dev        - 安装项目 + 开发工具
#   make install-vis        - 安装项目 + 可视化工具
#   make install-all        - 安装所有依赖
#   make train              - 开始训练（使用默认配置）
#   make train-test         - 快速测试训练（64 envs, 10 iters）
#   make clean              - 清理构建文件
#   make help               - 显示帮助信息
# ==============================================================================

# Isaac Lab 路径配置
ISAACLAB_PATH ?= ../dep/IsaacLab
ISAACLAB_PYTHON = $(ISAACLAB_PATH)/isaaclab.sh -p

.PHONY: help install install-dev install-vis install-all train train-test clean clean-logs verify
.PHONY: submodule-init submodule-update submodule-status submodule-update-remote
.PHONY: check-env check-isaaclab

# 默认目标
help:
	@echo "双足机器人 RL 项目 - 可用命令:"
	@echo ""
	@echo "子模块管理:"
	@echo "  make submodule-init          - 初始化git子模块"
	@echo "  make submodule-update        - 更新git子模块（推荐）"
	@echo "  make submodule-status        - 查看子模块状态"
	@echo "  make submodule-update-remote - 更新子模块到最新版本"
	@echo ""
	@echo "环境配置:"
	@echo "  ISAACLAB_PATH=$(ISAACLAB_PATH)"
	@echo "  (可通过环境变量 ISAACLAB_PATH 自定义路径)"
	@echo ""
	@echo "环境检查:"
	@echo "  make check-isaaclab          - 检查Isaac Lab环境"
	@echo "  make check-env               - 检查所有环境"
	@echo ""
	@echo "安装命令:"
	@echo "  make install                 - 安装项目（开发模式）"
	@echo "  make install-dev             - 安装项目 + 开发工具"
	@echo "  make install-vis             - 安装项目 + 可视化工具"
	@echo "  make install-all             - 安装所有依赖"
	@echo ""
	@echo "训练命令:"
	@echo "  make train                   - 开始训练（使用默认配置）"
	@echo "  make train-test              - 快速测试训练（64 envs, 10 iters）"
	@echo ""
	@echo "验证命令:"
	@echo "  make verify                  - 验证安装是否成功"
	@echo ""
	@echo "清理命令:"
	@echo "  make clean                   - 清理构建文件"
	@echo "  make clean-logs              - 清理日志文件"
	@echo ""

# ==============================================================================
# 子模块管理
# ==============================================================================

# 初始化子模块
submodule-init:
	@echo "初始化git子模块..."
	git submodule init
	@echo "✓ 子模块初始化完成"

# 更新子模块（推荐）
submodule-update:
	@echo "更新git子模块..."
	git submodule update --init --recursive
	@echo "✓ 子模块更新完成"

# 查看子模块状态
submodule-status:
	@echo "子模块状态:"
	git submodule status

# 更新所有子模块到最新
submodule-update-remote:
	@echo "更新子模块到远程最新版本..."
	git submodule update --remote
	@echo "✓ 子模块已更新到最新"

# ==============================================================================
# 环境检查
# ==============================================================================

# 检查Isaac Lab环境
check-isaaclab:
	@echo "检查Isaac Lab环境..."
	@if [ ! -d "$(ISAACLAB_PATH)" ]; then \
		echo "✗ Isaac Lab目录不存在: $(ISAACLAB_PATH)"; \
		echo "  请运行: make submodule-update"; \
		exit 1; \
	else \
		echo "✓ Isaac Lab目录存在: $(ISAACLAB_PATH)"; \
	fi
	@if [ ! -f "$(ISAACLAB_PATH)/isaaclab.sh" ]; then \
		echo "✗ isaaclab.sh脚本不存在"; \
		echo "  请确保Isaac Lab子模块已正确克隆"; \
		exit 1; \
	else \
		echo "✓ isaaclab.sh脚本存在"; \
	fi

# 检查所有环境
check-env: check-isaaclab
	@echo "✓ 所有环境检查通过"

# ==============================================================================
# 安装目标
# ==============================================================================

install: check-isaaclab
	@echo "正在安装项目（开发模式）..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e .
	@echo "✓ 安装完成！"

install-dev: check-isaaclab
	@echo "正在安装项目 + 开发工具..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e ".[dev]"
	@echo "✓ 安装完成（包含开发工具）！"

install-vis: check-isaaclab
	@echo "正在安装项目 + 可视化工具..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e ".[vis]"
	@echo "✓ 安装完成（包含可视化工具）！"

install-all: check-isaaclab
	@echo "正在安装所有依赖..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e ".[all]"
	@echo "✓ 安装完成（包含所有依赖）！"

# ==============================================================================
# 训练目标
# ==============================================================================

train: check-isaaclab
	@echo "开始训练（使用默认配置）..."
	$(ISAACLAB_PYTHON) scripts/train.py --config configs/train_config.yaml

train-test: check-isaaclab
	@echo "快速测试训练（64 envs, 10 iters）..."
	$(ISAACLAB_PYTHON) scripts/train.py --config configs/train_config.yaml --num_envs 64 --max_iterations 10

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
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ 清理完成！"

clean-logs:
	@echo "清理日志文件..."
	rm -rf logs/
	@echo "✓ 日志清理完成！"

# ==============================================================================
# 开发辅助目标
# ==============================================================================

# 格式化代码（需要安装 dev 依赖）
format: check-isaaclab
	@echo "格式化代码..."
	$(ISAACLAB_PYTHON) -m black jiyuan_tasks/ scripts/ --line-length 120
	@echo "✓ 代码格式化完成！"

# 运行测试（需要安装 dev 依赖）
test: check-isaaclab
	@echo "运行测试..."
	$(ISAACLAB_PYTHON) -m pytest tests/ -v
	@echo "✓ 测试完成！"
