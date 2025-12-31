# ==============================================================================
# Makefile - 双足机器人 RL 项目快速命令
# ==============================================================================
# 推荐使用方式（课程学习）:
#   make train-curriculum   - 全流程课程学习训练（推荐！）
#   make train-standing     - 站立平衡训练
#   make train-flat         - 平坦地形行走训练
#   make train-walking      - 正常速度行走训练
#   make train-rough        - 粗糙地形适应训练
#   make play               - 评估最新训练的策略
#   make play-video         - 录制策略视频
#
# 环境配置:
#   make submodule-update   - 初始化并更新git子模块
#   make install            - 安装项目（开发模式）
#   make install-all        - 安装所有依赖
#
# 其他命令:
#   make help               - 显示完整帮助信息
#   make clean              - 清理构建文件
# ==============================================================================

# Isaac Lab 路径配置
ISAACLAB_PATH ?= dep/IsaacLab
ISAACLAB_PYTHON = $(ISAACLAB_PATH)/isaaclab.sh -p

# AutoDL 云服务器平台专用配置
# 问题：AutoDL 环境下 NVIDIA Open Kernel Module 的 Vulkan 支持需要使用 EGL 库
# 解决方案：指定 NVIDIA Vulkan ICD 文件，使用 libEGL_nvidia.so.0
# 参考：AutoDL 官方文档 - https://www.autodl.com/docs/
export VK_ICD_FILENAMES = /usr/share/vulkan/icd.d/nvidia_icd.json

# 日志级别配置（屏蔽 Isaac Sim 的警告输出）
export CARB_LOG_LEVEL = ERROR


.PHONY: help install install-dev install-vis install-all clean clean-logs verify
.PHONY: submodule-init submodule-update submodule-status submodule-update-remote
.PHONY: check-env check-isaaclab convert-usd
.PHONY: train-curriculum train-standing train-flat train-walking train-rough
.PHONY: play play-velocity play-standing play-walking play-video play-video-velocity play-video-standing

# 默认目标
help:
	@echo "双足机器人 RL 项目 - 可用命令:"
	@echo ""
	@echo "==================================================================="
	@echo "📚 推荐工作流（课程学习）"
	@echo "==================================================================="
	@echo ""
	@echo "课程学习训练（推荐）:"
	@echo "  make train-curriculum        - 全流程课程学习（推荐！）"
	@echo "  make train-standing          - 站立平衡训练 (2000 iters)"
	@echo "  make train-flat              - 平坦地形行走训练 (5000 iters)"
	@echo "  make train-walking           - 正常速度行走训练 (10000 iters)"
	@echo "  make train-rough             - 粗糙地形适应训练 (30000 iters)"
	@echo ""
	@echo "策略评估和视频录制:"
	@echo "  make play                    - 加载最新模型并可视化（速度跟踪任务）"
	@echo "  make play-velocity           - 评估速度跟踪策略"
	@echo "  make play-standing           - 评估站立平衡策略"
	@echo "  make play-walking            - 评估行走步态策略"
	@echo "  make play-video              - 录制速度跟踪视频（默认500步）"
	@echo "  make play-video-velocity     - 录制速度跟踪视频"
	@echo "  make play-video-standing     - 录制站立平衡视频"
	@echo "    可选参数:"
	@echo "      CHECKPOINT=path/to/model.pt  指定检查点文件"
	@echo "      VIDEO_LENGTH=500             视频长度（步数）"
	@echo "      NUM_ENVS=1                   并行环境数"
	@echo ""
	@echo "==================================================================="
	@echo "🔧 环境配置"
	@echo "==================================================================="
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
	@echo "资产转换:"
	@echo "  make convert-usd             - 将MJCF转换为USD格式"
	@echo "    可选参数:"
	@echo "      MJCF=path/to/model.xml    指定输入MJCF文件"
	@echo "      USD_OUT=path/to/output.usd 指定输出USD文件"
	@echo ""
	@echo "安装命令:"
	@echo "  make install                 - 安装项目（开发模式）"
	@echo "  make install-dev             - 安装项目 + 开发工具"
	@echo "  make install-vis             - 安装项目 + 可视化工具"
	@echo "  make install-all             - 安装所有依赖"
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
# 资产转换
# ==============================================================================

# 将 MJCF 转换为 USD
# 用法：
#   make convert-usd                                    # 使用默认路径
#   make convert-usd MJCF=path/to/model.xml            # 指定输入文件
#   make convert-usd USD_OUT=path/to/output.usd        # 指定输出文件
#   make convert-usd MJCF=input.xml USD_OUT=output.usd # 同时指定
MJCF ?= assets/xmls/models/jiyuan_fit.xml
USD_OUT ?= assets/usd/jiyuan_fit/jiyuan_fit.usd

convert-usd: check-isaaclab
	@echo "转换 MJCF 到 USD..."
	@echo "  输入: $(MJCF)"
	@echo "  输出: $(USD_OUT)"
	@mkdir -p $(dir $(USD_OUT))
	$(ISAACLAB_PYTHON) dep/IsaacLab/scripts/tools/convert_mjcf.py \
		$(MJCF) \
		$(USD_OUT) \
		--import-sites
	@echo "✓ 转换完成: $(USD_OUT)"

# ==============================================================================
# 安装目标
# ==============================================================================

install: check-isaaclab
	@echo "正在安装项目（开发模式）..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e isaaclab_rl/
	@echo "✓ 安装完成！"

install-dev: check-isaaclab
	@echo "正在安装项目 + 开发工具..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl/[dev]"
	@echo "✓ 安装完成（包含开发工具）！"

install-vis: check-isaaclab
	@echo "正在安装项目 + 可视化工具..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl/[vis]"
	@echo "✓ 安装完成（包含可视化工具）！"

install-all: check-isaaclab
	@echo "正在安装所有依赖..."
	$(ISAACLAB_PYTHON) -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
	$(ISAACLAB_PYTHON) -m pip install -e "isaaclab_rl/[all]"
	@echo "✓ 安装完成（包含所有依赖）！"

# ==============================================================================
# 课程学习训练（推荐使用）
# ==============================================================================

# 站立平衡训练
# 目标：学会基本站立，摔倒率<20%
train-standing: check-isaaclab
	@echo ">>> [站立平衡] 开始训练 (2000 iterations)..."
	$(ISAACLAB_PYTHON) isaaclab_rl/scripts/train.py \
		--task standing \
		--num_envs 4096 \
		--max_iterations 2000 $(ARGS)

# 平坦地形行走训练
# 目标：学会迈步，保持平衡
# 自动加载站立训练最新的 checkpoint
train-flat: check-isaaclab
	@echo ">>> [平坦地形] 开始训练 (5000 iterations)..."
	@LATEST_STANDING=$$(ls -td logs/jiyuan_standing/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_STANDING" ]; then echo "Error: 未找到站立训练记录"; exit 1; fi; \
	echo "加载站立模型: jiyuan_standing/$$LATEST_STANDING"; \
	$(ISAACLAB_PYTHON) isaaclab_rl/scripts/train.py \
		--task flat \
		--num_envs 4096 \
		--max_iterations 5000 \
		--resume --load_run jiyuan_standing/$$LATEST_STANDING $(ARGS)

# 正常速度行走训练
# 目标：跟踪速度命令，提高动态稳定性
# 自动加载平坦地形训练最新的 checkpoint
train-walking: check-isaaclab
	@echo ">>> [正常行走] 开始训练 (10000 iterations)..."
	@LATEST_VEL=$$(ls -td logs/jiyuan_velocity_tracking/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_VEL" ]; then echo "Error: 未找到平坦地形训练记录"; exit 1; fi; \
	echo "加载平坦地形模型: jiyuan_velocity_tracking/$$LATEST_VEL"; \
	$(ISAACLAB_PYTHON) isaaclab_rl/scripts/train.py \
		--task velocity \
		--num_envs 4096 \
		--max_iterations 10000 \
		--resume --load_run jiyuan_velocity_tracking/$$LATEST_VEL $(ARGS)

# 粗糙地形适应训练
# 目标：在粗糙地形上稳定行走
train-rough: check-isaaclab
	@echo ">>> [粗糙地形] 开始训练 (30000 iterations)..."
	@LATEST_VEL=$$(ls -td logs/jiyuan_velocity_tracking/*/ 2>/dev/null | head -1 | xargs -I {} basename {}); \
	if [ -z "$$LATEST_VEL" ]; then echo "Error: 未找到行走训练记录"; exit 1; fi; \
	echo "加载行走模型: jiyuan_velocity_tracking/$$LATEST_VEL"; \
	$(ISAACLAB_PYTHON) isaaclab_rl/scripts/train.py \
		--task rough \
		--num_envs 4096 \
		--max_iterations 30000 \
		--resume --load_run jiyuan_velocity_tracking/$$LATEST_VEL $(ARGS)

# 一键启动全流程
train-curriculum: train-standing train-flat train-walking train-rough
	@echo "✓ 课程学习全流程训练完成！"

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
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task velocity \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task velocity \
			--num_envs $(NUM_ENVS) $(ARGS); \
	fi

# 评估站立平衡策略
play-standing: check-isaaclab
	@echo "评估站立平衡策略（GUI 可视化）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task standing \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task standing \
			--num_envs $(NUM_ENVS) $(ARGS); \
	fi

# 评估行走步态策略
play-walking: check-isaaclab
	@echo "评估行走步态策略（GUI 可视化）..."
	@if [ -n "$(CHECKPOINT)" ]; then \
		echo "使用指定检查点: $(CHECKPOINT)"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task walking \
			--checkpoint $(CHECKPOINT) \
			--num_envs $(NUM_ENVS) $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
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
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task velocity \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
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
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task standing \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
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
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
			--task walking \
			--checkpoint $(CHECKPOINT) \
			--video --video_length $(VIDEO_LENGTH) \
			--num_envs 1 $(ARGS); \
	else \
		echo "自动加载最新检查点"; \
		$(ISAACLAB_PYTHON) isaaclab_rl/scripts/play.py \
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
	
	@echo "✓ 日志清理完成！"

# ==============================================================================
# 开发辅助目标
# ==============================================================================

# 格式化代码（需要安装 dev 依赖）
format: check-isaaclab
	@echo "格式化代码..."
	$(ISAACLAB_PYTHON) -m black isaaclab_rl/jiyuan_tasks/ isaaclab_rl/scripts/ --line-length 120
	@echo "✓ 代码格式化完成！"

# 运行测试（需要安装 dev 依赖）
test: check-isaaclab
	@echo "运行测试..."
	$(ISAACLAB_PYTHON) -m pytest isaaclab_rl/tests/ -v
	@echo "✓ 测试完成！"
