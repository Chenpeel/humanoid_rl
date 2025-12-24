#!/bin/bash
# ==========================================
# RL 环境快速安装脚本
# ==========================================
#
# 使用方式:
#   bash scripts/install.sh
#
# 注意:
#   1. 需要先安装 Isaac Sim（见 docs/INSTALLATION.md 步骤 1）
#   2. 本脚本会自动安装 Isaac Lab、RSL_RL 和本项目
#   3. 需要 NVIDIA GPU 和 CUDA 环境
#
# ==========================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装"
        return 1
    else
        print_success "$1 已安装"
        return 0
    fi
}

# 主函数
main() {
    print_section "双足机器人 RL 环境安装脚本"
    print_info "开始安装..."

    # ========== 步骤 0：检查系统要求 ==========
    print_section "步骤 0：检查系统要求"

    # 检查 NVIDIA GPU
    print_info "检查 NVIDIA GPU..."
    if ! check_command nvidia-smi; then
        print_error "未检测到 NVIDIA GPU，请先安装 NVIDIA 驱动"
        exit 1
    fi
    nvidia-smi

    # 检查 Python 版本
    print_info "检查 Python 版本..."
    PYTHON_VERSION=$(python3 --version | grep -oP '3\.\d+')
    if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
        print_error "需要 Python 3.10, 3.11, 或 3.12，当前版本: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python 版本: $PYTHON_VERSION"

    # 检查 Git
    print_info "检查 Git..."
    check_command git || exit 1

    # ========== 步骤 1：设置路径 ==========
    print_section "步骤 1：设置路径"

    # 工作目录
    WORKSPACE_DIR="$HOME/workspace"
    print_info "工作目录: $WORKSPACE_DIR"
    mkdir -p $WORKSPACE_DIR

    # Isaac Sim 路径（需要用户手动安装）
    if [ -z "$ISAACSIM_PATH" ]; then
        print_warning "ISAACSIM_PATH 环境变量未设置"
        print_info "尝试自动检测 Isaac Sim 路径..."

        # 查找 Isaac Sim
        ISAAC_SIM_CANDIDATES=(
            "$HOME/.local/share/ov/pkg/isaac-sim-2024.1.1"
            "$HOME/.local/share/ov/pkg/isaac-sim-2024.1.0"
            "$HOME/.local/share/ov/pkg/isaac_sim-2024.1.1"
        )

        for candidate in "${ISAAC_SIM_CANDIDATES[@]}"; do
            if [ -d "$candidate" ]; then
                ISAACSIM_PATH="$candidate"
                print_success "找到 Isaac Sim: $ISAACSIM_PATH"
                break
            fi
        done

        if [ -z "$ISAACSIM_PATH" ]; then
            print_error "未找到 Isaac Sim，请先安装 Isaac Sim 2024.1.1+"
            print_info "参考文档: docs/INSTALLATION.md 步骤 1"
            exit 1
        fi
    else
        print_success "使用环境变量 ISAACSIM_PATH: $ISAACSIM_PATH"
    fi

    # Isaac Lab 路径
    ISAACLAB_PATH="$WORKSPACE_DIR/IsaacLab"
    print_info "Isaac Lab 路径: $ISAACLAB_PATH"

    # RSL_RL 路径
    RSL_RL_PATH="$WORKSPACE_DIR/rsl_rl"
    print_info "RSL_RL 路径: $RSL_RL_PATH"

    # 双足机器人 RL 路径（当前项目）
    BIPED_RL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    print_info "双足机器人 RL 路径: $BIPED_RL_PATH"

    # ========== 步骤 2：安装 Isaac Lab ==========
    print_section "步骤 2：安装 Isaac Lab"

    if [ -d "$ISAACLAB_PATH" ]; then
        print_warning "Isaac Lab 已存在: $ISAACLAB_PATH"
        read -p "是否重新安装？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "删除旧的 Isaac Lab..."
            rm -rf $ISAACLAB_PATH
        else
            print_info "跳过 Isaac Lab 安装"
            SKIP_ISAACLAB=true
        fi
    fi

    if [ -z "$SKIP_ISAACLAB" ]; then
        print_info "克隆 Isaac Lab..."
        cd $WORKSPACE_DIR
        git clone https://github.com/isaac-sim/IsaacLab.git
        cd IsaacLab

        print_info "安装 Isaac Lab（可能需要 10-20 分钟）..."
        ./isaaclab.sh --install

        print_info "安装 Isaac Lab Python 依赖..."
        ./isaaclab.sh --install-dependencies

        print_success "Isaac Lab 安装完成"
    fi

    # 验证 Isaac Lab
    print_info "验证 Isaac Lab 安装..."
    if ! $ISAACLAB_PATH/isaaclab.sh -p -c "import omni.isaac.lab; print('Isaac Lab OK')" 2>&1 | grep -q "Isaac Lab OK"; then
        print_error "Isaac Lab 验证失败"
        exit 1
    fi
    print_success "Isaac Lab 验证通过"

    # ========== 步骤 3：安装 RSL_RL ==========
    print_section "步骤 3：安装 RSL_RL"

    if [ -d "$RSL_RL_PATH" ]; then
        print_warning "RSL_RL 已存在: $RSL_RL_PATH"
        print_info "跳过克隆，更新现有仓库..."
        cd $RSL_RL_PATH
        git pull
    else
        print_info "克隆 RSL_RL..."
        cd $WORKSPACE_DIR
        git clone https://github.com/leggedrobotics/rsl_rl.git
        cd rsl_rl
    fi

    print_info "安装 RSL_RL..."
    $ISAACLAB_PATH/isaaclab.sh -p -m pip install -e .

    # 验证 RSL_RL
    print_info "验证 RSL_RL 安装..."
    if ! $ISAACLAB_PATH/isaaclab.sh -p -c "import rsl_rl; print('RSL_RL OK')" 2>&1 | grep -q "RSL_RL OK"; then
        print_error "RSL_RL 验证失败"
        exit 1
    fi
    print_success "RSL_RL 验证通过"

    # ========== 步骤 4：安装 双足机器人 RL ==========
    print_section "步骤 4：安装 双足机器人 RL"

    cd $BIPED_RL_PATH

    print_info "安装 双足机器人 RL（开发模式）..."
    $ISAACLAB_PATH/isaaclab.sh -p -m pip install -e .

    print_info "安装可视化工具（可选）..."
    $ISAACLAB_PATH/isaaclab.sh -p -m pip install -e ".[vis]" || print_warning "可视化工具安装失败（可选）"

    # 验证 双足机器人 RL
    print_info "验证 双足机器人 RL 安装..."
    if ! $ISAACLAB_PATH/isaaclab.sh -p -c "from jiyuan_tasks import *; print('双足机器人 Tasks OK')" 2>&1 | grep -q "双足机器人 Tasks OK"; then
        print_error "双足机器人 Tasks 验证失败"
        exit 1
    fi
    print_success "双足机器人 Tasks 验证通过"

    # ========== 步骤 5：配置环境变量 ==========
    print_section "步骤 5：配置环境变量"

    # 检查 ~/.bashrc 是否已有配置
    if ! grep -q "ISAACSIM_PATH" ~/.bashrc; then
        print_info "添加 ISAACSIM_PATH 到 ~/.bashrc..."
        echo "" >> ~/.bashrc
        echo "# Isaac Sim 环境变量（双足机器人 RL 安装脚本添加）" >> ~/.bashrc
        echo "export ISAACSIM_PATH=\"$ISAACSIM_PATH\"" >> ~/.bashrc
    fi

    if ! grep -q "ISAACLAB_PATH" ~/.bashrc; then
        print_info "添加 ISAACLAB_PATH 到 ~/.bashrc..."
        echo "export ISAACLAB_PATH=\"$ISAACLAB_PATH\"" >> ~/.bashrc
    fi

    if ! grep -q "BIPED_RL_PATH" ~/.bashrc; then
        print_info "添加 BIPED_RL_PATH 到 ~/.bashrc..."
        echo "export BIPED_RL_PATH=\"$BIPED_RL_PATH\"" >> ~/.bashrc
        echo "export PYTHONPATH=\"\$BIPED_RL_PATH:\$PYTHONPATH\"" >> ~/.bashrc
    fi

    print_success "环境变量配置完成"
    print_info "请运行 'source ~/.bashrc' 使配置生效"

    # ========== 步骤 6：最终验证 ==========
    print_section "步骤 6：最终验证"

    print_info "验证配置文件..."
    if $ISAACLAB_PATH/isaaclab.sh -p -c "
from jiyuan_tasks.utils.config_loader import load_train_config, validate_train_config
cfg = load_train_config('$BIPED_RL_PATH/configs/train_config.yaml')
validate_train_config(cfg)
print('配置文件验证通过')
" 2>&1 | grep -q "配置文件验证通过"; then
        print_success "配置文件验证通过"
    else
        print_warning "配置文件验证失败（可能需要手动检查）"
    fi

    # ========== 完成 ==========
    print_section "安装完成！"

    echo ""
    print_success "所有组件安装成功！"
    echo ""
    echo "关键路径:"
    echo "  - Isaac Sim:   $ISAACSIM_PATH"
    echo "  - Isaac Lab:   $ISAACLAB_PATH"
    echo "  - RSL_RL:      $RSL_RL_PATH"
    echo "  - 双足机器人 RL:   $BIPED_RL_PATH"
    echo ""
    echo "下一步:"
    echo "  1. 运行 'source ~/.bashrc' 使环境变量生效"
    echo "  2. 测试训练:"
    echo "     cd $BIPED_RL_PATH"
    echo "     $ISAACLAB_PATH/isaaclab.sh -p scripts/train.py --task velocity --num_envs 64 --max_iterations 10"
    echo "  3. 查阅文档:"
    echo "     - docs/QUICKSTART.md          - 快速开始"
    echo "     - docs/MINIMAL_SETUP_GUIDE.md - 配置系统"
    echo "     - docs/INSTALLATION.md        - 详细安装指南"
    echo ""
}

# 运行主函数
main "$@"
