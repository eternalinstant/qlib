#!/bin/bash
# ============================================================
#  Qlib 量化研究仓 - 新环境一键部署脚本
#  目标: Ubuntu 24.04 / Debian 12+
#  使用: chmod +x scripts/setup_new_env.sh && ./scripts/setup_new_env.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================
#  1. 系统依赖
# ============================================================
install_system_deps() {
    info "安装系统依赖..."
    sudo apt update
    sudo apt install -y \
        build-essential python3-dev python3-pip python3-venv \
        git curl wget \
        2>/dev/null || true
}

# ============================================================
#  2. Python 虚拟环境 + 依赖
# ============================================================
setup_python() {
    info "创建 Python 虚拟环境..."
    python3 -m venv "$PROJECT_DIR/.venv"
    source "$PROJECT_DIR/.venv/bin/activate"

    info "升级 pip..."
    pip install --upgrade pip setuptools wheel

    info "安装项目依赖..."
    pip install -e "$PROJECT_DIR[full]"

    info "Python 环境就绪: $($PROJECT_DIR/.venv/bin/python --version)"
}

# ============================================================
#  3. 首次数据初始化
# ============================================================
bootstrap_project_data() {
    if [ -z "${TUSHARE_TOKEN:-}" ]; then
        warn "未设置 TUSHARE_TOKEN，跳过首次数据初始化"
        warn "请先 export TUSHARE_TOKEN=your_token 后运行："
        warn "  source .venv/bin/activate"
        warn "  python main.py update"
        return
    fi

    info "执行首次数据初始化（Tushare -> Qlib -> 选股）..."
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    python main.py update
}

# ============================================================
#  4. 项目配置文件修正（路径适配新机器）
# ============================================================
fix_paths() {
    info "检查项目路径配置..."
    local user_home="$HOME"
    local old_path="/Users/sxt"

    # paths.yaml
    if [ -f "$PROJECT_DIR/config/paths.yaml" ]; then
        sed -i "s|$old_path|$user_home|g" "$PROJECT_DIR/config/paths.yaml"
    fi
    # trading.yaml
    if [ -f "$PROJECT_DIR/config/trading.yaml" ]; then
        sed -i "s|$old_path|$user_home|g" "$PROJECT_DIR/config/trading.yaml"
    fi
    # daily_run.sh
    if [ -f "$PROJECT_DIR/scripts/daily_run.sh" ]; then
        sed -i "s|PROJECT_DIR=.*|PROJECT_DIR=\"$PROJECT_DIR\"|" "$PROJECT_DIR/scripts/daily_run.sh"
    fi

    info "路径配置已修正为: $user_home"
}

# ============================================================
#  5. 创建必要目录
# ============================================================
create_dirs() {
    info "创建数据/输出目录..."
    mkdir -p "$PROJECT_DIR/data/tushare"
    mkdir -p "$PROJECT_DIR/data/qlib_data/cn_data"
    mkdir -p "$PROJECT_DIR/data/qlib_data/raw_data"
    mkdir -p "$PROJECT_DIR/data/selections"
    mkdir -p "$PROJECT_DIR/results"
    mkdir -p "$PROJECT_DIR/logs"
}

# ============================================================
#  6. 生成 activate 辅助脚本
# ============================================================
create_activate() {
    cat > "$PROJECT_DIR/env.sh" << 'EOF'
#!/bin/bash
# 激活 qlib 项目环境
# 使用: source env.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export TUSHARE_TOKEN="${TUSHARE_TOKEN:-}"

[ -z "$TUSHARE_TOKEN" ] && echo "[WARN] TUSHARE_TOKEN 未设置"
echo "Qlib 环境已激活 ($(python --version))"
EOF
    chmod +x "$PROJECT_DIR/env.sh"
    info "已生成 env.sh（source env.sh 即可激活环境）"
}

# ============================================================
#  7. 验证安装
# ============================================================
verify() {
    info "验证安装..."
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

    python -c "
import qlib
import pandas as pd
import numpy as np
import yaml
print('核心依赖导入正常')
" || error "核心依赖导入失败"

    if python -c "import pytest" >/dev/null 2>&1 && { [ -f "$PROJECT_DIR/tests/test_selection.py" ] || [ -d "$PROJECT_DIR/tests" ]; }; then
        info "运行测试..."
        python -m pytest "$PROJECT_DIR/tests" -q --tb=short 2>&1 | tail -5 || warn "部分测试未通过（可能因缺少数据）"
    fi

    info "安装验证完成"
}

# ============================================================
main() {
    echo ""
    echo "========================================="
    echo "  Qlib 量化研究仓 - 环境部署"
    echo "  项目目录: $PROJECT_DIR"
    echo "========================================="
    echo ""

    install_system_deps
    setup_python
    create_dirs
    fix_paths
    create_activate

    # 数据部分可选
    if [ "${SKIP_DATA:-}" != "true" ]; then
        bootstrap_project_data
    fi

    verify

    echo ""
    echo "========================================="
    echo -e "  ${GREEN}部署完成！${NC}"
    echo "========================================="
    echo ""
    echo "快速开始:"
    echo ""
    echo "  source env.sh                          # 激活环境"
    echo "  python main.py update                  # 首次初始化 / 日常更新数据"
    echo "  python main.py backtest --list          # 列出策略"
    echo "  python main.py backtest -s top15_core_trend -e qlib"
    echo ""
    if [ -z "${TUSHARE_TOKEN:-}" ]; then
        echo "⚠ 首次数据未初始化，请设置 token 后运行:"
        echo "  export TUSHARE_TOKEN=your_token"
        echo "  source env.sh"
        echo "  python main.py update"
        echo ""
    fi
    echo "首次数据初始化会从 2016-01-01 拉历史数据；之后重复执行 python main.py update 即可增量更新。"
    echo "数据目录约需 1.8GB 空间 (tushare ~500MB + qlib_data ~1.3GB)"
}

main "$@"
