#!/bin/bash
# 每日自动运行脚本
# 功能：数据更新 -> 选股 -> 回测验证

set -e

# 配置
PROJECT_DIR="/Users/sxt/code/qlib"
PYTHON_CMD="python3"
LOG_DIR="$PROJECT_DIR/logs"

# 导入环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "========================================"
echo "量化策略每日运行"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd "$PROJECT_DIR"

# 1. 数据更新
echo ""
echo "[1/3] 更新数据..."
$PYTHON_CMD -c "
from modules.data.updater import DataUpdater
updater = DataUpdater()
updater.update_daily()
" 2>&1 | tee -a "$LOG_DIR/update.log"

# 2. 选股
echo ""
echo "[2/3] 执行选股..."
$PYTHON_CMD -c "
from core.selection import compute_signal, generate_selections
generate_selections()
" 2>&1 | tee -a "$LOG_DIR/selection.log"

# 3. 回测验证
echo ""
echo "[3/3] 回测验证..."
$PYTHON_CMD -c "
from modules.backtest.qlib_engine import main as backtest_main
backtest_main()
" 2>&1 | tee -a "$LOG_DIR/backtest.log"

echo ""
echo "========================================"
echo "运行完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
