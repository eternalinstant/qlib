#!/bin/bash
# Tushare 数据下载脚本
# 下载 2016 年至今的因子数据

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Tushare Pro 因子数据下载"
echo "时间范围: 2016-01-01 至今"
echo "=========================================="

python3 -m modules.data.tushare_downloader --start 20160101 --type all

echo ""
echo "下载完成! 数据保存在: data/tushare/"
echo ""
echo "数据文件:"
ls -lh data/tushare/*.parquet 2>/dev/null || echo "暂无数据文件"
