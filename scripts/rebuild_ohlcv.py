#!/usr/bin/env python3
"""
全量重建前复权 OHLCV bin 文件

委托给 scripts.rebuild_with_adj_factor 使用 Tushare 官方 adj_factor 重建。
"""
import logging
from pathlib import Path

import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.rebuild_with_adj_factor as rebuild_adj


def rebuild_all():
    """全量重建并保持前复权口径一致。"""
    logger.info("rebuild_ohlcv.py 委托给 rebuild_with_adj_factor.py")
    return rebuild_adj.rebuild_all()

