"""本地数据仓库配置 — 与连接配置 config.py、实盘配置 live_config.py 解耦"""

import os

# ── 仓库根目录 ────────────────────────────────────────────
DATA_DIR = r"C:\Users\Administrator\Documents\qmt\data"

# ── 股票池（universe）─────────────────────────────────────
# 全 A 板块名；运行时用 qmt_data.get_sector_list() 校验确切名称。
# 常见候选: "沪深A股" / "沪深京A股" / "上证A股" / "深证A股"
UNIVERSE_SECTOR = "沪深A股"

# ── 历史范围 ──────────────────────────────────────────────
HISTORY_START = ""             # 空串 = 全历史
PERIODS = ["1d"]               # 当前只维护日线

# ── 复权口径 ──────────────────────────────────────────────
# 仓库存不复权（原始成交价）：append-only 真相源，新除权除息不会回溯改写历史，
# 增量追加才稳定一致。前复权会在每次分红时改写全部历史价，故此处显式用 none。
# 下游（qlib）若需复权用复权因子自行计算。
DIVIDEND_TYPE = "none"

# ── 财务报表 ──────────────────────────────────────────────
FINANCIAL_TABLES = ["Balance", "Income", "CashFlow"]

# ── 交易日历 ──────────────────────────────────────────────
CALENDAR_MARKETS = ["SH"]

# ── 批量参数 ──────────────────────────────────────────────
DOWNLOAD_BATCH = 300           # 下载/读回分批，约束内存与单次耗时
INCREMENTAL_LOOKBACK_DAYS = 5  # 增量回拉最近 N 个交易日，按日期去重合并以吸收订正


def daily_dir():
    return os.path.join(DATA_DIR, "daily")


def financial_dir():
    return os.path.join(DATA_DIR, "financial")


def meta_dir():
    return os.path.join(DATA_DIR, "meta")


def ensure_dirs():
    """确保仓库目录存在"""
    for d in (DATA_DIR, daily_dir(), financial_dir(), meta_dir()):
        os.makedirs(d, exist_ok=True)
