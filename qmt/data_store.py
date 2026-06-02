"""数据仓库存储层 — 纯文件/清单 IO + 缺口/覆盖率计算

不 import xtquant，便于离线测试。约定：
- 日线 DataFrame 的索引（index）是交易日字符串 "YYYYMMDD"，落盘时转成 'date' 列。
- CSV 以 date 为键去重合并，保证「只追加不重复」。
"""

import os
import json

import pandas as pd

import data_config


# ── 路径助手 ──────────────────────────────────────────────

def daily_path(code):
    return os.path.join(data_config.daily_dir(), f"{code}.csv")


def financial_path(code, table):
    return os.path.join(data_config.financial_dir(), f"{code}__{table}.csv")


def manifest_path():
    return os.path.join(data_config.meta_dir(), "manifest.json")


def calendar_path(market):
    return os.path.join(data_config.meta_dir(), f"calendar_{market}.csv")


def universe_path():
    return os.path.join(data_config.meta_dir(), "universe.json")


# ── 日线读写 ──────────────────────────────────────────────

def _to_date_frame(df):
    """把 xtquant 风格的 DataFrame（index=日期串）规整成含 'date' 列、date 为字符串的表"""
    out = df.copy()
    if "date" not in out.columns:
        out = out.reset_index()
        first = out.columns[0]
        if first != "date":
            out = out.rename(columns={first: "date"})
    out["date"] = out["date"].astype(str)
    return out


def read_daily(code):
    """读已落地的日线，返回 DataFrame（含 'date' 列）或 None"""
    path = daily_path(code)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"date": str})
    return df


def write_daily(code, df):
    """与已有 CSV 按交易日去重合并、排序后落盘。返回写入后的总行数。

    去重保留新数据（keep='last'），因增量回拉的最近几天可能含订正。
    """
    if df is None or len(df) == 0:
        existing = read_daily(code)
        return 0 if existing is None else len(existing)

    new = _to_date_frame(df)
    existing = read_daily(code)
    if existing is not None and len(existing) > 0:
        merged = pd.concat([existing, new], ignore_index=True)
    else:
        merged = new
    merged = merged.drop_duplicates(subset="date", keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)

    data_config.ensure_dirs()
    merged.to_csv(daily_path(code), index=False, encoding="utf-8")
    return len(merged)


def last_date(code):
    """返回已落地日线的最后交易日 'YYYYMMDD'，无数据返回 None"""
    df = read_daily(code)
    if df is None or len(df) == 0:
        return None
    return str(df["date"].max())


def first_date(code):
    df = read_daily(code)
    if df is None or len(df) == 0:
        return None
    return str(df["date"].min())


# ── 财务读写 ──────────────────────────────────────────────

def write_financial(code, table, df):
    """财务报表落盘（整表覆盖，财务为低频快照）。返回行数。"""
    if df is None or len(df) == 0:
        return 0
    out = df.copy()
    if out.index.name or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    data_config.ensure_dirs()
    out.to_csv(financial_path(code, table), index=False, encoding="utf-8")
    return len(out)


# ── 交易日历 ──────────────────────────────────────────────

def write_calendar(market, dates):
    """落盘交易日历 ['YYYYMMDD', ...]"""
    data_config.ensure_dirs()
    pd.DataFrame({"date": [str(d) for d in dates]}).to_csv(
        calendar_path(market), index=False, encoding="utf-8")


def read_calendar(market):
    path = calendar_path(market)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype={"date": str})
    return df["date"].astype(str).tolist()


# ── 清单 manifest ─────────────────────────────────────────

def load_manifest():
    path = manifest_path()
    if not os.path.exists(path):
        return {"daily": {}, "financial": {}}
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("daily", {})
    m.setdefault("financial", {})
    return m


def save_manifest(manifest):
    data_config.ensure_dirs()
    with open(manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def update_daily_entry(manifest, code, first, last, rows, updated):
    manifest.setdefault("daily", {})[code] = {
        "first": first, "last": last, "rows": rows, "updated": updated,
    }


def update_financial_entry(manifest, code, tables, updated):
    manifest.setdefault("financial", {})[code] = {
        "tables": list(tables), "updated": updated,
    }


# ── 缺口 & 覆盖率 ─────────────────────────────────────────

def find_gaps(stored_dates, calendar_dates):
    """在 [min, max] 区间内，对比交易日历找出缺失的交易日。

    stored_dates / calendar_dates: ['YYYYMMDD', ...]
    返回缺失日列表（升序）。stored 为空则返回 []。
    """
    stored = set(str(d) for d in stored_dates)
    if not stored:
        return []
    lo, hi = min(stored), max(stored)
    return [d for d in sorted(str(c) for c in calendar_dates)
            if lo <= d <= hi and d not in stored]


def coverage_summary(manifest, universe, latest_trading_day=None):
    """汇总仓库覆盖情况。

    universe: 期望维护的代码列表
    latest_trading_day: 最近交易日 'YYYYMMDD'，用于判断「过期」
    返回 dict。
    """
    daily = manifest.get("daily", {})
    uni = set(universe)
    stored_codes = set(daily.keys())

    missing = sorted(uni - stored_codes)
    stale = []
    if latest_trading_day is not None:
        for code in uni & stored_codes:
            last = daily[code].get("last")
            if last is not None and str(last) < str(latest_trading_day):
                stale.append(code)
        stale.sort()

    total_rows = sum(int(e.get("rows", 0)) for e in daily.values())
    return {
        "universe_size": len(uni),
        "stored": len(stored_codes),
        "in_universe_stored": len(uni & stored_codes),
        "missing": missing,
        "missing_count": len(missing),
        "stale": stale,
        "stale_count": len(stale),
        "total_rows": total_rows,
        "generated_at": manifest.get("generated_at"),
    }
