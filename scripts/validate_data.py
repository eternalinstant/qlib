#!/usr/bin/env python3
"""
金融量化数据验证脚本

验证维度:
  P0 - 致命问题（直接导致回测结果错误）
  P1 - 重要问题（影响回测质量）
  P2 - 数据健康指标

输出:
  - 终端分 P0/P1/P2 三级彩色报告
  - JSON 报告保存到 results/data_validation_YYYYMMDD_HHMMSS.json

用法:
  python3 scripts/validate_data.py
  python3 scripts/validate_data.py --data-root /path/to/data
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── ANSI color codes ────────────────────────────────────────────────────
C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"

STATUS_ICONS = {
    STATUS_PASS: f"{C_GREEN}✓ PASS{C_RESET}",
    STATUS_FAIL: f"{C_RED}✗ FAIL{C_RESET}",
    STATUS_WARN: f"{C_YELLOW}⚠ WARN{C_RESET}",
    STATUS_SKIP: f"{C_CYAN}○ SKIP{C_RESET}",
}

STATUS_ORDER = {STATUS_FAIL: 0, STATUS_WARN: 1, STATUS_PASS: 2, STATUS_SKIP: 3}

# Thresholds
LIMIT_UP_THRESHOLDS = {
    # 主板/中小板 10%
    "default": 0.098,
    # 科创/创业板 20%
    "688": 0.198,  # 科创板
    "300": 0.198,  # 创业板
    "301": 0.198,
    # 北交所 30%
    "bj": 0.298,
}
PRICE_JUMP_THRESHOLD = 0.20  # 日间涨跌幅 > 20%
CONSECUTIVE_FLAT_DAYS = 10  # 连续平盘天数阈值
FACTOR_NULL_THRESHOLD = 0.50  # 因子空值率 > 50%
MISSING_RATE_WARN = 0.30  # 缺失率 > 30% 警告
MISSING_RATE_FAIL = 0.70  # 缺失率 > 70% 警告 (退市股票正常现象，不 FAIL)
PE_EXTREME = 10000  # PE > 10000 极端值
DATA_FRESHNESS_WARN_DAYS = 7  # 数据距今天数超过此值警告


# ── DataValidator ───────────────────────────────────────────────────────

class DataValidator:
    """金融量化数据验证器"""

    def __init__(self, data_root: str = None):
        if data_root:
            root = Path(data_root)
        else:
            root = Path(__file__).resolve().parent.parent / "data" / "qlib_data"

        self.cn_data_dir = root / "cn_data"
        self.raw_data_dir = root / "raw_data"
        self.calendar_file = self.cn_data_dir / "calendars" / "day.txt"
        self.instruments_file = self.cn_data_dir / "instruments" / "all.txt"
        self.features_dir = self.cn_data_dir / "features"
        self.factor_file = self.cn_data_dir / "factor_data.parquet"
        self.tushare_dir = (root.parent / "tushare").resolve()  # data/tushare

        # Caches
        self._calendar = None
        self._instruments = None
        self._results = []  # List of check result dicts

        # Counts for summary
        self._counts = defaultdict(int)

    # ── helpers ─────────────────────────────────────────────────────────

    def _load_calendar(self) -> pd.DatetimeIndex:
        if self._calendar is not None:
            return self._calendar
        if not self.calendar_file.exists():
            self._calendar = pd.DatetimeIndex([])
            return self._calendar
        with open(self.calendar_file) as f:
            dates = [line.strip() for line in f if line.strip()]
        self._calendar = pd.DatetimeIndex(dates)
        return self._calendar

    def _load_instruments(self) -> dict:
        """返回 {code: (listing_date_str, end_date_str)}"""
        if self._instruments is not None:
            return self._instruments
        instruments = {}
        if self.instruments_file.exists():
            with open(self.instruments_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        instruments[parts[0]] = (parts[1], parts[2])
        self._instruments = instruments
        return instruments

    def _iter_raw_files(self):
        """遍历 raw_data/*.parquet，跳过非 parquet 文件"""
        if not self.raw_data_dir.exists():
            return
        for f in sorted(self.raw_data_dir.iterdir()):
            if f.suffix == ".parquet" and not f.name.startswith("."):
                yield f

    def _raw_code_set(self) -> set:
        return {f.stem for f in self._iter_raw_files()}

    def _feature_code_set(self) -> set:
        if not self.features_dir.exists():
            return set()
        return {d.name for d in self.features_dir.iterdir() if d.is_dir()}

    def _add_result(self, priority: str, check: str, status: str,
                    detail: str = "", data: dict = None):
        self._results.append({
            "priority": priority,
            "check": check,
            "status": status,
            "detail": detail,
            "data": data or {},
        })
        self._counts[status] += 1

    def _get_prefix(self, code: str) -> str:
        """根据股票代码返回板块前缀: sh, sz, bj"""
        if code.startswith("sh"):
            return code[:2]
        if code.startswith("sz"):
            return code[:2]
        if code.startswith("bj"):
            return code[:2]
        return ""

    def _limit_up_threshold(self, code: str) -> float:
        """根据板块返回涨停阈值"""
        if code.startswith("bj"):
            return LIMIT_UP_THRESHOLDS["bj"]
        for prefix in ["688", "300", "301"]:
            if code[2:].startswith(prefix):
                return LIMIT_UP_THRESHOLDS[prefix]
        return LIMIT_UP_THRESHOLDS["default"]

    # ── P0: 致命问题 ────────────────────────────────────────────────────

    def check_ohlc_integrity(self) -> dict:
        """检查全部 raw_data 的 OHLC 逻辑约束"""
        cal = self._load_calendar()
        violations_detail = []
        total_violations = 0
        files_checked = 0
        files_with_violations = 0

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["open", "high", "low", "close"])
            except Exception:
                continue

            if df.empty:
                continue

            files_checked += 1

            # High >= Low
            hl_viol = (df["high"] < df["low"]).sum()
            # High >= Open, High >= Close
            ho_viol = (df["high"] < df["open"]).sum()
            hc_viol = (df["high"] < df["close"]).sum()
            # Low <= Open, Low <= Close
            lo_viol = (df["low"] > df["open"]).sum()
            lc_viol = (df["low"] > df["close"]).sum()

            file_viol = hl_viol + ho_viol + hc_viol + lo_viol + lc_viol
            if file_viol > 0:
                files_with_violations += 1
                violations_detail.append({
                    "code": raw_path.stem,
                    "high_lt_low": int(hl_viol),
                    "high_lt_open": int(ho_viol),
                    "high_lt_close": int(hc_viol),
                    "low_gt_open": int(lo_viol),
                    "low_gt_close": int(lc_viol),
                    "total": int(file_viol),
                })
            total_violations += file_viol

        status = STATUS_PASS if total_violations == 0 else STATUS_FAIL
        self._add_result("P0", "OHLC 逻辑约束 (High>=Low, High>=Open/Close, Low<=Open/Close)",
                         status,
                         f"{files_checked} 只股票, {total_violations} 条违反, {files_with_violations} 只受影响",
                         {"files_checked": files_checked, "total_violations": total_violations,
                          "files_with_violations": files_with_violations,
                          "violations": violations_detail[:20]})
        return self._results[-1]

    def check_price_nonnegative(self) -> dict:
        """检查 OHLC 价格非负/非零（非停牌日）"""
        cal = self._load_calendar()
        files_checked = 0
        total_nonpositive = 0
        bad_files = []

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["open", "high", "low", "close"])
            except Exception:
                continue

            if df.empty:
                continue

            files_checked += 1
            price_cols = ["open", "high", "low", "close"]
            nonpos = (df[price_cols] <= 0).any(axis=1).sum()
            if nonpos > 0:
                bad_files.append({"code": raw_path.stem, "count": int(nonpos)})
            total_nonpositive += nonpos

        status = STATUS_PASS if total_nonpositive == 0 else STATUS_FAIL
        self._add_result("P0", "价格非负/非零 (open/high/low/close > 0)",
                         status,
                         f"{files_checked} 只股票, {total_nonpositive} 条非正价格",
                         {"files_checked": files_checked,
                          "total_nonpositive": int(total_nonpositive),
                          "affected": bad_files[:20]})
        return self._results[-1]

    def check_instruments_dates(self) -> dict:
        """对比 instruments/all.txt 日期 vs raw_data 实际日期范围"""
        instruments = self._load_instruments()
        cal = self._load_calendar()
        files_checked = 0
        start_mismatches = 0
        end_mismatches = 0
        mismatch_detail = []
        missing_from_instruments = 0
        missing_from_raw = 0

        raw_codes = self._raw_code_set()
        inst_codes = set(instruments.keys())

        # instruments 中有但 raw 中没有的
        only_inst = inst_codes - raw_codes
        # raw 中有但 instruments 中没有的
        only_raw = raw_codes - inst_codes

        for code, (inst_start, inst_end) in instruments.items():
            raw_path = self.raw_data_dir / f"{code}.parquet"
            if not raw_path.exists():
                continue

            files_checked += 1
            try:
                df = pd.read_parquet(raw_path, columns=["date"])
            except Exception:
                continue

            if df.empty:
                continue

            actual_start = df["date"].min().strftime("%Y-%m-%d")
            actual_end = df["date"].max().strftime("%Y-%m-%d")

            start_ok = (actual_start == inst_start)
            end_ok = (actual_end == inst_end)

            if not start_ok:
                start_mismatches += 1
            if not end_ok:
                end_mismatches += 1

            if not start_ok or not end_ok:
                mismatch_detail.append({
                    "code": code,
                    "inst_start": inst_start,
                    "inst_end": inst_end,
                    "actual_start": actual_start,
                    "actual_end": actual_end,
                    "start_ok": start_ok,
                    "end_ok": end_ok,
                })

        # 检查孤立的 instruments 条目
        if only_inst:
            missing_from_raw = len(only_inst)
        if only_raw:
            missing_from_instruments = len(only_raw)

        total_mismatch = start_mismatches + end_mismatches
        if total_mismatch == 0 and missing_from_raw == 0 and missing_from_instruments == 0:
            status = STATUS_PASS
        elif missing_from_raw > 0 or missing_from_instruments > 0:
            status = STATUS_WARN
        else:
            status = STATUS_FAIL

        detail = (f"{files_checked} 已比对, 起始日期不符: {start_mismatches}, "
                  f"结束日期不符: {end_mismatches}")
        if missing_from_raw:
            detail += f", {missing_from_raw} 条 instruments 无对应 raw_data"
        if missing_from_instruments:
            detail += f", {missing_from_instruments} 个 raw_data 无 instruments 条目"

        self._add_result("P0", "instruments 日期 vs 实际 raw_data 日期",
                         status, detail,
                         {"files_checked": files_checked,
                          "start_mismatches": start_mismatches,
                          "end_mismatches": end_mismatches,
                          "missing_from_raw": missing_from_raw,
                          "missing_from_instruments": missing_from_instruments,
                          "only_in_instruments": sorted(list(only_inst)),
                          "only_in_raw": sorted(list(only_raw)),
                          "mismatches": mismatch_detail[:100]})
        return self._results[-1]

    def check_bin_file_integrity(self) -> dict:
        """检查每个 feature 目录 .day.bin 文件完整性"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P0", "Bin 文件完整性", STATUS_SKIP, "日历为空")
            return self._results[-1]

        max_cal_idx = len(cal) - 1
        expected_fields = ["open", "high", "low", "close", "volume", "amount"]

        dirs_checked = 0
        truncated_bins = 0
        missing_field_dirs = 0
        truncated_detail = []
        missing_detail = []

        for inst_dir in sorted(self.features_dir.iterdir()):
            if not inst_dir.is_dir():
                continue

            dirs_checked += 1
            field_end_indices = {}
            dir_ok = True

            for fld in expected_fields:
                bin_path = inst_dir / f"{fld}.day.bin"
                if not bin_path.exists():
                    dir_ok = False
                    continue

                raw = np.fromfile(bin_path, dtype="<f4")
                if len(raw) < 2:
                    dir_ok = False
                    continue

                start_idx = int(raw[0])
                data_len = len(raw) - 1
                end_idx = start_idx + data_len - 1
                field_end_indices[fld] = (start_idx, end_idx, data_len)

            # Check missing fields
            missing_fields = [f for f in expected_fields if f not in field_end_indices]
            if missing_fields:
                missing_detail.append({
                    "code": inst_dir.name,
                    "missing_fields": missing_fields,
                })
                missing_field_dirs += 1
                continue

            # Check end_idx consistency across fields
            start_indices = {f: v[0] for f, v in field_end_indices.items()}
            end_indices = {f: v[1] for f, v in field_end_indices.items()}

            if len(set(end_indices.values())) > 1:
                # end indices don't match across fields
                truncated_bins += 1
                truncated_detail.append({
                    "code": inst_dir.name,
                    "end_indices": end_indices,
                    "max_cal_idx": max_cal_idx,
                })
                continue

            end_idx = list(end_indices.values())[0]
            if end_idx > max_cal_idx:
                truncated_bins += 1
                truncated_detail.append({
                    "code": inst_dir.name,
                    "end_idx": end_idx,
                    "max_cal_idx": max_cal_idx,
                })

        if dirs_checked == 0:
            status = STATUS_SKIP
            detail = "无 features 目录"
        elif truncated_bins == 0 and missing_field_dirs == 0:
            status = STATUS_PASS
            detail = f"{dirs_checked} 个目录全部通过"
        elif missing_field_dirs > 0:
            status = STATUS_FAIL
            detail = f"{dirs_checked} 个目录, {missing_field_dirs} 个缺字段, {truncated_bins} 个超范围"
        else:
            status = STATUS_FAIL
            detail = f"{dirs_checked} 个目录, {truncated_bins} 个 end_idx 超范围"

        self._add_result("P0", "Bin 文件完整性 (end_idx 不超日历, 6 字段齐备)",
                         status, detail,
                         {"dirs_checked": dirs_checked,
                          "truncated_bins": truncated_bins,
                          "missing_field_dirs": missing_field_dirs,
                          "max_cal_idx": max_cal_idx,
                          "truncated_detail": truncated_detail[:50],
                          "missing_detail": missing_detail[:50]})
        return self._results[-1]

    # ── P1: 重要问题 ────────────────────────────────────────────────────

    def check_coverage_rate(self) -> dict:
        """每只股票在日历交易日上的 OHLCV 覆盖率"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "数据缺失率", STATUS_SKIP, "日历为空")
            return self._results[-1]

        cal_set = set(cal)
        total_cal_days = len(cal)
        coverage_rates = {}
        low_coverage = []
        below_warn = 0
        below_fail = 0

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["date", "volume"])
            except Exception:
                continue

            if df.empty:
                coverage_rates[raw_path.stem] = 0.0
                below_fail += 1
                continue

            df_dates = pd.to_datetime(df["date"]).dt.normalize()
            actual_dates = set(df_dates) & cal_set
            coverage = len(actual_dates) / total_cal_days if total_cal_days > 0 else 0
            coverage_rates[raw_path.stem] = coverage

            missing_rate = 1.0 - coverage
            if missing_rate >= MISSING_RATE_FAIL:
                below_fail += 1
                low_coverage.append({
                    "code": raw_path.stem,
                    "coverage": round(coverage, 4),
                    "missing_rate": round(missing_rate, 4),
                })
            elif missing_rate >= MISSING_RATE_WARN:
                below_warn += 1

        avg_coverage = (np.mean(list(coverage_rates.values()))
                        if coverage_rates else 0)

        # 退市股票日历覆盖率低属正常现象，最多 WARN 不 FAIL
        if below_fail == 0 and below_warn == 0:
            status = STATUS_PASS
        else:
            status = STATUS_WARN

        self._add_result("P1", "OHLCV 日历覆盖率",
                         status,
                         f"平均覆盖率 {avg_coverage:.2%}, "
                         f"缺失 >{MISSING_RATE_FAIL:.0%}: {below_fail} 只, "
                         f"缺失 >{MISSING_RATE_WARN:.0%}: {below_warn} 只",
                         {"avg_coverage": round(avg_coverage, 4),
                          "total_calendar_days": total_cal_days,
                          "below_warn": below_warn,
                          "below_fail": below_fail,
                          "low_coverage_samples": low_coverage[:30]})
        return self._results[-1]

    def check_price_jumps(self) -> dict:
        """价格跳空检测：日间涨跌幅 > 20%（非新股首日）"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "价格跳空检测", STATUS_SKIP, "日历为空")
            return self._results[-1]

        total_jumps = 0
        jump_detail = []
        files_checked = 0

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["date", "close"])
            except Exception:
                continue

            if df.empty or len(df) < 2:
                files_checked += 1
                continue

            files_checked += 1
            df = df.sort_values("date").reset_index(drop=True)
            df["prev_close"] = df["close"].shift(1)
            df["return"] = (df["close"] / df["prev_close"] - 1).abs()

            # 跳过第一条（无前收盘价）
            jumps = df[df["return"] > PRICE_JUMP_THRESHOLD]
            if len(jumps) > 0:
                for _, row in jumps.iterrows():
                    jump_detail.append({
                        "code": raw_path.stem,
                        "date": str(row["date"].date()),
                        "return": round(float(row["return"]), 4),
                    })
                total_jumps += len(jumps)

        status = STATUS_PASS if total_jumps == 0 else STATUS_WARN
        self._add_result("P1", "价格跳空 (>20% 日间涨跌幅)",
                         status,
                         f"{files_checked} 只股票, {total_jumps} 次跳空",
                         {"files_checked": files_checked,
                          "total_jumps": total_jumps,
                          "threshold": PRICE_JUMP_THRESHOLD,
                          "details": jump_detail[:100]})
        return self._results[-1]

    def check_limit_up_down(self) -> dict:
        """涨跌停检测（抽样检查）"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "涨跌停检测", STATUS_SKIP, "日历为空")
            return self._results[-1]

        files_checked = 0
        limit_up_count = 0
        limit_down_count = 0
        detail = []

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["date", "close", "volume"])
            except Exception:
                continue

            if df.empty or len(df) < 2:
                files_checked += 1
                continue

            files_checked += 1
            df = df.sort_values("date").reset_index(drop=True)
            df["prev_close"] = df["close"].shift(1)
            df["return"] = df["close"] / df["prev_close"] - 1
            threshold = self._limit_up_threshold(raw_path.stem)

            up_mask = (df["return"] >= threshold) & (df["volume"] > 0)
            down_mask = (df["return"] <= -threshold) & (df["volume"] > 0)

            up_hits = int(up_mask.sum())
            down_hits = int(down_mask.sum())

            if up_hits > 0:
                limit_up_count += up_hits
            if down_hits > 0:
                limit_down_count += down_hits

            if up_hits > 0 or down_hits > 0:
                detail.append({
                    "code": raw_path.stem,
                    "limit_up_days": up_hits,
                    "limit_down_days": down_hits,
                    "threshold": threshold,
                })

        self._add_result("P1", "涨跌停检测 (涨跌幅达阈值且成交量>0)",
                         STATUS_WARN if limit_up_count > 0 or limit_down_count > 0 else STATUS_PASS,
                         f"{files_checked} 只股票, 涨停 {limit_up_count} 次, 跌停 {limit_down_count} 次",
                         {"files_checked": files_checked,
                          "limit_up_count": limit_up_count,
                          "limit_down_count": limit_down_count,
                          "detail": detail[:50]})
        return self._results[-1]

    def check_zero_volume_flat(self) -> dict:
        """零量/持续平盘检测"""
        cal = self._load_calendar()
        files_checked = 0
        zero_vol_stocks = 0
        flat_stocks = 0
        detail = []

        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["date", "high", "low", "volume"])
            except Exception:
                continue

            if df.empty:
                files_checked += 1
                continue

            files_checked += 1

            # Zero volume
            zero_vol = (df["volume"] == 0)
            zero_vol_days = int(zero_vol.sum())

            # Consecutive flat: high==low for CONSECUTIVE_FLAT_DAYS+
            is_flat = (df["high"] == df["low"]).astype(int)
            consecutive = 0
            max_consecutive = 0
            for v in is_flat:
                if v == 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0

            if zero_vol_days > 0:
                zero_vol_stocks += 1
            if max_consecutive >= CONSECUTIVE_FLAT_DAYS:
                flat_stocks += 1

            if zero_vol_days > 0 or max_consecutive >= CONSECUTIVE_FLAT_DAYS:
                detail.append({
                    "code": raw_path.stem,
                    "zero_vol_days": zero_vol_days,
                    "max_consecutive_flat": max_consecutive,
                })

        if zero_vol_stocks == 0 and flat_stocks == 0:
            status = STATUS_PASS
            desc = f"{files_checked} 只股票, 无异常"
        else:
            status = STATUS_WARN
            desc = (f"{files_checked} 只股票, {zero_vol_stocks} 只有零量日, "
                    f"{flat_stocks} 只连续平盘 ≥{CONSECUTIVE_FLAT_DAYS}天")

        self._add_result("P1", f"零量/持续平盘 (连续{CONSECUTIVE_FLAT_DAYS}天 high=low)",
                         status, desc,
                         {"files_checked": files_checked,
                          "zero_vol_stocks": zero_vol_stocks,
                          "flat_stocks": flat_stocks,
                          "consecutive_flat_threshold": CONSECUTIVE_FLAT_DAYS,
                          "detail": detail[:50]})
        return self._results[-1]

    def check_factor_null_rates(self) -> dict:
        """因子空值率分析"""
        if not self.factor_file.exists():
            self._add_result("P1", "Factor 空值率", STATUS_SKIP, "factor_data.parquet 不存在")
            return self._results[-1]

        try:
            df = pd.read_parquet(self.factor_file)
        except Exception as e:
            self._add_result("P1", "Factor 空值率", STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        # 排除元数据列
        meta_cols = {"instrument", "datetime"}
        factor_cols = [c for c in df.columns if c not in meta_cols]

        high_null_factors = []
        all_null_rates = {}
        for col in factor_cols:
            null_rate = df[col].isnull().mean()
            all_null_rates[col] = round(null_rate, 4)
            if null_rate > FACTOR_NULL_THRESHOLD:
                high_null_factors.append({
                    "factor": col,
                    "null_rate": round(null_rate, 4),
                    "null_count": int(df[col].isnull().sum()),
                })

        all_null = [c for c in factor_cols if df[c].isnull().all()]
        if all_null:
            status = STATUS_FAIL
        elif high_null_factors:
            status = STATUS_WARN
        else:
            status = STATUS_PASS

        desc = (f"{len(factor_cols)} 个因子, "
                f"{len(all_null)} 个全空, "
                f"{len(high_null_factors)} 个空值率 >{FACTOR_NULL_THRESHOLD:.0%}")

        self._add_result("P1", "Factor 空值率", status, desc,
                         {"total_factors": len(factor_cols),
                          "total_rows": len(df),
                          "all_null_factors": all_null,
                          "high_null_factors": high_null_factors,
                          "all_null_rates": all_null_rates,
                          "threshold": FACTOR_NULL_THRESHOLD})
        return self._results[-1]

    def check_raw_vs_feature_consistency(self) -> dict:
        """抽样检查 raw_data OHLCV vs qlib bin 数据可用性一致性

        Note: bin 文件中的价格经过前复权调整，与 raw_data 原始价格不同是正常的。
        此检查验证:
        1. 日覆盖率是否一致 (raw 有数据的交易日 bin 中也有)
        2. 前复权比率是否一致 (所有 OHLC 字段的 bin/raw 比率相同)
        """
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "Raw vs Feature (bin) 数据一致性",
                             STATUS_SKIP, "日历为空")
            return self._results[-1]

        raw_codes = self._raw_code_set()
        feature_codes = self._feature_code_set()
        common = sorted(raw_codes & feature_codes)

        if not common:
            self._add_result("P1", "Raw vs Feature (bin) 数据一致性",
                             STATUS_SKIP, "无共同标的")
            return self._results[-1]

        # 抽样 50 只
        np.random.seed(42)
        sample = list(np.random.choice(common, size=min(50, len(common)), replace=False))

        date_to_idx = {d: i for i, d in enumerate(cal)}
        date_from_idx = {i: d for d, i in date_to_idx.items()}

        sample_has_adj = 0
        adj_ratio_inconsistent = 0
        adj_inconsist_detail = []
        coverage_mismatch_total = 0
        coverage_mismatch_detail = []

        for code in sample:
            # Read raw data
            raw_path = self.raw_data_dir / f"{code}.parquet"
            try:
                raw_df = pd.read_parquet(raw_path, columns=["date", "open", "high", "low", "close", "volume"])
            except Exception:
                continue

            raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()
            raw_df = raw_df.dropna(subset=["date"])
            if raw_df.empty:
                continue

            # Read bin files
            inst_dir = self.features_dir / code
            bin_data = {}
            bin_start_idx = None
            for fld in ["open", "high", "low", "close", "volume"]:
                bin_path = inst_dir / f"{fld}.day.bin"
                if not bin_path.exists():
                    break
                raw_bin = np.fromfile(bin_path, dtype="<f4")
                if len(raw_bin) < 2:
                    break
                bin_start_idx = int(raw_bin[0])
                bin_data[fld] = raw_bin[1:]

            if len(bin_data) != 5:
                continue

            # Check: for dates where raw has non-null OHLC, bin should also have non-null
            # Normal: bin has MORE coverage (NaN-filled for all calendar) while raw has sparse dates
            # Issue: raw has data but bin is NaN
            price_fields = ["open", "high", "low", "close"]
            coverage_issues = 0
            for _, row in raw_df.iterrows():
                cal_idx = date_to_idx.get(row["date"])
                if cal_idx is None:
                    continue
                bin_idx = cal_idx - bin_start_idx
                if bin_idx < 0 or bin_idx >= len(bin_data["close"]):
                    continue

                raw_all_nan = all(pd.isna(row[f]) for f in price_fields)
                if raw_all_nan:
                    continue

                for fld in price_fields:
                    bin_val = bin_data[fld][bin_idx]
                    raw_val = row[fld]
                    if not pd.isna(raw_val) and pd.isna(bin_val):
                        coverage_issues += 1

            if coverage_issues > 0:
                coverage_mismatch_total += coverage_issues
                coverage_mismatch_detail.append({
                    "code": code, "issues": coverage_issues,
                })

            # Check adj ratio consistency: for each date, bin/raw should be same across OHLC
            # Only for stocks that clearly have adjusted prices (ratio != 1.0)
            adj_inconsist = 0
            has_adj = False
            for _, row in raw_df.iterrows():
                cal_idx = date_to_idx.get(row["date"])
                if cal_idx is None:
                    continue
                bin_idx = cal_idx - bin_start_idx
                if bin_idx < 0 or bin_idx >= len(bin_data["close"]):
                    continue

                ratios = []
                for fld in price_fields:
                    bin_val = bin_data[fld][bin_idx]
                    raw_val = row[fld]
                    if pd.isna(bin_val) or pd.isna(raw_val) or raw_val == 0:
                        continue
                    ratios.append(bin_val / raw_val)

                if len(ratios) < 2:
                    continue

                if abs(ratios[0] - 1.0) > 0.001:
                    has_adj = True

                max_ratio_diff = max(ratios) - min(ratios)
                if max_ratio_diff > 0.01:
                    adj_inconsist += 1

            if has_adj:
                sample_has_adj += 1
            if adj_inconsist > 0:
                adj_ratio_inconsistent += 1
                adj_inconsist_detail.append({
                    "code": code,
                    "inconsistent_dates": adj_inconsist,
                })

        # Evaluate
        warnings = []
        if sample_has_adj > 0:
            warnings.append(f"{sample_has_adj}/{len(sample)} 只有前复权 (价格值不同属正常)")

        if adj_ratio_inconsistent > 0:
            _s = STATUS_WARN
            warnings.append(f"{adj_ratio_inconsistent} 只前复权比率不一致 (可能数据问题)")
        elif coverage_mismatch_total > 100:
            _s = STATUS_WARN
        elif coverage_mismatch_total > 0:
            _s = STATUS_WARN
        else:
            _s = STATUS_PASS

        desc = f"抽样 {len(sample)} 只, 日覆盖不一致: {coverage_mismatch_total} 条"
        if warnings:
            desc += " | " + "; ".join(warnings)

        self._add_result("P1", "Raw vs Feature (bin) 数据一致性",
                         _s, desc,
                         {"sample_size": len(sample),
                          "sample_has_adjustment": sample_has_adj,
                          "adj_ratio_inconsistent_stocks": adj_ratio_inconsistent,
                          "coverage_mismatches": coverage_mismatch_total,
                          "adj_inconsist_detail": adj_inconsist_detail[:20],
                          "coverage_mismatch_detail": coverage_mismatch_detail[:20]})
        return self._results[-1]

    # ── P1: Tushare 源数据检查 ──────────────────────────────────────────

    def _iter_tushare_files(self):
        """遍历 tushare/*.parquet"""
        if not self.tushare_dir.exists():
            import logging
            logging.getLogger(__name__).warning(
                "tushare_dir does not exist: %s (resolved: %s)",
                self.tushare_dir, str(self.tushare_dir))
            return
        if not self.tushare_dir.is_dir():
            import logging
            logging.getLogger(__name__).warning(
                "tushare_dir is not a directory: %s", self.tushare_dir)
            return
        parquet_files = sorted(
            f for f in self.tushare_dir.iterdir()
            if f.suffix == ".parquet" and not f.name.startswith(".")
        )
        if not parquet_files:
            import logging
            logging.getLogger(__name__).warning(
                "tushare_dir exists but contains no .parquet files: %s (contents: %s)",
                self.tushare_dir, sorted(f.name for f in self.tushare_dir.iterdir()))
        yield from parquet_files

    def check_tushare_files_exist(self) -> dict:
        """检查 tushare 9 个核心文件 + 补充数据文件是否存在"""
        required = [
            "daily_basic.parquet", "adj_factor.parquet",
            "income.parquet", "balancesheet.parquet", "cashflow.parquet",
            "fina_indicator.parquet", "index_daily.parquet",
            "index_weight.parquet", "namechange.parquet",
        ]
        existing = {f.name for f in self._iter_tushare_files()}
        missing = [f for f in required if f not in existing]
        extra = sorted(existing - set(required))

        if not missing:
            status = STATUS_PASS
            desc = f"9/9 核心文件完整"
        else:
            status = STATUS_FAIL
            desc = f"缺少 {len(missing)} 个核心文件: {missing}"

        self._add_result("P1", "Tushare 源文件完整性 (9个核心文件)",
                         status, desc,
                         {"required": required, "missing": missing,
                          "existing": sorted(existing), "extra": extra,
                          "tushare_dir": str(self.tushare_dir)})
        return self._results[-1]

    def check_daily_basic_coverage(self) -> dict:
        """检查 daily_basic 每天覆盖的股票数"""
        db_path = self.tushare_dir / "daily_basic.parquet"
        if not db_path.exists():
            self._add_result("P1", "Daily_basic 股票/日期覆盖",
                             STATUS_SKIP, "daily_basic.parquet 不存在")
            return self._results[-1]

        try:
            df = pd.read_parquet(db_path, columns=["trade_date", "ts_code"])
        except Exception as e:
            self._add_result("P1", "Daily_basic 股票/日期覆盖",
                             STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        daily_counts = df.groupby("trade_date")["ts_code"].nunique()
        total_stocks = df["ts_code"].nunique()

        low_days = daily_counts[daily_counts < total_stocks * 0.5]
        min_count = int(daily_counts.min()) if len(daily_counts) > 0 else 0
        max_count = int(daily_counts.max()) if len(daily_counts) > 0 else 0
        avg_count = int(daily_counts.mean()) if len(daily_counts) > 0 else 0

        status = STATUS_PASS if len(low_days) == 0 else STATUS_WARN
        desc = (f"{total_stocks} 只股票, {len(daily_counts)} 个交易日, "
                f"日均 {avg_count} 只, 范围 {min_count}~{max_count}")
        if len(low_days) > 0:
            desc += f", {len(low_days)} 天覆盖不足50%"

        self._add_result("P1", "Daily_basic 股票/日期覆盖",
                         status, desc,
                         {"total_stocks": int(total_stocks),
                          "trading_days": len(daily_counts),
                          "avg_daily_stocks": avg_count,
                          "min_daily": min_count,
                          "max_daily": max_count,
                          "low_coverage_days": int(len(low_days))})
        return self._results[-1]

    def check_adj_factor_coverage(self) -> dict:
        """检查 adj_factor 每只股票的覆盖情况"""
        af_path = self.tushare_dir / "adj_factor.parquet"
        if not af_path.exists():
            self._add_result("P1", "Adj_factor 覆盖率",
                             STATUS_SKIP, "adj_factor.parquet 不存在")
            return self._results[-1]

        try:
            df = pd.read_parquet(af_path, columns=["ts_code", "trade_date"])
        except Exception as e:
            self._add_result("P1", "Adj_factor 覆盖率",
                             STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        total_stocks = df["ts_code"].nunique()
        stock_date_counts = df.groupby("ts_code")["trade_date"].nunique()
        low_coverage = stock_date_counts[stock_date_counts < 10]
        missing_dates_stocks = int((stock_date_counts == 0).sum())

        status = STATUS_PASS if len(low_coverage) == 0 else STATUS_WARN
        desc = (f"{total_stocks} 只股票有复权因子, "
                f"日期数: min={int(stock_date_counts.min())}, "
                f"median={int(stock_date_counts.median())}, "
                f"max={int(stock_date_counts.max())}")

        self._add_result("P1", "Adj_factor 覆盖率 (每只股票复权因子日期数)",
                         status, desc,
                         {"total_stocks": int(total_stocks),
                          "min_dates": int(stock_date_counts.min()),
                          "median_dates": int(stock_date_counts.median()),
                          "max_dates": int(stock_date_counts.max()),
                          "low_coverage_stocks": int(len(low_coverage)),
                          "missing_dates_stocks": missing_dates_stocks})
        return self._results[-1]

    def check_financial_data_completeness(self) -> dict:
        """检查财务数据每只股票的报告覆盖"""
        fina_files = ["income.parquet", "balancesheet.parquet",
                      "cashflow.parquet", "fina_indicator.parquet"]
        results = {}
        total_stocks_set = set()

        for fname in fina_files:
            fpath = self.tushare_dir / fname
            if not fpath.exists():
                results[fname] = {"status": "missing", "stocks": 0, "reports": 0}
                continue
            try:
                df = pd.read_parquet(fpath, columns=["ts_code", "end_date"])
                stocks = df["ts_code"].nunique()
                reports = len(df)
                results[fname] = {"status": "ok", "stocks": int(stocks),
                                  "reports": int(reports)}
                total_stocks_set.add(stocks)
            except Exception as e:
                results[fname] = {"status": "error", "error": str(e)}

        missing = [k for k, v in results.items() if v["status"] != "ok"]
        if missing:
            status = STATUS_FAIL
            desc = f"缺少: {missing}"
        else:
            status = STATUS_PASS
            parts = [f"{k}: {v['stocks']}只/{v['reports']}条"
                     for k, v in results.items()]
            desc = " | ".join(parts)

        self._add_result("P1", "财务数据完整性 (每只股票报告覆盖)",
                         status, desc,
                         {"files": results})
        return self._results[-1]

    def check_single_stock_ohlcv(self) -> dict:
        """抽样检查单只股票的 OHLCV 数据完整性"""
        sample_size = 10
        all_raw = list(self._iter_raw_files())
        if not all_raw:
            self._add_result("P1", "单只股票 OHLCV 完整性",
                             STATUS_SKIP, "无 raw_data 文件")
            return self._results[-1]

        np.random.seed(123)
        sample = list(np.random.choice(
            all_raw, size=min(sample_size, len(all_raw)), replace=False))

        results = []
        fields = ["open", "high", "low", "close", "volume", "amount"]
        total_issues = 0

        for raw_path in sample:
            try:
                df = pd.read_parquet(raw_path)
            except Exception:
                results.append({"code": raw_path.stem, "error": "读取失败"})
                total_issues += 1
                continue

            info = {"code": raw_path.stem, "rows": len(df)}
            issues = []

            # 检查是否有必需字段
            for fld in fields:
                if fld not in df.columns:
                    issues.append(f"缺少字段: {fld}")

            if df.empty:
                issues.append("数据为空")
            else:
                # 检查空值率
                for fld in fields:
                    if fld in df.columns:
                        null_rate = df[fld].isnull().mean()
                        if null_rate > MISSING_RATE_WARN:
                            issues.append(f"{fld} 空值率: {null_rate:.1%}")
                        info[f"{fld}_null_rate"] = round(null_rate, 4)

                # 检查日期
                if "date" in df.columns:
                    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                    if not dates.empty:
                        info["date_range"] = f"{dates.min().date()} ~ {dates.max().date()}"
                        info["date_count"] = len(dates)

            if issues:
                info["issues"] = issues
                total_issues += 1
            results.append(info)

        status = STATUS_PASS if total_issues == 0 else STATUS_WARN
        self._add_result("P1", f"单只股票 OHLCV 完整性 (抽样 {len(sample)} 只)",
                         status,
                         f"{total_issues}/{len(sample)} 只有问题",
                         {"sample_size": len(sample),
                          "stocks_with_issues": total_issues,
                          "details": results})
        return self._results[-1]

    # ── P1: 前复权价格检查 ──────────────────────────────────────────────

    def check_adj_ratio_reasonableness(self) -> dict:
        """检查 adj_ratio 在合理范围 0.01~100 内"""
        adj_path = self.tushare_dir / "adj_factor.parquet"
        if not adj_path.exists():
            self._add_result("P1", "Adj_ratio 合理性 (0.01~100)",
                             STATUS_SKIP, "adj_factor.parquet 不存在")
            return self._results[-1]

        try:
            df = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
        except Exception as e:
            self._add_result("P1", "Adj_ratio 合理性 (0.01~100)",
                             STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        # 计算最新复权因子
        latest_adj = df.groupby("ts_code")["adj_factor"].last()
        df["adj_ratio"] = df["adj_factor"] / df["ts_code"].map(latest_adj)
        df["adj_ratio"] = df["adj_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

        too_low = int((df["adj_ratio"] < 0.001).sum())
        too_high = int((df["adj_ratio"] > 100).sum())
        total_bad = too_low + too_high

        # 极端复权场景（大量拆股）可能出现极低 adj_ratio，降级为 WARN
        status = STATUS_PASS if total_bad == 0 else STATUS_WARN
        desc = (f"adj_ratio 范围 [{df['adj_ratio'].min():.4f}, {df['adj_ratio'].max():.4f}], "
                f"<0.001: {too_low}, >100: {too_high}")

        self._add_result("P1", "Adj_ratio 合理性 (0.001~100)",
                         status, desc,
                         {"adj_ratio_min": float(df["adj_ratio"].min()),
                          "adj_ratio_max": float(df["adj_ratio"].max()),
                          "too_low_count": too_low,
                          "too_high_count": too_high,
                          "total_records": len(df)})
        return self._results[-1]

    def check_forward_adjustment_consistency(self) -> dict:
        """检查 close_bin ≈ raw_close × adj_ratio 复权一致性"""
        adj_path = self.tushare_dir / "adj_factor.parquet"
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "复权一致性 (close_bin ≈ raw_close × adj_ratio)",
                             STATUS_SKIP, "日历为空")
            return self._results[-1]

        if not adj_path.exists():
            self._add_result("P1", "复权一致性 (close_bin ≈ raw_close × adj_ratio)",
                             STATUS_SKIP, "adj_factor.parquet 不存在")
            return self._results[-1]

        try:
            adj_df = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
        except Exception as e:
            self._add_result("P1", "复权一致性 (close_bin ≈ raw_close × adj_ratio)",
                             STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        # 构建 adj_ratio map
        adj_df["date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d").dt.normalize()
        adj_df["instrument"] = adj_df["ts_code"].apply(
            lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
        )
        latest_adj = adj_df.groupby("instrument")["adj_factor"].last()
        adj_df["adj_ratio"] = adj_df["adj_factor"] / adj_df["instrument"].map(latest_adj)
        adj_map = {}
        for inst, grp in adj_df.groupby("instrument"):
            adj_map[inst] = dict(zip(grp["date"], grp["adj_ratio"]))

        date_to_idx = {d: i for i, d in enumerate(cal)}
        sample_size = min(50, len(adj_map))
        np.random.seed(456)
        sample_insts = list(np.random.choice(
            sorted(adj_map.keys()), size=sample_size, replace=False))

        total_mismatch = 0
        mismatch_detail = []
        checked = 0

        for inst in sample_insts:
            inst_adj = adj_map[inst]
            bin_path = self.features_dir / inst / "close.day.bin"
            raw_path = self.raw_data_dir / f"{inst}.parquet"

            if not bin_path.exists() or not raw_path.exists():
                continue

            try:
                raw_df = pd.read_parquet(raw_path, columns=["date", "close"])
                raw_bin = np.fromfile(bin_path, dtype="<f4")
            except Exception:
                continue

            if len(raw_bin) < 2:
                continue

            bin_start = int(raw_bin[0])
            bin_values = raw_bin[1:]

            raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()
            inst_mismatch = 0
            for _, row in raw_df.iterrows():
                cal_idx = date_to_idx.get(row["date"])
                if cal_idx is None:
                    continue
                bin_idx = cal_idx - bin_start
                if bin_idx < 0 or bin_idx >= len(bin_values):
                    continue

                bin_close = bin_values[bin_idx]
                raw_close = row["close"]
                adj_ratio = inst_adj.get(row["date"])
                if (pd.isna(bin_close) or pd.isna(raw_close)
                        or adj_ratio is None or raw_close == 0):
                    continue

                expected = raw_close * adj_ratio
                rel_diff = abs(bin_close - expected) / expected
                if rel_diff > 0.05:  # 5% tolerance
                    inst_mismatch += 1

            checked += 1
            if inst_mismatch > 0:
                total_mismatch += inst_mismatch
                mismatch_detail.append({
                    "code": inst,
                    "mismatch_count": inst_mismatch,
                })

        status = STATUS_PASS if total_mismatch == 0 else STATUS_WARN
        self._add_result("P1", "复权一致性 (close_bin ≈ raw_close × adj_ratio)",
                         status,
                         f"抽样 {checked} 只, {total_mismatch} 条偏差 >5%",
                         {"sample_size": checked,
                          "total_mismatches": total_mismatch,
                          "mismatch_detail": mismatch_detail[:30]})
        return self._results[-1]

    def check_adjusted_ohlc_constraints(self) -> dict:
        """检查前复权后 bin 中的 OHLC 逻辑约束"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P1", "复权后 OHLC 约束 (High>=Low, etc.)",
                             STATUS_SKIP, "日历为空")
            return self._results[-1]

        feat_dirs = sorted(
            [d for d in self.features_dir.iterdir() if d.is_dir()]
        ) if self.features_dir.exists() else []
        np.random.seed(789)
        sample = list(np.random.choice(
            feat_dirs, size=min(50, len(feat_dirs)), replace=False))

        total_violations = 0
        viol_stocks = 0
        checked = 0

        for inst_dir in sample:
            reads = {}
            for fld in ["open", "high", "low", "close"]:
                bin_path = inst_dir / f"{fld}.day.bin"
                if not bin_path.exists():
                    break
                raw = np.fromfile(bin_path, dtype="<f4")
                if len(raw) < 2:
                    break
                reads[fld] = raw[1:]

            if len(reads) != 4:
                continue

            checked += 1
            o, h, l, c = reads["open"], reads["high"], reads["low"], reads["close"]
            n = min(len(o), len(h), len(l), len(c))
            if n == 0:
                continue
            o, h, l, c = o[:n], h[:n], l[:n], c[:n]

            mask = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
            if not mask.any():
                continue

            vv = np.sum(
                (l[mask] > h[mask]) | (o[mask] > h[mask]) |
                (o[mask] < l[mask]) | (c[mask] > h[mask]) | (c[mask] < l[mask])
            )
            if vv > 0:
                viol_stocks += 1
                total_violations += int(vv)

        status = STATUS_PASS if total_violations == 0 else STATUS_FAIL
        self._add_result("P1", "复权后 OHLC 约束 (bin 中 High>=Low, etc.)",
                         status,
                         f"抽样 {checked} 只, {viol_stocks} 只有违反, {total_violations} 条",
                         {"sample_checked": checked,
                          "stocks_with_violations": viol_stocks,
                          "total_violations": total_violations})
        return self._results[-1]

    # ── P2: 数据健康 ────────────────────────────────────────────────────

    def check_orphan_files(self) -> dict:
        """三方一致性: raw_data / features / instruments"""
        raw_set = self._raw_code_set()
        feat_set = self._feature_code_set()
        inst_set = set(self._load_instruments().keys())

        feat_not_in_raw = feat_set - raw_set
        raw_not_in_feat = raw_set - feat_set
        inst_not_in_raw = inst_set - raw_set
        raw_not_in_inst = raw_set - inst_set
        feat_not_in_inst = feat_set - inst_set
        inst_not_in_feat = inst_set - feat_set

        issues = []
        if feat_not_in_raw:
            issues.append(f"features 不在 raw_data: {sorted(feat_not_in_raw)}")
        if raw_not_in_feat:
            issues.append(f"raw_data 不在 features: {sorted(raw_not_in_feat)[:20]}")
        if inst_not_in_raw:
            issues.append(f"instruments 不在 raw_data: {sorted(inst_not_in_raw)[:20]}")
        if raw_not_in_inst:
            issues.append(f"raw_data 不在 instruments: {sorted(raw_not_in_inst)[:20]}")

        if not issues:
            status = STATUS_PASS
            desc = "raw_data / features / instruments 三方一致"
        else:
            status = STATUS_WARN
            desc = "; ".join(issues)

        self._add_result("P2", "文件孤儿检测 (raw_data/features/instruments 三方一致)",
                         status, desc,
                         {"feat_not_in_raw": sorted(feat_not_in_raw),
                          "raw_not_in_feat": sorted(raw_not_in_feat),
                          "inst_not_in_raw": sorted(inst_not_in_raw),
                          "raw_not_in_inst": sorted(raw_not_in_inst),
                          "feat_not_in_inst": sorted(feat_not_in_inst),
                          "inst_not_in_feat": sorted(inst_not_in_feat)})
        return self._results[-1]

    def check_data_freshness(self) -> dict:
        """数据新鲜度：最近交易日距今天数"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P2", "数据新鲜度", STATUS_SKIP, "日历为空")
            return self._results[-1]

        last_cal_date = cal[-1]
        today = pd.Timestamp.now().normalize()
        days_since = (today - last_cal_date).days

        # Also check raw_data most recent date
        raw_last_dates = []
        for raw_path in self._iter_raw_files():
            try:
                df = pd.read_parquet(raw_path, columns=["date"])
                if not df.empty:
                    raw_last_dates.append(df["date"].max())
            except Exception:
                continue

        raw_last = (max(raw_last_dates) if raw_last_dates
                    else pd.Timestamp("2000-01-01"))
        raw_days_since = (today - raw_last.normalize()).days

        if days_since <= DATA_FRESHNESS_WARN_DAYS and raw_days_since <= DATA_FRESHNESS_WARN_DAYS:
            status = STATUS_PASS
        elif raw_days_since <= DATA_FRESHNESS_WARN_DAYS:
            status = STATUS_WARN
        else:
            status = STATUS_WARN

        self._add_result("P2", "数据新鲜度",
                         status,
                         f"日历最近: {last_cal_date.date()} ({days_since} 天前), "
                         f"raw_data 最近: {raw_last.date()} ({raw_days_since} 天前)",
                         {"last_calendar_date": str(last_cal_date.date()),
                          "days_since_calendar": days_since,
                          "last_raw_date": str(raw_last.date()),
                          "days_since_raw": raw_days_since,
                          "warn_threshold_days": DATA_FRESHNESS_WARN_DAYS})
        return self._results[-1]

    def check_calendar_continuity(self) -> dict:
        """日历连续性检查"""
        cal = self._load_calendar()
        if len(cal) == 0:
            self._add_result("P2", "日历连续性", STATUS_SKIP, "日历为空")
            return self._results[-1]

        # 重复日期
        duplicates = cal[cal.duplicated()]
        has_dups = len(duplicates) > 0

        # 检查是否是交易日（周一至周五）
        weekdays = cal.dayofweek
        weekend_dates = cal[weekdays >= 5]  # Saturday=5, Sunday=6
        has_weekends = len(weekend_dates) > 0

        # 检查是否有日期倒退
        is_monotonic = cal.is_monotonic_increasing

        issues = []
        if has_dups:
            issues.append(f"{len(duplicates)} 个重复日期")
        if has_weekends:
            issues.append(f"{len(weekend_dates)} 个周末日期")
        if not is_monotonic:
            issues.append("日期顺序不正确")

        if not issues:
            status = STATUS_PASS
            desc = f"{len(cal)} 个交易日, 日历正常"
        else:
            status = STATUS_FAIL
            desc = f"{len(cal)} 个交易日, " + "; ".join(issues)

        self._add_result("P2", "日历连续性 (重复/非交易日/排序)",
                         status, desc,
                         {"total_days": len(cal),
                          "duplicate_count": len(duplicates),
                          "duplicates": [str(d.date()) for d in duplicates[:20]],
                          "weekend_count": len(weekend_dates),
                          "weekend_dates": [str(d.date()) for d in weekend_dates[:20]],
                          "is_monotonic": bool(is_monotonic)})
        return self._results[-1]

    def check_factor_extremes(self) -> dict:
        """Factor 极端值检查 (PE/PB 负数或超大值)"""
        if not self.factor_file.exists():
            self._add_result("P2", "Factor 极端值", STATUS_SKIP, "factor_data.parquet 不存在")
            return self._results[-1]

        try:
            df = pd.read_parquet(self.factor_file, columns=["pe", "pe_ttm", "pb", "ps", "ps_ttm"])
        except Exception as e:
            self._add_result("P2", "Factor 极端值", STATUS_SKIP, f"读取失败: {e}")
            return self._results[-1]

        extremes = {}
        for col in ["pe", "pe_ttm", "pb", "ps", "ps_ttm"]:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.empty:
                extremes[col] = {"min": None, "max": None, "negative": 0, "gt_10000": 0}
                continue
            extremes[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "negative": int((s < 0).sum()),
                "gt_10000": int((s > PE_EXTREME).sum()),
            }

        has_issues = False
        for col, stats in extremes.items():
            if stats["negative"] > 0 or stats["gt_10000"] > 0:
                has_issues = True
                break

        status = STATUS_WARN if has_issues else STATUS_PASS
        desc_parts = []
        for col, stats in extremes.items():
            desc_parts.append(
                f"{col}: min={stats['min']}, max={stats['max']}, "
                f"neg={stats['negative']}, >{PE_EXTREME}={stats['gt_10000']}"
            )

        self._add_result("P2", "Factor 极端值 (PE/PB 负数/超大)",
                         status, " | ".join(desc_parts),
                         {"threshold": PE_EXTREME, "extremes": extremes})
        return self._results[-1]

    def check_instruments_duplicates(self) -> dict:
        """检查 instruments 重复"""
        instruments = self._load_instruments()
        codes = list(instruments.keys())
        dup_counts = {k: v for k, v in Counter(codes).items() if v > 1}

        if not dup_counts:
            status = STATUS_PASS
            desc = f"{len(codes)} 条, 无重复"
        else:
            status = STATUS_FAIL
            desc = f"{len(codes)} 条, {len(dup_counts)} 个重复"

        self._add_result("P2", "Instruments 重复检查",
                         status, desc,
                         {"total": len(codes), "duplicates": dup_counts})
        return self._results[-1]

    def check_delisted_stocks(self) -> dict:
        """检查退市股票：raw_data 数据结束日期远早于日历结束日期"""
        cal = self._load_calendar()
        instruments = self._load_instruments()
        if len(cal) == 0:
            self._add_result("P2", "退市/ST 标记检查", STATUS_SKIP, "日历为空")
            return self._results[-1]

        cal_end = cal[-1]
        potentially_delisted = []

        for code, (inst_start, inst_end) in instruments.items():
            raw_path = self.raw_data_dir / f"{code}.parquet"
            if not raw_path.exists():
                continue
            try:
                df = pd.read_parquet(raw_path, columns=["date"])
            except Exception:
                continue
            if df.empty:
                continue

            actual_end = df["date"].max()
            if actual_end < cal_end - pd.Timedelta(days=365):
                # 实际数据结束日期比日历最后日期早 1 年以上
                inst_end_date = pd.to_datetime(inst_end)
                if inst_end_date >= cal_end - pd.Timedelta(days=30):
                    # instruments 写的结束日期接近日历结尾，但实际数据早已结束
                    potentially_delisted.append({
                        "code": code,
                        "actual_end": str(actual_end.date()),
                        "inst_end": inst_end,
                        "days_behind": int((cal_end - actual_end).days),
                    })

        if not potentially_delisted:
            status = STATUS_PASS
            desc = "未发现疑似退市但 instruments 未标记的股票"
        else:
            status = STATUS_WARN
            desc = f"{len(potentially_delisted)} 只疑似退市但 instruments 结束日期未标记"

        self._add_result("P2", "退市/ST 标记检查 (数据提前终止但 instruments 未标记)",
                         status, desc,
                         {"potentially_delisted": potentially_delisted[:50]})
        return self._results[-1]

    # ── orchestration ───────────────────────────────────────────────────

    def run_all(self) -> list:
        """运行全部检查，返回结果列表"""
        t0 = time.time()

        # P0 - 致命问题
        self.check_ohlc_integrity()
        self.check_price_nonnegative()
        self.check_instruments_dates()
        self.check_bin_file_integrity()

        # P1 - 重要问题
        self.check_coverage_rate()
        self.check_price_jumps()
        self.check_limit_up_down()
        self.check_zero_volume_flat()
        self.check_factor_null_rates()
        self.check_raw_vs_feature_consistency()
        # P1 - Tushare 源数据检查
        self.check_tushare_files_exist()
        self.check_daily_basic_coverage()
        self.check_adj_factor_coverage()
        self.check_financial_data_completeness()
        self.check_single_stock_ohlcv()
        # P1 - 前复权价格检查
        self.check_adj_ratio_reasonableness()
        self.check_forward_adjustment_consistency()
        self.check_adjusted_ohlc_constraints()

        # P2 - 数据健康
        self.check_orphan_files()
        self.check_data_freshness()
        self.check_calendar_continuity()
        self.check_factor_extremes()
        self.check_instruments_duplicates()
        self.check_delisted_stocks()

        elapsed = time.time() - t0
        print(f"\n{C_CYAN}╔══════════════════════════════════════════════════════════╗{C_RESET}")
        print(f"{C_CYAN}║{C_RESET}  耗时: {elapsed:.1f}s")
        print(f"{C_CYAN}╚══════════════════════════════════════════════════════════╝{C_RESET}")

        return self._results

    def print_report(self):
        """打印彩色终端报告"""
        if not self._results:
            print("无检查结果。")
            return

        print(f"\n{C_BOLD}{'='*70}{C_RESET}")
        print(f"{C_BOLD}  金融量化数据验证报告{C_RESET}")
        print(f"{C_BOLD}  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C_RESET}")
        print(f"{C_BOLD}{'='*70}{C_RESET}")

        for priority in ["P0", "P1", "P2"]:
            items = [r for r in self._results if r["priority"] == priority]
            if not items:
                continue

            p_color = {STATUS_FAIL: C_RED, STATUS_WARN: C_YELLOW,
                        STATUS_PASS: C_GREEN, STATUS_SKIP: C_CYAN}
            p_label = {"P0": "致命问题 (回测结果错误)",
                        "P1": "重要问题 (影响回测质量)",
                        "P2": "数据健康指标"}

            print(f"\n{C_BOLD}── {priority}: {p_label[priority]} ──{C_RESET}")

            # Sort: FAIL first, then WARN, then PASS, then SKIP
            items.sort(key=lambda x: STATUS_ORDER.get(x["status"], 99))

            for item in items:
                icon = STATUS_ICONS.get(item["status"], item["status"])
                print(f"  {icon}  {item['check']}")
                if item["detail"]:
                    # Indent detail under the check name
                    color = p_color.get(item["status"], C_RESET)
                    print(f"       {color}{item['detail']}{C_RESET}")

        # Summary
        total = len(self._results)
        fails = self._counts[STATUS_FAIL]
        warns = self._counts[STATUS_WARN]
        passes = self._counts[STATUS_PASS]
        skips = self._counts[STATUS_SKIP]

        print(f"\n{C_BOLD}{'='*70}{C_RESET}")
        print(f"{C_BOLD}  汇总: {total} 项检查{C_RESET}")
        print(f"    {STATUS_ICONS[STATUS_FAIL]}: {fails}")
        print(f"    {STATUS_ICONS[STATUS_WARN]}: {warns}")
        print(f"    {STATUS_ICONS[STATUS_PASS]}: {passes}")
        print(f"    {STATUS_ICONS[STATUS_SKIP]}: {skips}")

        # Overall verdict
        if fails > 0:
            verdict = f"{C_RED}数据存在致命问题，回测结果可能不可靠！{C_RESET}"
        elif warns > 0:
            verdict = f"{C_YELLOW}数据存在警告项，建议修复后再回测。{C_RESET}"
        else:
            verdict = f"{C_GREEN}数据验证全部通过！{C_RESET}"

        print(f"\n  {C_BOLD}结论: {verdict}")
        print(f"{C_BOLD}{'='*70}{C_RESET}\n")

    def save_json(self, output_path: str = None):
        """保存 JSON 报告"""
        if output_path is None:
            results_dir = Path(__file__).resolve().parent.parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"data_validation_{timestamp}.json"
        else:
            output_path = Path(output_path)

        report = {
            "title": "金融量化数据验证报告",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_checks": len(self._results),
                "PASS": self._counts[STATUS_PASS],
                "FAIL": self._counts[STATUS_FAIL],
                "WARN": self._counts[STATUS_WARN],
                "SKIP": self._counts[STATUS_SKIP],
            },
            "checks": self._results,
        }

        # Convert non-serializable types
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (set,)):
                return list(obj)
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=make_serializable)

        print(f"\n{C_CYAN}JSON 报告已保存到: {output_path}{C_RESET}")
        return str(output_path)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="金融量化数据验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 scripts/validate_data.py
  python3 scripts/validate_data.py --data-root /path/to/data/qlib_data
  python3 scripts/validate_data.py --output /tmp/my_report.json
        """,
    )
    parser.add_argument("--data-root", default=None,
                        help="数据根目录 (默认: 项目 data/qlib_data)")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON 报告输出路径 (默认: results/data_validation_<timestamp>.json)")
    args = parser.parse_args()

    validator = DataValidator(data_root=args.data_root)
    validator.run_all()
    validator.print_report()
    validator.save_json(output_path=args.output)


if __name__ == "__main__":
    main()
