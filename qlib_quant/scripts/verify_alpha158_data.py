#!/usr/bin/env python3
"""
Verify that the local Qlib provider can serve the official Alpha158 feature set.

This script does not reimplement Alpha158. It asks pyqlib's official
Alpha158DL for the expressions, verifies the required base bins, and loads a
small sample through qlib.data.D.features.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ALPHA158_BASE_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


def _read_calendar(provider_uri: Path) -> pd.Series:
    cal_path = provider_uri / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"missing calendar: {cal_path}")
    return pd.read_csv(cal_path, header=None, names=["date"], parse_dates=["date"])["date"]


def _read_instruments(provider_uri: Path):
    inst_path = provider_uri / "instruments" / "all.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"missing instruments file: {inst_path}")

    records = []
    with inst_path.open() as fp:
        for line in fp:
            parts = line.strip().split()
            if not parts:
                continue
            inst = parts[0]
            start = pd.to_datetime(parts[1], errors="coerce") if len(parts) > 1 else pd.NaT
            end = pd.to_datetime(parts[2], errors="coerce") if len(parts) > 2 else pd.NaT
            records.append((inst, start, end))
    return records


def _missing_base_bins(provider_uri: Path, instruments) -> list[tuple[str, list[str]]]:
    features_dir = provider_uri / "features"
    missing = []
    for inst, _, _ in instruments:
        inst_dir = features_dir / inst
        missing_fields = [
            field
            for field in ALPHA158_BASE_FIELDS
            if not (inst_dir / f"{field}.day.bin").exists()
        ]
        if missing_fields:
            missing.append((inst, missing_fields))
    return missing


def _pick_sample(instruments, start_time: pd.Timestamp, end_time: pd.Timestamp, sample_size: int):
    sample = []
    for inst, inst_start, inst_end in instruments:
        if pd.isna(inst_start) or pd.isna(inst_end):
            continue
        if inst_start <= end_time and inst_end >= start_time:
            sample.append(inst)
        if len(sample) >= sample_size:
            break
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify official pyqlib Alpha158 data readiness")
    parser.add_argument(
        "--provider-uri",
        default=str(PROJECT_ROOT / "data" / "qlib_data" / "cn_data"),
        help="Qlib provider directory",
    )
    parser.add_argument("--start", default=None, help="sample load start date, default: last 120 calendar rows")
    parser.add_argument("--end", default=None, help="sample load end date, default: latest calendar date")
    parser.add_argument("--sample-size", type=int, default=32, help="number of instruments for smoke load")
    args = parser.parse_args()

    provider_uri = Path(args.provider_uri).expanduser().resolve()
    calendar = _read_calendar(provider_uri)
    instruments = _read_instruments(provider_uri)

    missing = _missing_base_bins(provider_uri, instruments)
    if missing:
        preview = ", ".join(
            f"{inst}:{'/'.join(fields)}" for inst, fields in missing[:20]
        )
        suffix = " ..." if len(missing) > 20 else ""
        print(f"[FAIL] Alpha158 base bins missing for {len(missing)} instruments: {preview}{suffix}")
        return 1

    end_time = pd.to_datetime(args.end) if args.end else calendar.max()
    if args.start:
        start_time = pd.to_datetime(args.start)
    else:
        end_pos = int(calendar.searchsorted(end_time, side="right") - 1)
        start_pos = max(0, end_pos - 119)
        start_time = calendar.iloc[start_pos]

    sample = _pick_sample(instruments, start_time, end_time, args.sample_size)
    if not sample:
        print(f"[FAIL] no instruments overlap {start_time.date()} ~ {end_time.date()}")
        return 1

    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D

    fields, names = Alpha158DL.get_feature_config()
    if len(fields) != 158 or len(names) != 158:
        print(f"[FAIL] official Alpha158DL returned {len(fields)} fields / {len(names)} names")
        return 1

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    df = D.features(sample, fields, start_time=start_time, end_time=end_time, freq="day")
    if df.empty:
        print("[FAIL] Alpha158 sample load returned empty DataFrame")
        return 1

    non_null_ratio = float(df.notna().mean().mean())
    print(
        "[OK] Alpha158 ready: "
        f"{len(fields)} official fields, sample={len(sample)} instruments, "
        f"rows={len(df):,}, non_null_ratio={non_null_ratio:.2%}, "
        f"window={start_time.date()}~{end_time.date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
