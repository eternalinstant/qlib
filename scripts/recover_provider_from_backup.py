#!/usr/bin/env python3
"""Recover provider bins from a backup, then rebuild with safe logic."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data.tushare_to_qlib import TushareToQlibConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-29", help="mtime date to recover, YYYY-MM-DD")
    parser.add_argument(
        "--backup-dir",
        default="data/qlib_data/cn_data/features.bak.0404",
        help="backup features dir",
    )
    parser.add_argument(
        "--features-dir",
        default="data/qlib_data/cn_data/features",
        help="live features dir",
    )
    return parser.parse_args()


def _mtime_window(target_date: str) -> tuple[float, float]:
    start = datetime.strptime(target_date, "%Y-%m-%d")
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def changed_instruments(features_dir: Path, target_date: str) -> list[str]:
    start_ts, end_ts = _mtime_window(target_date)
    changed: list[str] = []
    for inst_dir in sorted(features_dir.iterdir()):
        close_path = inst_dir / "close.day.bin"
        if not close_path.exists():
            continue
        mtime = close_path.stat().st_mtime
        if start_ts <= mtime < end_ts:
            changed.append(inst_dir.name)
    return changed


def restore_from_backup(features_dir: Path, backup_dir: Path, instruments: list[str]) -> tuple[int, list[str]]:
    restored = 0
    missing: list[str] = []
    for inst in instruments:
        src_dir = backup_dir / inst
        dst_dir = features_dir / inst
        if not src_dir.exists() or not dst_dir.exists():
            missing.append(inst)
            continue
        copied_any = False
        for src_file in src_dir.glob("*.day.bin"):
            shutil.copy2(src_file, dst_dir / src_file.name)
            copied_any = True
        if copied_any:
            restored += 1
        else:
            missing.append(inst)
    return restored, missing


def main() -> None:
    args = parse_args()
    features_dir = (PROJECT_ROOT / args.features_dir).resolve()
    backup_dir = (PROJECT_ROOT / args.backup_dir).resolve()

    instruments = changed_instruments(features_dir, args.date)
    restored, missing = restore_from_backup(features_dir, backup_dir, instruments)

    converter = TushareToQlibConverter(
        tushare_dir=str(PROJECT_ROOT / "data" / "tushare"),
        qlib_dir=str(PROJECT_ROOT / "data" / "qlib_data" / "cn_data"),
    )
    close_updated = converter.update_close_bins()
    ohlcv_updated = converter.update_ohlcv_bins()
    repair_stats = converter.repair_price_provider()

    print("[OK] provider recovery finished")
    print(f"[INFO] target_date={args.date} changed_instruments={len(instruments)} restored={restored}")
    print(f"[INFO] missing_backup={len(missing)}")
    if missing:
        preview = ",".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        print(f"[INFO] missing_backup_preview={preview}{suffix}")
    print(f"[INFO] close_updated={close_updated}")
    print(
        "[INFO] ohlcv_updated="
        + ",".join(f"{field}:{count}" for field, count in sorted(ohlcv_updated.items()))
    )
    print(
        "[INFO] repair_stats="
        + ",".join(f"{key}:{value}" for key, value in sorted(repair_stats.items()))
    )


if __name__ == "__main__":
    main()
