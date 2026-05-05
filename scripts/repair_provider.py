#!/usr/bin/env python3
"""Safely repair the local Qlib provider.

This script preserves the existing close-series continuity instead of
rebuilding close.day.bin directly from raw quotes.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data.tushare_to_qlib import TushareToQlibConverter


def main() -> None:
    converter = TushareToQlibConverter(
        tushare_dir=str(PROJECT_ROOT / "data" / "tushare"),
        qlib_dir=str(PROJECT_ROOT / "data" / "qlib_data" / "cn_data"),
    )

    close_updated = converter.update_close_bins()
    ohlcv_updated = converter.update_ohlcv_bins()
    repair_stats = converter.repair_price_provider()

    print("[OK] safe provider repair finished")
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
