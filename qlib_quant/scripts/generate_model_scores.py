#!/usr/bin/env python3
"""基于已训练模型生成分数与选股结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import load_predictive_config, score_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/models/lgb_10d.yaml",
        help="模型配置 YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_predictive_config(args.config)
    summary = score_from_config(cfg)
    print(f"[OK] 分数已生成: {summary['score_path']}")
    print(
        "[INFO] "
        f"score_rows={summary['score_rows']} "
        f"selection_dates={summary['selection_dates']} "
        f"selection_path={summary['selection_path']}"
    )


if __name__ == "__main__":
    main()

