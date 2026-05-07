#!/usr/bin/env python3
"""回测预测式模型信号。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import backtest_from_config, load_predictive_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/models/lgb_10d.yaml",
        help="模型配置 YAML",
    )
    parser.add_argument(
        "--engine",
        default="qlib",
        choices=["qlib", "pybroker"],
        help="回测引擎",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_predictive_config(args.config)
    _, summary = backtest_from_config(cfg, engine=args.engine)
    print(
        "[OK] 回测完成: "
        f"engine={summary.get('engine', args.engine)} "
        f"annual_return={summary['annual_return']:.2%} "
        f"max_drawdown={summary['max_drawdown']:.2%} "
        f"sharpe={summary['sharpe_ratio']:.3f}"
    )
    if summary.get("results_file"):
        print(f"[INFO] 结果文件: {summary['results_file']}")


if __name__ == "__main__":
    main()
