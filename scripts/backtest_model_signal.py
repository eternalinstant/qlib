#!/usr/bin/env python3
"""回测预测式模型信号。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.producers.predictive_signal import backtest_from_config, load_predictive_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/strategy/configs/models/lgb_10d.yaml",
        help="模型配置 YAML",
    )
    parser.add_argument(
        "--engine",
        default="qlib",
        choices=["qlib", "pybroker"],
        help="回测引擎",
    )
    parser.add_argument(
        "--universe", "-u",
        choices=["all", "csi300", "csi500", "csi800"],
        default=None,
        help="覆盖股票池（默认使用配置文件值）",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="覆盖初始资金（元），不传则用 common/configs/trading.yaml 的默认值",
    )
    parser.add_argument(
        "--enforce-lot-size",
        action="store_true",
        help="启用最低买入手数约束（默认 1 手=100 股）",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=100,
        help="每手股数（默认 100）",
    )
    parser.add_argument(
        "--min-lots",
        type=int,
        default=1,
        help="单标的最少买入手数（默认 1）",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="附加到 cfg.output.root 末尾的后缀，避免覆盖基线结果（例如 _cap20w_lot1）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_predictive_config(args.config)
    if args.universe:
        cfg["selection"]["universe"] = args.universe
        if cfg.get("data", {}).get("source", "parquet") in {"alpha158", "alpha360", "hybrid"}:
            cfg["data"]["alpha158_universe"] = args.universe

    trading_overrides = cfg.setdefault("trading", {})
    if args.initial_capital is not None:
        trading_overrides["initial_capital"] = float(args.initial_capital)
    if args.enforce_lot_size:
        trading_overrides["enforce_lot_size"] = True
        trading_overrides["lot_size"] = int(args.lot_size)
        trading_overrides["min_lots"] = int(args.min_lots)

    if args.output_suffix:
        original_root = cfg["output"]["root"]
        cfg["output"]["root"] = f"{original_root.rstrip('/')}{args.output_suffix}"
        print(f"[INFO] 输出目录已改写: {cfg['output']['root']}")

    # selection_path 复用 output.root；但选股文件是在 frozen 基线里生成的，
    # 需要把基线 selections.csv 软链/复制过来再回测。
    from pathlib import Path as _Path
    from strategy.producers.predictive_signal import selection_path as _selection_path
    sel_path = _selection_path(cfg)
    if args.output_suffix and not sel_path.exists():
        base_sel = _Path(original_root) / "selections.csv"
        if base_sel.exists():
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(base_sel, sel_path)
            print(f"[INFO] 复制选股文件: {base_sel} -> {sel_path}")

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
