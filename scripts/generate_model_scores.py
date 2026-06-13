#!/usr/bin/env python3
"""基于已训练模型生成分数与选股结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.producers.predictive_signal import load_predictive_config, score_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/strategy/configs/models/lgb_10d.yaml",
        help="模型配置 YAML",
    )
    parser.add_argument(
        "--universe", "-u",
        choices=["all", "csi300", "csi500", "csi800"],
        default=None,
        help="覆盖股票池（默认使用配置文件值）",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="附加到 cfg.output.root 末尾的后缀，避免覆盖基线打分（如 _csi800）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_predictive_config(args.config)
    if args.universe:
        cfg["selection"]["universe"] = args.universe
        if cfg.get("data", {}).get("source", "parquet") in {"alpha158", "alpha360", "hybrid"}:
            cfg["data"]["alpha158_universe"] = args.universe
    original_root = cfg["output"]["root"]
    if args.output_suffix:
        cfg["output"]["root"] = f"{original_root.rstrip('/')}{args.output_suffix}"
        print(f"[INFO] 输出目录已改写: {cfg['output']['root']}")
        new_root = Path(cfg["output"]["root"]).expanduser()
        new_root.mkdir(parents=True, exist_ok=True)
        base_bundle = Path(original_root).expanduser() / "model_bundle.pkl"
        new_bundle = new_root / "model_bundle.pkl"
        if base_bundle.exists() and not new_bundle.exists():
            import shutil
            shutil.copy2(base_bundle, new_bundle)
            print(f"[INFO] 复制 model_bundle: {base_bundle} -> {new_bundle}")
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

