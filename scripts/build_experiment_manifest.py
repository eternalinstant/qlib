#!/usr/bin/env python3
"""
构建实验 manifest：汇总训练、评分、回测元数据到统一 JSON。

用法:
    python3 scripts/build_experiment_manifest.py --config strategy/configs/models/qvf_alpha158_core12.yaml
    python3 scripts/build_experiment_manifest.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _file_hash(path: Path, max_bytes: int = 10 << 20) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
            if f.tell() > max_bytes:
                break
    return h.hexdigest()[:16]


def build_manifest(config_path: Path) -> dict | None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from strategy.producers.predictive_signal import (
        backtest_summary_path,
        load_predictive_config,
        output_root,
        scores_path,
        selection_path,
        scoring_summary_path,
        training_summary_path,
        overlay_results_path,
        model_bundle_path,
    )

    cfg = load_predictive_config(str(config_path))
    name = cfg["name"]
    root = output_root(cfg)

    # 读取 YAML 原文计算 hash
    yaml_content = config_path.read_text()
    yaml_hash = hashlib.sha256(yaml_content.encode()).hexdigest()[:16]

    manifest = {
        "experiment_name": name,
        "config_path": str(config_path),
        "config_yaml_hash": yaml_hash,
        "frozen": bool(cfg.get("frozen", False)),
        "frozen_date": cfg.get("frozen_date", None),
        "created_at": datetime.now().isoformat(),
        "data": {
            "source": cfg["data"].get("source", "unknown"),
            "parquet_feature_columns": cfg["data"].get("parquet_feature_columns", []),
            "alpha158_feature_columns": cfg["data"].get("alpha158_feature_columns", []),
            "start_date": cfg["data"].get("start_date", ""),
            "end_date": cfg["data"].get("end_date", ""),
        },
        "training": {
            "train_start": cfg["training"].get("train_start", ""),
            "train_end": cfg["training"].get("train_end", ""),
            "valid_start": cfg["training"].get("valid_start", ""),
            "valid_end": cfg["training"].get("valid_end", ""),
        },
        "model": {
            "backend": cfg.get("model", {}).get("preferred_backend", "unknown"),
            "params": cfg.get("model", {}).get("params", {}),
        },
        "files": {},
    }

    # 计算 feature_count
    pq_feats = manifest["data"]["parquet_feature_columns"]
    a158_feats = manifest["data"]["alpha158_feature_columns"]
    manifest["data"]["feature_count"] = len(pq_feats) + len(a158_feats)

    # 文件状态
    files_to_check = {
        "model_bundle": model_bundle_path(cfg),
        "scores": scores_path(cfg),
        "selections": selection_path(cfg),
        "training_summary": training_summary_path(cfg),
        "scoring_summary": scoring_summary_path(cfg),
        "backtest_summary": backtest_summary_path(cfg),
        "overlay_results": overlay_results_path(cfg),
    }
    for label, fpath in files_to_check.items():
        manifest["files"][label] = {
            "path": str(fpath),
            "exists": fpath.exists(),
            "hash": _file_hash(fpath) if fpath.exists() else None,
            "size_bytes": fpath.stat().st_size if fpath.exists() else None,
            "mtime": datetime.fromtimestamp(fpath.stat().st_mtime).isoformat() if fpath.exists() else None,
        }

    # 合并 training_summary.json
    ts_path = training_summary_path(cfg)
    if ts_path.exists():
        with open(ts_path) as f:
            ts = json.load(f)
        manifest["training"]["actual_feature_count"] = ts.get("feature_count")
        manifest["training"]["actual_feature_columns"] = ts.get("feature_columns", [])
        manifest["training"]["metrics"] = ts.get("metrics", {})
        # 验证因子一致性
        if manifest["training"]["actual_feature_columns"]:
            expected = sorted(pq_feats + a158_feats)
            actual = sorted(manifest["training"]["actual_feature_columns"])
            manifest["training"]["features_match"] = expected == actual
            if expected != actual:
                manifest["training"]["missing_features"] = sorted(set(expected) - set(actual))
                manifest["training"]["extra_features"] = sorted(set(actual) - set(expected))

    # 合并 backtest_summary.json
    bs_path = backtest_summary_path(cfg)
    if bs_path.exists():
        with open(bs_path) as f:
            bs = json.load(f)
        manifest["backtest"] = {
            "annual_return": bs.get("annual_return"),
            "max_drawdown": bs.get("max_drawdown"),
            "sharpe_ratio": bs.get("sharpe_ratio"),
            "avg_exposure": bs.get("avg_exposure"),
            "results_file": bs.get("results_file"),
            "strategy_name": bs.get("strategy_name"),
        }

    # 输出
    out_path = root / "experiment_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="构建实验 manifest")
    parser.add_argument("--config", type=str, help="YAML 配置路径")
    parser.add_argument("--all", action="store_true", help="扫描 strategy/configs/models/ 下所有 YAML")
    args = parser.parse_args()

    if args.all:
        configs = sorted((PROJECT_ROOT / "src" / "strategy" / "configs" / "models").glob("*.yaml"))
    elif args.config:
        configs = [Path(args.config)]
    else:
        parser.error("请指定 --config 或 --all")

    for cfg_path in configs:
        print(f"\n{'=' * 50}")
        print(f"构建 manifest: {cfg_path.name}")
        manifest = build_manifest(cfg_path)
        if manifest is None:
            print("  SKIP: 无法加载配置")
            continue

        print(f"  实验名称: {manifest['experiment_name']}")
        print(f"  配置哈希: {manifest['config_yaml_hash']}")
        print(f"  冻结: {manifest['frozen']}")
        print(f"  因子数: {manifest['data']['feature_count']}")

        # 因子一致性检查
        features_match = manifest.get("training", {}).get("features_match")
        if features_match is not None:
            status = "OK" if features_match else "MISMATCH"
            print(f"  因子一致性: {status}")
            if not features_match:
                missing = manifest["training"].get("missing_features", [])
                extra = manifest["training"].get("extra_features", [])
                if missing:
                    print(f"    缺失因子: {missing}")
                if extra:
                    print(f"    多余因子: {extra}")

        # 回测结果
        bt = manifest.get("backtest")
        if bt:
            print(f"  回测: 年化={bt.get('annual_return', 'N/A')}, "
                  f"夏普={bt.get('sharpe_ratio', 'N/A')}, "
                  f"回撤={bt.get('max_drawdown', 'N/A')}")

        # 文件完整性
        missing_files = [k for k, v in manifest.get("files", {}).items() if not v["exists"]]
        if missing_files:
            print(f"  缺失文件: {missing_files}")

        out_path = Path(manifest["files"].get("training_summary", {}).get("path", "")).parent / "experiment_manifest.json"
        print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
