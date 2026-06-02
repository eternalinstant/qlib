"""【Mac 侧参考模板，不在本 Windows 机运行/测试】

在跑 qlib 的那台机器上，从模型预测结果取某策略「最新调仓日」的 top-K，
按目标持仓契约写出 target_YYYYMMDD.json，再同步到 Windows 的 live_config.TARGET_DIR。

需按你的 qlib 工程实际改两处：
  1. 预测分数怎么拿（下面用 pred.pkl 举例：MultiIndex [datetime, instrument] -> score）
  2. 策略的 topk / invested_pct / 调仓日历

本机无 qlib，此脚本仅为规范示例，请在 Mac 上落地。
"""

import json
import time
import pandas as pd


def qlib_to_qmt(inst):
    """qlib 代码 SH600519 / 600519.SH -> QMT 600519.SH"""
    s = str(inst).strip().upper()
    if "." in s:
        return s
    if s.startswith(("SH", "SZ")):
        return f"{s[2:]}.{s[:2]}"
    return f"{s}.{'SH' if s.startswith('6') else 'SZ'}"


def export(pred_pkl, strategy, topk=6, invested_pct=0.8, out_dir="."):
    pred = pd.read_pickle(pred_pkl)          # Series, index=[datetime, instrument]
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    last_day = pred.index.get_level_values(0).max()
    day_scores = pred.loc[last_day].sort_values(ascending=False)
    picks = day_scores.head(topk)

    weight = 1.0 / len(picks)
    positions = [{"symbol": qlib_to_qmt(inst), "weight": round(weight, 4)}
                 for inst in picks.index]

    date_str = pd.Timestamp(last_day).strftime("%Y%m%d")
    doc = {
        "date": date_str,
        "strategy": strategy,
        "invested_pct": invested_pct,
        "positions": positions,
    }
    path = f"{out_dir}/target_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[OK] {path}  ({len(positions)} 只, 调仓日 {date_str})")
    return path


if __name__ == "__main__":
    # 示例：export("/Users/sxt/code/qlib/.../pred.pkl",
    #              "alpha158_momentum_volume_k6", topk=6)
    print(__doc__)
