"""目标持仓文件 — 加载、校验、代码归一化

契约 target_YYYYMMDD.json (UTF-8):
{
  "date": "20260527",
  "strategy": "alpha158_momentum_volume_k6",
  "invested_pct": 0.8,
  "positions": [
    {"symbol": "600641.SH", "weight": 0.1667},
    {"symbol": "002322.SZ", "weight": 0.1667}
  ]
}
"""

import os
import json
import time

import live_config


def normalize_symbol(sym):
    """统一成 QMT 后缀格式 600641.SH

    接受 SH600641 / 600641.SH / 600641 三种写法。
    交易所判定：6 开头=SH（含 688 科创），其余（0/3）=SZ。
    """
    s = str(sym).strip().upper()
    if "." in s:                          # 已是 600641.SH
        code, suffix = s.split(".", 1)
        return f"{code}.{suffix}"
    if s.startswith(("SH", "SZ")):        # 前缀式 SH600641
        return f"{s[2:]}.{s[:2]}"
    suffix = "SH" if s.startswith("6") else "SZ"   # 纯代码 600641
    return f"{s}.{suffix}"


def _target_path_for(date_str):
    return os.path.join(live_config.TARGET_DIR, f"target_{date_str}.json")


def load_today_target():
    """读取并校验当天目标文件，返回规整后的 dict

    返回 {date, strategy, invested_pct, positions:[{symbol, weight}]}。
    校验失败抛 ValueError；文件缺失抛 FileNotFoundError。
    """
    today = time.strftime("%Y%m%d")
    path = _target_path_for(today)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到当天目标文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 日期必须 == 今天，防止用了陈旧文件
    if str(data.get("date")) != today:
        raise ValueError(
            f"目标文件日期 {data.get('date')} 与今天 {today} 不符，拒绝执行"
        )

    positions = data.get("positions") or []
    if not positions:
        raise ValueError("目标文件 positions 为空")

    norm = []
    total_w = 0.0
    for p in positions:
        w = float(p["weight"])
        if w < 0:
            raise ValueError(f"权重为负: {p}")
        norm.append({"symbol": normalize_symbol(p["symbol"]), "weight": w})
        total_w += w

    if abs(total_w - 1.0) > 0.02:
        raise ValueError(f"权重之和 {total_w:.4f} 偏离 1.0 过大")

    invested = float(data.get("invested_pct", live_config.DEFAULT_INVESTED_PCT))
    if not (0 < invested <= 1.0):
        raise ValueError(f"invested_pct 非法: {invested}")

    return {
        "date": today,
        "strategy": data.get("strategy", "unknown"),
        "invested_pct": invested,
        "positions": norm,
    }
