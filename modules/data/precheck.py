"""
正式数据预检

在生成选股、回测或正式验证前，检查核心数据是否齐备。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.config import CONFIG
from core.universe import _iter_index_weight_paths, _iter_namechange_paths


CORE_TUSHARE_FILES = [
    "daily_basic.parquet",
    "income.parquet",
    "balancesheet.parquet",
    "cashflow.parquet",
    "fina_indicator.parquet",
    "index_daily.parquet",
    "stock_basic.csv",
    "stock_industry.csv",
]


@dataclass
class DataPrecheckResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resolved_paths: Dict[str, str] = field(default_factory=dict)

    def raise_if_failed(self):
        if self.ok:
            return
        lines = ["数据预检失败："]
        lines.extend(f"- {msg}" for msg in self.errors)
        if self.warnings:
            lines.append("警告：")
            lines.extend(f"- {msg}" for msg in self.warnings)
        raise FileNotFoundError("\n".join(lines))


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _check_table_columns(path: Path, required_cols: List[str]) -> Optional[str]:
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        return f"{path} 无法读取: {exc}"

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return f"{path} 缺少字段: {missing}"
    return None


def _qlib_root() -> Path:
    return Path(CONFIG.get("paths.data.qlib_data", CONFIG.get("qlib_data_path", ""))).expanduser()


def _tushare_root() -> Path:
    qlib_root = _qlib_root()
    if qlib_root.name == "cn_data":
        return qlib_root.parent.parent / "tushare"
    return Path("data/tushare").resolve()


def _check_exists(errors: List[str], resolved: Dict[str, str], key: str, path: Path):
    if path.exists():
        resolved[key] = str(path)
    else:
        errors.append(f"缺少文件: {path}")


def _backtest_period():
    start_date = pd.Timestamp(CONFIG.get("start_date", "2019-01-01"))
    end_date = pd.Timestamp(CONFIG.get("end_date", "2026-02-26"))
    return start_date, end_date


def run_data_precheck(universe: str = "all", require_st_history: bool = False) -> DataPrecheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    resolved: Dict[str, str] = {}

    qlib_root = _qlib_root()
    tushare_root = _tushare_root()
    start_date, end_date = _backtest_period()

    # Qlib provider
    _check_exists(errors, resolved, "qlib_cal", qlib_root / "calendars" / "day.txt")
    _check_exists(errors, resolved, "qlib_instruments", qlib_root / "instruments" / "all.txt")
    _check_exists(errors, resolved, "qlib_factor_data", qlib_root / "factor_data.parquet")
    cal_path = qlib_root / "calendars" / "day.txt"
    if cal_path.exists():
        try:
            cal = pd.read_csv(cal_path, header=None, names=["date"])
            cal_dates = pd.to_datetime(cal["date"], errors="coerce").dropna()
            if cal_dates.empty:
                errors.append(f"{cal_path} 交易日历为空或无法解析")
            elif cal_dates.max() < end_date - pd.Timedelta(days=7):
                errors.append(
                    f"Qlib 交易日历最新日期 {cal_dates.max().date()} 距离回测终点 {end_date.date()} 过远"
                )
        except Exception as exc:
            errors.append(f"{cal_path} 无法检查日期覆盖: {exc}")

    # Core Tushare raw data
    for filename in CORE_TUSHARE_FILES:
        _check_exists(errors, resolved, filename, tushare_root / filename)

    daily_basic_path = tushare_root / "daily_basic.parquet"
    if daily_basic_path.exists():
        try:
            daily_basic = pd.read_parquet(daily_basic_path, columns=["trade_date"])
            daily_dates = pd.to_datetime(daily_basic["trade_date"], errors="coerce").dropna()
            if daily_dates.empty:
                errors.append(f"{daily_basic_path} trade_date 全部无效")
            elif daily_dates.max() < end_date - pd.Timedelta(days=7):
                errors.append(
                    f"daily_basic 最新日期 {daily_dates.max().date()} 距离回测终点 {end_date.date()} 过远"
                )
        except Exception as exc:
            errors.append(f"{daily_basic_path} 日期检查失败: {exc}")

    index_daily_path = tushare_root / "index_daily.parquet"
    if index_daily_path.exists():
        try:
            index_daily = pd.read_parquet(index_daily_path, columns=["trade_date"])
            index_dates = pd.to_datetime(index_daily["trade_date"], errors="coerce").dropna()
            if index_dates.empty:
                errors.append(f"{index_daily_path} trade_date 全部无效")
            elif index_dates.max() < end_date - pd.Timedelta(days=7):
                errors.append(
                    f"index_daily 最新日期 {index_dates.max().date()} 距离回测终点 {end_date.date()} 过远"
                )
        except Exception as exc:
            errors.append(f"{index_daily_path} 日期检查失败: {exc}")

    # Historical index constituents
    if universe == "csi300":
        index_weight = _first_existing(_iter_index_weight_paths())
        if index_weight is None:
            errors.append("缺少历史指数成分文件 index_weight.parquet/csv（csi300 股票池必需）")
        else:
            resolved["index_weight"] = str(index_weight)
            err = _check_table_columns(index_weight, ["index_code", "con_code", "trade_date"])
            if err:
                errors.append(err)
            else:
                if index_weight.suffix == ".parquet":
                    df = pd.read_parquet(index_weight, columns=["index_code", "trade_date"])
                else:
                    df = pd.read_csv(index_weight, usecols=["index_code", "trade_date"])
                if "000300.SH" not in set(df["index_code"].astype(str)):
                    errors.append(f"{index_weight} 不包含 000300.SH 成分数据")
                if not df.empty:
                    trade_dates = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
                    if trade_dates.empty:
                        errors.append(f"{index_weight} trade_date 全部无效")
                    else:
                        if trade_dates.min() > start_date:
                            warnings.append(
                                f"index_weight 最早快照 {trade_dates.min().date()} 晚于回测起点 {start_date.date()}"
                            )
                        # 指数成分通常按月更新，允许 62 天窗口
                        if trade_dates.max() < end_date - pd.Timedelta(days=62):
                            errors.append(
                                f"index_weight 最新快照 {trade_dates.max().date()} 距离回测终点 {end_date.date()} 过远"
                            )

    # Historical ST
    if require_st_history:
        namechange = _first_existing(_iter_namechange_paths())
        if namechange is None:
            errors.append("缺少历史 ST 文件 namechange.parquet/csv（exclude_st=true 必需）")
        else:
            resolved["namechange"] = str(namechange)
            err = _check_table_columns(namechange, ["ts_code", "name", "start_date"])
            if err:
                errors.append(err)
            else:
                try:
                    if namechange.suffix == ".parquet":
                        df = pd.read_parquet(namechange, columns=["start_date"])
                    else:
                        df = pd.read_csv(namechange, usecols=["start_date"])
                    start_dates = pd.to_datetime(df["start_date"], errors="coerce").dropna()
                    if start_dates.empty:
                        errors.append(f"{namechange} start_date 全部无效")
                    elif start_dates.min() > start_date:
                        warnings.append(
                            f"namechange 最早记录 {start_dates.min().date()} 晚于回测起点 {start_date.date()}"
                        )
                except Exception as exc:
                    errors.append(f"{namechange} 日期检查失败: {exc}")

    return DataPrecheckResult(ok=not errors, errors=errors, warnings=warnings, resolved_paths=resolved)


def ensure_strategy_data_ready(strategy) -> DataPrecheckResult:
    result = run_data_precheck(
        universe=getattr(strategy, "universe", "all"),
        require_st_history=bool(getattr(strategy, "exclude_st", False)),
    )
    result.raise_if_failed()
    return result
