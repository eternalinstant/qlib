#!/usr/bin/env python3
"""
批量策略回测 - 数据只加载一次，计算跑50个策略
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG
from core.factors import FactorInfo, FactorRegistry, create_default_registry
from core.selection import (
    compute_signal, extract_topk, _load_parquet_factors,
    _fill_cross_sectional, _get_rebalance_dates,
)
from core.compute import compute_layer_score, neutralize_by_industry
from core.universe import filter_instruments
from core.position import MarketPositionController
from core.qlib_init import init_qlib, load_features_safe
from modules.backtest.base import BacktestResult


@dataclass
class StrategyConfig:
    name: str
    factor_names: List[str]
    weights: Dict[str, float]
    freq: str
    topk: int
    sticky: int
    position_model: str
    description: str = ""  # 策略描述
    stock_pct: float = 0.8
    neutralize_industry: bool = True
    open_cost: float = 0.0003
    close_cost: float = 0.0013

    def get_full_name(self) -> str:
        return f"{self.name}_f{self.freq}_k{self.topk}s{self.sticky}_pm-{self.position_model}"


class PreloadedDataManager:
    def __init__(self, start_date: str = None, end_date: str = None):
        # 固定使用最近2年数据（2019数据太旧，2024-2026足够测试）
        self.start_date = start_date or "2024-01-01"
        self.end_date = end_date or "2026-03-02"
        self.daily_df: pd.DataFrame = None
        self.df_ret: pd.DataFrame = None
        self.valid_instruments: List[str] = None
        self.all_dates: pd.DatetimeIndex = None
        self.controller: MarketPositionController = None
        self.bond_daily_ret: float = 0.0
        self.industry_map: Dict[str, str] = None
        # 缓存
        self._registry = create_default_registry()
        self._filtered_df_cache: Dict[str, pd.DataFrame] = {}

    def load_all(self):
        print(f"[Preload] 数据: {self.start_date} ~ {self.end_date}")
        self._init_qlib()
        self._load_instruments()
        self._load_factor_data()
        self._load_return_data()
        self._load_position_controller()
        self._load_industry_map()
        print(f"[OK] 股票: {len(self.valid_instruments)}, 交易日: {len(self.all_dates)}, 因子: {len(self.daily_df.columns)}")

    def _init_qlib(self):
        init_qlib()

    def _load_instruments(self):
        from qlib.data import D
        instruments = D.instruments(market="all")
        df_close = load_features_safe(
            instruments, ["$close"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day"
        )
        self.valid_instruments = filter_instruments(
            df_close.index.get_level_values("instrument").unique().tolist()
        )
        self.all_dates = df_close.index.get_level_values("datetime").unique().sort_values()

    def _load_factor_data(self):
        qlib_factors = self._registry.get_by_source("qlib")
        parquet_factors = self._registry.get_by_source("parquet")
        qlib_fields = [f.expression for f in qlib_factors]
        qlib_names = [f"{f.category}_{f.name}" for f in qlib_factors]

        df_qlib = load_features_safe(
            self.valid_instruments, qlib_fields,
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day"
        )
        df_qlib.columns = qlib_names

        df_parquet = _load_parquet_factors(
            self.valid_instruments,
            self.start_date,
            self.end_date,
            registry=self._registry
        )

        if df_parquet.empty:
            self.daily_df = df_qlib
        else:
            self.daily_df = df_qlib.join(df_parquet, how="left")
        self.daily_df = _fill_cross_sectional(self.daily_df)

    def _load_return_data(self):
        ret_field = ["$close / Ref($close, 1) - 1"]
        self.df_ret = load_features_safe(
            self.valid_instruments, ret_field,
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day"
        )
        self.df_ret.columns = ["daily_ret"]

    def _load_position_controller(self):
        self.controller = MarketPositionController()
        self.controller.load_market_data()
        self.bond_daily_ret = self.controller.get_bond_daily_return()

    def _load_industry_map(self):
        from core.selection import _load_industry_map
        self.industry_map = _load_industry_map()

    def get_rebalance_dates(self, freq: str) -> pd.DatetimeIndex:
        return _get_rebalance_dates(self.daily_df, freq=freq)

    def get_filtered_df(self, freq: str) -> pd.DataFrame:
        if freq not in self._filtered_df_cache:
            rebalance_dates = self.get_rebalance_dates(freq)
            self._filtered_df_cache[freq] = self.daily_df.loc[
                self.daily_df.index.get_level_values("datetime").isin(rebalance_dates)
            ]
        return self._filtered_df_cache[freq]


def inline_backtest(
    date_to_symbols: Dict[pd.Timestamp, set],
    df_ret: pd.DataFrame,
    controller: Optional[MarketPositionController],
    config: StrategyConfig,
    all_dates: pd.DatetimeIndex,
    bond_daily_ret: float = 0.0,
) -> BacktestResult:
    portfolio_returns = []
    monthly_dates_list = sorted(date_to_symbols.keys())

    if len(monthly_dates_list) < 2:
        return BacktestResult(daily_returns=pd.Series(dtype=float), portfolio_value=pd.Series(dtype=float))

    prev_selected = set()

    for i, rebal_date in enumerate(monthly_dates_list[:-1]):
        selected = date_to_symbols.get(rebal_date, set())
        if len(selected) < config.topk:
            continue
        next_date = monthly_dates_list[i + 1]
        holding_dates = all_dates[(all_dates > rebal_date) & (all_dates <= next_date)]

        for j, hd in enumerate(holding_dates):
            try:
                daily_ret = df_ret.xs(hd, level="datetime")
                stock_ret = daily_ret.loc[daily_ret.index.isin(selected), "daily_ret"].mean()
                if np.isnan(stock_ret):
                    stock_ret = 0.0

                is_rebal = (j == 0)
                cost_deduction = 0.0
                if is_rebal:
                    # 计算实际换手（排除sticky保留的部分）
                    kept = prev_selected & selected if prev_selected else set()
                    sell_count = len(prev_selected - kept) if prev_selected else 0
                    buy_count = len(selected - kept)
                    # 只对实际买卖的部分收费
                    cost = (sell_count * config.close_cost + buy_count * config.open_cost)
                    cost_deduction = cost / len(selected) if selected else 0

                if controller is not None and config.position_model == "trend":
                    alloc = controller.get_allocation(hd, is_rebalance_day=is_rebal)
                    port_ret = alloc.stock_pct * stock_ret + alloc.cash_pct * bond_daily_ret - cost_deduction
                elif config.position_model == "fixed":
                    port_ret = config.stock_pct * stock_ret + (1 - config.stock_pct) * bond_daily_ret - cost_deduction
                else:
                    port_ret = stock_ret - cost_deduction

                portfolio_returns.append({"date": hd, "return": port_ret})

                if is_rebal:
                    prev_selected = selected.copy()
            except (KeyError, IndexError):
                continue

    if not portfolio_returns:
        return BacktestResult(daily_returns=pd.Series(dtype=float), portfolio_value=pd.Series(dtype=float))

    df_result = pd.DataFrame(portfolio_returns).set_index("date")
    df_result.index = pd.to_datetime(df_result.index)
    daily_returns = df_result["return"]
    portfolio_value = (1 + daily_returns).cumprod()

    return BacktestResult(daily_returns=daily_returns, portfolio_value=portfolio_value)


class StrategyExecutor:
    def __init__(self, data_manager: PreloadedDataManager):
        self.data_manager = data_manager
        self.full_registry = data_manager._registry

    def execute(self, config: StrategyConfig) -> Tuple[BacktestResult, Dict[str, Any]]:
        registry = self._build_registry(config.factor_names)
        rebalance_dates = self.data_manager.get_rebalance_dates(config.freq)
        filtered_df = self.data_manager.get_filtered_df(config.freq)

        if config.neutralize_industry and self.data_manager.industry_map:
            df = neutralize_by_industry(filtered_df, self.data_manager.industry_map)
        else:
            df = filtered_df

        signal = compute_signal(df, registry=registry, weights=config.weights,
                                neutralize_industry=False)

        df_sel = extract_topk(signal, rebalance_dates, topk=config.topk, sticky=config.sticky)

        date_to_symbols = {}
        for dt, grp in df_sel.groupby("date"):
            date_to_symbols[pd.Timestamp(dt)] = set(grp["symbol"].tolist())

        controller = self.data_manager.controller if config.position_model == "trend" else None
        result = inline_backtest(
            date_to_symbols=date_to_symbols,
            df_ret=self.data_manager.df_ret,
            controller=controller,
            config=config,
            all_dates=self.data_manager.all_dates,
            bond_daily_ret=self.data_manager.bond_daily_ret,
        )

        metadata = {
            "name": config.get_full_name(),
            "base_name": config.name,
            "description": config.description,
            "freq": config.freq,
            "topk": config.topk,
            "sticky": config.sticky,
            "position_model": config.position_model,
            "num_factors": len(config.factor_names),
            "num_periods": len(date_to_symbols),
            "weights": config.weights,
        }
        return result, metadata

    def _build_registry(self, factor_names: List[str]) -> FactorRegistry:
        registry = FactorRegistry()
        for name in factor_names:
            factor = self.full_registry.get(name)
            if factor:
                registry.register(factor)
        return registry


class StrategyConfigGenerator:
    def __init__(self, registry=None):
        self.full_registry = registry or create_default_registry()
        self.alpha_factors = [f.name for f in self.full_registry.get_by_category("alpha")]
        self.risk_factors = [f.name for f in self.full_registry.get_by_category("risk")]
        self.enhance_factors = [f.name for f in self.full_registry.get_by_category("enhance")]
        self.all_factors = self.alpha_factors + self.risk_factors + self.enhance_factors

    def generate_all(self) -> List[StrategyConfig]:
        configs = []
        configs.extend(self._generate_existing_strategies())
        configs.extend(self._generate_weight_scan())
        configs.extend(self._generate_factor_subset_scan())
        configs.extend(self._generate_frequency_scan())
        configs.extend(self._generate_topk_scan())
        configs.extend(self._generate_position_model_scan())
        configs.extend(self._generate_sticky_scan())
        configs.extend(self._generate_high_ir_strategies())
        configs.extend(self._generate_cross_combinations())
        configs.extend(self._generate_extreme_configs())
        print(f"[Generate] {len(configs)} 个策略配置")
        return configs

    def _generate_existing_strategies(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="default", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=5, position_model="trend", description="基准策略：alpha50%+risk20%+enhance30%，双周调仓"))
        configs.append(StrategyConfig(name="value", factor_names=self.all_factors, weights={"alpha": 0.70, "risk": 0.10, "enhance": 0.20}, freq="biweek", topk=15, sticky=5, position_model="trend", description="价值优先：alpha70%+risk10%+enhance20%，偏重基本面"))
        configs.append(StrategyConfig(name="defensive", factor_names=self.all_factors, weights={"alpha": 0.30, "risk": 0.50, "enhance": 0.20}, freq="biweek", topk=15, sticky=5, position_model="trend", description="防御优先：risk50%+alpha30%+enhance20%，偏重低波动"))
        configs.append(StrategyConfig(name="momentum", factor_names=self.all_factors, weights={"alpha": 0.30, "risk": 0.20, "enhance": 0.50}, freq="biweek", topk=15, sticky=5, position_model="trend", description="动量优先：enhance50%+alpha30%+risk20%，偏重趋势"))
        return configs

    def _generate_weight_scan(self) -> List[StrategyConfig]:
        configs = []
        weights_list = [
            ("balanced", {"alpha": 0.34, "risk": 0.33, "enhance": 0.33}, "均衡配置：三层各1/3"),
            ("alpha_heavy", {"alpha": 0.70, "risk": 0.15, "enhance": 0.15}, "Alpha重：70%基本面+15%风控+15%增强"),
            ("enhance_heavy", {"alpha": 0.15, "risk": 0.15, "enhance": 0.70}, "增强重：70%趋势+15%基本面+15%风控"),
            ("risk_heavy", {"alpha": 0.15, "risk": 0.70, "enhance": 0.15}, "风控重：70%低波+15%基本面+15%趋势"),
            ("no_risk", {"alpha": 0.60, "risk": 0.00, "enhance": 0.40}, "无风控：60%基本面+40%趋势，忽略低波因子"),
        ]
        for name, weights, desc in weights_list:
            configs.append(StrategyConfig(name=f"weight_{name}", factor_names=self.all_factors, weights=weights, freq="biweek", topk=15, sticky=5, position_model="trend", description=desc))
        return configs

    def _generate_factor_subset_scan(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="value_core", factor_names=self.alpha_factors, weights={"alpha": 1.0, "risk": 0.0, "enhance": 0.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="纯价值：只用基本面因子"))
        reverse_factors = ["bb_lower_dist", "ema26_dev", "bbi_momentum"]
        configs.append(StrategyConfig(name="reverse_core", factor_names=reverse_factors, weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="纯反转：布林下轨+EMA26偏离+BBI反转"))
        power_factors = ["bull_power", "bear_power"]
        configs.append(StrategyConfig(name="power_core", factor_names=power_factors, weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="纯多空力道：bull+bear power"))
        high_ir_factors = ["bull_power", "amt_std_6d", "vol_std_10d", "vol_ema26_ratio", "bb_lower_dist"]
        configs.append(StrategyConfig(name="high_ir_core", factor_names=high_ir_factors, weights={"alpha": 0.0, "risk": 0.4, "enhance": 0.6}, freq="biweek", topk=15, sticky=5, position_model="trend", description="高IR精选：v2扫描前5强因子"))
        return configs

    def _generate_frequency_scan(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="freq_week", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="week", topk=15, sticky=5, position_model="trend", description="周频调仓：每周换手，更灵活但交易成本高"))
        configs.append(StrategyConfig(name="freq_month", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="month", topk=15, sticky=5, position_model="trend", description="月频调仓：每月换手，更稳健"))
        return configs

    def _generate_topk_scan(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="small_topk", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=10, sticky=3, position_model="trend", description="小集中：Top10精选，高波动高收益"))
        configs.append(StrategyConfig(name="large_topk", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=20, sticky=7, position_model="trend", description="大分散：Top20分散，降低个股风险"))
        return configs

    def _generate_position_model_scan(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="full_position", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=5, position_model="full", description="满仓模式：不做仓位控制，始终100%持仓"))
        return configs

    def _generate_sticky_scan(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="no_sticky", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=0, position_model="trend", description="无粘性：每次完全按信号调仓"))
        configs.append(StrategyConfig(name="high_sticky", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=10, position_model="trend", description="强粘性：保留10只上期持仓，降低换手"))
        return configs

    def _generate_high_ir_strategies(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="single_bull_power", factor_names=["bull_power"], weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="单因子之王：bull_power(IR=0.63)"))
        configs.append(StrategyConfig(name="single_bb_lower", factor_names=["bb_lower_dist"], weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="单因子：布林下轨(IR=0.39)"))
        top3_ir = ["bull_power", "amt_std_6d", "vol_std_10d"]
        configs.append(StrategyConfig(name="top3_ir", factor_names=top3_ir, weights={"risk": 0.5, "enhance": 0.5}, freq="biweek", topk=15, sticky=5, position_model="trend", description="Top3 IR因子：bull_power+量波动"))
        top5_ir = ["bull_power", "amt_std_6d", "vol_std_10d", "vol_ema26_ratio", "bear_power"]
        configs.append(StrategyConfig(name="top5_ir", factor_names=top5_ir, weights={"risk": 0.4, "enhance": 0.6}, freq="biweek", topk=15, sticky=5, position_model="trend", description="Top5 IR因子"))
        return configs

    def _generate_cross_combinations(self) -> List[StrategyConfig]:
        configs = []
        value_reverse = self.alpha_factors + ["bb_lower_dist", "ema26_dev"]
        configs.append(StrategyConfig(name="value_reverse", factor_names=value_reverse, weights={"alpha": 0.6, "risk": 0.0, "enhance": 0.4}, freq="biweek", topk=15, sticky=5, position_model="trend", description="价值+反转：基本面为主+少量反转"))
        power_reverse = ["bull_power", "bear_power", "bb_lower_dist", "ema26_dev"]
        configs.append(StrategyConfig(name="power_reverse_mix", factor_names=power_reverse, weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="力道+反转混合"))
        low_vol_value = self.alpha_factors + ["ret_vol_20d"]
        configs.append(StrategyConfig(name="low_vol_value", factor_names=low_vol_value, weights={"alpha": 0.7, "risk": 0.3, "enhance": 0.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="低波价值：价值+低波动防守"))
        configs.append(StrategyConfig(name="high_freq", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="week", topk=10, sticky=0, position_model="trend", description="高频激进：周频+小topk+无粘性"))
        configs.append(StrategyConfig(name="low_freq", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="month", topk=20, sticky=10, position_model="trend", description="低频保守：月频+大topk+高粘性"))
        configs.append(StrategyConfig(name="enhance_sticky", factor_names=self.enhance_factors, weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=10, position_model="trend", description="增强层+粘性：增强层因子+保留持仓"))
        configs.append(StrategyConfig(name="value_full", factor_names=self.alpha_factors, weights={"alpha": 1.0}, freq="biweek", topk=15, sticky=5, position_model="full", description="纯价值满仓：只用基本面+满仓"))
        configs.append(StrategyConfig(name="balanced_mix", factor_names=self.all_factors, weights={"alpha": 0.34, "risk": 0.33, "enhance": 0.33}, freq="biweek", topk=15, sticky=5, position_model="trend", description="均衡混合：三层各1/3"))
        configs.append(StrategyConfig(name="enhance_boost", factor_names=self.all_factors, weights={"alpha": 0.2, "risk": 0.1, "enhance": 0.7}, freq="week", topk=12, sticky=3, position_model="trend", description="增强层重：70%增强+周频"))
        low_vol_defense = self.risk_factors + self.alpha_factors[:3]
        configs.append(StrategyConfig(name="low_vol_defense", factor_names=low_vol_defense, weights={"alpha": 0.5, "risk": 0.5, "enhance": 0.0}, freq="month", topk=15, sticky=7, position_model="trend", description="低波防御：风控层+价值前3"))
        return configs

    def _generate_extreme_configs(self) -> List[StrategyConfig]:
        configs = []
        configs.append(StrategyConfig(name="extreme_alpha", factor_names=self.alpha_factors, weights={"alpha": 1.0}, freq="biweek", topk=15, sticky=0, position_model="full", description="极端价值：纯基本面+满仓+双周调仓"))
        reverse_only = ["bb_lower_dist", "ema26_dev", "bbi_momentum"]
        configs.append(StrategyConfig(name="extreme_reverse", factor_names=reverse_only, weights={"enhance": 1.0}, freq="week", topk=10, sticky=0, position_model="full", description="极端反转：只用反转因子+周频+满仓"))
        power_only = ["bull_power", "bear_power", "vol_ema26_ratio"]
        configs.append(StrategyConfig(name="extreme_power", factor_names=power_only, weights={"enhance": 1.0}, freq="week", topk=10, sticky=0, position_model="full", description="极端力道：bull/bear power+量比+周频+满仓"))
        configs.append(StrategyConfig(name="large_diversified", factor_names=self.all_factors, weights={"alpha": 0.34, "risk": 0.33, "enhance": 0.33}, freq="month", topk=30, sticky=15, position_model="trend", description="极度分散：30只+15粘性+月频"))
        configs.append(StrategyConfig(name="small_concentrated", factor_names=self.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="week", topk=8, sticky=0, position_model="trend", description="极度集中：8只+周频+无粘性"))
        configs.append(StrategyConfig(name="pure_risk", factor_names=self.risk_factors, weights={"risk": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="纯风控：只用低波低换手因子"))
        return configs


class ResultsSummarizer:
    @staticmethod
    def summarize(results: Dict[str, Tuple[BacktestResult, Dict]]) -> pd.DataFrame:
        rows = []
        for name, (result, metadata) in results.items():
            if result.portfolio_value.empty:
                continue
            win_rate = (result.daily_returns > 0).mean() if not result.daily_returns.empty else 0.0
            volatility = result.daily_returns.std() * np.sqrt(252) if not result.daily_returns.empty else 0.0
            rows.append({
                "策略名": name,
                "策略描述": metadata.get("description", ""),
                "总收益率": f"{result.total_return:.2%}",
                "年化收益率": f"{result.annual_return:.2%}",
                "夏普比率": f"{result.sharpe_ratio:.4f}",
                "最大回撤": f"{result.max_drawdown:.2%}",
                "日胜率": f"{win_rate:.2%}",
                "波动率": f"{volatility:.2%}",
                "调仓频率": metadata.get("freq", ""),
                "TopK": metadata.get("topk", ""),
                "Sticky": metadata.get("sticky", ""),
                "仓位模型": metadata.get("position_model", ""),
                "因子数": metadata.get("num_factors", ""),
                "期数": metadata.get("num_periods", ""),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("夏普比率", key=lambda x: x.str.rstrip("%").astype(float), ascending=False)
        return df

    @staticmethod
    def save_results(df: pd.DataFrame, output_path: Path = None):
        if output_path is None:
            output_path = PROJECT_ROOT / "results" / "batch_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 结果已保存: {output_path}")

    @staticmethod
    def print_summary(df: pd.DataFrame, top_n: int = 20):
        if df.empty:
            print("[WARN] 无有效回测结果")
            return
        print("\n" + "="*120)
        print(f"  批量回测结果 (Top {min(top_n, len(df))})")
        print("="*120)
        df_display = df.head(top_n).copy()
        print(df_display.drop(columns=[c for c in df_display.columns if c.startswith("_")]).to_string(index=False))
        print("\n" + "="*120)


def _top10_configs(generator: StrategyConfigGenerator) -> List[StrategyConfig]:
    """v2因子重建后 Top 策略 — 验证跨期稳健性"""
    g = generator
    return [
        # Top 5 from 2024-2026
        StrategyConfig(name="power_core", factor_names=["bull_power", "bear_power"], weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="纯多空力道：bull+bear power"),
        StrategyConfig(name="single_bull", factor_names=["bull_power"], weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=5, position_model="trend", description="单因子：bull_power"),
        StrategyConfig(name="top3_ir", factor_names=["bull_power", "amt_std_6d", "vol_std_10d"], weights={"risk": 0.5, "enhance": 0.5}, freq="biweek", topk=15, sticky=5, position_model="trend", description="Top3 IR因子"),
        StrategyConfig(name="default", factor_names=g.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=5, position_model="trend", description="基准16因子"),
        StrategyConfig(name="high_sticky", factor_names=g.all_factors, weights={"alpha": 0.50, "risk": 0.20, "enhance": 0.30}, freq="biweek", topk=15, sticky=10, position_model="trend", description="强粘性16因子"),
        # Variants for robustness check
        StrategyConfig(name="power_low_freq", factor_names=["bull_power", "bear_power"], weights={"enhance": 1.0}, freq="month", topk=15, sticky=5, position_model="trend", description="力道月频"),
        StrategyConfig(name="power_sticky10", factor_names=["bull_power", "bear_power"], weights={"enhance": 1.0}, freq="biweek", topk=15, sticky=10, position_model="trend", description="力道+高粘性"),
        StrategyConfig(name="power_k20", factor_names=["bull_power", "bear_power"], weights={"enhance": 1.0}, freq="biweek", topk=20, sticky=7, position_model="trend", description="力道+分散20只"),
        StrategyConfig(name="power_value", factor_names=["bull_power", "bear_power"] + g.alpha_factors, weights={"alpha": 0.3, "enhance": 0.7}, freq="biweek", topk=15, sticky=5, position_model="trend", description="力道+基本面"),
        StrategyConfig(name="low_vol_defense", factor_names=g.risk_factors + g.alpha_factors[:3], weights={"alpha": 0.5, "risk": 0.5}, freq="month", topk=15, sticky=7, position_model="trend", description="低波防御"),
    ]


def main(start_date: str = None, end_date: str = None, top10_only: bool = False):
    import time
    start_time = time.time()
    print("\n" + "="*60)
    print("  批量策略回测系统")
    print("="*60)

    data_manager = PreloadedDataManager(start_date=start_date, end_date=end_date)
    data_manager.load_all()

    generator = StrategyConfigGenerator(registry=data_manager._registry)
    if top10_only:
        configs = _top10_configs(generator)
        print(f"[Generate] {len(configs)} 个策略配置 (Top10)")
    else:
        configs = generator.generate_all()

    executor = StrategyExecutor(data_manager)

    print("\n" + "="*60)
    print("  开始批量回测")
    print("="*60)

    results = {}
    failed = []

    for i, config in enumerate(configs, 1):
        full_name = config.get_full_name()
        print(f"\n[{i}/{len(configs)}] {full_name}")
        try:
            result, metadata = executor.execute(config)
            if not result.portfolio_value.empty:
                results[full_name] = (result, metadata)
                print(f"  [OK] {result.total_return:+.2%} 夏普:{result.sharpe_ratio:.4f}")
            else:
                failed.append(full_name)
        except Exception as e:
            failed.append(full_name)
            print(f"  [ERROR] {e}")

    summarizer = ResultsSummarizer()
    df_results = summarizer.summarize(results)
    output_name = "batch_results_top10.csv" if top10_only else "batch_results.csv"
    summarizer.save_results(df_results, PROJECT_ROOT / "results" / output_name)
    summarizer.print_summary(df_results, top_n=20)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟), 每策略:{elapsed/len(configs):.1f}秒")

    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-08")
    parser.add_argument("--top10", action="store_true")
    args = parser.parse_args()
    df = main(start_date=args.start, end_date=args.end, top10_only=args.top10)
