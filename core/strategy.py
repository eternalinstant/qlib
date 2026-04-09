"""
策略定义模块 — 从 YAML 加载策略配置，驱动因子注册、选股、仓位控制

支持两种放置方式：
1. 兼容旧结构：config/strategies/*.yaml
2. 分层结构：config/strategies/<layer>/<group>/<strategy>.yaml

Strategy 类是整个多策略架构的核心入口。
"""

import json
import yaml
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from core.factors import FactorInfo, FactorRegistry, default_registry
from core.validity import ValidityConfig, build_validity_config

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGIES_DIR = PROJECT_ROOT / "config" / "strategies"
SELECTIONS_DIR = PROJECT_ROOT / "data" / "selections"

VALID_POSITION_MODELS = {"trend", "fixed", "full", "gate"}
VALID_REBALANCE_FREQS = {"day", "week", "biweek", "month"}
VALID_SOURCES = {"qlib", "parquet"}
VALID_SELECTION_UNIVERSES = {"all", "csi300"}
VALID_SELECTION_MODES = {"factor_topk", "stoploss_replace"}
SELECTION_CACHE_VERSION = 1


def is_composite_strategy(strategy: Any) -> bool:
    """判断对象是否为组合策略，避免对 Mock 误判。"""
    components = getattr(strategy, "composition_components", None)
    return isinstance(components, list) and len(components) > 0


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_strategy_defaults() -> Dict[str, Any]:
    """加载全局策略默认值，供单策略 YAML 继承"""
    from config.config import load_yaml

    defaults = load_yaml("strategy.yaml")
    if not isinstance(defaults, dict):
        return {}
    return defaults


def _strategy_key_from_path(path: Path) -> str:
    """将策略 YAML 路径转成相对 key。"""
    return path.relative_to(STRATEGIES_DIR).with_suffix("").as_posix()


def _resolve_strategy_path(name: str) -> Tuple[str, Path]:
    """解析策略名，支持递归目录与 basename 回退。"""
    explicit_path = STRATEGIES_DIR / f"{name}.yaml"
    if explicit_path.exists():
        return _strategy_key_from_path(explicit_path), explicit_path

    if "/" not in name:
        legacy_path = STRATEGIES_DIR / f"{name}.yaml"
        if legacy_path.exists():
            return name, legacy_path

    matches = sorted(p for p in STRATEGIES_DIR.rglob("*.yaml") if p.stem == name)
    if len(matches) == 1:
        path = matches[0]
        return _strategy_key_from_path(path), path
    if len(matches) > 1:
        candidates = ", ".join(_strategy_key_from_path(p) for p in matches)
        raise ValueError(f"策略名 '{name}' 不唯一，请改用完整路径：{candidates}")
    raise FileNotFoundError(f"策略文件不存在: {explicit_path}")


def _validate_strategy(cfg: Dict[str, Any], name: str) -> None:
    """验证策略 YAML 配置"""
    errors = []
    composition = cfg.get("composition", {}) or {}

    if composition:
        components = composition.get("components", [])
        if not isinstance(components, list) or len(components) < 2:
            errors.append("composition.components 至少需要 2 个成员策略")
        else:
            seen = set()
            total_weight = 0.0
            for idx, item in enumerate(components):
                if not isinstance(item, dict):
                    errors.append(f"composition.components[{idx}] 必须是对象")
                    continue

                member_name = item.get("strategy")
                if not member_name:
                    errors.append(f"composition.components[{idx}] 缺少 strategy")
                elif member_name == name:
                    errors.append("组合策略不能直接引用自身")
                elif member_name in seen:
                    errors.append(f"composition.components[{idx}].strategy='{member_name}' 重复")
                else:
                    seen.add(member_name)
                    try:
                        _resolve_strategy_path(str(member_name))
                    except Exception as exc:
                        errors.append(
                            f"composition.components[{idx}].strategy='{member_name}' 无法解析: {exc}"
                        )

                weight = item.get("weight")
                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    errors.append(f"composition.components[{idx}].weight 必须是数值")
                    continue

                if weight <= 0 or weight > 1:
                    errors.append(
                        f"composition.components[{idx}].weight={weight} 超出范围，应在 (0, 1]"
                    )
                total_weight += weight

            if total_weight > 1 + 1e-9:
                errors.append(f"composition.components 权重和为 {total_weight:.4f}，不能超过 1.0")

        cash_weight = composition.get("cash_weight")
        if cash_weight is not None:
            try:
                cash_weight = float(cash_weight)
            except (TypeError, ValueError):
                errors.append("composition.cash_weight 必须是数值")
            else:
                if cash_weight < 0 or cash_weight > 1:
                    errors.append("composition.cash_weight 超出范围，应在 [0, 1]")
                elif components:
                    total_weight = sum(
                        float(item.get("weight", 0.0))
                        for item in components
                        if isinstance(item, dict)
                    )
                    if abs(total_weight + cash_weight - 1.0) > 1e-6:
                        errors.append("composition.cash_weight 与成员权重之和必须等于 1.0")

    position = cfg.get("position", {})
    position_model = position.get("model", "trend")
    if position_model not in VALID_POSITION_MODELS:
        errors.append(f"position.model='{position_model}' 无效，可选: {VALID_POSITION_MODELS}")

    rebalance = cfg.get("rebalance", {})
    rebalance_freq = rebalance.get("freq", "month")
    if rebalance_freq not in VALID_REBALANCE_FREQS:
        errors.append(f"rebalance.freq='{rebalance_freq}' 无效，可选: {VALID_REBALANCE_FREQS}")

    selection = cfg.get("selection", {})
    selection_mode = selection.get("mode", "factor_topk")
    if selection_mode not in VALID_SELECTION_MODES:
        errors.append(f"selection.mode='{selection_mode}' 无效，可选: {VALID_SELECTION_MODES}")
    selection_universe = selection.get("universe", "all")
    if selection_universe not in VALID_SELECTION_UNIVERSES:
        errors.append(
            f"selection.universe='{selection_universe}' 无效，可选: {VALID_SELECTION_UNIVERSES}"
        )
    hard_filter_quantiles = selection.get("hard_filter_quantiles", {})
    if hard_filter_quantiles:
        for factor_name, quantile in hard_filter_quantiles.items():
            try:
                q = float(quantile)
            except (TypeError, ValueError):
                errors.append(f"selection.hard_filter_quantiles.{factor_name} 必须是数值")
                continue
            if not 0.0 <= q <= 1.0:
                errors.append(
                    f"selection.hard_filter_quantiles.{factor_name}={quantile} 超出范围，应在 [0, 1]"
                )
    industry_leader_field = selection.get("industry_leader_field")
    industry_leader_top_n = selection.get("industry_leader_top_n")
    if industry_leader_field is not None and not str(industry_leader_field).strip():
        errors.append("selection.industry_leader_field 不能为空字符串")
    if industry_leader_top_n is not None:
        try:
            value = int(industry_leader_top_n)
        except (TypeError, ValueError):
            errors.append("selection.industry_leader_top_n 必须是整数")
        else:
            if value < 1:
                errors.append("selection.industry_leader_top_n 不能小于 1")
    if (industry_leader_field is None) ^ (industry_leader_top_n is None):
        errors.append(
            "selection.industry_leader_field 与 selection.industry_leader_top_n 必须同时配置"
        )
    for name in (
        "score_smoothing_days",
        "entry_persist_days",
        "exit_persist_days",
        "min_hold_days",
    ):
        if name in selection:
            try:
                value = int(selection[name])
            except (TypeError, ValueError):
                errors.append(f"selection.{name} 必须是整数")
                continue
            if name == "min_hold_days" and value < 0:
                errors.append(f"selection.{name} 不能小于 0")
            elif name != "min_hold_days" and value < 1:
                errors.append(f"selection.{name} 不能小于 1")
    for name in ("entry_rank", "exit_rank"):
        if name in selection:
            try:
                value = int(selection[name])
            except (TypeError, ValueError):
                errors.append(f"selection.{name} 必须是整数")
                continue
            if value < 1:
                errors.append(f"selection.{name} 不能小于 1")
    if selection_mode == "stoploss_replace":
        if "stoploss_lookback_days" in selection:
            try:
                value = int(selection["stoploss_lookback_days"])
            except (TypeError, ValueError):
                errors.append("selection.stoploss_lookback_days 必须是整数")
            else:
                if value < 1:
                    errors.append("selection.stoploss_lookback_days 不能小于 1")
        if "replacement_pool_size" in selection:
            try:
                value = int(selection["replacement_pool_size"])
            except (TypeError, ValueError):
                errors.append("selection.replacement_pool_size 必须是整数")
            else:
                if value < 0:
                    errors.append("selection.replacement_pool_size 不能小于 0")
        if "stoploss_drawdown" in selection:
            try:
                value = float(selection["stoploss_drawdown"])
            except (TypeError, ValueError):
                errors.append("selection.stoploss_drawdown 必须是数值")
            else:
                if value <= 0 or value >= 1:
                    errors.append("selection.stoploss_drawdown 必须在 (0, 1) 之间")

    factors = cfg.get("factors", {})
    for category, factor_list in factors.items():
        if not isinstance(factor_list, list):
            continue
        for i, f_cfg in enumerate(factor_list):
            if "name" not in f_cfg:
                errors.append(f"factors.{category}[{i}]: 缺少 'name' 字段")
            if (
                "expression" not in f_cfg
                and "name" in f_cfg
                and f_cfg["name"] not in default_registry.all()
            ):
                errors.append(
                    f"factors.{category}[{i}]: 缺少 'expression' 字段且因子 '{f_cfg['name']}' 不在默认注册表中"
                )
            source = f_cfg.get("source", "qlib")
            if source not in VALID_SOURCES:
                errors.append(
                    f"factors.{category}[{i}].source='{source}' 无效，可选: {VALID_SOURCES}"
                )

    validity = cfg.get("validity", {})
    if validity:
        from core.validity import VALID_VALIDITY_ACTIONS

        action = validity.get("action", "review")
        if action not in VALID_VALIDITY_ACTIONS:
            errors.append(
                f"validity.action='{action}' 无效，可选: {sorted(VALID_VALIDITY_ACTIONS)}"
            )

    if "factor_window_scale" in cfg:
        try:
            scale = int(cfg["factor_window_scale"])
        except (TypeError, ValueError):
            errors.append("factor_window_scale 必须是整数")
        else:
            if scale < 1:
                errors.append("factor_window_scale 不能小于 1")

    if errors:
        raise ValueError(f"策略 {name} 配置错误:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class Strategy:
    """策略定义：从 YAML 加载，驱动因子注册、选股、仓位"""

    name: str
    display_name: str
    description: str
    registry: FactorRegistry
    weights: Dict[str, float]
    topk: int
    neutralize_industry: bool
    universe: str  # "all" | "csi300"
    min_market_cap: float  # 市值下限（亿元），0 = 不过滤
    exclude_st: bool  # 排除历史 ST（无本地历史文件时自动 no-op）
    exclude_new_days: int  # 排除上市 N 天内股票
    sticky: int  # 持仓粘性：从上期保留的股票数量
    buffer: int  # 排名缓冲区：持仓股在 topk+buffer 内保留
    position_model: str  # "trend" | "fixed" | "full"
    position_params: Dict[str, Any]
    rebalance_freq: str  # "month" | "biweek" | "week"
    selection_mode: str = "factor_topk"
    score_smoothing_days: int = 1  # 因子得分平滑窗口
    entry_rank: Optional[int] = None
    exit_rank: Optional[int] = None
    entry_persist_days: int = 1
    exit_persist_days: int = 1
    min_hold_days: int = 0
    stoploss_lookback_days: int = 20
    stoploss_drawdown: float = 0.10
    replacement_pool_size: int = 0
    threshold: float = 0.0  # 稳定性阈值
    churn_limit: int = 0  # 单次最大换仓数
    margin_stable: bool = False  # 边缘持仓稳定性
    trading_cost: Dict[str, float] = field(default_factory=dict)
    validity: Optional[ValidityConfig] = None
    config_path: Optional[Path] = None
    hard_filters: Dict[str, float] = field(default_factory=dict)  # 财务因子硬过滤
    hard_filter_quantiles: Dict[str, float] = field(default_factory=dict)  # 分位过滤
    industry_leader_field: Optional[str] = None
    industry_leader_top_n: Optional[int] = None
    composition_components: List[Dict[str, Any]] = field(default_factory=list)
    cash_weight: float = 0.0
    factor_window_scale: int = 1

    @classmethod
    def load(cls, name: str) -> "Strategy":
        """从 config/strategies 递归加载策略。"""
        resolved_name, yaml_path = _resolve_strategy_path(name)

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        cfg = _deep_merge_dict(_load_strategy_defaults(), raw_cfg)

        _validate_strategy(cfg, resolved_name)

        registry = cls._build_registry(cfg.get("factors", {}))
        weights = cfg.get("weights", {})
        selection = cfg.get("selection", {})
        stability = cfg.get("stability", {})
        position = cfg.get("position", {})
        rebalance = cfg.get("rebalance", {})
        trading = cfg.get("trading", {})
        validity = build_validity_config(cfg.get("validity", {}))
        composition = cfg.get("composition", {}) or {}

        raw_selection = raw_cfg.get("selection", {}) if isinstance(raw_cfg, dict) else {}
        raw_stability = raw_cfg.get("stability", {}) if isinstance(raw_cfg, dict) else {}
        raw_trading = raw_cfg.get("trading", {}) if isinstance(raw_cfg, dict) else {}

        if "threshold" in raw_stability:
            threshold = raw_stability["threshold"]
        elif "threshold" in raw_selection:
            threshold = raw_selection["threshold"]
        else:
            threshold = stability.get("threshold", selection.get("threshold", 0.0))

        if "churn_limit" in raw_stability:
            churn_limit = raw_stability["churn_limit"]
        elif "churn_limit" in raw_selection:
            churn_limit = raw_selection["churn_limit"]
        else:
            churn_limit = stability.get("churn_limit", selection.get("churn_limit", 0))

        if "margin_stable" in raw_stability:
            margin_stable = raw_stability["margin_stable"]
        elif "margin_stable" in raw_selection:
            margin_stable = raw_selection["margin_stable"]
        else:
            margin_stable = stability.get("margin_stable", selection.get("margin_stable", False))

        score_smoothing_days = int(selection.get("score_smoothing_days", 1))
        selection_mode = str(selection.get("mode", "factor_topk"))
        entry_rank = selection.get("entry_rank")
        exit_rank = selection.get("exit_rank")
        entry_persist_days = int(selection.get("entry_persist_days", 1))
        exit_persist_days = int(selection.get("exit_persist_days", 1))
        min_hold_days = int(selection.get("min_hold_days", 0))
        stoploss_lookback_days = int(selection.get("stoploss_lookback_days", 20))
        stoploss_drawdown = float(selection.get("stoploss_drawdown", 0.10))
        replacement_pool_size = int(selection.get("replacement_pool_size", 0))
        if entry_rank is not None:
            entry_rank = int(entry_rank)
        if exit_rank is not None:
            exit_rank = int(exit_rank)

        if "buy_commission_rate" in raw_trading:
            buy_commission_rate = raw_trading["buy_commission_rate"]
        elif "open_cost" in raw_trading:
            buy_commission_rate = raw_trading["open_cost"]
        else:
            buy_commission_rate = trading.get(
                "buy_commission_rate", trading.get("open_cost", 0.0003)
            )

        if "sell_stamp_tax_rate" in raw_trading:
            sell_stamp_tax_rate = raw_trading["sell_stamp_tax_rate"]
        else:
            sell_stamp_tax_rate = trading.get("sell_stamp_tax_rate", 0.001)

        if "sell_commission_rate" in raw_trading:
            sell_commission_rate = raw_trading["sell_commission_rate"]
        elif "close_cost" in raw_trading:
            sell_commission_rate = raw_trading["close_cost"] - sell_stamp_tax_rate
        else:
            sell_commission_rate = trading.get(
                "sell_commission_rate",
                trading.get("close_cost", 0.0013) - sell_stamp_tax_rate,
            )

        components = []
        cash_weight = 0.0
        if composition:
            total_weight = 0.0
            for item in composition.get("components", []):
                member_name, _ = _resolve_strategy_path(str(item["strategy"]))
                weight = float(item["weight"])
                total_weight += weight
                components.append({"strategy": member_name, "weight": weight})
            cash_weight = float(composition.get("cash_weight", max(0.0, 1.0 - total_weight)))

        return cls(
            name=resolved_name,
            display_name=cfg.get("name", Path(resolved_name).name),
            description=cfg.get("description", ""),
            registry=FactorRegistry() if composition else registry,
            weights=weights,
            topk=selection.get("topk", 20),
            neutralize_industry=selection.get("neutralize_industry", True),
            universe=selection.get("universe", "all"),
            min_market_cap=selection.get("min_market_cap", 0.0),
            exclude_st=selection.get("exclude_st", False),
            exclude_new_days=selection.get("exclude_new_days", 0),
            sticky=selection.get("sticky", stability.get("sticky", 0)),
            buffer=selection.get("buffer", 0),
            position_model=position.get("model", "trend"),
            position_params={k: v for k, v in position.items() if k != "model"},
            rebalance_freq=rebalance.get("freq", "month"),
            selection_mode=selection_mode,
            score_smoothing_days=score_smoothing_days,
            entry_rank=entry_rank,
            exit_rank=exit_rank,
            entry_persist_days=entry_persist_days,
            exit_persist_days=exit_persist_days,
            min_hold_days=min_hold_days,
            stoploss_lookback_days=stoploss_lookback_days,
            stoploss_drawdown=stoploss_drawdown,
            replacement_pool_size=replacement_pool_size,
            threshold=threshold,
            churn_limit=churn_limit,
            margin_stable=margin_stable,
            trading_cost={
                "open_cost": buy_commission_rate,
                "close_cost": sell_commission_rate + sell_stamp_tax_rate,
                "buy_commission_rate": buy_commission_rate,
                "sell_commission_rate": sell_commission_rate,
                "sell_stamp_tax_rate": sell_stamp_tax_rate,
                "min_buy_commission": trading.get("min_buy_commission", 5.0),
                "min_sell_commission": trading.get("min_sell_commission", 5.0),
                **{
                    k: deepcopy(v)
                    for k, v in trading.items()
                    if k
                    not in {
                        "open_cost",
                        "close_cost",
                        "buy_commission_rate",
                        "sell_commission_rate",
                        "sell_stamp_tax_rate",
                        "min_buy_commission",
                        "min_sell_commission",
                    }
                },
            },
            validity=validity,
            config_path=yaml_path,
            hard_filters=selection.get("hard_filters", {}),
            hard_filter_quantiles=selection.get("hard_filter_quantiles", {}),
            industry_leader_field=selection.get("industry_leader_field"),
            industry_leader_top_n=(
                int(selection["industry_leader_top_n"])
                if selection.get("industry_leader_top_n") is not None
                else None
            ),
            composition_components=components,
            cash_weight=cash_weight,
            factor_window_scale=int(cfg.get("factor_window_scale", 1)),
        )

    @classmethod
    def load_metadata(cls, name: str) -> Dict[str, str]:
        """轻量加载：仅读取 name 和 description，不构建 registry"""
        resolved_name, yaml_path = _resolve_strategy_path(name)

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        return {
            "name": resolved_name,
            "display_name": cfg.get("name", Path(resolved_name).name),
            "description": cfg.get("description", ""),
        }

    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用策略名"""
        if not STRATEGIES_DIR.exists():
            return []
        return sorted(_strategy_key_from_path(p) for p in STRATEGIES_DIR.rglob("*.yaml"))

    @classmethod
    def list_grouped(cls) -> Dict[str, List[str]]:
        """按顶层目录分组列出策略。根目录下的策略视为最终 winners。"""
        grouped: Dict[str, List[str]] = defaultdict(list)
        for name in cls.list_available():
            parts = Path(name).parts
            layer = parts[0] if len(parts) > 1 else "winners"
            grouped[layer].append(name)

        layer_order = {"winners": 0, "fixed": 1, "experimental": 2, "research": 3}
        return {
            key: sorted(grouped[key])
            for key in sorted(grouped.keys(), key=lambda x: (layer_order.get(x, 99), x))
        }

    @staticmethod
    def _build_registry(factors_cfg: Dict[str, list]) -> FactorRegistry:
        """从 YAML factors 配置构建 FactorRegistry。

        若 YAML 未定义 factors 段，则克隆 core/factors.py 的 default_registry。
        若有 factors 段，则按 YAML 定义构建（向后兼容），支持省略 expression 以引用默认注册表中的因子。
        """
        if not factors_cfg:
            registry = FactorRegistry()
            for factor in default_registry.all().values():
                registry.register(factor)
            return registry

        registry = FactorRegistry()
        for category, factor_list in factors_cfg.items():
            if not isinstance(factor_list, list):
                continue
            for f_cfg in factor_list:
                name = f_cfg["name"]
                if "expression" not in f_cfg and name in default_registry.all():
                    # 引用默认注册表中的因子，仅覆盖 category
                    base = default_registry.get(name)
                    factor = FactorInfo(
                        name=base.name,
                        expression=base.expression,
                        description=f_cfg.get("description", base.description),
                        category=category,
                        source=base.source,
                        negate=f_cfg.get("negate", base.negate),
                        ir=f_cfg.get("ir", base.ir),
                    )
                else:
                    factor = FactorInfo(
                        name=name,
                        expression=f_cfg["expression"],
                        description=f_cfg.get("description", ""),
                        category=category,
                        source=f_cfg.get("source", "qlib"),
                        negate=f_cfg.get("negate", False),
                        ir=f_cfg.get("ir", 0.0),
                    )
                registry.register(factor)
        return registry

    def build_position_controller(self):
        """根据 position.model 返回对应仓位控制器（或 None）

        Returns:
            - MarketPositionController for model="trend"
            - MarketGatePositionController for model="gate"
            - _FixedPositionController for model="fixed"
            - None for model="full" (100% stock)
        """
        if self.position_model == "trend":
            from core.position import MarketPositionController, MarketConfig

            config_kwargs = {
                key: value
                for key, value in self.position_params.items()
                if key in MarketConfig.__dataclass_fields__
            }
            return MarketPositionController(
                config=MarketConfig(**config_kwargs) if config_kwargs else None
            )
        elif self.position_model == "gate":
            from core.position import MarketGatePositionController, MarketGateConfig

            config_kwargs = {
                key: value
                for key, value in self.position_params.items()
                if key in MarketGateConfig.__dataclass_fields__
            }
            return MarketGatePositionController(
                config=MarketGateConfig(**config_kwargs) if config_kwargs else None
            )
        elif self.position_model == "fixed":
            stock_pct = self.position_params.get("stock_pct", 0.8)
            return _FixedPositionController(stock_pct)
        elif self.position_model == "full":
            return None
        else:
            raise ValueError(f"未知仓位模型: {self.position_model}")

    @property
    def is_composite(self) -> bool:
        return bool(self.composition_components)

    def component_weights(self) -> Dict[str, float]:
        """返回组合策略成员权重。"""
        return {item["strategy"]: float(item["weight"]) for item in self.composition_components}

    def load_component_strategies(
        self,
        stack: Optional[List[str]] = None,
    ) -> List[Tuple["Strategy", float]]:
        """加载组合策略成员，并检测循环引用。"""
        if not self.is_composite:
            return []

        stack = stack or []
        if self.name in stack:
            chain = " -> ".join(stack + [self.name])
            raise ValueError(f"检测到组合策略循环引用: {chain}")

        next_stack = stack + [self.name]
        members: List[Tuple["Strategy", float]] = []
        for item in self.composition_components:
            child = Strategy.load(item["strategy"])
            if child.name in next_stack:
                chain = " -> ".join(next_stack + [child.name])
                raise ValueError(f"检测到组合策略循环引用: {chain}")
            if child.is_composite:
                child.load_component_strategies(stack=next_stack)
            members.append((child, float(item["weight"])))
        return members

    def effective_universe(self) -> str:
        """返回单策略/组合策略的有效股票池标签。"""
        if not self.is_composite:
            return self.universe

        universes = {child.effective_universe() for child, _ in self.load_component_strategies()}
        if len(universes) == 1:
            return next(iter(universes))
        return "mixed"

    def get_rebalance_dates(self, trade_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """按 freq 从交易日历中生成调仓日期"""
        from core.selection import compute_rebalance_dates

        return compute_rebalance_dates(pd.Series(trade_dates), freq=self.rebalance_freq)

    def selections_path(self) -> Path:
        """策略专属选股文件路径"""
        return SELECTIONS_DIR / Path(f"{self.name}.csv")

    def selections_meta_path(self) -> Path:
        """选股缓存元数据路径。"""
        return self.selections_path().with_suffix(".meta.json")

    def artifact_slug(self) -> str:
        """用于结果文件名的安全标识。"""
        return self.name.replace("/", "__")

    def selection_dependency_paths(self) -> List[Path]:
        """返回选股结果依赖的文件列表，用于缓存失效判断"""
        if self.is_composite:
            paths = [
                self.config_path
                if self.config_path is not None
                else _resolve_strategy_path(self.name)[1],
                PROJECT_ROOT / "config" / "strategy.yaml",
            ]
            for child, _ in self.load_component_strategies():
                paths.extend(child.selection_dependency_paths())
            # 保持顺序，去重
            seen = set()
            ordered = []
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                ordered.append(path)
            return ordered
        return [
            self.config_path
            if self.config_path is not None
            else _resolve_strategy_path(self.name)[1],
            PROJECT_ROOT / "config" / "strategy.yaml",
            PROJECT_ROOT / "core" / "factors.py",
            PROJECT_ROOT / "core" / "selection.py",
            PROJECT_ROOT / "core" / "universe.py",
            PROJECT_ROOT / "data" / "tushare" / "index_weight.parquet",
            PROJECT_ROOT / "data" / "tushare" / "index_weight.csv",
            PROJECT_ROOT / "data" / "tushare" / "namechange.parquet",
            PROJECT_ROOT / "data" / "tushare" / "namechange.csv",
        ]

    def selection_cache_metadata(self) -> Dict[str, Any]:
        """返回选股缓存的最小校验元数据。"""
        return {
            "cache_version": SELECTION_CACHE_VERSION,
            "strategy_name": self.name,
            "selection_mode": self.selection_mode,
            "factor_window_scale": self.factor_window_scale,
            "industry_leader_field": self.industry_leader_field,
            "industry_leader_top_n": self.industry_leader_top_n,
        }

    def _write_selection_cache_metadata(self) -> None:
        """写入选股缓存元数据。"""
        meta_path = self.selections_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(self.selection_cache_metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def selections_are_stale(self) -> bool:
        """当依赖文件比选股结果新时，判定缓存过期"""
        csv_path = self.selections_path()
        meta_path = self.selections_meta_path()
        if not csv_path.exists() or not meta_path.exists():
            return True

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return True

        for key, expected in self.selection_cache_metadata().items():
            if meta.get(key) != expected:
                return True

        cache_mtime = min(csv_path.stat().st_mtime, meta_path.stat().st_mtime)
        for dep in self.selection_dependency_paths():
            if dep.exists() and dep.stat().st_mtime > cache_mtime:
                return True
        return False

    def generate_selections(self, force=False, update_start_date=None) -> pd.DataFrame:
        """用本策略的因子/权重/topk 生成选股列表

        Parameters
        ----------
        force : bool
            是否强制重新计算
        update_start_date : str, optional
            增量更新起始日期，只计算此日期之后的选股
        """
        if self.is_composite:
            frames = []
            for child, _ in self.load_component_strategies():
                df = child.generate_selections(
                    force=force, update_start_date=update_start_date
                ).copy()
                if not df.empty:
                    df["strategy"] = child.name
                frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        from core.selection import generate_selections

        self.selections_path().parent.mkdir(parents=True, exist_ok=True)
        df = generate_selections(
            force=force,
            output_path=self.selections_path(),
            update_start_date=update_start_date,
            registry=self.registry,
            weights=self.weights,
            topk=self.topk,
            rebalance_freq=self.rebalance_freq,
            neutralize_industry=self.neutralize_industry,
            universe=self.universe,
            min_market_cap=self.min_market_cap,
            exclude_st=self.exclude_st,
            exclude_new_days=self.exclude_new_days,
            sticky=self.sticky,
            threshold=self.threshold,
            churn_limit=self.churn_limit,
            margin_stable=self.margin_stable,
            buffer=self.buffer,
            selection_mode=self.selection_mode,
            score_smoothing_days=self.score_smoothing_days,
            entry_rank=self.entry_rank,
            exit_rank=self.exit_rank,
            entry_persist_days=self.entry_persist_days,
            exit_persist_days=self.exit_persist_days,
            min_hold_days=self.min_hold_days,
            stoploss_lookback_days=self.stoploss_lookback_days,
            stoploss_drawdown=self.stoploss_drawdown,
            replacement_pool_size=self.replacement_pool_size,
            factor_window_scale=self.factor_window_scale,
            hard_filters=self.hard_filters if self.hard_filters else None,
            hard_filter_quantiles=self.hard_filter_quantiles
            if self.hard_filter_quantiles
            else None,
            industry_leader_field=self.industry_leader_field,
            industry_leader_top_n=self.industry_leader_top_n,
        )
        self._write_selection_cache_metadata()
        return df

    def validate_data_requirements(self):
        """正式生成选股/回测前的数据预检。"""
        if self.is_composite:
            results = []
            for child, _ in self.load_component_strategies():
                results.append(child.validate_data_requirements())
            return results
        from modules.data.precheck import ensure_strategy_data_ready

        return ensure_strategy_data_ready(self)

    def load_selections(self):
        """加载本策略的选股列表

        Returns:
            date_to_symbols: {Timestamp: set of symbols}
            rebalance_dates: set of Timestamps
        """
        if self.is_composite:
            raise ValueError("组合策略不直接提供单一选股列表，请走组合回测入口。")
        csv_path = self.selections_path()
        if self.selections_are_stale():
            print(f"[INFO] 策略 {self.name} 选股列表缺失或已过期，正在重新生成...")
            self.generate_selections(force=True)
        from core.selection import load_selections

        return load_selections(csv_path=csv_path)

    def evaluate_validity(self, daily_returns: pd.Series):
        """基于策略 validity 配置评估最近一段时间是否仍然有效。"""
        if self.validity is None:
            return None
        from core.validity import evaluate_strategy_validity

        return evaluate_strategy_validity(daily_returns, self.validity)


class _FixedPositionController:
    """固定仓位控制器"""

    def __init__(self, stock_pct: float = 0.8):
        self.stock_pct = stock_pct

    def load_market_data(self):
        """无需加载市场数据"""
        pass

    def get_allocation(self, date, is_rebalance_day=False):
        from core.position import AllocationResult

        return AllocationResult(
            stock_pct=self.stock_pct,
            cash_pct=round(1 - self.stock_pct, 4),
            regime="fixed",
            opportunity_level="none",
            market_drawdown=0.0,
            trend_score=0.0,
        )

    def get_bond_daily_return(self) -> float:
        return 0.03 / 252
