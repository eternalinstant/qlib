"""
配置加载模块
统一管理所有配置
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(filename: str, data: Dict[str, Any]):
    """保存 YAML 配置文件"""
    path = CONFIG_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


@dataclass(frozen=True)
class AppConfig:
    """不可变配置对象"""
    _data: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @property
    def strategy(self) -> Dict[str, Any]:
        return self._data.get("strategy", {})

    @property
    def trading(self) -> Dict[str, Any]:
        return self._data.get("trading", {})

    @property
    def paths(self) -> Dict[str, Any]:
        return self._data.get("paths", {})


class ConfigManager:
    """配置管理器（非单例，支持依赖注入）"""

    def __init__(self, env: str = "default", strategy_file: str = "strategy.yaml"):
        self._env = env
        self._strategy_file = strategy_file
        self._data: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self):
        """加载所有配置文件"""
        self._data["strategy"] = load_yaml(self._strategy_file)
        self._data["trading"] = load_yaml("trading.yaml")
        self._data["paths"] = load_yaml("paths.yaml")
        self._expand_paths()
        self._apply_env_overrides()
        self._flatten_keys()

    def _flatten_keys(self):
        """扁平化配置键，支持向后兼容"""
        strategy = self._data.get("strategy", {})
        trading = self._data.get("trading", {})
        paths = self._data.get("paths", {})

        weights = strategy.get("weights", {})
        self._data["w_alpha"] = weights.get("alpha", 0.55)
        self._data["w_risk"] = weights.get("risk", 0.20)
        self._data["w_enhance"] = weights.get("enhance", 0.25)

        selection = strategy.get("selection", {})
        self._data["topk"] = selection.get("topk", 20)
        self._data["min_market_cap"] = selection.get("min_market_cap", 50)

        # 读取持仓稳定性配置（用下划线避免与嵌套dict冲突）
        stability = strategy.get("stability", {})
        self._data["stability_sticky"] = stability.get("sticky", 0)
        self._data["stability_threshold"] = stability.get("threshold", 0.0)
        self._data["stability_churn_limit"] = stability.get("churn_limit", 0)
        self._data["stability_margin_stable"] = stability.get("margin_stable", False)

        period = strategy.get("backtest_period", {})
        self._data["start_date"] = period.get("start_date", "2019-01-01")
        from datetime import date
        end_date = period.get("end_date", "auto")
        self._data["end_date"] = date.today().isoformat() if end_date == "auto" else end_date

        capital = trading.get("capital", {})
        self._data["initial_capital"] = capital.get("initial", 500000)

        cost = trading.get("cost", {})
        self._data["open_cost"] = cost.get("open_cost", 0.0003)
        self._data["close_cost"] = cost.get("close_cost", 0.0013)
        self._data["min_cost"] = cost.get("min_cost", 5)

        self._data["qlib_data_path"] = paths.get("data", {}).get("qlib_data", "")

    def _expand_paths(self):
        """展开路径中的 ~ 和环境变量（递归处理嵌套字典）"""
        def expand_dict(d):
            if not isinstance(d, dict):
                return
            for key, value in list(d.items()):
                if isinstance(value, str) and ("~" in value or "$" in value):
                    d[key] = os.path.expanduser(os.path.expandvars(value))
                elif isinstance(value, dict):
                    expand_dict(value)

        for config in self._data.values():
            expand_dict(config)

    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_prefix = "QLIB_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                self._data.setdefault("env", {})[config_key] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    def get_config(self) -> AppConfig:
        """返回不可变配置对象"""
        return AppConfig(self._data.copy())

    def reload(self):
        """重新加载配置"""
        self._load_all()


_default_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取默认配置管理器（全局单例，供兼容使用）"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConfigManager()
    return _default_manager


def load_config() -> AppConfig:
    """加载配置（便捷函数）"""
    return get_config_manager().get_config()


config = load_config()

CONFIG = config
