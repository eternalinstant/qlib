"""
回测结果缓存模块
按 (strategy_name, engine, start_date, end_date, yaml_hash) 缓存回测结果
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd

from config.config import CONFIG


class BacktestCache:
    """回测结果缓存器"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = CONFIG.get("paths.cache", "~/code/qlib/data/cache")
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        strategy_name: str,
        engine: str,
        start_date: str,
        end_date: str,
        yaml_content: str,
    ) -> str:
        """生成缓存键"""
        yaml_hash = hashlib.md5(yaml_content.encode()).hexdigest()[:8]
        key = f"{strategy_name}_{engine}_{start_date}_{end_date}_{yaml_hash}"
        return key

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self,
        strategy_name: str,
        engine: str,
        start_date: str,
        end_date: str,
        yaml_content: str,
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的回测结果"""
        cache_key = self._get_cache_key(
            strategy_name, engine, start_date, end_date, yaml_content
        )
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            import pickle
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(
        self,
        strategy_name: str,
        engine: str,
        start_date: str,
        end_date: str,
        yaml_content: str,
        result: Dict[str, Any],
    ) -> None:
        """保存回测结果到缓存"""
        cache_key = self._get_cache_key(
            strategy_name, engine, start_date, end_date, yaml_content
        )
        cache_path = self._get_cache_path(cache_key)

        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass

    def clear(self, strategy_name: str = None) -> None:
        """清除缓存"""
        if strategy_name:
            pattern = f"{strategy_name}_*"
        else:
            pattern = "*.pkl"

        for p in self.cache_dir.glob(pattern):
            p.unlink()


_default_cache: Optional[BacktestCache] = None


def get_cache() -> BacktestCache:
    """获取默认缓存实例"""
    global _default_cache
    if _default_cache is None:
        _default_cache = BacktestCache()
    return _default_cache
