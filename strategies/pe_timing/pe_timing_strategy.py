"""Backward-compat shim → strategy.producers.pe_timing_strategy（五层迁移）。

实体已迁至 strategy.producers.pe_timing_strategy。sys.modules 别名让旧路径与新模块指向同一对象
（含 default_registry 等单例、YAML strategy_class 动态 import）。
"""
import sys as _sys
import strategy.producers.pe_timing_strategy as _impl

_sys.modules[__name__] = _impl
