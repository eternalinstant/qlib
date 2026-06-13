"""Backward-compat shim → engine/rule_engine.py（五层迁移）。"""
import sys as _sys
import engine.rule_engine as _impl

_sys.modules[__name__] = _impl
