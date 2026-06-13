"""Backward-compat shim → engine/pybroker_engine.py（五层迁移）。"""
import sys as _sys
import engine.pybroker_engine as _impl

_sys.modules[__name__] = _impl
