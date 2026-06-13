"""Backward-compat shim → engine/qlib_engine.py（五层迁移）。"""
import sys as _sys
import engine.qlib_engine as _impl

_sys.modules[__name__] = _impl
