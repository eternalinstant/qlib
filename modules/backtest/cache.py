"""Backward-compat shim → engine/cache.py（五层迁移）。"""
import sys as _sys
import engine.cache as _impl

_sys.modules[__name__] = _impl
