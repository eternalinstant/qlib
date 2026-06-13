"""Backward-compat shim → app.composite_runner（五层迁移）。"""
import sys as _sys
import app.composite_runner as _impl

_sys.modules[__name__] = _impl
