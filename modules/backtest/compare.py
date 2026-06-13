"""Backward-compat shim → app.compare（五层迁移）。"""
import sys as _sys
import app.compare as _impl

_sys.modules[__name__] = _impl
