"""Backward-compat shim → app.factor_mining（五层迁移）。"""
import sys as _sys
import app.factor_mining as _impl

_sys.modules[__name__] = _impl
