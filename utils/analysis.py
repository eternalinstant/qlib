"""Backward-compat shim → app.report.analysis（五层迁移）。"""
import sys as _sys
import app.report.analysis as _impl

_sys.modules[__name__] = _impl
