"""Backward-compat shim → app.report.compare_plot（五层迁移）。"""
import sys as _sys
import app.report.compare_plot as _impl

_sys.modules[__name__] = _impl
