"""Backward-compat shim → app.report.quantstats_report（五层迁移）。"""
import sys as _sys
import app.report.quantstats_report as _impl

_sys.modules[__name__] = _impl
