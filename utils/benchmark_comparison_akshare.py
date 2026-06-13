"""Backward-compat shim → app.report.benchmark_akshare（五层迁移）。"""
import sys as _sys
import app.report.benchmark_akshare as _impl

_sys.modules[__name__] = _impl
