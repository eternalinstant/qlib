"""Backward-compat shim → app.leaderboard（五层迁移）。"""
import sys as _sys
import app.leaderboard as _impl

_sys.modules[__name__] = _impl
