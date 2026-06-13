"""Backward-compat shim → engine/base.py（五层迁移）。

实体已迁至 engine/base.py。用 sys.modules 别名让旧路径
`modules.backtest.base` 与新模块指向同一对象（import 与 patch 均生效）。
"""
import sys as _sys
import engine.base as _impl

_sys.modules[__name__] = _impl
