"""Backward-compat shim → data.provider（五层迁移）。

实体已迁至 data.provider。sys.modules 别名让旧路径与新模块指向同一对象。
"""
import sys as _sys
import data.provider as _impl

_sys.modules[__name__] = _impl
