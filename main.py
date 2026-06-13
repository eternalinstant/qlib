#!/usr/bin/env python3
"""入口 shim → app/cli.py（五层迁移）。

CLI 实体已迁至 app/cli.py。用 sys.modules 别名让 `import main` 拿到 app.cli 的
全部符号（_load_strategy / cmd_* 及其 patch 目标），并保留 `python main.py <cmd>` 入口。
"""
import sys as _sys
import app.cli as _cli

_sys.modules[__name__] = _cli

if __name__ == "__main__":
    _cli.main()
