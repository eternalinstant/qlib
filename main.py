#!/usr/bin/env python3
"""入口 → src/app/cli.py（src-layout）。"""
import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent
for _p in (str(_root / "src"), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import app.cli as _cli
sys.modules[__name__] = _cli
if __name__ == "__main__":
    _cli.main()
