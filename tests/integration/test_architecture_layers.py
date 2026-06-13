"""架构守卫：五层(src-layout)迁移后，全仓不得再引用旧路径。

代码层在 src/{app,strategy,engine,data,common}（import 仍为 from strategy.x，src 在 path）。
禁止任何 from/import 指向旧的 core / modules.* / config.config / utils / strategies。
"""
import ast
import pathlib

FORBIDDEN = [
    ("core",),
    ("modules", "backtest"),
    ("modules", "modeling"),
    ("modules", "data"),
    ("modules", "risk"),
    ("config", "config"),
    ("utils",),
    ("strategies",),
]

SCAN_DIRS = ["src", "scripts", "tests"]

# 上溯到含 pytest.ini 的仓库根（深度无关）
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pytest.ini").exists())


def _is_forbidden(module: str) -> bool:
    if not module:
        return False
    parts = module.split(".")
    return any(parts[: len(pref)] == list(pref) for pref in FORBIDDEN)


def _iter_py_files():
    seen = set()
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            seen.add(p)
            yield p
    for p in ROOT.glob("*.py"):
        if p not in seen:
            yield p


def _legacy_imports(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_forbidden(node.module):
                bad.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    bad.append(alias.name)
    return bad


def test_no_legacy_layer_imports():
    violations = {}
    for py in _iter_py_files():
        if py.name == "test_architecture_layers.py":
            continue
        bad = _legacy_imports(py)
        if bad:
            violations[str(py.relative_to(ROOT))] = sorted(set(bad))
    assert not violations, (
        f"{len(violations)} 个文件仍引用旧路径（应为 src/ 下的 app/strategy/engine/data/common）:\n"
        + "\n".join(f"  {f}: {mods}" for f, mods in sorted(violations.items())[:30])
    )
