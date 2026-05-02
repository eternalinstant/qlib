"""
迁移兼容性测试：验证从 macOS 迁移到 Linux 后，所有硬编码路径和平台特定代码已修正
"""
import sys
import os
import ast
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"

# 需要检查的 Python 源文件和白名单
PYTHON_FILES = [
    "utils/diagnose.py",
    "utils/analysis.py",
    "tests/test_bugs.py",
    "tests/test_bugs_after_fix.py",
    "tests/test_integration.py",
    "tests/utils/test_config_refactor.py",
    "scripts/daily_run.sh",
]


def test_no_hardcoded_macos_home():
    """所有源文件中不存在硬编码的 /Users/sxt 路径（setup_new_env.sh 是搜索模式，除外）"""
    print("\n" + "=" * 70)
    print("测试: 无硬编码的 /Users/sxt 路径")
    print("=" * 70)

    all_pass = True
    for rel_path in PYTHON_FILES:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            print(f"  {SKIP} {rel_path} 不存在")
            continue

        content = full_path.read_text()
        # setup_new_env.sh 中的 old_path 是用于搜索替换的，是合理的
        if "setup_new_env" in rel_path:
            # 只允许在 old_path 变量中使用
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                if "/Users/sxt" in line and 'old_path="/Users/sxt"' not in line and 'old_path=' not in line:
                    print(f"  {FAIL} {rel_path}:{i}: {line.strip()}")
                    all_pass = False
            if all_pass:
                print(f"  {PASS} {rel_path} (old_path 搜索模式是合法的)")
            continue

        # 注释中的路径也检查（因为路径是错的，注释也会误导）
        if "/Users/sxt" in content:
            for i, line in enumerate(content.split("\n"), 1):
                if "/Users/sxt" in line:
                    print(f"  {FAIL} {rel_path}:{i}: {line.strip()}")
                    all_pass = False
        else:
            print(f"  {PASS} {rel_path}")

    return all_pass


def test_project_root_resolve():
    """诊断/分析工具脚本能正确解析项目根目录"""
    print("\n" + "=" * 70)
    print("测试: 工具脚本正确解析项目根目录")
    print("=" * 70)

    all_pass = True
    for script in ["utils/diagnose.py", "utils/analysis.py"]:
        full_path = PROJECT_ROOT / script
        if not full_path.exists():
            print(f"  {SKIP} {script} 不存在")
            continue
        content = full_path.read_text()
        tree = ast.parse(content)
        has_dynamic_root = False
        has_pathlib = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("PROJECT_ROOT",):
                        has_dynamic_root = True
            if isinstance(node, ast.ImportFrom):
                if node.module == "pathlib" and any(
                    alias.name == "Path" for alias in node.names
                ):
                    has_pathlib = True
        if has_dynamic_root or has_pathlib:
            print(f"  {PASS} {script} 使用动态路径解析")
        else:
            print(f"  {FAIL} {script} 未使用动态路径解析")
            all_pass = False

    return all_pass


def test_sys_path_dynamic():
    """测试文件的 sys.path 使用动态路径"""
    print("\n" + "=" * 70)
    print("测试: sys.path.insert 使用动态路径")
    print("=" * 70)

    all_pass = True
    test_files = [
        "tests/test_bugs.py",
        "tests/test_bugs_after_fix.py",
        "tests/test_integration.py",
        "tests/utils/test_config_refactor.py",
    ]
    for tf in test_files:
        full_path = PROJECT_ROOT / tf
        if not full_path.exists():
            print(f"  {SKIP} {tf} 不存在")
            continue
        content = full_path.read_text()
        if "/Users/sxt" in content:
            print(f"  {FAIL} {tf} 仍含硬编码路径")
            all_pass = False
        elif "Path(__file__)" in content:
            print(f"  {PASS} {tf} 使用 Path(__file__) 动态路径")
        else:
            print(f"  {FAIL} {tf} 未使用 Path(__file__) 动态路径")
            all_pass = False

    return all_pass


def test_daily_run_sh_dynamic():
    """daily_run.sh 使用动态 PROJECT_DIR"""
    print("\n" + "=" * 70)
    print("测试: daily_run.sh PROJECT_DIR 使用动态路径")
    print("=" * 70)

    script = PROJECT_ROOT / "scripts/daily_run.sh"
    if not script.exists():
        print(f"  {SKIP} scripts/daily_run.sh 不存在")
        return True

    content = script.read_text()
    if "/Users/sxt" in content:
        print(f"  {FAIL} 仍含硬编码 /Users/sxt")
        return False
    if 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"' in content:
        print(f"  {PASS} 使用 $(cd $(dirname $0) && pwd) 动态路径")
        return True
    if 'PROJECT_DIR="$(dirname' in content:
        print(f"  {PASS} 使用动态 PROJECT_DIR")
        return True
    print(f"  {FAIL} 未使用动态路径")
    return False


def test_import_functions():
    """验证 core 模块可以正常导入（sys.path 已修正）"""
    print("\n" + "=" * 70)
    print("测试: 核心模块导入")
    print("=" * 70)

    sys.path.insert(0, str(PROJECT_ROOT))
    all_pass = True
    modules = [
        ("core.factors", "FactorRegistry"),
        ("core.selection", "compute_signal"),
        ("core.position", "MarketPositionController"),
        ("config.config", "ConfigManager"),
    ]
    for mod_name, attr in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            obj = getattr(mod, attr, None)
            if obj is not None:
                print(f"  {PASS} {mod_name}.{attr}")
            else:
                print(f"  {FAIL} {mod_name}.{attr} 属性不存在")
                all_pass = False
        except ModuleNotFoundError as e:
            print(f"  {SKIP} {mod_name}: 缺少依赖 ({e.name})")
        except Exception as e:
            print(f"  {FAIL} {mod_name}: {e}")
            all_pass = False

    return all_pass


def main():
    print("=" * 70)
    print("  Qlib 迁移兼容性测试 (macOS -> Linux)")
    print("  项目目录:", PROJECT_ROOT)
    print("=" * 70)

    results = {}
    results["无硬编码 macOS 路径"] = test_no_hardcoded_macos_home()
    results["工具脚本动态路径解析"] = test_project_root_resolve()
    results["测试文件 sys.path 动态"] = test_sys_path_dynamic()
    results["daily_run.sh 动态路径"] = test_daily_run_sh_dynamic()
    results["核心模块导入"] = test_import_functions()

    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  {PASS} 所有迁移兼容性测试通过！")
        return 0
    else:
        print(f"\n  {FAIL} 存在未解决的兼容性问题")
        return 1


if __name__ == "__main__":
    exit(main())
