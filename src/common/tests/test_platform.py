"""TDD: common/platform.py — 跨平台路径/OS 工具（从 utils/platform.py 下沉到公共库）"""
from pathlib import Path
from common.platform import (
    resolve_path,
    project_root,
    system_name,
    is_macos,
    is_linux,
    is_windows,
    temp_dir,
    safe_cpu_count,
)


def test_project_root_finds_repo_root():
    """common/platform.py 在 repo 根下一层，project_root 应仍定位到含 pytest.ini 的根"""
    root = project_root()
    assert (root / "pytest.ini").exists() or (root / ".git").exists()


def test_resolve_absolute_path_unchanged():
    assert resolve_path("/tmp/x") == Path("/tmp/x")


def test_resolve_relative_path_anchored_to_root():
    p = resolve_path("data/foo")
    assert p.is_absolute()
    assert str(p).endswith("data/foo")


def test_system_name_is_one_of_known():
    assert system_name() in {"Darwin", "Linux", "Windows"}


def test_os_predicates_mutually_consistent():
    assert sum([is_macos(), is_linux(), is_windows()]) <= 1


def test_safe_cpu_count_positive():
    assert safe_cpu_count() >= 1


def test_temp_dir_created():
    d = temp_dir("qlib_test_common")
    assert d.exists() and d.is_dir()
