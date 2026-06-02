import os

from core import qlib_init


def test_windows_runtime_overrides_disable_fork_and_parallel_processes(monkeypatch):
    monkeypatch.setattr(qlib_init.platform, "system", lambda: "Windows")
    monkeypatch.setenv("JOBLIB_START_METHOD", "fork")

    overrides = qlib_init._qlib_runtime_overrides()

    assert os.environ["JOBLIB_START_METHOD"] == "spawn"
    assert overrides["kernels"] == 1
    assert overrides["joblib_backend"] == "threading"
