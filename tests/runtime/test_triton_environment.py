from types import SimpleNamespace

from solarwm.runtime import triton_environment


def test_non_windows_does_not_change_environment(monkeypatch):
    monkeypatch.setattr(triton_environment, "os", SimpleNamespace(name="posix"))
    triton_environment.configure()


def test_windows_sdk_and_existing_compiler(monkeypatch, tmp_path):
    sdk = tmp_path / "level-zero-sdk" / "1.34.0"
    header = sdk / "include" / "level_zero" / "ze_api.h"
    header.parent.mkdir(parents=True)
    header.touch()
    monkeypatch.setattr(triton_environment.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        triton_environment, "os",
        SimpleNamespace(name="nt", environ={}),
    )
    monkeypatch.setattr(triton_environment.shutil, "which", lambda name: "compiler")
    triton_environment.configure()
    assert triton_environment.os.environ["LEVEL_ZERO_V1_SDK_PATH"] == str(sdk)


def test_windows_missing_sdk_warns(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(triton_environment.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(triton_environment, "os", SimpleNamespace(name="nt", environ={}))
    with pytest.warns(RuntimeWarning, match="SDK is missing"):
        triton_environment.configure()
