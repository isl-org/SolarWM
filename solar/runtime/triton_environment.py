"""Configure the installed Windows XPU development environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings


def configure() -> None:
    if os.name != "nt":
        return
    sdk = Path(sys.prefix) / "level-zero-sdk" / "1.34.0"
    if not (sdk / "include" / "level_zero" / "ze_api.h").is_file():
        warnings.warn(f"SolarWM Level Zero SDK is missing: {sdk}", RuntimeWarning)
        return
    os.environ.setdefault("LEVEL_ZERO_V1_SDK_PATH", str(sdk))
    if os.environ.get("CXX") or shutil.which("icpx") or shutil.which("cl"):
        return
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        warnings.warn("SolarWM Triton compilation needs MSVC C++ Build Tools.", RuntimeWarning)
        return
    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        check=True, capture_output=True, text=True,
    )
    installation = result.stdout.strip()
    if not installation:
        warnings.warn("SolarWM could not locate the MSVC C++ toolset.", RuntimeWarning)
        return
    vcvars = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    environment = dict(os.environ)
    environment = {
        key: value for key, value in environment.items()
        if key.upper() != "NODEFAULTCURRENTDIRECTORYINEXEPATH"
    }
    comspec = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
    result = subprocess.run(
        f'"{comspec}" /d /s /c "call "{vcvars}" >nul && set"',
        check=True, capture_output=True, text=True, env=environment,
    )
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.startswith("="):
            os.environ[key] = value
    os.environ.setdefault("CXX", "cl")
