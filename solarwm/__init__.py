"""Backward-compatibility shim redirecting solarwm imports to solar."""

from __future__ import annotations

import sys
import solar

sys.modules["solarwm"] = solar
__path__ = list(solar.__path__)
