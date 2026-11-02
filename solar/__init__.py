"""SolarWM public package."""

import sys

from .version import __version__

# Compatibility alias: allow 'import solarwm' and 'from solarwm... import ...'
sys.modules.setdefault("solarwm", sys.modules[__name__])

__all__ = ["__version__"]
