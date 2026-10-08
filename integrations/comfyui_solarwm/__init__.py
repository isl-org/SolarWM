"""SolarWM interactive ComfyUI custom node."""

from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .factory import build_provider_inputs
from .manager import configure_provider_factory
from .server import register_routes

WEB_DIRECTORY = "./web"

configure_provider_factory(build_provider_inputs)
try:
    register_routes()
except Exception:
    # ComfyUI may import custom nodes while its PromptServer is still being
    # constructed.  The node remains usable; route registration retries when
    # the package is explicitly loaded by the extension entry point.
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
