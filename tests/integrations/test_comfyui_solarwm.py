from __future__ import annotations

from queue import Queue

import numpy as np

from integrations.comfyui_solarwm.manager import SessionState
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera


class _Runtime:
    done = False
    latent_index = 0
    emitted_pixel_frames = 0
    provider = None

    def close(self):
        pass


def test_session_controls_are_monotonic_and_focus_safe():
    session = SessionState(
        "test",
        _Runtime(),
        FreeFlyCamera(),
        object(),
        events=Queue(),
    )
    session.control({"sequence": 3, "pressed": ["KeyW"]})
    session.control({"sequence": 2, "pressed": ["KeyS"]})
    assert session.controls == {"KeyW"}
    assert session.control_sequence == 3
    session.control({"sequence": 4, "action": "brake", "pressed": []})
    assert np.allclose(session.camera._velocity, 0.0)
    session.control({"sequence": 5, "action": "drop", "pressed": ["KeyA"]})
    assert not session.controls
    assert np.allclose(session.camera._velocity, 0.0)
