"""ComfyUI HTTP/WebSocket routes for interactive SolarWM sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .manager import manager

_REGISTERED = False


def register_routes() -> None:
    """Register routes when imported inside a running ComfyUI server."""

    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    prompt_server = getattr(PromptServer, "instance", None)
    routes = getattr(prompt_server, "routes", None)
    if routes is None:
        return

    @routes.post("/solarwm/sessions")
    async def create_session(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            session = await asyncio.to_thread(manager.create, payload)
        except Exception as exc:
            logging.exception("SolarWM session creation failed")
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=400,
            )
        return web.json_response({"session_id": session.session_id})

    @routes.get("/solarwm/image-metadata")
    async def image_metadata(request: web.Request) -> web.Response:
        from .factory import load_dataset_camera_metadata

        image_path = request.query.get("path", "")
        if not image_path:
            return web.json_response({"error": "missing image path"}, status=400)
        try:
            metadata = await asyncio.to_thread(
                load_dataset_camera_metadata, image_path
            )
        except Exception as exc:
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=400,
            )
        if metadata is None:
            return web.json_response(
                {"prompt": "", "has_trajectory": False},
            )
        return web.json_response(
            {
                "prompt": metadata["prompt"],
                "has_trajectory": True,
                "trajectory_fps": metadata["fps"],
                "sample_id": metadata["sample_id"],
            }
        )

    @routes.get("/solarwm/sessions/{session_id}")
    async def session_status(request: web.Request) -> web.Response:
        try:
            session = manager.get(request.match_info["session_id"])
        except KeyError:
            return web.json_response({"error": "unknown session"}, status=404)
        return web.json_response(
            {
                "session_id": session.session_id,
                "running": session.running,
                "latent_index": session.runtime.latent_index,
                "pixel_frames": session.runtime.emitted_pixel_frames,
                "done": session.runtime.done,
                "video_path": (
                    str(session.recording_path)
                    if session.recording_path is not None
                    else None
                ),
            }
        )

    @routes.delete("/solarwm/sessions/{session_id}")
    async def delete_session(request: web.Request) -> web.Response:
        manager.delete(request.match_info["session_id"])
        return web.json_response({"ok": True})

    @routes.get("/solarwm/model/status")
    async def model_status(request: web.Request) -> web.Response:
        from .factory import is_provider_cached

        return web.json_response({"cached": is_provider_cached()})

    @routes.post("/solarwm/model/unload")
    async def unload_model(request: web.Request) -> web.Response:
        from .factory import clear_cached_provider

        with manager._lock:
            active_ids = list(manager._sessions.keys())
        for sid in active_ids:
            manager.delete(sid)
        await asyncio.to_thread(clear_cached_provider)
        return web.json_response({"ok": True, "message": "Model unloaded from VRAM"})

    @routes.get("/solarwm/sessions/{session_id}/stream")
    async def stream_session(request: web.Request) -> web.WebSocketResponse:
        try:
            session = manager.get(request.match_info["session_id"])
        except KeyError:
            return web.json_response({"error": "unknown session"}, status=404)
        websocket = web.WebSocketResponse(heartbeat=20)
        await websocket.prepare(request)
        try:
            while not websocket.closed:
                try:
                    message = await asyncio.to_thread(session.events.get, True, 0.05)
                except Exception:
                    message = None
                if isinstance(message, dict):
                    await websocket.send_json(message)
                elif isinstance(message, tuple) and message[0] == "jpeg":
                    await websocket.send_bytes(message[1])
                try:
                    incoming = await websocket.receive(timeout=0.01)
                except asyncio.TimeoutError:
                    continue
                if incoming.type == web.WSMsgType.TEXT:
                    try:
                        session.control(await _json_message(incoming.data))
                    except (TypeError, ValueError):
                        await websocket.send_json(
                            {"type": "error", "error": "invalid control message"}
                        )
                elif incoming.type in {
                    web.WSMsgType.CLOSE,
                    web.WSMsgType.CLOSED,
                    web.WSMsgType.ERROR,
                }:
                    break
        finally:
            session.control({"sequence": session.control_sequence + 1, "action": "stop"})
        return websocket

    _REGISTERED = True


async def _json_message(value: str) -> dict[str, Any]:
    import json

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("control message must be an object")
    return parsed


__all__ = ["register_routes"]
