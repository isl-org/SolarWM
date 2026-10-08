"""ComfyUI HTTP/WebSocket routes for interactive SolarWM sessions."""

from __future__ import annotations

import asyncio
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
            session = manager.create(await request.json())
        except Exception as exc:
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=400,
            )
        return web.json_response({"session_id": session.session_id})

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
            }
        )

    @routes.delete("/solarwm/sessions/{session_id}")
    async def delete_session(request: web.Request) -> web.Response:
        manager.delete(request.match_info["session_id"])
        return web.json_response({"ok": True})

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
