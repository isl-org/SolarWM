/* Keyboard-only SolarWM viewer. Loaded by ComfyUI's custom-node web loader. */

const SOLARWM_CODES = new Set([
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
  "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
  "KeyZ", "KeyC", "KeyX", "ShiftLeft", "ShiftRight", "KeyV",
]);

function makeSolarWMViewer(sessionId, root, sessionConfig = null, onSession = null) {
  const image = document.createElement("img");
  const status = document.createElement("output");
  const help = document.createElement("pre");
  help.textContent =
    "WASD move · QE vertical · arrows look · ZC roll · X level · " +
    "Shift boost · V precision · G pause/resume · Enter step · Space brake · " +
    "K drop input · Esc clear";
  help.hidden = true;
  root.append(image, status);
  root.append(help);
  root.tabIndex = 0;
  root.setAttribute("aria-label", "SolarWM keyboard camera viewer");
  const pressed = new Set();
  let sequence = 0;
  let socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}` +
      `/solarwm/sessions/${encodeURIComponent(sessionId)}/stream`,
  );
  socket.binaryType = "blob";
  let running = true;

  function send(action = null) {
    if (socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({
      sequence: sequence++,
      pressed: [...pressed],
      ...(action ? { action } : {}),
      sent_at: performance.now(),
      sent_at_epoch: Date.now() / 1000,
    }));
  }

  function clearKeys() {
    pressed.clear();
    send();
  }

  root.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.code === "Space") {
      event.preventDefault();
      send("brake");
      return;
    }
    if (event.code === "KeyG") {
      event.preventDefault();
      running = !running;
      send(running ? "resume" : "pause");
      return;
    }
    if (event.code === "Enter") {
      event.preventDefault();
      send("step");
      return;
    }
    if (event.code === "KeyK") {
      event.preventDefault();
      send("drop");
      return;
    }
    if (event.code === "Escape") {
      event.preventDefault();
      clearKeys();
      root.blur();
      return;
    }
    if (event.code === "Slash" && event.shiftKey) {
      event.preventDefault();
      help.hidden = !help.hidden;
      return;
    }
    if (event.code === "Home" && event.shiftKey && sessionConfig) {
      event.preventDefault();
      if (!window.confirm("Start a new SolarWM session from the initial pose?")) return;
      fetch("/solarwm/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sessionConfig),
      }).then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "restart failed");
        socket.close();
        root.replaceChildren();
        const nextViewer = makeSolarWMViewer(
          payload.session_id, root, sessionConfig, onSession,
        );
        if (onSession) onSession(payload.session_id, nextViewer);
        root.focus();
      }).catch((error) => { status.value = error.message; });
      return;
    }
    if (event.code === "KeyX") {
      event.preventDefault();
      send("level_roll");
      return;
    }
    if (event.code === "Minus") {
      event.preventDefault();
      send("speed_down");
      return;
    }
    if (event.code === "Equal") {
      event.preventDefault();
      send("speed_up");
      return;
    }
    if (event.code === "BracketLeft") {
      event.preventDefault();
      send("look_down");
      return;
    }
    if (event.code === "BracketRight") {
      event.preventDefault();
      send("look_up");
      return;
    }
    if (event.code === "Digit0") {
      event.preventDefault();
      send("defaults");
      return;
    }
    if (SOLARWM_CODES.has(event.code)) {
      event.preventDefault();
      pressed.add(event.code);
      send();
    }
  });
  root.addEventListener("keyup", (event) => {
    if (SOLARWM_CODES.has(event.code)) {
      event.preventDefault();
      pressed.delete(event.code);
      send();
    }
  });
  root.addEventListener("blur", clearKeys);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearKeys();
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      const value = JSON.parse(event.data);
      if (value.type === "frame") {
        status.value =
          `latent ${value.latent_index} · ${value.latency_seconds.toFixed(2)} s`;
        if (value.control_sent_at) {
          status.value +=
            ` · input→frame ${(Date.now() / 1000 - value.control_sent_at).toFixed(2)} s`;
        }
      } else if (value.type === "error") {
        status.value = value.error;
      }
      return;
    }
    const url = URL.createObjectURL(event.data);
    const previous = image.src;
    image.src = url;
    image.onload = () => {
      if (previous) URL.revokeObjectURL(previous);
    };
  });
  socket.addEventListener("close", clearKeys);
  return { close: () => socket.close(), clearKeys };
}

export { makeSolarWMViewer };
