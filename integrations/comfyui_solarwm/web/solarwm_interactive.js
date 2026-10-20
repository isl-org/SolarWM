/* Keyboard-only SolarWM viewer. Loaded by ComfyUI's custom-node web loader. */
import { api } from "../../../scripts/api.js";

const SOLARWM_CODES = new Set([
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
  "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
  "KeyZ", "KeyC", "KeyX", "ShiftLeft", "ShiftRight", "KeyV",
]);

function makeSolarWMViewer(sessionId, root, sessionConfig = null, onSession = null) {
  const configuredScale = ["1x", "2x", "fit"].includes(sessionConfig?.display_scale)
    ? sessionConfig.display_scale
    : "2x";
  const configuredFps = [4, 8, 12, 16, 24, 30].includes(
    Number(sessionConfig?.display_fps),
  )
    ? Number(sessionConfig.display_fps)
    : 8;
  const toolbar = document.createElement("div");
  toolbar.style.cssText =
    "display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;";

  const leftControls = document.createElement("div");
  leftControls.style.cssText = "display:flex;align-items:center;gap:8px;";

  const scaleLabel = document.createElement("span");
  scaleLabel.textContent = "Scale:";
  scaleLabel.style.cssText = "font-size:12px;color:#a1a1aa;font-weight:600;";

  function makeScaleBtn(label) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText =
      "background:#27272a;color:#d4d4d8;border:1px solid #3f3f46;padding:4px 10px;" +
      "border-radius:4px;cursor:pointer;font-size:12px;font-weight:600;";
    return btn;
  }

  const btn1x = makeScaleBtn("1x");
  const btn2x = makeScaleBtn("2x");
  const btnFit = makeScaleBtn("Fit");
  const displayLabel = document.createElement("span");
  displayLabel.textContent = "Display:";
  displayLabel.style.cssText =
    "font-size:12px;color:#a1a1aa;font-weight:600;margin-left:8px;";
  const displayFps = document.createElement("select");
  displayFps.title = "Playback rate for decoded frames; generation runs independently";
  displayFps.style.cssText =
    "background:#27272a;color:#d4d4d8;border:1px solid #3f3f46;padding:4px 6px;" +
    "border-radius:4px;cursor:pointer;font-size:12px;";
  [4, 8, 12, 16, 24, 30].forEach((fps) => {
    const option = document.createElement("option");
    option.value = String(fps);
    option.textContent = `${fps} fps`;
    displayFps.append(option);
  });
  displayFps.value = String(configuredFps);
  leftControls.append(
    scaleLabel, btn1x, btn2x, btnFit, displayLabel, displayFps,
  );

  const status = document.createElement("output");
  status.style.cssText = "font-family:monospace;font-size:13px;color:#a1a1aa;";

  const helpToggle = document.createElement("button");
  helpToggle.textContent = "? Controls";
  helpToggle.style.cssText =
    "background:#27272a;color:#d4d4d8;border:1px solid #3f3f46;padding:4px 10px;" +
    "border-radius:4px;cursor:pointer;font-size:12px;";

  const rightControls = document.createElement("div");
  rightControls.style.cssText = "display:flex;align-items:center;gap:12px;";
  rightControls.append(status, helpToggle);

  toolbar.append(leftControls, rightControls);

  const viewport = document.createElement("div");
  viewport.style.cssText =
    "flex:1;overflow:auto;display:flex;justify-content:center;align-items:flex-start;" +
    "background:#09090b;border-radius:6px;border:1px solid #27272a;min-height:300px;padding:8px;";

  const image = document.createElement("img");
  image.style.cssText = "border-radius:4px;display:block;transition:width 0.1s ease, height 0.1s ease;";
  viewport.append(image);

  const help = document.createElement("pre");
  help.textContent =
    "Controls:\n" +
    "  Scale: 1 (1x) · 2 (2x scaled) · 3/F (Fit window)\n" +
    "  Display: choose playback rate for decoded frames (default 8 fps)\n" +
    "  Motion: WASD move · QE vertical · arrows look · ZC roll · X level\n" +
    "  Speed: Shift boost (3x) · V precision (0.25x) · -/+ adjust translation · [/] adjust rotation\n" +
    "  Playback: G pause/resume · Enter step · Space brake to halt · K drop input · 0 reset defaults\n" +
    "  Session: Shift+Home restart session · Esc blur/release keys · ? toggle help\n" +
    "  VRAM: Model stays resident in VRAM across sessions; click 'Unload VRAM' in header to release";
  help.style.cssText =
    "background:#27272a;color:#e4e4e7;padding:10px;margin:0;border-radius:4px;" +
    "font-family:monospace;font-size:12px;line-height:1.5;border:1px solid #3f3f46;";
  help.hidden = true;

  root.append(toolbar, viewport, help);
  root.tabIndex = 0;
  root.setAttribute("aria-label", "SolarWM keyboard camera viewer");

  let currentScale = configuredScale;

  function applyScale(scale) {
    currentScale = scale;
    [btn1x, btn2x, btnFit].forEach((btn) => {
      btn.style.background = "#27272a";
      btn.style.color = "#d4d4d8";
      btn.style.borderColor = "#3f3f46";
    });
    if (scale === "1x") {
      btn1x.style.background = "#2563eb";
      btn1x.style.color = "#ffffff";
      btn1x.style.borderColor = "#3b82f6";
      image.style.width = image.naturalWidth ? `${image.naturalWidth}px` : "auto";
      image.style.height = image.naturalHeight ? `${image.naturalHeight}px` : "auto";
      image.style.maxWidth = "none";
      image.style.maxHeight = "none";
      image.style.imageRendering = "auto";
    } else if (scale === "2x") {
      btn2x.style.background = "#2563eb";
      btn2x.style.color = "#ffffff";
      btn2x.style.borderColor = "#3b82f6";
      const w = image.naturalWidth ? image.naturalWidth * 2 : 1728;
      const h = image.naturalHeight ? image.naturalHeight * 2 : 960;
      image.style.width = `${w}px`;
      image.style.height = `${h}px`;
      image.style.maxWidth = "none";
      image.style.maxHeight = "none";
      image.style.imageRendering = "pixelated";
    } else if (scale === "fit") {
      btnFit.style.background = "#2563eb";
      btnFit.style.color = "#ffffff";
      btnFit.style.borderColor = "#3b82f6";
      const naturalWidth = image.naturalWidth || 1;
      const naturalHeight = image.naturalHeight || 1;
      const maxWidth = Math.max(1, viewport.clientWidth - 16);
      const maxHeight = Math.max(1, viewport.clientHeight - 16);
      const fitScale = Math.min(
        maxWidth / naturalWidth,
        maxHeight / naturalHeight,
      );
      image.style.width = `${Math.max(1, naturalWidth * fitScale)}px`;
      image.style.height = `${Math.max(1, naturalHeight * fitScale)}px`;
      image.style.maxWidth = "none";
      image.style.maxHeight = "none";
      image.style.objectFit = "contain";
      image.style.imageRendering = "auto";
    }
  }

  btn1x.onclick = (e) => { e.preventDefault(); applyScale("1x"); root.focus(); };
  btn2x.onclick = (e) => { e.preventDefault(); applyScale("2x"); root.focus(); };
  btnFit.onclick = (e) => { e.preventDefault(); applyScale("fit"); root.focus(); };
  helpToggle.onclick = (e) => { e.preventDefault(); help.hidden = !help.hidden; root.focus(); };
  window.addEventListener("resize", () => {
    if (currentScale === "fit") applyScale("fit");
  });
  applyScale(configuredScale);

  const pressed = new Set();
  let sequence = 0;
  const frameQueue = [];
  const maxQueuedFrames = 32;
  let displayTimer = null;
  let currentObjectUrl = null;
  let latestFrameStatus = "";
  const streamUrl = new URL(
    api.apiURL(`/solarwm/sessions/${encodeURIComponent(sessionId)}/stream`),
    window.location.href,
  );
  streamUrl.protocol = streamUrl.protocol === "https:" ? "wss:" : "ws:";
  let socket = new WebSocket(streamUrl);
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

  function refreshStatus() {
    if (!latestFrameStatus) return;
    status.value =
      `${latestFrameStatus} · display ${displayFps.value} fps · buffered ${frameQueue.length}`;
  }

  function renderNextFrame() {
    if (frameQueue.length === 0) return;
    const blob = frameQueue.shift();
    const url = URL.createObjectURL(blob);
    const previous = currentObjectUrl;
    currentObjectUrl = url;
    image.src = url;
    image.onload = () => {
      if (previous) URL.revokeObjectURL(previous);
      applyScale(currentScale);
    };
    refreshStatus();
  }

  function restartDisplayTimer() {
    if (displayTimer !== null) clearInterval(displayTimer);
    displayTimer = setInterval(
      renderNextFrame,
      1000 / Number(displayFps.value),
    );
  }

  displayFps.addEventListener("change", restartDisplayTimer);
  restartDisplayTimer();

  root.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.code === "Digit1" || event.code === "Numpad1") {
      event.preventDefault();
      applyScale("1x");
      return;
    }
    if (event.code === "Digit2" || event.code === "Numpad2") {
      event.preventDefault();
      applyScale("2x");
      return;
    }
    if (event.code === "Digit3" || event.code === "Numpad3" || event.code === "KeyF") {
      event.preventDefault();
      applyScale("fit");
      return;
    }
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
    if (event.code === "Slash" && (event.shiftKey || event.key === "?")) {
      event.preventDefault();
      help.hidden = !help.hidden;
      return;
    }
    if (event.code === "Home" && event.shiftKey && sessionConfig) {
      event.preventDefault();
      if (!window.confirm("Start a new SolarWM session from the initial pose?")) return;
      api.fetchApi("/solarwm/sessions", {
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
        const dim = image.naturalWidth && image.naturalHeight ? ` · ${image.naturalWidth}×${image.naturalHeight}` : "";
        latestFrameStatus =
          `latent ${value.latent_index} · ${value.latency_seconds.toFixed(2)} s${dim} · [${currentScale}]`;
        if (value.control_sent_at) {
          latestFrameStatus +=
            ` · input→frame ${(Date.now() / 1000 - value.control_sent_at).toFixed(2)} s`;
        }
        refreshStatus();
      } else if (value.type === "error") {
        status.value = value.error;
      }
      return;
    }
    frameQueue.push(event.data);
    while (frameQueue.length > maxQueuedFrames) frameQueue.shift();
    refreshStatus();
  });
  socket.addEventListener("close", () => {
    clearKeys();
    if (displayTimer !== null) clearInterval(displayTimer);
    frameQueue.length = 0;
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  });
  return { close: () => socket.close(), clearKeys };
}

export { makeSolarWMViewer };
