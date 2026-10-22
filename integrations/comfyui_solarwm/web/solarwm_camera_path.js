import { app } from "../../../scripts/app.js";

const PATH_NODE = "SolarWMCameraPath";
const VIEWER_NODE = "SolarWMInteractive";
const MIN_STEP = 0.012;
const CONTINUE_RADIUS = 0.08;
const MAX_POINTS = 2000;
const SECONDS_PER_LATENT = 0.25;
const SETTLE_SECONDS = 0.5;

function widgetByName(node, name) {
  return node?.widgets?.find((widget) => widget.name === name);
}

function readPoints(node) {
  try {
    const points = JSON.parse(widgetByName(node, "path_json")?.value || "{}").points;
    return Array.isArray(points) ? points : [];
  } catch {
    return [];
  }
}

function pathLength(points) {
  let length = 0;
  for (let i = 1; i < points.length; i += 1) {
    length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  return length;
}

function niceStep(extent) {
  const raw = extent / 4;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * power >= raw) * power;
  return step;
}

/** Model-space path of a SolarWM Camera Path node, or null when nothing is drawn. */
export function solarwmCameraPath(node) {
  if (node?.comfyClass !== PATH_NODE) return null;
  const points = readPoints(node);
  if (points.length < 2) return null;
  const extent = Number(widgetByName(node, "extent")?.value ?? 1);
  return {
    schema: "solarwm.comfyui.camera-path.v1",
    points: points.map(([x, y]) => [x * extent, y * extent]),
  };
}

function connectedViewer(node) {
  const targets = node.getOutputNodes?.(0) ?? [];
  return targets.find((target) => target?.comfyClass === VIEWER_NODE) ?? null;
}

function setupPathNode(node) {
  const pathWidget = widgetByName(node, "path_json");
  const extentWidget = widgetByName(node, "extent");
  pathWidget.hidden = true;
  pathWidget.options = { ...(pathWidget.options ?? {}), hidden: true };
  pathWidget.computeSize = () => [0, -4];

  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;box-sizing:border-box;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "flex:1;min-height:0;width:100%;border-radius:4px;cursor:crosshair;touch-action:none;";
  const toolbar = document.createElement("div");
  toolbar.style.cssText =
    "display:flex;align-items:center;gap:6px;font:11px monospace;color:#a1a1aa;";
  const makeButton = (label) => {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText =
      "background:#27272a;color:#e4e4e7;border:1px solid #3f3f46;border-radius:3px;" +
      "padding:1px 8px;cursor:pointer;font:11px sans-serif;";
    return button;
  };
  const undo = makeButton("Undo");
  const clear = makeButton("Clear");
  const info = document.createElement("span");
  info.style.cssText = "flex:1;text-align:right;white-space:nowrap;overflow:hidden;";
  toolbar.append(undo, clear, info);
  container.append(canvas, toolbar);

  let points = readPoints(node);
  const strokes = [];
  let drawing = false;

  const commit = () => {
    const rounded = points.map(([x, y]) => [Number(x.toFixed(4)), Number(y.toFixed(4))]);
    pathWidget.value = JSON.stringify({ points: rounded });
    node.setDirtyCanvas?.(true, true);
  };

  const toPath = (event) => {
    const rect = canvas.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height);
    const x = (event.clientX - rect.left - rect.width / 2) / (size / 2);
    const y = -(event.clientY - rect.top - rect.height / 2) / (size / 2);
    return [Math.max(-1, Math.min(1, x)), Math.max(-1, Math.min(1, y))];
  };

  let lastInfo = "";
  const updateInfo = () => {
    const extent = Number(extentWidget?.value ?? 1);
    const length = pathLength(points) * extent;
    let text = points.length < 2 ? "drag to draw from the start marker" : `${length.toFixed(2)} u`;
    const viewer = connectedViewer(node);
    if (viewer && points.length >= 2) {
      const speed = Number(widgetByName(viewer, "translation_speed")?.value ?? 0.1);
      const seconds = length / Math.max(speed, 1e-6) + SETTLE_SECONDS;
      const latents = Math.ceil(seconds / SECONDS_PER_LATENT);
      const budget = Number(widgetByName(viewer, "max_latent_frames")?.value ?? 0);
      text += ` · ≈${seconds.toFixed(0)} s / ${latents} latents`;
      if (budget && latents > budget) text += " > max_latent_frames";
    }
    if (text !== lastInfo) {
      info.textContent = text;
      info.style.color = text.includes(">") ? "#f59e0b" : "#a1a1aa";
      lastInfo = text;
    }
  };

  const redraw = () => {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    const size = Math.min(width, height);
    const half = size / 2;
    const cx = width / 2;
    const cy = height / 2;
    const px = ([x, y]) => [cx + x * half, cy - y * half];
    const extent = Number(extentWidget?.value ?? 1);

    ctx.fillStyle = "#0f0f12";
    ctx.fillRect(0, 0, width, height);
    const step = niceStep(extent) / extent;
    ctx.lineWidth = 1;
    for (let v = -Math.floor(1 / step) * step; v <= 1 + 1e-9; v += step) {
      const major = Math.abs(v) < 1e-9;
      ctx.strokeStyle = major ? "#3f3f46" : "#1f1f24";
      ctx.beginPath();
      ctx.moveTo(cx + v * half, cy - half);
      ctx.lineTo(cx + v * half, cy + half);
      ctx.moveTo(cx - half, cy - v * half);
      ctx.lineTo(cx + half, cy - v * half);
      ctx.stroke();
    }
    ctx.strokeStyle = "#52525b";
    ctx.strokeRect(cx - half + 0.5, cy - half + 0.5, size - 1, size - 1);

    ctx.fillStyle = "#71717a";
    ctx.font = `${Math.round(11 * ratio)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const pad = 9 * ratio;
    ctx.fillText("W", cx, cy - half + pad);
    ctx.fillText("S", cx, cy + half - pad);
    ctx.fillText("A", cx - half + pad, cy);
    ctx.fillText("D", cx + half - pad, cy);
    ctx.textAlign = "left";
    ctx.fillText(`grid ${niceStep(extent)} u`, cx - half + 4 * ratio, cy + half - pad);

    if (points.length >= 2) {
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2.5 * ratio;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      points.forEach((point, index) => {
        const [x, y] = px(point);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const [ex, ey] = px(points[points.length - 1]);
      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.arc(ex, ey, 4 * ratio, 0, 2 * Math.PI);
      ctx.fill();
    }

    const marker = 8 * ratio;
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(cx, cy - marker);
    ctx.lineTo(cx - marker * 0.7, cy + marker * 0.6);
    ctx.lineTo(cx + marker * 0.7, cy + marker * 0.6);
    ctx.closePath();
    ctx.fill();
    updateInfo();
  };

  const stop = (event) => event.stopPropagation();
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    canvas.setPointerCapture(event.pointerId);
    const point = toPath(event);
    const end = points[points.length - 1];
    strokes.push(points.slice());
    if (!end || points.length < 2 || Math.hypot(point[0] - end[0], point[1] - end[1]) > CONTINUE_RADIUS) {
      points = [[0, 0]];
    }
    const last = points[points.length - 1];
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) >= MIN_STEP) points.push(point);
    drawing = true;
    redraw();
  });
  canvas.addEventListener("pointermove", (event) => {
    event.stopPropagation();
    if (!drawing || points.length >= MAX_POINTS) return;
    const point = toPath(event);
    const end = points[points.length - 1];
    if (Math.hypot(point[0] - end[0], point[1] - end[1]) < MIN_STEP) return;
    points.push(point);
    redraw();
  });
  const finish = (event) => {
    event.stopPropagation();
    if (!drawing) return;
    drawing = false;
    commit();
    redraw();
  };
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("wheel", stop);
  for (const button of [undo, clear]) button.addEventListener("pointerdown", stop);
  undo.onclick = () => {
    points = strokes.pop() ?? [];
    commit();
    redraw();
  };
  clear.onclick = () => {
    if (points.length) strokes.push(points.slice());
    points = [];
    commit();
    redraw();
  };

  node.addDOMWidget("path_canvas", "solarwm_camera_path", container, {
    serialize: false,
    getMinHeight: () => 240,
  });
  new ResizeObserver(redraw).observe(canvas);

  const extentCallback = extentWidget?.callback;
  if (extentWidget) {
    extentWidget.callback = function (...args) {
      const result = extentCallback?.apply(this, args);
      redraw();
      return result;
    };
  }
  const onConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = onConfigure?.apply(this, args);
    points = readPoints(node);
    strokes.length = 0;
    redraw();
    return result;
  };
  const onDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (...args) {
    const result = onDrawForeground?.apply(this, args);
    updateInfo();
    return result;
  };
  node.setSize?.([Math.max(node.size?.[0] ?? 0, 300), Math.max(node.size?.[1] ?? 0, 360)]);
  redraw();
}

app.registerExtension({
  name: "SolarWM.CameraPath",
  async nodeCreated(node) {
    if (node.comfyClass === PATH_NODE) setupPathNode(node);
  },
});
