import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { makeSolarWMViewer } from "./solarwm_interactive.js";

async function loadSelectedImageMetadata(path) {
  if (!path) return null;
  const response = await api.fetchApi(
    `/solarwm/image-metadata?path=${encodeURIComponent(path)}`,
  );
  if (!response.ok) return null;
  return response.json();
}

app.registerExtension({
  name: "SolarWM.InteractiveCamera",
  async nodeCreated(node) {
    if (node.comfyClass !== "SolarWMInteractive") return;
    const imageWidget = node.widgets?.find(
      (widget) => widget.name === "initial_image",
    );
    const promptWidget = node.widgets?.find(
      (widget) => widget.name === "prompt",
    );
    const cameraWidget = node.widgets?.find(
      (widget) => widget.name === "camera_motion",
    );
    const syncSelectedImage = async (value) => {
      const metadata = await loadSelectedImageMetadata(value);
      if (!metadata || !metadata.has_trajectory) return;
      if (promptWidget && metadata.prompt) {
        promptWidget.value = metadata.prompt;
      }
      if (cameraWidget) {
        cameraWidget.value = "dataset_trajectory";
      }
      node.graph?.setDirtyCanvas(true, true);
    };
    if (imageWidget) {
      const previousCallback = imageWidget.callback;
      imageWidget.callback = async (value) => {
        if (previousCallback) previousCallback.call(imageWidget, value);
        await syncSelectedImage(value);
      };
      void syncSelectedImage(imageWidget.value);
    }
    node.addWidget("button", "Open keyboard viewer", null, async () => {
      const values = Object.fromEntries(
        node.widgets
          .filter((widget) => widget.name !== "Open keyboard viewer")
          .map((widget) => [widget.name, widget.value]),
      );
      const root = document.createElement("div");
      root.style.cssText =
        "position:fixed;inset:3% 4%;z-index:10000;background:#18181b;color:#f4f4f5;padding:12px;" +
        "display:flex;flex-direction:column;gap:8px;box-shadow:0 12px 48px rgba(0,0,0,0.85);" +
        "border-radius:8px;border:1px solid #3f3f46;box-sizing:border-box;max-height:94vh;";
      const close = document.createElement("button");
      close.textContent = "Close viewer";
      close.style.cssText =
        "background:#ef4444;color:white;border:none;padding:6px 14px;border-radius:4px;" +
        "cursor:pointer;font-weight:600;font-size:13px;align-self:flex-start;";
      const unloadBtn = document.createElement("button");
      unloadBtn.textContent = "Unload VRAM";
      unloadBtn.title = "Release SolarWM model weights from VRAM to free GPU memory";
      unloadBtn.style.cssText =
        "background:#27272a;color:#d4d4d8;border:1px solid #3f3f46;padding:6px 14px;border-radius:4px;" +
        "cursor:pointer;font-weight:600;font-size:13px;align-self:flex-start;";
      const status = document.createElement("output");
      status.style.cssText = "font-family:monospace;font-size:13px;color:#a1a1aa;";
      status.textContent = "Starting SolarWM interactive session…";
      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;gap:12px;";
      header.append(close, unloadBtn, status);
      root.append(header);
      document.body.append(root);
      let activeSessionId = null;
      let viewer = null;
      let cancelled = false;
      close.onclick = async () => {
        cancelled = true;
        if (viewer) viewer.close();
        if (activeSessionId) {
          await api.fetchApi(`/solarwm/sessions/${activeSessionId}`, { method: "DELETE" });
        }
        root.remove();
      };
      unloadBtn.onclick = async () => {
        if (!window.confirm("Unload SolarWM model from VRAM to free GPU memory? This stops any active session.")) return;
        status.textContent = "Unloading model from VRAM…";
        if (viewer) viewer.close();
        try {
          const resp = await api.fetchApi("/solarwm/model/unload", { method: "POST" });
          const res = await resp.json();
          status.textContent = res.message || "Model unloaded from VRAM";
        } catch (e) {
          status.textContent = `Unload failed: ${e.message}`;
        }
      };
      try {
        const response = await api.fetchApi("/solarwm/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(values),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "SolarWM session failed");
        activeSessionId = payload.session_id;
        if (cancelled) {
          await api.fetchApi(`/solarwm/sessions/${activeSessionId}`, { method: "DELETE" });
          return;
        }
        const viewerRoot = document.createElement("div");
        viewerRoot.style.cssText =
          "flex:1;display:flex;flex-direction:column;gap:8px;overflow:hidden;outline:none;";
        root.replaceChildren(header, viewerRoot);
        viewer = makeSolarWMViewer(
          activeSessionId,
          viewerRoot,
          values,
          (sessionId, nextViewer) => {
            activeSessionId = sessionId;
            viewer = nextViewer;
          },
        );
        viewerRoot.focus();
      } catch (error) {
        status.textContent = `SolarWM failed to start: ${error.message}`;
        console.error("SolarWM interactive session failed", error);
      }
    });
  },
});
