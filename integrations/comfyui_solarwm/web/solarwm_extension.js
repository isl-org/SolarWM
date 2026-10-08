import { app } from "../../../scripts/app.js";
import { makeSolarWMViewer } from "./solarwm_interactive.js";

app.registerExtension({
  name: "SolarWM.InteractiveCamera",
  async nodeCreated(node) {
    if (node.comfyClass !== "SolarWMInteractive") return;
    node.addWidget("button", "Open keyboard viewer", null, async () => {
      const values = Object.fromEntries(
        node.widgets
          .filter((widget) => widget.name !== "Open keyboard viewer")
          .map((widget) => [widget.name, widget.value]),
      );
      const response = await fetch("/solarwm/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "SolarWM session failed");
      const root = document.createElement("div");
      root.style.cssText =
        "position:fixed;inset:10%;z-index:10000;background:#111;padding:12px;" +
        "display:flex;flex-direction:column;gap:8px;";
      const close = document.createElement("button");
      close.textContent = "Close viewer";
      const viewerRoot = document.createElement("div");
      root.append(close, viewerRoot);
      document.body.append(root);
      let activeSessionId = payload.session_id;
      let viewer = makeSolarWMViewer(
        activeSessionId,
        viewerRoot,
        values,
        (sessionId, nextViewer) => {
          activeSessionId = sessionId;
          viewer = nextViewer;
        },
      );
      viewerRoot.focus();
      close.onclick = async () => {
        viewer.close();
        await fetch(`/solarwm/sessions/${activeSessionId}`, { method: "DELETE" });
        root.remove();
      };
    });
  },
});
