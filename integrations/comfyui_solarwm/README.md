# SolarWM interactive ComfyUI extension

Copy or symlink `integrations/comfyui_solarwm` into ComfyUI's
`custom_nodes` directory, then restart ComfyUI. The `SolarWM Interactive
Camera` node opens a separate keyboard-controlled viewer and keeps model
execution outside ComfyUI's prompt queue.

The default bootstrap expects:

- `config_path`: an absolute or working-directory-relative path to
  `configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive.yaml`;
- `initial_image`: a ComfyUI uploadable image picker with preview, or a local
  RGB anchor image path. The dropdown lists images in the remote ComfyUI
  input directory; the upload button copies a local browser file there.
- model/checkpoint paths and XPU environment configured as described in
  `AGENTS.local.md`.

The node exposes these Stage2 runtime choices and applies them to the loaded
configuration before model construction:

| Node field | Values | Default |
| --- | --- | --- |
| `attention_backend` | `sdpa`, `sageattnv1` | `sdpa` |
| `linear_quant` | `full_precision`, `svdquant_k1`, `svdquant_k2`, `svdquant_k4` | `full_precision` |
| `svdquant_weights_root` | SVDQuant weight directory | `${SOLAR_MODEL_ROOT}/SolarWM-5B-sgf-stage2-81f/svdquant_int4` |
| `stage2_vae_quant` | `full_precision`, `quarot_w8a8` | `full_precision` |
| `camera_motion` | `straight_about_turn_10s`, `circle_twice_10s`, `manual_only` | `straight_about_turn_10s` |
| `translation_speed` | float | `0.1` |
| `rotation_speed` | float | `0.1` |
| `display_scale` | `1x`, `2x`, `fit` | `2x` |
| `display_fps` | `4`, `8`, `12`, `16`, `24`, `30` | `8` |

The **Open keyboard viewer** button loads the provider in a worker thread, so
ComfyUI remains responsive while model weights are loading. A loading dialog
is shown immediately; do not use **Queue Prompt** to start the viewer.

The autonomous presets are a 10-second straight walk with an about-turn
(`straight_about_turn_10s`) or two slow circles (`circle_twice_10s`). Any
keyboard inputs steer or override the preset, and pressing `Space` brakes to a
halt.

Every interactive session is also recorded as an H.264 MP4. Recordings are
written under the configured runtime output directory, or
`outputs/solarwm_interactive/` when the example config still contains a
placeholder output path. The saved path is returned by the session status and
stopped events.

The viewer must have focus. Controls:

| Keys | Action |
| --- | --- |
| `1` / `2` / `3` (`F`) | Scale display: 1x, 2x pixelated, or Fit window |
| Display selector | Playback rate for decoded frames; default `8 fps` |
| `W/S` | forward/back |
| `A/D` | strafe left/right |
| `Q/E` | down/up |
| `Z/C` | roll |
| `X` | level roll |
| `Shift` | temporary speed boost (3x) |
| `V` | precision movement (0.25x) |
| `-`/`=` | translation speed down/up |
| `[`/`]` | look speed down/up |
| `0` | restore default speeds and camera trajectory |
| `G` | pause/resume generation |
| `Enter` | generate one latent while paused |
| `Space` | brake motion |
| `K` | drop current control input |
| `Shift+Home` | start a new confirmed session from the initial pose |
| `?` | show/hide the control overlay |
| `Esc` or focus loss | clear held keys |

### VRAM Lifecycle & Caching

The loaded Stage2 model weights (diffusion transformer, VAE, text encoder) are kept resident in VRAM across sessions. When closing and reopening the viewer, restarting generation, or modifying camera parameters, the model is reused immediately (<0.2 s start) without the ~1 minute reload from disk.

- If model architecture options change (e.g. `linear_quant` or `attention_backend`), the cache automatically reloads with the new settings.
- To explicitly release GPU memory, click the **Unload VRAM** button in the viewer header (or send `POST /solarwm/model/unload`).

The extension streams each decoded JPEG frame from a generated latent tile and
buffers them for client-side playback at the selected display rate. This
separates browser painting from bursty generation, but cannot display frames
before the VAE has decoded them. It does not claim
game-engine latency: the current block-1 measurement is approximately 1.56 s
best case from camera input to decoded pixels, before browser paint, and
arbitrary keypress phase adds up to one generation step. The released
checkpoint was trained with three-latent blocks; block size 1 is opt-in and
must pass a fixed-seed quality comparison before production use.

For deployments that already have tensors, replace the default bootstrap:

```python
from integrations.comfyui_solarwm.manager import configure_provider_factory

configure_provider_factory(my_provider_factory)
```

The factory returns `provider`, `first_latent`, and `condition`. This makes
the route testable without putting model state in a ComfyUI node instance.
