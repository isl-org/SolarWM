# SolarWM interactive ComfyUI extension

Copy or symlink `integrations/comfyui_solarwm` into ComfyUI's
`custom_nodes` directory, then restart ComfyUI. The `SolarWM Interactive
Camera` node opens a separate keyboard-controlled viewer and keeps model
execution outside ComfyUI's prompt queue.

The default bootstrap expects:

- `config_path`: an absolute or working-directory-relative path to
  `configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive.yaml`;
- `initial_image`: a local RGB anchor image;
- model/checkpoint paths and XPU environment configured as described in
  `AGENTS.local.md`.

The viewer must have focus. Controls:

| Keys | Action |
| --- | --- |
| `W/S` | forward/back |
| `A/D` | strafe left/right |
| `Q/E` | down/up |
| Arrow keys | keyboard yaw/pitch |
| `Z/C` | roll |
| `X` | level roll |
| `Shift` | temporary speed boost |
| `V` | precision movement |
| `-`/`=` | translation speed down/up |
| `[`/`]` | look speed down/up |
| `0` | restore default speeds |
| `G` | pause/resume generation |
| `Enter` | generate one latent while paused |
| `Space` | brake motion |
| `K` | drop current control input |
| `Shift+Home` | start a new confirmed session from the initial pose |
| `?` | show/hide the control overlay |
| `Esc` or focus loss | clear held keys |

The extension streams each decoded JPEG frame from a generated latent tile. It does not claim
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
