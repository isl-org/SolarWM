# Wan2.2 interactive ComfyUI viewer

The interactive viewer is an optional ComfyUI custom node under
`integrations/comfyui_solarwm`. It runs a persistent Stage2 session outside
ComfyUI's prompt queue. The existing batch sampler in `runtime/stage2.py` is
unchanged.

## Runtime behavior

`runtime/interactive.py` keeps the diffusion KV/cross-attention caches and
one continuous VAE decode session alive between steps. Each step:

1. reads the newest coalesced keyboard state;
2. integrates one absolute C2W pose;
3. converts it to relative W2C PRoPE tokens;
4. runs four NFE4 denoise forwards and one detached cache commit;
5. decodes and streams the one-latent tile immediately.

The dedicated interactive config uses `model.num_frame_per_block: 1`.
This is allowed only for camera inference by the contract. The released
checkpoint was trained with three-latent blocks, so block 1 remains an
experimental quality mode. Existing configs retain block size 3.

At 16 fps, one generated latent advances four pixel frames after the initial
anchor, or 0.25 seconds of video. The repository's block-1 microbenchmark
reports 0.99 seconds for four denoise forwards and 0.57 seconds for VAE
decode: 1.56 seconds best-case pose-to-pixel latency, measured in isolation
on the earlier timing fixture. An arbitrary keypress can arrive after a
latent has started, so the viewer must report end-to-end input timestamps
instead of promising a fixed latency. The browser receives
`control_sent_at`, `control_received_at`, `diffusion_seconds`, and
`decode_seconds` for this purpose.

## Installation

1. Make SolarWM importable from the ComfyUI process, for example by adding
   `SolarWM/src` to `PYTHONPATH`.
2. Copy or symlink `integrations/comfyui_solarwm` into ComfyUI's
   `custom_nodes` directory.
3. Configure the model/checkpoint paths in
   `configs/examples/wan22_ti2v_5b/infer_stage2_sgf_interactive.yaml`.
4. Add the `SolarWM Interactive Camera` node and provide a local anchor image.
5. Click `Open keyboard viewer`, then focus the viewer.

The default route bootstrap loads the provider once per session. Deployments
with pre-encoded tensors can replace it with
`configure_provider_factory()`; the factory must return `provider`,
`first_latent`, and `condition`.

## Controls

| Keys | Behavior |
| --- | --- |
| `W` / `S` | forward / backward |
| `A` / `D` | strafe left / right |
| `Q` / `E` | down / up |
| Arrow keys | yaw / pitch |
| `Z` / `C` | roll left / right |
| `X` | level roll |
| `Shift` | temporary speed boost |
| `V` | precision movement |
| `-` / `=` | decrease / increase translation speed |
| `[` / `]` | decrease / increase look speed |
| `0` | restore default speeds |
| `G` | pause / resume continual generation |
| `Enter` | generate one latent while paused |
| `Space` | brake camera motion |
| `K` | clear current control input |
| `Shift+Home` | confirm and start a new session at the initial pose |
| `?` | show / hide the help overlay |
| `Esc` | clear held keys and remove focus |

Controls are active only while the viewer is focused. Ctrl/Alt/Meta
combinations are left to browser and ComfyUI shortcuts. Blur, hidden-tab,
disconnect, and pointer/focus loss clear held keys.

## Verification

The CPU/mocked verification suite covers:

- persistent cache identity and one-latent decode callback ordering;
- SE(3) validity, diagonal normalization, and relative-W2C token conversion;
- monotonic control sequencing and focus-safe clearing;
- acceptance of the dedicated one-latent camera config.

The sandboxed environment reports zero XPU devices, but the host environment
outside the sandbox reports `torch 2.12.1+xpu` with one Intel Graphics XPU.
The outside-sandbox softmax/SDPA guard passed all 19 cases. A full model
smoke run and end-to-end browser-paint latency measurement still require
model weights and an interactive session, and adoption remains gated on a
fixed-seed block-1 versus block-3 visual comparison.
