"""Build the Intel-template summary deck for the Wan 2.2-5B XPU report.

Numbers are copied from docs/backends/wan22-xpu-porting-and-optimization-report.md (Part I).

    PYTHONPATH=<python-pptx site dir> python scripts/report/build_summary_deck.py
"""

import sys
from pathlib import Path

SKILL = Path.home() / ".agents/skills/intel-pptx/scripts"
sys.path.insert(0, str(SKILL))
import intel_pptx as ip  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/report_slides/wan22-xpu-summary.pptx"
C, CS = ip.CONTENT, ip.CONTENT_SUB

deck = ip.IntelDeck(footer="SolarWM | Wan 2.2-5B on Intel Arc Pro B70")

deck.title_slide("SolarWM Wan 2.2-5B on Intel Arc Pro B70",
                 "XPU port and optimization", event="October 2026")

# ---------------------------------------------------------------- summary
s = deck.title_only("Summary", subtitle="Toward real time on one Arc Pro B70: 750 ms per chunk (12 frames at 16 FPS)")
ip.stat_row(s, [("2.22×", "Faster at 480×864", "3,617 → 1,628 ms per chunk"),
                ("7.37", "FPS at 480×864", "Real time needs 2.17× more"),
                ("15.40", "FPS at 320×576", "Within 4 % of real time")],
            box=ip.Box(CS.x, CS.y, CS.w, 2.3))
ip.text_box(s, CS.x, 4.2, CS.w, 2.3, [
    "The GPU was already saturated: gains came from removing redundant GPU work, fusing bandwidth-bound operations and low precision",
    "Our SVDQuant W4A4 kernel is 2.15× faster than llm-scaler's on the same weights",
    "Compilers did not help: torch.compile and OpenVINO were slower or barely faster",
], size=16, bullets=True, space_after=8)

# ---------------------------------------------------------------- model
s = deck.title_only("The model: SolarWM Wan 2.2-5B",
                    subtitle="Autoregressive: first frame + prompt + 6-DoF camera → video")
ip.icon_features(s, [
    ("processor", "DiT, 5B parameters", "30 layers. Per chunk: 4 denoise forwards + 1 KV-cache commit forward over 3 latent frames"),
    ("video-player", "Causal 3D VAE", "Decodes each chunk's 3 latent frames into 12 pixel frames"),
    ("camera-tripod", "Camera conditioning", "Projective RoPE (PRoPE) on attention over an 18-latent-frame KV window: 7,290 tokens at 480p"),
], box=ip.Box(CS.x + 0.5, CS.y + 0.4, CS.w - 1.0, 3.8), cols=3)

# ---------------------------------------------------------------- initial port
s = deck.title_only("Initial port: CUDA → XPU", subtitle="Native PyTorch XPU, no other GPU used")
ip.card_grid(s, [
    {"title": "XPU-only environment", "body": "No CUDA packages. torch.cuda → XPU device selection and synchronize.", "icon": "settings"},
    {"title": "Attention", "body": "flash-attn (CUDA-only) → PyTorch SDPA.", "icon": "code-file"},
    {"title": "Memory", "body": "UMT5-XXL text encoder on the host CPU, so DiT + VAE fit in 32 GB.", "icon": "memory-chip"},
], cols=3, box=ip.Box(CS.x, CS.y, CS.w, 2.5))
ip.stat_row(s, [("3,617 ms", "Per chunk", "DiT 1,898 ms + VAE 1,720 ms; 3.32 FPS")],
            box=ip.Box(CS.x, 4.45, CS.w, 2.0))

# ---------------------------------------------------------------- optimizations
DIT = [("Host-int KV-cache indices", 1.02),
       ("Circular KV cache + PRoPE memo", 1.12),
       ("RoPEd KV ring", 1.25),
       ("Fused RoPE+PRoPE+SDPA", 1.13),
       ("DiT operator fusions", 1.18),
       ("Drop raw KV window copy", 1.10),
       ("Other merged changes", 1.03),
       ("SVDQuant W4A4 k4 linears", 1.07)]
VAE = [("Elementwise in bf16", 1.19),
       ("Channels-last convolutions", 1.06),
       ("Fused Triton RMSNorm+SiLU", 1.27),
       ("QuaRot dynamic W8A8", 1.18),
       ("oneDNN epilogue fusion", 1.11)]

s = deck.title_only("Optimizations: speedup per step",
                    subtitle="Speedup over the previous step, 480×864; steps ≤ 1.01× for that network left out")
half = (CS.w - 0.4) / 2
for i, (head, steps) in enumerate([("DiT: 1,898 → 811 ms (2.34×)", DIT), ("VAE: 1,720 → 817 ms (2.11×)", VAE)]):
    x = CS.x + i * (half + 0.4)
    ip.text_box(s, x, CS.y, half, 0.45, head, size=18, font="medium", color="energy")
    steps = steps[::-1]  # bar charts draw the first category at the bottom
    gf = ip.add_chart(s, "barh", [n for n, _ in steps], {"Speedup": [v for _, v in steps]},
                      x, CS.y + 0.5, half, 4.4, legend=False, data_labels=True,
                      number_format='0.00"×"', font_size=13)
    ax = gf.chart.value_axis
    ax.minimum_scale, ax.maximum_scale, ax.major_unit = 1.0, 1.3, 0.1
ip.source_note(s, "Each step adds one change to the previous one; same input, seed and workload. Details: report Part II §5.")

# ---------------------------------------------------------------- final results
s = deck.title_only("Final results", subtitle="2.22× per chunk at 480×864; 320×576 within 4 % of real time")
ip.add_table(s, [
    ["", "DiT ms", "VAE ms", "Chunk ms", "FPS", "vs 750 ms target"],
    ["Plain port, 480×864", "1,897.7", "1,719.5", "3,617.2", "3.32", "4.82× over"],
    ["Optimized, 480×864", "811.2", "816.7", "1,627.9", "7.37", "2.17× over"],
    ["Speedup", "2.34×", "2.11×", "2.22×", "", ""],
    ["Optimized, 320×576", "374.7", "404.6", "779.2", "15.40", "1.04× over"],
], CS.x, CS.y + 0.1, CS.w, col_widths=[3.13, 1.7, 1.7, 1.8, 1.4, 2.2], row_h=0.6, size=16,
   highlight_row=2)
ip.text_box(s, CS.x, CS.y + 3.4, CS.w, 1.4, [
    "Intel Arc Pro B70 (Xe2, 32 GB), torch 2.12.1+xpu, Triton XPU, oneDNN, SYCL",
    "Steady-state chunk: median of chunks 5–7 of an 8-chunk rollout. 320×576 is a different computation, not a chain step",
], size=15, bullets=True, space_after=6)

# ---------------------------------------------------------------- quantization kernels
s = deck.title_only("Quantization kernels", subtitle="Each kernel alone vs its BF16 kernel, 480×864 steady-state shapes")
ip.card_grid(s, [
    {"title": "SageAttention v1", "body": [
        "INT8 Q, K (Triton DPAS); softmax, P·V in BF16",
        "Q₈ = ⌊QH / s_Q⌉,  K₈ = ⌊(K − μ_K)H / s_K⌉",
        "S ≈ s_Q s_K · Q₈K₈ᵀ; μ_K leaves softmax unchanged",
        "H: 4×4 Hadamard folded into PRoPE"]},
    {"title": "SageAttention v2", "body": [
        "INT4 Q, K (SYCL DPAS); softmax, P·V in BF16",
        "Q₄ = ⌊(Q − μ_Q) / s_Q⌉,  K₄ = ⌊(K − μ_K) / s_K⌉",
        "S ≈ s_Q s_K · Q₄K₄ᵀ + μ_Q(K − μ_K)ᵀ, mean term exact",
        "16 levels: 11–25× v1's error"]},
    {"title": "SVDQuant W4A4", "body": [
        "Smooth: X̂ = X / λ,  Ŵ = λW",
        "Ŵ ≈ L₁L₂ + R: rank-32 BF16 branch absorbs outliers",
        "Y = X̂L₁L₂ + Q₄(X̂)·Q₄(R), INT4 group 64",
        "GPTQ residual; clipping per layer and timestep"]},
], cols=3, box=ip.Box(CS.x, CS.y, CS.w, 2.1), title_size=16, body_size=12, bullets=True, space_after=3)
ip.add_table(s, [
    ["Kernel", "Problem size", "BF16 ms", "Quant ms", "Speedup", "Error, real", "Error, random"],
    ["Sage v1 attention", "1,215 q × 7,290 k, 24 heads × 128", "1.760", "1.247", "1.41×", "2.2–3.6 %", "1.35 %"],
    ["Sage v2 attention", "1,215 q × 7,290 k, 24 heads × 128", "1.760", "0.964", "1.83×", "31–55 %", "17.1 %"],
    ["SVDQuant Q/K/V", "1,215 × 3,072 → 3 × 3,072", "0.771", "0.658", "1.17×", "3.5–11 %", "16–22 %"],
    ["SVDQuant FFN", "1,215 × 3,072 → 14,336 → 3,072", "2.084", "1.800", "1.16×", "11–22 %", "14–22 %"],
    ["SVDQuant, a block's linears", "Q/K/V + FFN", "2.855", "2.458", "1.16×", "", ""],
], CS.x, CS.y + 2.25, CS.w, col_widths=[2.6, 3.2, 1.0, 1.05, 1.0, 1.5, 1.58], row_h=0.36, size=12, header_size=13)
ip.text_box(s, CS.x, CS.y + 4.5, CS.w, 0.5, [
    "Real Q/K are harder for Sage than random data; SVDQuant's smoothing and clipping are calibrated on real activations, so they are easier",
], size=13, bullets=True)
ip.source_note(s, "GPU locked at 2,000 MHz, median of 50; actual speedups may be lower due to GPU throttling. Error ‖y − y_BF16‖ / ‖y_BF16‖: real = "
                  "steady-state inputs of DiT layers 0, 15, 29; random = Gaussian, same shapes. BF16 SDPA vs FP32: 0.12–0.23 %. Report §6.2, §6.4.")

# ---------------------------------------------------------------- quantization
QUANT = [  # configuration, chunk ms, chunk speedup, DOVER++ overall / aesthetic / technical (first, last chunk; change vs BF16)
    ("BF16 reference", "2,107.2", "1.00×", "0.421", "0.466", "0.947", "0.956", "0.087", "0.098"),
    ("SageAttention v1", "2,033.2", "1.04×", "0.421 (+0.1%)", "0.463 (−0.5%)", "0.947 (+0.0%)", "0.954 (−0.3%)", "0.087 (+0.2%)", "0.099 (+0.5%)"),
    ("SageAttention v2", "2,025.8", "1.04×", "0.414 (−1.6%)", "0.299 (−35.8%)", "0.947 (+0.0%)", "0.911 (−4.7%)", "0.083 (−4.0%)", "0.051 (−48.7%)"),
    ("SVDQuant W4A4 k1", "2,053.4", "1.03×", "0.415 (−1.3%)", "0.445 (−4.5%)", "0.947 (+0.0%)", "0.952 (−0.5%)", "0.084 (−3.3%)", "0.092 (−6.3%)"),
    ("SVDQuant W4A4 k2", "2,052.4", "1.03×", "0.417 (−1.0%)", "0.437 (−6.1%)", "0.947 (+0.0%)", "0.951 (−0.6%)", "0.085 (−2.4%)", "0.091 (−7.7%)"),
    ("SVDQuant W4A4 k4", "2,052.5", "1.03×", "0.416 (−1.1%)", "0.438 (−5.9%)", "0.947 (+0.0%)", "0.951 (−0.6%)", "0.084 (−3.0%)", "0.091 (−7.2%)"),
    ("QuaRot dynamic W8A8 VAE", "1,903.0", "1.11×", "0.420 (−0.1%)", "0.466 (−0.1%)", "0.947 (−0.0%)", "0.956 (+0.0%)", "0.086 (−0.1%)", "0.098 (−0.1%)"),
    ("SVDQuant + W8A8", "1,854.0", "1.14×", "0.416 (−1.2%)", "0.438 (−5.9%)", "0.947 (−0.0%)", "0.951 (−0.6%)", "0.084 (−3.1%)", "0.091 (−7.2%)"),
    ("Sage v1 + SVDQuant", "1,978.4", "1.07×", "0.417 (−1.0%)", "0.433 (−6.9%)", "0.947 (+0.0%)", "0.948 (−0.8%)", "0.084 (−2.5%)", "0.089 (−9.4%)"),
    ("Sage v1 + SVDQuant + W8A8", "1,797.3", "1.17×", "0.416 (−1.1%)", "0.433 (−7.1%)", "0.947 (−0.0%)", "0.948 (−0.9%)", "0.084 (−2.6%)", "0.089 (−9.5%)"),
]
s = deck.title_only("Quantization: speed and quality",
                    subtitle="153-frame rollouts, 480×864; DOVER++ mean of 117 test samples (change vs BF16)")
ip.add_table(s, [["Configuration", "Chunk ms", "Chunk ×", "Overall 1st", "Overall last", "Aesthetic 1st", "Aesthetic last",
                  "Technical 1st", "Technical last"]]
             + [list(r) for r in QUANT], CS.x, CS.y, CS.w,
             col_widths=[2.6, 0.95, 0.9] + [1.2467] * 6, row_h=0.32, size=10.5, header_size=11,
             highlight_row=10)
ip.text_box(s, CS.x, CS.y + 3.8, CS.w, 1.0, [
    "**Sage v2 collapses over long rollouts:** last chunk overall −35.8%, aesthetic −4.7%, technical −48.7% vs BF16",
    "Sage v1 and W8A8 match BF16 in all three scores; SVDQuant costs −5.9% overall, −0.6% aesthetic, −7.2% technical",
    "**Our SVDQuant kernel is 2.15× faster than llm-scaler's** on the same weights (a block's five linears, 6.29 → 2.92 ms)",
], size=14, bullets=True)
ip.source_note(s, "DOVER++: no-reference video quality, 0–1, higher is better; ground truth overall 0.422 | 0.455. Speed: one rollout, GPU locked "
                  "at 2,000 MHz; actual speedups may be lower due to throttling. SVDQuant k4 unless noted; Sage v1 + SVDQuant uses sageattnv1_k4 weights. Sage v2 "
                  "with BF16 linears (no matching SVDQuant weights).")

# ---------------------------------------------------------------- quantization video
GIFS = OUT.parent / "quality_gifs"
NAMES = ["bf16", "sage1", "sage2", "svdq_k1", "svdq_k2", "svdq", "w8a8", "svdq_w8a8",
         "sage1_svdq", "sage1_svdq_w8a8"]
s = deck.title_only("Quantization: first and last chunk",
                    subtitle="Each pair: first chunk (frames 0–8) | last chunk (frames 141–152) of 153, at 16 FPS")
gif_w, gif_h, label_h, col_gap, row_gap = 3.4, 3.4 * 180 / 654, 0.22, 0.25, 0.04
x0 = CS.x + (CS.w - 3 * gif_w - 2 * col_gap) / 2
for i, (name, row) in enumerate(zip(NAMES, QUANT)):
    x = x0 + (i % 3) * (gif_w + col_gap)
    y = CS.y + (i // 3) * (label_h + gif_h + row_gap)
    label = row[0].replace("QuaRot dynamic W8A8 VAE", "QuaRot W8A8 VAE").replace("SVDQuant +", "SVDQ +").replace("+ SVDQuant", "+ SVDQ")
    ip.text_box(s, x, y, gif_w, label_h, [f"**{label}**"], size=11, anchor="b")
    s.shapes.add_picture(str(GIFS / f"{name}.gif"), ip.Inches(x), ip.Inches(y + label_h), ip.Inches(gif_w))
ip.source_note(s, "One benchmark sample, same seed and prompt for all; DOVER++ over 117 test samples is on the previous slide. "
                  "GIFs from scripts/report/quality_gifs.py.")

# ---------------------------------------------------------------- did not work
s = deck.title_only("What did not work")
ip.add_table(s, [
    ["Approach", "Result", "Why"],
    ["Host-sync removal, DiT/VAE stream overlap", "1.00×", "The GPU was already saturated, and it serializes the two compute streams"],
    ["torch.compile DiT", "DiT 0.21×", "Recompiles every chunk because cache indices change"],
    ["torch.compile VAE", "VAE 1.05×", "Small gain for a long JIT warmup"],
    ["OpenVINO GPU VAE", "VAE 0.71×", "Its 3D convolutions lack oneDNN's DPAS channels_last_3d path"],
    ["SageAttention v1", "DiT 1.01×", "Kernel gain mostly lost to a lower GPU clock"],
    ["Radial sparse attention", "DiT 1.01×", "No gain at 1,215 queries / 7,290 keys"],
    ["EchoRoPE folded into PRoPE", "rejected", "Failed the video quality gate"],
], C.x, C.y + 0.1, C.w, col_widths=[4.0, 1.7, 6.23], row_h=0.6, size=15)

OUT.parent.mkdir(parents=True, exist_ok=True)
print(deck.save(OUT))
