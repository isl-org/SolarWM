"""Wan2.2-I2V-A14B (diffusers) on one Arc Pro B70: INT8 experts, default SDPA vs Sage self-attention.

    scripts/rss_guard.sh 24 logs/wan_i2v_sdpa.log -- ./env.sh python wan/run_wan_i2v.py --arm sdpa --tag cat480
    scripts/rss_guard.sh 24 logs/wan_i2v_sage.log -- ./env.sh python wan/run_wan_i2v.py --arm sage --tag cat480

Per run (outputs in --out/<tag>_<arm>.*): frames .npy (uint8 [F, H, W, 3]), .mp4, and a JSON with total /
denoising / self-attention / expert-load times and peak VRAM. Settings follow the model card: 81 frames,
40 UniPC steps, guidance 3.5, the official negative prompt, area 480x832 (or --max-area 720*1280).
Only the transformer's self-attention changes between arms; cross-attention stays on torch SDPA.
Requires PYTHONPATH to include the isolated diffusers install (see env in the runbook).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAGE_BUILD_DIR", "/home/gta/omni_bench/kernel/build"))
import sage_causal  # noqa: E402
import wan_int8     # noqa: E402

MODEL = os.environ.get("WAN_DIR", "/home/gta/models/wan22-i2v-a14b-diffusers")
INT8 = os.environ.get("WAN_INT8_DIR", "/home/gta/models/wan22-i2v-a14b-int8")
PROMPT = ("Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline "
          "gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring "
          "crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
          "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's "
          "intricate details and the refreshing atmosphere of the seaside.")
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

STATE = dict(arm="sdpa", timing=False, events=[])


def install_attention_hook():
    """Route the transformer's self-attention through Sage when arm == 'sage' (cross-attention unchanged)."""
    import diffusers.models.transformers.transformer_wan as tw
    orig = tw.dispatch_attention_fn

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kwargs):
        self_attn = (attn_mask is None and query.shape[1] == key.shape[1] and query.shape[1] > 4096
                     and query.shape[-1] == 128 and query.dtype == torch.bfloat16)
        if not self_attn:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, *args, **kwargs)
        if STATE["timing"]:
            a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            a.record()
        if STATE["arm"] == "sage":
            out = sage_causal.forward(query.contiguous(), key.contiguous(), value.contiguous(), False)
        else:
            out = orig(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False, *args, **kwargs)
        if STATE["timing"]:
            b.record()
            STATE["events"].append((a, b))
        return out

    tw.dispatch_attention_fn = dispatch


class LazyExperts:
    """Keep at most one INT8 expert on the GPU: load on first forward, unloading the other."""

    def __init__(self, models, dirs, device):
        self.models, self.dirs, self.device, self.active, self.load_s = models, dirs, device, None, 0.0
        for i, m in enumerate(models):
            m.register_forward_pre_hook(lambda mod, args, kw, i=i: self._ensure(i), with_kwargs=True)

    def _ensure(self, i):
        if self.active == i:
            return
        t = time.perf_counter()
        if self.active is not None:
            wan_int8.unload_expert(self.models[self.active])
        wan_int8.load_expert_weights(self.models[i], self.dirs[i], self.device)
        torch.xpu.synchronize()
        self.active = i
        self.load_s += time.perf_counter() - t
        print(f"[expert] loaded {os.path.basename(self.dirs[i])} in {time.perf_counter() - t:.1f}s "
              f"({torch.xpu.memory_allocated() / 2**30:.1f} GiB allocated)", flush=True)

    def unload_all(self):
        if self.active is not None:
            wan_int8.unload_expert(self.models[self.active])
            self.active = None


def write_mp4(frames, path, fps=16):
    f, h, w, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path],
                         stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes()); p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sdpa", "sage"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--image", default=f"{MODEL}/examples/i2v_input.JPG")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-area", type=int, default=480 * 832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/home/gta/omni_bench/wan/outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.xpu.set_per_process_memory_fraction(0.94)
    dev = torch.device("xpu")
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    from diffusers.utils import load_image
    from transformers import AutoTokenizer, UMT5EncoderModel

    class Pipe(WanImageToVideoPipeline):
        @property
        def _execution_device(self):
            return dev

    install_attention_hook()
    STATE["arm"] = a.arm
    t_all = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer")
    te = UMT5EncoderModel.from_pretrained(f"{MODEL}/text_encoder", torch_dtype=torch.bfloat16).to(dev)
    vae = AutoencoderKLWan.from_pretrained(f"{MODEL}/vae", torch_dtype=torch.float32).to(dev)
    sched = UniPCMultistepScheduler.from_pretrained(f"{MODEL}/scheduler")
    experts = [wan_int8.build_expert(f"{INT8}/transformer"), wan_int8.build_expert(f"{INT8}/transformer_2")]
    lazy = LazyExperts(experts, [f"{INT8}/transformer", f"{INT8}/transformer_2"], dev)
    boundary = json.load(open(f"{MODEL}/model_index.json"))["boundary_ratio"]
    pipe = Pipe(tokenizer=tok, text_encoder=te, vae=vae, scheduler=sched, transformer=experts[0],
                transformer_2=experts[1], boundary_ratio=boundary)

    image = load_image(a.image)
    ratio = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * experts[0].config.patch_size[1]
    height = round(np.sqrt(a.max_area * ratio)) // mod * mod
    width = round(np.sqrt(a.max_area / ratio)) // mod * mod
    image = image.resize((width, height))
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, NEGATIVE, do_classifier_free_guidance=True, device=dev)
    pipe.text_encoder = None
    del te
    torch.xpu.empty_cache()
    tokens = ((a.frames - 1) // 4 + 1) * (height // 16) * (width // 16)
    print(f"[run] arm={a.arm} {width}x{height} frames={a.frames} tokens/attention={tokens} steps={a.steps}", flush=True)

    torch.xpu.reset_peak_memory_stats()
    STATE["timing"] = True
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        latents = pipe(image=image, prompt_embeds=pe, negative_prompt_embeds=ne, height=height, width=width,
                       num_frames=a.frames, guidance_scale=a.guidance, num_inference_steps=a.steps,
                       generator=gen, output_type="latent").frames
    torch.xpu.synchronize()
    t_denoise = time.perf_counter() - t0
    STATE["timing"] = False
    attn_s = sum(x.elapsed_time(y) for x, y in STATE["events"]) / 1e3
    peak = torch.xpu.max_memory_allocated() / 2**30
    lazy.unload_all()
    t1 = time.perf_counter()
    with torch.no_grad():
        lat = latents.to(dev, vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
        video = vae.decode(lat / std + mean, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    t_decode = time.perf_counter() - t1
    frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    base = f"{a.out}/{a.tag}_{a.arm}"
    np.save(f"{base}.npy", frames)
    write_mp4(frames, f"{base}.mp4")
    rec = dict(tag=a.tag, arm=a.arm, width=width, height=height, frames=a.frames, tokens_per_attention=tokens,
               steps=a.steps, guidance=a.guidance, seed=a.seed, denoise_s=round(t_denoise, 1),
               denoise_excl_load_s=round(t_denoise - lazy.load_s, 1),
               self_attention_s=round(attn_s, 1), expert_load_s=round(lazy.load_s, 1), decode_s=round(t_decode, 1),
               total_s=round(time.perf_counter() - t_all, 1), peak_denoise_gib=round(peak, 2),
               self_attention_calls=len(STATE["events"]),
               sage_build=os.path.dirname(os.path.abspath(sage_causal.__file__)) if a.arm == "sage" else None)
    json.dump(rec, open(f"{base}.json", "w"), indent=1)
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
