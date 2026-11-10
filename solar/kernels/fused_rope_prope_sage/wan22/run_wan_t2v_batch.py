"""Wan2.2-T2V-A14B (diffusers) batch generation for VBench-2.0 on one Arc Pro B70.

Same integration as run_wan_i2v.py (INT8 or SVDQuant experts, one on the GPU at a time; self-attention on
torch SDPA or Sage), for a list of prompts. All prompts are encoded once with UMT5, which is then freed.
Videos go to <out>/<arm_name>/<Dimension>/<prompt[:180]>-<index>.mp4 (the VBench-2.0 naming); per-video
timings are appended to <out>/<arm_name>/timings.jsonl. Existing videos are skipped (resumable).

    # one arm over the subset (arm_name defaults to <arm>[_svdq])
    scripts/rss_guard.sh 24 logs/vb_sdpa.log -- ./env.sh python wan/run_wan_t2v_batch.py --arm sdpa
    SAGE_BUILD_DIR=kernel/build_int4 ... --arm sage --arm-name sage_int4
    ... --arm sdpa --linear svdq
    # SVDQuant calibration (one prompt that is not in the subset)
    ... --arm sdpa --prompt "<text>" --seed 7 --frames 33 --calib-out wan/outputs/t2v_calib_amax.pt

Settings follow the model card: 81 frames, 40 UniPC steps, guidance 4.0 (high-noise expert) / 3.0
(low-noise expert), the official negative prompt, 16 fps; 832x480.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_wan_i2v as R  # noqa: E402  (attention hook, lazy experts, calibration, mp4 writer)
import wan_int8           # noqa: E402
import wan_svdq           # noqa: E402

MODEL = os.environ.get("WAN_T2V_DIR", "/home/gta/models/wan22-t2v-a14b-diffusers")
INT8 = os.environ.get("WAN_T2V_INT8_DIR", "/home/gta/models/wan22-t2v-a14b-int8")
SVDQ = os.environ.get("WAN_T2V_SVDQ_DIR", "/home/gta/models/wan22-t2v-a14b-svdq")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sdpa", "sage"], required=True)
    ap.add_argument("--arm-name", help="output folder name (default <arm>[_svdq])")
    ap.add_argument("--linear", choices=["int8", "svdq"], default="int8")
    ap.add_argument("--subset", default=f"{HERE}/vbench/subset.json")
    ap.add_argument("--prompt", help="single prompt instead of the subset (e.g. calibration)")
    ap.add_argument("--seed", type=int, default=7, help="seed for --prompt")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--guidance-2", type=float, default=3.0)
    ap.add_argument("--out", default=f"{HERE}/vbench/videos")
    ap.add_argument("--calib-out")
    ap.add_argument("--limit", type=int, help="only the first N videos of the subset")
    a = ap.parse_args()
    name = a.arm_name or (a.arm + ("_svdq" if a.linear == "svdq" else ""))
    root = os.path.join(a.out, name)
    if a.prompt:
        jobs = [dict(dimension="custom", prompt=a.prompt, index=0, seed=a.seed, file=f"custom/{a.prompt[:180]}-0.mp4")]
    else:
        jobs = json.load(open(a.subset))[: a.limit]
    todo = [j for j in jobs if not os.path.isfile(os.path.join(root, j["file"]))]
    print(f"[batch] arm={name} {len(jobs)} videos, {len(todo)} to generate", flush=True)
    if not todo:
        return

    torch.xpu.set_per_process_memory_fraction(0.94)
    dev = torch.device("xpu")
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    class Pipe(WanPipeline):
        @property
        def _execution_device(self):
            return dev

    R.install_attention_hook()
    R.STATE["arm"] = a.arm
    tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer")
    te = UMT5EncoderModel.from_pretrained(f"{MODEL}/text_encoder", torch_dtype=torch.bfloat16).to(dev)
    vae = AutoencoderKLWan.from_pretrained(f"{MODEL}/vae", torch_dtype=torch.float32).to(dev)
    sched = UniPCMultistepScheduler.from_pretrained(f"{MODEL}/scheduler")
    wdir, build = (SVDQ, wan_svdq.build_expert) if a.linear == "svdq" else (INT8, wan_int8.build_expert)
    experts = [build(f"{wdir}/transformer"), build(f"{wdir}/transformer_2")]
    lazy = R.LazyExperts(experts, [f"{wdir}/transformer", f"{wdir}/transformer_2"], dev)
    calib = R.install_calibration(experts, ["transformer", "transformer_2"]) if a.calib_out else None
    boundary = json.load(open(f"{MODEL}/model_index.json"))["boundary_ratio"]
    pipe = Pipe(tokenizer=tok, text_encoder=te, vae=vae, scheduler=sched, transformer=experts[0],
                transformer_2=experts[1], boundary_ratio=boundary)

    embeds = {}
    with torch.no_grad():
        for j in todo:
            if j["prompt"] not in embeds:
                pe, ne = pipe.encode_prompt(j["prompt"], R.NEGATIVE, do_classifier_free_guidance=True, device=dev)
                embeds[j["prompt"]] = (pe.cpu(), ne.cpu())
    pipe.text_encoder = None
    del te
    torch.xpu.empty_cache()
    tokens = ((a.frames - 1) // 4 + 1) * (a.height // 16) * (a.width // 16)
    print(f"[batch] {len(embeds)} prompts encoded; {a.width}x{a.height} frames={a.frames} tokens/attention={tokens}", flush=True)

    for n, j in enumerate(todo):
        pe, ne = (t.to(dev) for t in embeds[j["prompt"]])
        R.STATE["events"].clear()
        R.STATE["timing"] = True
        load0 = lazy.load_s
        torch.xpu.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            latents = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=a.height, width=a.width,
                           num_frames=a.frames, guidance_scale=a.guidance, guidance_scale_2=a.guidance_2,
                           num_inference_steps=a.steps, generator=torch.Generator("cpu").manual_seed(j["seed"]),
                           output_type="latent").frames
        torch.xpu.synchronize()
        t_den = time.perf_counter() - t0
        R.STATE["timing"] = False
        attn_s = sum(x.elapsed_time(y) for x, y in R.STATE["events"]) / 1e3
        peak = torch.xpu.max_memory_allocated() / 2**30
        lazy.unload_all()
        t1 = time.perf_counter()
        with torch.no_grad():
            lat = latents.to(dev, vae.dtype)
            mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
            std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
            video = vae.decode(lat / std + mean, return_dict=False)[0]
            frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
        del latents, lat, video
        torch.xpu.empty_cache()
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
        path = os.path.join(root, j["file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        R.write_mp4(frames, path + ".part.mp4")
        os.replace(path + ".part.mp4", path)
        rec = dict(arm=name, attention=a.arm, linear=a.linear, dimension=j["dimension"], prompt=j["prompt"],
                   index=j["index"], seed=j["seed"], width=a.width, height=a.height, frames=a.frames,
                   steps=a.steps, denoise_s=round(t_den, 1), denoise_excl_load_s=round(t_den - (lazy.load_s - load0), 1),
                   self_attention_s=round(attn_s, 1), decode_s=round(time.perf_counter() - t1, 1),
                   peak_denoise_gib=round(peak, 2), self_attention_calls=len(R.STATE["events"]),
                   sage_build=os.path.dirname(os.path.abspath(R.sage_causal.__file__)) if a.arm == "sage" else None)
        with open(os.path.join(root, "timings.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[batch] {n + 1}/{len(todo)} {j['dimension']} seed {j['seed']}: denoise {rec['denoise_excl_load_s']} s, "
              f"attention {rec['self_attention_s']} s", flush=True)
    if calib is not None:
        torch.save({k: v.cpu() for k, v in calib.items()}, a.calib_out)
        print(f"[calib] {len(calib)} channel-max vectors -> {a.calib_out}", flush=True)


if __name__ == "__main__":
    main()
