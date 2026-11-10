"""After conversion: check every fp32 source tensor exists in the int8 index, then delete the fp32 expert folder."""
import json, os, shutil, sys
src_root, int8_root = sys.argv[1], sys.argv[2]
for sub in ("transformer", "transformer_2"):
    src = json.load(open(f"{src_root}/{sub}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    dst = json.load(open(f"{int8_root}/{sub}/int8_index.json"))["weight_map"]
    missing = [k for k in src if k not in dst]
    shards = set(dst.values())
    absent = [s for s in shards if not os.path.isfile(f"{int8_root}/{sub}/{s}")]
    n_scale = sum(k.endswith(".weight_scale") for k in dst)
    if missing or absent or n_scale == 0 or not os.path.isfile(f"{int8_root}/{sub}/config.json"):
        sys.exit(f"[prune] {sub}: verification FAILED missing={missing[:5]} absent={absent} scales={n_scale}; nothing deleted")
    print(f"[prune] {sub}: {len(src)} source tensors all present, {n_scale} int8 weights, {len(shards)} shards — OK", flush=True)
for sub in ("transformer", "transformer_2"):
    shutil.rmtree(f"{src_root}/{sub}")
    print(f"[prune] deleted {src_root}/{sub}", flush=True)
