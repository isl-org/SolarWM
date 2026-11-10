"""Self-test of the INT8 expert path and the Sage hook on a tiny random WanTransformer3DModel."""
import json, os, sys, shutil, torch
sys.path.insert(0, "/home/gta/omni_bench/wan")
import wan_int8, run_wan_i2v as R
from diffusers import WanTransformer3DModel
from safetensors.torch import save_file
root = "/tmp/wan_tiny"; shutil.rmtree(root, ignore_errors=True); os.makedirs(f"{root}/src/transformer")
cfg = dict(patch_size=[1, 2, 2], num_attention_heads=4, attention_head_dim=128, in_channels=36, out_channels=16,
           text_dim=4096, freq_dim=256, ffn_dim=1024, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
           eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024, pos_embed_seq_len=None)
torch.manual_seed(0)
ref = WanTransformer3DModel(**cfg).float()
for p in ref.parameters(): p.data.normal_(0, 0.02)
sd = {k: v.contiguous() for k, v in ref.state_dict().items()}
keys = list(sd); half = len(keys) // 2
save_file({k: sd[k] for k in keys[:half]}, f"{root}/src/transformer/s1.safetensors")
save_file({k: sd[k] for k in keys[half:]}, f"{root}/src/transformer/s2.safetensors")
json.dump({"weight_map": {**{k: "s1.safetensors" for k in keys[:half]}, **{k: "s2.safetensors" for k in keys[half:]}}},
          open(f"{root}/src/transformer/diffusion_pytorch_model.safetensors.index.json", "w"))
ref.save_config(f"{root}/src/transformer")
wan_int8.convert_expert(f"{root}/src/transformer", f"{root}/int8/transformer")
dev = torch.device("xpu")
m = wan_int8.load_expert_weights(wan_int8.build_expert(f"{root}/int8/transformer"), f"{root}/int8/transformer", dev)
ref = ref.to(dev, torch.bfloat16)
# 5 latent frames x 24 x 64 latent -> 5 * 12 * 32 = 1920 tokens; use 9 frames x 32 x 64 -> 9*16*32 = 4608 tokens (> 4096: Sage path)
x = torch.randn(1, 36, 9, 32, 64, device=dev, dtype=torch.bfloat16); t = torch.tensor([500.0], device=dev)
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
R.install_attention_hook()
def run(model, arm):
    R.STATE["arm"] = arm
    with torch.no_grad():
        return model(hidden_states=x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0].float()
o_ref = run(ref, "sdpa"); o_int8 = run(m, "sdpa"); o_sage = run(m, "sage")
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
print(f"int8 vs bf16 (SDPA): cos={cos(o_int8, o_ref):.5f} | int8+Sage vs int8+SDPA: cos={cos(o_sage, o_int8):.5f} | finite={torch.isfinite(o_sage).all().item()}")
wan_int8.unload_expert(m); m = wan_int8.load_expert_weights(m, f"{root}/int8/transformer", dev)
print(f"reload after unload: cos={cos(run(m, 'sdpa'), o_int8):.5f}")
