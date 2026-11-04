"""Qwen3-VL-8B-Instruct (INT4 W4A16 text decoder) for long-context text prefill on one Arc Pro B70.

Pieces:
  load()            INT4 checkpoint (auto-round, Triton XPU backend), VRAM cap, small-tensor packing
  Int8KVCache       preallocated INT8 KV cache, one scale per (token, KV head); each layer is
                    dequantised on demand into one bf16 buffer shared by all layers
  prefill()         chunked prefill (every chunk but the last is a multiple of the chunk size, so the
                    causal diagonal offset is always tile-aligned); logits for the last position only
  greedy_decode()   token-by-token decode on the same cache

KV cache at 256K: bf16 would be 36 GiB; INT8 is 18 GiB + 0.28 GiB of scales. The shared bf16
dequant buffers are 2 x Smax x Hkv x D x 2 bytes (1 GiB at 256K).
"""
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, CacheLayerMixin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vl_attention  # noqa: E402

QDIR = os.environ.get("QWEN3VL_INT4_DIR", "/home/gta/models/qwen3vl8b-w4a16/Qwen3-VL-8B-Instruct-w4g128")
VRAM_FRACTION = 0.94
PACK_BELOW_BYTES = 2 * 2**20
PACK_ALIGN_BYTES = 256


def _allow_triton_xpu():
    """auto-round's Triton backends require the pip package `triton`; the XPU build is `triton-xpu`."""
    from auto_round.inference.backend import BackendInfos
    for name in ("auto_round:tritonv2", "auto_round:tritonv2_zp"):
        if name in BackendInfos:
            BackendInfos[name].requirements = [r for r in BackendInfos[name].requirements if not r.startswith("triton")]


def load(quant_backend="auto_round:tritonv2_zp"):
    # The Xe driver spills VRAM into host RAM instead of failing; cap torch so it raises OOM instead.
    torch.xpu.set_per_process_memory_fraction(VRAM_FRACTION)
    _allow_triton_xpu()
    vl_attention.register()
    cfg = AutoConfig.from_pretrained(QDIR)
    # auto_round_kernel_xpu uses AVX-512 host code and SIGILLs on this CPU: pin the Triton backend.
    cfg.quantization_config["backend"] = quant_backend
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QDIR, config=cfg, dtype=torch.bfloat16, device_map="xpu",
        attn_implementation={"text_config": vl_attention.NAME, "vision_config": "sdpa"})
    model.eval()
    _defragment(model)
    return model, AutoProcessor.from_pretrained(QDIR)


def _defragment(model):
    """Pack < 2 MB tensors (INT4 scales / zeros) into one buffer per (decoder layer, dtype): the
    caching allocator serves them from 2 MB segments and would strand the rest."""
    torch.xpu.synchronize()
    before = torch.xpu.memory_reserved() - torch.xpu.memory_allocated()
    for layer in model.model.language_model.layers:
        groups = {}
        for mod in layer.modules():
            for store in (mod._parameters, mod._buffers):
                for name, t in store.items():
                    if t is not None and t.device.type == "xpu" and t.nbytes < PACK_BELOW_BYTES:
                        groups.setdefault(t.dtype, []).append((mod, store, name, t))
        for dtype, items in groups.items():
            esize = torch.empty((), dtype=dtype).element_size()
            align = max(PACK_ALIGN_BYTES // esize, 1)
            offsets, total = [], 0
            for *_, t in items:
                offsets.append(total)
                total += (t.numel() + align - 1) // align * align
            flat = torch.empty(total, dtype=dtype, device="xpu")
            for (mod, store, name, t), off in zip(items, offsets):
                view = flat[off:off + t.numel()].view(t.shape)
                view.copy_(t)
                store[name] = torch.nn.Parameter(view, requires_grad=False) if isinstance(t, torch.nn.Parameter) else view
                for attr, val in list(vars(mod).items()):
                    if val is t:
                        setattr(mod, attr, store[name])
        torch.xpu.empty_cache()
    torch.xpu.synchronize()
    after = torch.xpu.memory_reserved() - torch.xpu.memory_allocated()
    print(f"[defragment] slack {before / 2**30:.2f} -> {after / 2**30:.2f} GiB, "
          f"allocated {torch.xpu.memory_allocated() / 2**30:.2f} GiB", flush=True)


class _DequantBuffers:
    """bf16 K/V buffers [1, Smax, Hkv, D] shared by every layer (each layer's attention consumes them
    before the next layer overwrites them)."""
    def __init__(self, max_len, kv_heads, head_dim, device):
        self.k = torch.empty(1, max_len, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)


class Int8KVLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, max_len, kv_heads, head_dim, device, buffers):
        super().__init__()
        self.max_len, self.n, self.buf = max_len, 0, buffers
        self.kq = torch.empty(1, max_len, kv_heads, head_dim, dtype=torch.int8, device=device)
        self.vq = torch.empty_like(self.kq)
        self.ks = torch.empty(1, max_len, kv_heads, 1, dtype=torch.bfloat16, device=device)
        self.vs = torch.empty_like(self.ks)
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        pass

    @staticmethod
    def _quantize(x, q_out, s_out):
        """x [1, c, H, D] -> symmetric int8 per (token, head)."""
        amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8)
        scale = amax / 127.0
        q_out.copy_(torch.round(x.float() / scale).clamp_(-127, 127).to(torch.int8))
        s_out.copy_(scale.to(torch.bfloat16))

    def update(self, key_states, value_states, *args, **kwargs):
        """key/value_states [1, Hkv, c, D] -> full dequantised K/V [1, Hkv, n, D] (views of [1,S,H,D])."""
        c = key_states.shape[2]
        p, n = self.n, self.n + c
        assert n <= self.max_len, f"cache full ({n} > {self.max_len})"
        self._quantize(key_states.transpose(1, 2), self.kq[:, p:n], self.ks[:, p:n])
        self._quantize(value_states.transpose(1, 2), self.vq[:, p:n], self.vs[:, p:n])
        self.n = n
        k, v = self.buf.k[:, :n], self.buf.v[:, :n]
        torch.mul(self.kq[:, :n], self.ks[:, :n], out=k)
        torch.mul(self.vq[:, :n], self.vs[:, :n], out=v)
        return k.transpose(1, 2), v.transpose(1, 2)

    def get_mask_sizes(self, query_length):
        return self.n + query_length, 0

    def get_seq_length(self):
        return self.n

    def get_max_length(self):
        return self.max_len

    def reset(self):
        self.n = 0


class Int8KVCache(Cache):
    def __init__(self, model, max_len):
        tc = model.config.text_config
        dev = next(model.parameters()).device
        max_len = (max_len + 255) // 256 * 256
        buf = _DequantBuffers(max_len, tc.num_key_value_heads, tc.head_dim, dev)
        super().__init__(layers=[Int8KVLayer(max_len, tc.num_key_value_heads, tc.head_dim, dev, buf)
                                 for _ in range(tc.num_hidden_layers)])

    @staticmethod
    def bytes_for(model, max_len):
        tc = model.config.text_config
        per_tok = tc.num_hidden_layers * 2 * tc.num_key_value_heads * (tc.head_dim + 2) + 2 * 2 * tc.num_key_value_heads * tc.head_dim
        return per_tok * ((max_len + 255) // 256 * 256)


@torch.no_grad()
def prefill(model, input_ids, cache, chunk=4096):
    """Chunked prefill of input_ids [1, S]; returns logits [1, 1, V] for the last position."""
    s = input_ids.shape[1]
    out = None
    for p in range(0, s, chunk):
        ids = input_ids[:, p:p + chunk]
        cp = torch.arange(p, p + ids.shape[1], device=input_ids.device)
        out = model.model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True)
    return model.lm_head(out.last_hidden_state[:, -1:])


@torch.no_grad()
def greedy_decode(model, logits, cache, max_new_tokens, eos_ids):
    tokens, times = [], []
    pos = cache.get_seq_length()
    dev = logits.device
    for _ in range(max_new_tokens):
        tok = int(logits[0, -1].argmax())
        tokens.append(tok)
        if tok in eos_ids:
            break
        t = time.perf_counter()
        out = model.model(input_ids=torch.tensor([[tok]], device=dev), past_key_values=cache,
                          cache_position=torch.tensor([pos], device=dev), use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:])
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t)
        pos += 1
    return tokens, times
