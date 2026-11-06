"""Load the Intel INT4 AutoRound Qwen3-Omni checkpoint on XPU (thinker only, no talker).

The checkpoint's quantization_config names modules as `thinker.model.layers...`, but
transformers 5.x calls them `thinker.model.language_model.layers...`. Without the remap
auto-round replaces nothing, every expert is allocated as a bf16 Linear (~58 GB) and the
GPU spills into host RAM until the OOM killer fires.
"""
import torch
from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MID = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
OLD, NEW = "thinker.model.layers", "thinker.model.language_model.layers"

def _allow_triton_xpu():
    """auto-round's tritonv2 backends require the pip package `triton`; the XPU build ships as
    `triton-xpu` (still `import triton`), so the requirement check rejects it. Drop that check."""
    from auto_round.inference.backend import BackendInfos
    for name in ("auto_round:tritonv2", "auto_round:tritonv2_zp"):
        BackendInfos[name].requirements = [r for r in BackendInfos[name].requirements
                                           if not r.startswith("triton")]

VRAM_FRACTION = 0.94

class _TrimCacheWhileLoading:
    """transformers 5 materializes every tensor on the device and then converts / regroups it,
    leaving freed blocks the caching allocator cannot reuse (seen: 11.6 GiB allocated, 17.7 GiB
    reserved-but-free mid-load). Return them to the driver whenever slack exceeds 1 GiB."""
    def __enter__(self):
        import transformers.core_model_loading as cml
        self.cml, self.orig = cml, cml._materialize_copy
        def wrapped(tensor, device=None, dtype=None):
            if torch.xpu.memory_reserved() - torch.xpu.memory_allocated() > 2**30:
                torch.xpu.empty_cache()
            return self.orig(tensor, device, dtype)
        cml._materialize_copy = wrapped
    def __exit__(self, *exc):
        self.cml._materialize_copy = self.orig
        torch.xpu.empty_cache()

def load(attn_implementation="sdpa", device="xpu", quant_backend="auto_round:tritonv2_zp"):
    # The Xe driver over-commits VRAM into system RAM instead of failing, which on this 30 GB
    # host ends in the global OOM killer. Cap the caching allocator so torch raises OOM instead.
    torch.xpu.set_per_process_memory_fraction(VRAM_FRACTION)
    _allow_triton_xpu()
    cfg = AutoConfig.from_pretrained(MID)
    qc = cfg.quantization_config
    qc["block_name_to_quantize"] = NEW
    qc["extra_config"] = {k.replace(OLD.replace(".", "\\."), NEW.replace(".", "\\.")): v
                          for k, v in qc.get("extra_config", {}).items()}
    # auto_round_kernel_xpu (auto-round-lib) uses AVX-512 host code and SIGILLs on this
    # Arrow Lake CPU, so pin the Triton XPU INT4 backend.
    qc["backend"] = quant_backend
    cfg.enable_audio_output = False
    with _TrimCacheWhileLoading():
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MID, config=cfg, dtype=torch.bfloat16, device_map=device, attn_implementation=attn_implementation)
    model.eval()
    _defragment(model)
    _last_token_logits(model.thinker.lm_head)
    _chunk_moe(model)
    _trim_cache_per_layer(model)
    return model, Qwen3OmniMoeProcessor.from_pretrained(MID)


def _last_token_logits(lm_head):
    """The thinker ignores logits_to_keep and projects every prefill position onto the 152K vocab
    (16K tokens -> 4.6 GiB, 64K -> ~19 GiB, OOM). Generation only reads the last position, so
    project just that one. Applied identically to the SDPA and Sage arms."""
    orig = lm_head.forward
    def forward(h):
        return orig(h[:, -1:, :]) if h.dim() == 3 and h.shape[1] > 1 else orig(h)
    lm_head.forward = forward


def _chunk_moe(model):
    """Replace auto-round's sequential MoE expert loop with a memory-lean, exact equivalent.

    auto-round materialises (tokens * top_k, hidden) buffers (gathered inputs, outputs, weighted
    outputs: ~6 GiB at 63K tokens) and does one nonzero() device sync per expert. Here: one stable
    sort of the routing assignments (single host sync for the per-expert counts), then per expert
    gather -> expert MLP -> scale by routing weight -> index_add_ into an fp32 (tokens, hidden)
    accumulator. Each token occurs at most once per expert, so the adds are deterministic.
    Applied identically to both attention arms. (Token chunking was tried first: chunk 4096 cost
    +7.5 s TTFT at 32K from repeating the 128-expert loop per chunk.)"""
    patched = 0
    for m in model.thinker.modules():
        if hasattr(m, "experts_forward") and hasattr(m, "gate") and hasattr(m, "experts"):
            experts, n_exp = m.experts, m.num_experts
            def experts_forward(h, top_k_index, top_k_weights, _experts=experts, _n=n_exp):
                k = top_k_index.shape[-1]
                flat = top_k_index.reshape(-1)
                order = torch.argsort(flat, stable=True)
                counts = torch.bincount(flat, minlength=_n).tolist()
                tok = order // k
                w = top_k_weights.reshape(-1)[order].to(torch.float32)
                acc = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
                start = 0
                for e, c in enumerate(counts):
                    if c == 0:
                        continue
                    t = tok[start:start + c]
                    y = _experts[e](h.index_select(0, t)).to(torch.float32)
                    acc.index_add_(0, t, y * w[start:start + c, None])
                    start += c
                return acc.to(h.dtype)
            m.experts_forward = experts_forward
            patched += 1
    assert patched == model.config.thinker_config.text_config.num_hidden_layers, patched


TRIM_ABOVE_TOKENS = 16384


def _trim_cache_per_layer(model):
    """At ~63K tokens every layer allocates ~1 GiB transients (fp32 norm upcasts, MoE accumulator,
    attention temporaries); the caching allocator ends up with ~5.7 GiB of unusable free blocks and
    OOMs. Release cached blocks before each decoder layer for long prefills only (one sync per
    layer, negligible vs ~1.5 s of layer compute at 63K). Same for both arms. Expandable segments
    fix fragmentation too but slowed TTFT 2.4x and decode 5x on XPU."""
    def pre_hook(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and h.dim() == 3 and h.shape[1] > TRIM_ABOVE_TOKENS:
            torch.xpu.empty_cache()
    for layer in model.thinker.model.language_model.layers:
        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)


PACK_BELOW_BYTES = 2 * 2**20
PACK_ALIGN_BYTES = 256


def _defragment(model):
    """Pack small weight tensors into one buffer per (decoder layer, dtype).

    Each expert's INT4 weights are ~0.75 MB; the caching allocator serves < 1 MB requests from
    2 MB segments, so two fit per segment and 0.5 MB is wasted: ~4 GiB reserved-but-unusable over
    6144 experts (reserved 22.2 vs allocated 18.2 GiB before the first layer; a CPU round trip
    re-creates the same pattern). That slack made the 63K prefill OOM. Modules keep views into
    the packed buffers (256-byte aligned)."""
    torch.xpu.synchronize()
    before = torch.xpu.memory_reserved() - torch.xpu.memory_allocated()
    for layer in model.thinker.model.language_model.layers:
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
                if isinstance(t, torch.nn.Parameter):
                    store[name] = torch.nn.Parameter(view, requires_grad=t.requires_grad)
                else:
                    store[name] = view
                # some quantized linears cache their tensors as plain attributes too
                for attr, val in list(vars(mod).items()):
                    if val is t:
                        setattr(mod, attr, store[name])
        del groups
        torch.xpu.empty_cache()
    torch.xpu.synchronize()
    after = torch.xpu.memory_reserved() - torch.xpu.memory_allocated()
    print(f"[defragment] reserved-but-free slack {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB, "
          f"allocated {torch.xpu.memory_allocated() / 2**30:.2f} GiB", flush=True)
