"""Q*K^T quantisation schemes for SageAttention-style kernels, evaluated on captured activations.

All functions work on one KV head at a time:
  q    [R, G, D] fp32   sampled query rows of the G query heads that share this KV head
  qpos [R]              absolute positions of the rows (causal: row r sees keys <= qpos[r])
  qmean[R, G, D] fp32   the 256-row block mean of each row's block (only for Q smoothing)
  k, v [S, D] fp32
Scores use scale = D^-0.5; P is rounded to bf16 before P*V and the row sum, as the kernels do.
`rot` (optional, [D, D] orthogonal) is applied to Q and K after K centring: (qR)(kR)^T = qk^T exactly.
"""
import math

import torch

K_MEAN_FULL, K_MEAN_STRIDE, K_MEAN_TILE = 2048, 512, 64


def k_mean_sampled(k):
    s = k.shape[0]
    if s < K_MEAN_FULL:
        return k.mean(0)
    idx = torch.arange(s, device=k.device)
    return k[(idx % K_MEAN_STRIDE) < K_MEAN_TILE].mean(0)


def quant_tok(x, levels, ste=False):
    """Symmetric per-row (last dim) quantisation, round half to even. ste: straight-through gradient."""
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / levels
    if ste:
        scale = scale.detach()
        return x + (torch.round(x / scale) * scale - x).detach()
    return torch.round(x / scale) * scale


def hadamard(d, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(d)


def causal_attend(scores, v, qpos):
    """scores [R, G, S] (already scaled); returns [R, G, D] with P rounded to bf16."""
    keys = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.masked_fill(keys[None, None, :] > qpos[:, None, None], float("-inf"))
    p = torch.exp(scores - scores.amax(-1, keepdim=True)).to(torch.bfloat16).float()
    return torch.einsum("rgs,sd->rgd", p, v) / p.sum(-1, keepdim=True)


def reference(q, k, v, qpos):
    return causal_attend(torch.einsum("rgd,sd->rgs", q, k) * q.shape[-1] ** -0.5, v, qpos)


def scheme_scores(q, k, bits, qpos=None, qmean=None, rot=None, q_smooth=False, ste=False):
    """Scaled scores [R, G, S] for a quantised scheme (K always centred by its sampled mean)."""
    levels = 7 if bits == 4 else 127
    kc = k - k_mean_sampled(k)
    qq = q - qmean if q_smooth else q
    if rot is not None:
        kc, qq = kc @ rot, qq @ rot
    s = torch.einsum("rgd,sd->rgs", quant_tok(qq, levels, ste), quant_tok(kc, levels, ste))
    if q_smooth:   # exact fp32 correction dS = qmean . K'  (rotation-invariant)
        s = s + torch.einsum("rgd,sd->rgs", qmean, k - k_mean_sampled(k))
    return s * q.shape[-1] ** -0.5


def cos_per_head(out, ref):
    """[R, G, D] -> cosine per head [G] over rows and channels."""
    o, r = out.flatten(0, 0).transpose(0, 1).flatten(1), ref.transpose(0, 1).flatten(1)
    return torch.nn.functional.cosine_similarity(o.double(), r.double(), dim=1)
