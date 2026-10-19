"""Unit tests for the INT4 SVDQuant toolkit."""

import json
import torch

from scripts.quantization.svdquant_int4 import (
    compute_joint_smoothing_scales,
    compute_smoothing_scale,
    dequantize_int4_weights,
    export_svdquant_safetensors,
    extract_low_rank_svd,
    gptq_quantize_residual,
    pack_int4_signed,
    quantize_activations_dynamic,
    quantize_weights_rtn_g64,
    reshard_svdquant_directory,
    search_act_clip_ratio,
    unpack_int4_signed,
)


def test_pack_unpack_int4():
    unpacked = torch.tensor([[-8, -7, 0, 7, -1, 3], [1, -2, 5, -6, 2, -4]], dtype=torch.int8)
    packed = pack_int4_signed(unpacked)
    assert packed.shape == (2, 3)

    restored = unpack_int4_signed(packed)
    assert torch.equal(restored, unpacked)


def test_quantize_weights_rtn_g64():
    oc, ic = 16, 128
    w = torch.randn(oc, ic, dtype=torch.bfloat16) * 0.1
    qweight, w_scale = quantize_weights_rtn_g64(w, group_size=64)

    assert qweight.shape == (oc, ic // 2)
    assert w_scale.shape == (oc, ic // 64)
    assert qweight.dtype == torch.int8
    assert w_scale.dtype == torch.bfloat16

    w_deq = dequantize_int4_weights(qweight, w_scale, group_size=64)
    # Cosine similarity between original and dequantized weight
    cos = torch.nn.functional.cosine_similarity(w.float().flatten(), w_deq.float().flatten(), dim=0)
    assert cos.item() > 0.95


def test_quantize_activations_dynamic():
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    x_qdq = quantize_activations_dynamic(x, group_size=64, clip_ratio=0.8)
    assert x_qdq.shape == x.shape
    assert x_qdq.dtype == torch.bfloat16

    cos = torch.nn.functional.cosine_similarity(x.float().flatten(), x_qdq.float().flatten(), dim=0)
    assert cos.item() > 0.95


def test_svd_exact_reconstruction_before_quantization():
    oc, ic, rank = 128, 128, 16
    w = torch.randn(oc, ic, dtype=torch.bfloat16)
    s = torch.exp(torch.randn(ic, dtype=torch.bfloat16) * 0.5).clamp(0.1, 10.0)
    ws = w.float() * s.float().unsqueeze(0)

    lora_a, lora_b, resid = extract_low_rank_svd(ws.to(torch.bfloat16), s, rank=rank)
    assert lora_a.shape == (rank, ic)
    assert lora_b.shape == (oc, rank)
    assert resid.shape == (oc, ic)

    # Verify: X @ W.T == (X / s) @ resid.T + X @ lora_a.T @ lora_b.T
    x = torch.randn(32, ic, dtype=torch.bfloat16)
    y_true = x.float() @ w.float().T
    s_inv = 1.0 / s.float()
    xs = x.float() * s_inv.unsqueeze(0)
    y_recon = xs @ resid.float().T + (x.float() @ lora_a.float().T) @ lora_b.float().T

    rel_diff = torch.norm(y_true - y_recon) / torch.norm(y_true)
    # bfloat16 precision threshold (~0.003 is typical bfloat16 relative rounding error)
    assert rel_diff.item() < 5e-3


def test_joint_smoothing():
    ic = 128
    act_max = torch.ones(ic) * 5.0
    w1 = torch.ones(64, ic) * 0.1
    w2 = torch.ones(128, ic) * 0.2
    s = compute_joint_smoothing_scales(act_max, [w1, w2], alpha=0.5)
    assert s.shape == (ic,)
    assert (s > 0).all()


def test_gptq_quantize_residual():
    oc, ic = 64, 128
    resid = torch.randn(oc, ic, dtype=torch.bfloat16) * 0.05
    x = torch.randn(256, ic)
    h = x.T @ x

    qweight, w_scale, w_recon = gptq_quantize_residual(resid, h, group_size=64, damping=0.01)
    assert qweight.shape == (oc, ic // 2)
    assert w_scale.shape == (oc, ic // 64)
    assert w_recon.shape == (oc, ic)

    # GPTQ reconstruction error should be small
    rel_err = torch.norm(resid.float() - w_recon.float()) / torch.norm(resid.float())
    assert rel_err.item() < 0.35


def test_search_act_clip_ratio():
    tokens, oc, ic, rank = 128, 64, 128, 16
    x = torch.randn(tokens, ic, dtype=torch.bfloat16)
    w = torch.randn(oc, ic, dtype=torch.bfloat16) * 0.1
    b = torch.randn(oc, dtype=torch.bfloat16) * 0.01
    s = torch.ones(ic, dtype=torch.bfloat16)

    lora_a, lora_b, resid = extract_low_rank_svd(w, s, rank=rank)
    _, _, r_recon = gptq_quantize_residual(resid, x.float().T @ x.float(), group_size=64)

    best_rho, errors = search_act_clip_ratio(x, w, b, r_recon, s, lora_a, lora_b)
    assert best_rho in (0.6, 0.7, 0.8, 0.9, 1.0)
    assert len(errors) == 5


def test_export_and_manifest_roundtrip(tmp_path):
    oc, ic = 64, 128
    dummy_tensors = {
        "blocks.0.self_attn.q.qweight": torch.zeros((oc, ic // 2), dtype=torch.int8),
        "blocks.0.self_attn.q.w_scale": torch.ones((oc, ic // 64), dtype=torch.bfloat16),
        "blocks.0.self_attn.q.smooth_scale": torch.ones(ic, dtype=torch.bfloat16),
        "blocks.0.self_attn.q.lora_a": torch.zeros((32, ic), dtype=torch.bfloat16),
        "blocks.0.self_attn.q.lora_b": torch.zeros((oc, 32), dtype=torch.bfloat16),
    }
    quantized_sets = {"set0": dummy_tensors}
    manifest = {
        "format_version": "solarwm-svdquant-v1",
        "rank": 32,
        "group_size": 64,
        "num_sets": 1,
    }

    manifest_path = export_svdquant_safetensors(tmp_path, quantized_sets, manifest)
    assert manifest_path.exists()

    with open(manifest_path) as f:
        loaded = json.load(f)
    assert loaded["format_version"] == "solarwm-svdquant-v1"
    assert "svdquant_weights.safetensors" in loaded["shards"]


def test_export_multiset_safetensors(tmp_path):
    oc, ic = 64, 128
    dummy_tensors_0 = {
        "blocks.0.self_attn.q.qweight": torch.zeros((oc, ic // 2), dtype=torch.int8),
    }
    dummy_tensors_1 = {
        "blocks.0.self_attn.q.qweight": torch.ones((oc, ic // 2), dtype=torch.int8),
    }
    quantized_sets = {"set0": dummy_tensors_0, "set1": dummy_tensors_1}
    manifest = {
        "format_version": "solarwm-svdquant-v1",
        "rank": 32,
        "group_size": 64,
        "num_sets": 2,
    }

    manifest_path = export_svdquant_safetensors(tmp_path, quantized_sets, manifest)
    assert manifest_path.exists()

    with open(manifest_path) as f:
        loaded = json.load(f)
    assert loaded["shards"] == [
        "svdquant_weights_set0.safetensors",
        "svdquant_weights_set1.safetensors",
    ]
    assert loaded["weight_map"]["set0.blocks.0.self_attn.q.qweight"] == "svdquant_weights_set0.safetensors"
    assert loaded["weight_map"]["set1.blocks.0.self_attn.q.qweight"] == "svdquant_weights_set1.safetensors"
    assert (tmp_path / "svdquant_weights_set0.safetensors").is_file()
    assert (tmp_path / "svdquant_weights_set1.safetensors").is_file()


def test_reshard_svdquant_directory(tmp_path):
    oc, ic = 64, 128
    from safetensors.torch import save_file

    # Simulate old arbitrary chunking where set0 and set1 were combined into a single file
    combined = {
        "set0.blocks.0.self_attn.q.qweight": torch.zeros((oc, ic // 2), dtype=torch.int8),
        "set1.blocks.0.self_attn.q.qweight": torch.ones((oc, ic // 2), dtype=torch.int8),
    }
    save_file(combined, str(tmp_path / "svdquant_weights-00000-of-00001.safetensors"))
    manifest = {
        "format_version": "solarwm-svdquant-v1",
        "shards": ["svdquant_weights-00000-of-00001.safetensors"],
        "weight_map": {
            "set0.blocks.0.self_attn.q.qweight": "svdquant_weights-00000-of-00001.safetensors",
            "set1.blocks.0.self_attn.q.qweight": "svdquant_weights-00000-of-00001.safetensors",
        },
    }
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    shards = reshard_svdquant_directory(tmp_path)
    assert shards == [
        "svdquant_weights_set0.safetensors",
        "svdquant_weights_set1.safetensors",
    ]
    assert not (tmp_path / "svdquant_weights-00000-of-00001.safetensors").exists()
    assert (tmp_path / "svdquant_weights_set0.safetensors").is_file()
    assert (tmp_path / "svdquant_weights_set1.safetensors").is_file()

    with open(tmp_path / "manifest.json") as f:
        updated = json.load(f)
    assert updated["shards"] == shards
    assert updated["weight_map"]["set0.blocks.0.self_attn.q.qweight"] == "svdquant_weights_set0.safetensors"
    assert updated["weight_map"]["set1.blocks.0.self_attn.q.qweight"] == "svdquant_weights_set1.safetensors"

