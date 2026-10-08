#!/usr/bin/env python3
"""Bit-exactness test for WS13 (Encoded KV Ring) vs Old Mirrored Circular Cache."""

import torch
import torch.nn as nn
from solarwm.backends.wan22.runtime.modeling.causal_model import (
    CausalWanSelfAttention,
    CausalWanModel,
    _write_mirrored_ring,
    _cache_index,
    _set_cache_index,
    echorope_apply,
    block_relativistic_rope,
)
from solarwm.backends.wan22.runtime.modeling.camera_prope import (
    _prepare_apply_fns_all_dim,
    transform_relative_viewmats,
    prope_apply_fns_separate_cached,
)
from solarwm.backends.wan22.runtime.modeling.attention import attention


def _read_ring_slice(ring: torch.Tensor, start: int, length: int) -> torch.Tensor:
    capacity = ring.shape[1]
    start = start % capacity
    if start + length <= capacity:
        return ring[:, start : start + length]
    part1 = capacity - start
    part2 = length - part1
    return torch.cat([ring[:, start:], ring[:, :part2]], dim=1)


def _write_ring(ring: torch.Tensor, start: int, values: torch.Tensor) -> None:
    capacity = ring.shape[1]
    start = start % capacity
    length = values.shape[1]
    if start + length <= capacity:
        ring[:, start : start + length].copy_(values)
    else:
        part1 = capacity - start
        part2 = length - part1
        ring[:, start:].copy_(values[:, :part1])
        ring[:, :part2].copy_(values[:, part1:])


def run_test():
    torch.manual_seed(42)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    dtype = torch.bfloat16

    dim = 64
    num_heads = 4
    head_dim = dim // num_heads  # 16
    frame_seq_length = 16
    local_attn_size = 6  # 6 frames max window = 96 tokens
    chunk_frames = 2     # 2 frames per chunk = 32 tokens
    num_chunks = 6
    chunk_tokens = chunk_frames * frame_seq_length  # 32
    capacity_tokens = local_attn_size * frame_seq_length  # 96

    attn = CausalWanSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=local_attn_size,
        sink_size=0,
        qk_norm=False,
        frame_seq_length=frame_seq_length,
        use_echorope=True,
        camera_attention_mode="fused_prope",
    ).to(device=device, dtype=dtype)

    # Old Mirrored Cache
    cache_old = {
        "k": torch.zeros((1, 2 * capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "v": torch.zeros((1, 2 * capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "global_end_index": 0,
        "local_end_index": 0,
        "_circular_kv_cache": True,
        "_circular_capacity": capacity_tokens,
        "_circular_ring_start": 0,
    }
    cam_meta_old = {
        "viewmats": torch.zeros((1, 2 * capacity_tokens, 4, 4), dtype=torch.float32, device=device),
        "K": torch.zeros((1, 2 * capacity_tokens, 3, 3), dtype=torch.float32, device=device),
    }

    # New WS13 Unmirrored + Encoded Cache
    cache_new = {
        "k": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "v": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "k_encoded": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "v_encoded": torch.zeros((1, capacity_tokens, num_heads, head_dim), dtype=dtype, device=device),
        "global_end_index": 0,
        "local_end_index": 0,
        "_circular_kv_cache": True,
        "_circular_capacity": capacity_tokens,
        "_circular_ring_start": 0,
    }
    cam_meta_new = {
        "viewmats": torch.zeros((1, capacity_tokens, 4, 4), dtype=torch.float32, device=device),
        "K": torch.zeros((1, capacity_tokens, 3, 3), dtype=torch.float32, device=device),
    }

    # Frequencies for RoPE
    freqs = torch.randn((100, head_dim // 2), dtype=torch.complex128, device=device)
    grid_sizes = torch.tensor([[chunk_frames, 4, 4]], device=device)

    print("Beginning multi-chunk rollout test...")

    for chunk_idx in range(num_chunks):
        # 5 forward steps per chunk (4 denoise + 1 commit)
        # Shared camera viewmats for this chunk
        chunk_viewmats = torch.eye(4, dtype=torch.float32, device=device).repeat(1, chunk_tokens, 1, 1)
        chunk_viewmats[..., 0, 3] = torch.linspace(chunk_idx * 0.1, (chunk_idx + 1) * 0.1, chunk_tokens, device=device)
        chunk_Ks = torch.eye(3, dtype=torch.float32, device=device).repeat(1, chunk_tokens, 1, 1)

        prope_cache_old = {}
        prope_cache_new = {}

        for step_idx in range(5):
            is_commit = (step_idx == 4)
            policy = "commit_detached" if is_commit else "inference_direct"

            # Input tokens
            x_in = torch.randn((1, chunk_tokens, dim), dtype=dtype, device=device)

            # --- OLD EXECUTION ---
            # Forward projection
            q_old = attn.norm_q(attn.q(x_in)).view(1, chunk_tokens, num_heads, head_dim)
            k_old = attn.norm_k(attn.k(x_in)).view(1, chunk_tokens, num_heads, head_dim)
            v_old = attn.v(x_in).view(1, chunk_tokens, num_heads, head_dim)

            # Circular cache ring write
            g_end_old = _cache_index(cache_old, "global_end_index")
            l_end_old = _cache_index(cache_old, "local_end_index")
            r_start_old = int(cache_old["_circular_ring_start"])
            cap_old = int(cache_old["_circular_capacity"])
            num_new = chunk_tokens

            cur_end_old = g_end_old + num_new
            num_evict_old = max(0, l_end_old + num_new - cap_old)
            new_l_end_old = min(cap_old, l_end_old + num_new)
            new_r_start_old = (r_start_old + num_evict_old) % cap_old
            w_start_old = (r_start_old + l_end_old) % cap_old

            with torch.no_grad():
                _write_mirrored_ring(cache_old["k"], start=w_start_old, values=k_old.detach())
                _write_mirrored_ring(cache_old["v"], start=w_start_old, values=v_old.detach())
                _write_mirrored_ring(cam_meta_old["viewmats"], start=w_start_old, values=chunk_viewmats.detach())
                _write_mirrored_ring(cam_meta_old["K"], start=w_start_old, values=chunk_Ks.detach())

            win_start_old = new_r_start_old + max(0, new_l_end_old - attn.max_attention_size)
            win_end_old = new_r_start_old + new_l_end_old
            k_win_old = cache_old["k"][:, win_start_old:win_end_old]
            v_win_old = cache_old["v"][:, win_start_old:win_end_old]
            vm_win_old = cam_meta_old["viewmats"][:, win_start_old:win_end_old]
            k_cam_win_old = cam_meta_old["K"][:, win_start_old:win_end_old]

            # RoPE Q and K window
            roped_q_old, roped_k_win_old, _ = attn._rope_q_and_window_k(
                q=q_old,
                k_window=k_win_old,
                grid_sizes=grid_sizes,
                freqs=freqs,
                frame_seqlen=frame_seq_length,
                num_new_tokens=num_new,
            )

            # PRoPE
            attn_q_old, attn_k_old, attn_v_old, apply_fn_o_old = attn._apply_fused_prope(
                roped_q_old,
                roped_k_win_old,
                v_win_old,
                chunk_viewmats,
                chunk_Ks,
                kv_cam_viewmats=vm_win_old,
                kv_cam_K=k_cam_win_old,
                prope_cache=prope_cache_old,
                kv_window=(win_start_old, win_end_old),
            )
            out_old = attention(attn_q_old, attn_k_old, attn_v_old)
            if apply_fn_o_old is not None:
                out_old = apply_fn_o_old(out_old.transpose(1, 2)).transpose(1, 2)
            out_old = attn.o(out_old.flatten(2))

            # --- NEW WS13 EXECUTION ---
            # Forward projection (exact same x_in)
            q_new = attn.norm_q(attn.q(x_in)).view(1, chunk_tokens, num_heads, head_dim)
            k_new = attn.norm_k(attn.k(x_in)).view(1, chunk_tokens, num_heads, head_dim)
            v_new = attn.v(x_in).view(1, chunk_tokens, num_heads, head_dim)

            g_end_new = _cache_index(cache_new, "global_end_index")
            l_end_new = _cache_index(cache_new, "local_end_index")
            r_start_new = int(cache_new["_circular_ring_start"])
            cap_new = int(cache_new["_circular_capacity"])

            cur_end_new = g_end_new + num_new
            num_evict_new = max(0, l_end_new + num_new - cap_new)
            new_l_end_new = min(cap_new, l_end_new + num_new)
            new_r_start_new = (r_start_new + num_evict_new) % cap_new
            w_start_new = (r_start_new + l_end_new) % cap_new

            # Unmirrored ring writes
            with torch.no_grad():
                _write_ring(cache_new["k"], start=w_start_new, values=k_new.detach())
                _write_ring(cache_new["v"], start=w_start_new, values=v_new.detach())
                _write_ring(cam_meta_new["viewmats"], start=w_start_new, values=chunk_viewmats.detach())
                _write_ring(cam_meta_new["K"], start=w_start_new, values=chunk_Ks.detach())

            num_window_tokens = new_l_end_new - max(0, new_l_end_new - attn.max_attention_size)
            num_hist_tokens = max(0, num_window_tokens - num_new)
            num_window_frames = num_window_tokens // frame_seq_length
            num_new_frames = num_new // frame_seq_length
            num_hist_frames = num_hist_tokens // frame_seq_length

            history_key = (g_end_new, new_r_start_new, num_hist_tokens)
            if num_hist_tokens > 0 and cache_new.get("_encoded_history_key") != history_key:
                k_hist_raw = _read_ring_slice(cache_new["k"], start=new_r_start_new, length=num_hist_tokens)
                v_hist_raw = _read_ring_slice(cache_new["v"], start=new_r_start_new, length=num_hist_tokens)
                hist_cam_viewmats = _read_ring_slice(cam_meta_new["viewmats"], start=new_r_start_new, length=num_hist_tokens)
                hist_cam_K = _read_ring_slice(cam_meta_new["K"], start=new_r_start_new, length=num_hist_tokens)

                # RoPE history (start_frame = 0)
                hist_grid = grid_sizes.clone()
                hist_grid[:, 0] = num_hist_frames
                rope_fn = echorope_apply if attn.use_echorope else block_relativistic_rope
                roped_k_hist = rope_fn(k_hist_raw, hist_grid, freqs, start_frame=0).type_as(v_hist_raw)

                # PRoPE history
                hist_kv_key = ("hist_kv", history_key)
                apply_fn_kv_hist = prope_cache_new.get(hist_kv_key)
                if apply_fn_kv_hist is None:
                    _, apply_fn_kv_hist, _ = _prepare_apply_fns_all_dim(
                        head_dim=attn.head_dim,
                        viewmats=transform_relative_viewmats(hist_cam_viewmats, attn.camera_translation_transform),
                        Ks=hist_cam_K,
                        patches_x=None, patches_y=None, image_width=None, image_height=None,
                    )
                    prope_cache_new[hist_kv_key] = apply_fn_kv_hist

                k_hist_encoded = apply_fn_kv_hist(roped_k_hist.transpose(1, 2)).transpose(1, 2)
                v_hist_encoded = apply_fn_kv_hist(v_hist_raw.transpose(1, 2)).transpose(1, 2)

                cache_new["k_encoded"][:, :num_hist_tokens].copy_(k_hist_encoded)
                cache_new["v_encoded"][:, :num_hist_tokens].copy_(v_hist_encoded)
                cache_new["_encoded_history_key"] = history_key

            # RoPE Q and new K (start_frame = num_hist_frames)
            q_grid = grid_sizes.clone()
            q_grid[:, 0] = num_new_frames
            rope_fn = echorope_apply if attn.use_echorope else block_relativistic_rope
            roped_q_new = rope_fn(q_new, q_grid, freqs, start_frame=num_hist_frames).type_as(v_new)
            roped_k_new = rope_fn(k_new, q_grid, freqs, start_frame=num_hist_frames).type_as(v_new)

            # PRoPE Q and new K/V
            entry_q = prope_cache_new.get("q")
            if entry_q is None:
                apply_fn_q_new, _, apply_fn_o_new = _prepare_apply_fns_all_dim(
                    head_dim=attn.head_dim,
                    viewmats=transform_relative_viewmats(chunk_viewmats, attn.camera_translation_transform),
                    Ks=chunk_Ks,
                    patches_x=None, patches_y=None, image_width=None, image_height=None,
                )
                prope_cache_new["q"] = (apply_fn_q_new, apply_fn_o_new)
            else:
                apply_fn_q_new, apply_fn_o_new = entry_q

            apply_fn_kv_new = prope_cache_new.get("new_kv")
            if apply_fn_kv_new is None:
                _, apply_fn_kv_new, _ = _prepare_apply_fns_all_dim(
                    head_dim=attn.head_dim,
                    viewmats=transform_relative_viewmats(chunk_viewmats, attn.camera_translation_transform),
                    Ks=chunk_Ks,
                    patches_x=None, patches_y=None, image_width=None, image_height=None,
                )
                prope_cache_new["new_kv"] = apply_fn_kv_new

            attn_q_new = apply_fn_q_new(roped_q_new.transpose(1, 2)).transpose(1, 2)
            k_new_encoded = apply_fn_kv_new(roped_k_new.transpose(1, 2)).transpose(1, 2)
            v_new_encoded = apply_fn_kv_new(v_new.transpose(1, 2)).transpose(1, 2)

            cache_new["k_encoded"][:, num_hist_tokens:num_window_tokens].copy_(k_new_encoded)
            cache_new["v_encoded"][:, num_hist_tokens:num_window_tokens].copy_(v_new_encoded)

            attn_k_new = cache_new["k_encoded"][:, :num_window_tokens]
            attn_v_new = cache_new["v_encoded"][:, :num_window_tokens]

            out_new = attention(attn_q_new, attn_k_new, attn_v_new)
            if apply_fn_o_new is not None:
                out_new = apply_fn_o_new(out_new.transpose(1, 2)).transpose(1, 2)
            out_new = attn.o(out_new.flatten(2))

            # Compare outputs
            diff = (out_old - out_new).abs().max().item()
            print(f"Chunk {chunk_idx} Step {step_idx} ({policy}): diff = {diff}")
            assert diff == 0.0, f"Mismatch at chunk {chunk_idx} step {step_idx}: max diff {diff}"

            # Apply cache updates on commit
            if is_commit:
                cache_old["_circular_ring_start"] = new_r_start_old
                _set_cache_index(cache_old, "global_end_index", cur_end_old)
                _set_cache_index(cache_old, "local_end_index", new_l_end_old)

                cache_new["_circular_ring_start"] = new_r_start_new
                _set_cache_index(cache_new, "global_end_index", cur_end_new)
                _set_cache_index(cache_new, "local_end_index", new_l_end_new)

    print("\nALL 6 CHUNKS AND 30 STEPS PASSED WITH 0.0000000 DIFF (BIT-EXACT MATCH)!")


if __name__ == "__main__":
    run_test()
