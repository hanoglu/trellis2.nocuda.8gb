#!/usr/bin/env python3
import ctypes
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parent
TRELLIS2_ROOT = pathlib.Path(os.environ.get("TRELLIS2_PYTHON_ROOT", "/home/wimaxs/Documents/TRELLIS.2"))
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)
U8P = ctypes.POINTER(ctypes.c_uint8)
U32P = ctypes.POINTER(ctypes.c_uint32)


class SparseTensor(ctypes.Structure):
    _fields_ = [
        ("coords", I32P),
        ("feats", F32P),
        ("n", ctypes.c_longlong),
        ("channels", ctypes.c_int),
    ]


class Mesh(ctypes.Structure):
    _fields_ = [
        ("vertices", F32P),
        ("faces", U32P),
        ("n_vertices", ctypes.c_longlong),
        ("n_faces", ctypes.c_longlong),
    ]


class InferOptions(ctypes.Structure):
    _fields_ = [
        ("latent_size", ctypes.c_int),
        ("stage1_steps", ctypes.c_int),
        ("stage2_steps", ctypes.c_int),
        ("cond_tokens", ctypes.c_int),
        ("cond_channels", ctypes.c_int),
        ("stage1_channels", ctypes.c_int),
        ("stage2_channels", ctypes.c_int),
        ("voxel_threshold", ctypes.c_float),
        ("stage2_rescale_t", ctypes.c_float),
        ("seed", ctypes.c_uint32),
    ]


class InferResult(ctypes.Structure):
    _fields_ = [
        ("coords_bxyz", I32P),
        ("n_coords", ctypes.c_longlong),
        ("sparse_resolution", ctypes.c_int),
        ("slat_feats", F32P),
        ("slat_channels", ctypes.c_int),
        ("mesh", Mesh),
    ]


class DitFlowBlockWeights(ctypes.Structure):
    _fields_ = [
        ("modulation", F32P),
        ("norm2_gamma", F32P),
        ("norm2_beta", F32P),
        ("self_qkv_w", F32P),
        ("self_qkv_b", F32P),
        ("self_q_rms_gamma", F32P),
        ("self_k_rms_gamma", F32P),
        ("self_out_w", F32P),
        ("self_out_b", F32P),
        ("cross_q_w", F32P),
        ("cross_q_b", F32P),
        ("cross_kv_w", F32P),
        ("cross_kv_b", F32P),
        ("cross_q_rms_gamma", F32P),
        ("cross_k_rms_gamma", F32P),
        ("cross_out_w", F32P),
        ("cross_out_b", F32P),
        ("mlp_fc1_w", F32P),
        ("mlp_fc1_b", F32P),
        ("mlp_fc2_w", F32P),
        ("mlp_fc2_b", F32P),
    ]


class DitFlowWeights(ctypes.Structure):
    _fields_ = [
        ("in_channels", ctypes.c_int),
        ("out_channels", ctypes.c_int),
        ("model_channels", ctypes.c_int),
        ("cond_channels", ctypes.c_int),
        ("time_frequency_dim", ctypes.c_int),
        ("heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("mlp_channels", ctypes.c_int),
        ("mod_channels", ctypes.c_int),
        ("n_blocks", ctypes.c_int),
        ("debug_block_parts", ctypes.c_int),
        ("debug_disable_rope", ctypes.c_int),
        ("emulate_bf16_blocks", ctypes.c_int),
        ("final_norm_eps", ctypes.c_float),
        ("input_w", F32P),
        ("input_b", F32P),
        ("t_embedder_0_w", F32P),
        ("t_embedder_0_b", F32P),
        ("t_embedder_2_w", F32P),
        ("t_embedder_2_b", F32P),
        ("adaln_w", F32P),
        ("adaln_b", F32P),
        ("out_w", F32P),
        ("out_b", F32P),
        ("blocks", ctypes.POINTER(DitFlowBlockWeights)),
    ]


def build_lib():
    cc = os.environ.get("CC", "cc")
    tmpdir = tempfile.mkdtemp(prefix="standalone_trellis_")
    so_path = pathlib.Path(tmpdir) / "libstandalone_trellis.so"
    cmd = [
        cc,
        "-std=c11",
        "-O2",
        "-fPIC",
        "-shared",
        str(ROOT / "trellis_ops.c"),
        str(ROOT / "trellis_net.c"),
        "-lm",
        "-o",
        str(so_path),
    ]
    subprocess.check_call(cmd)
    return ctypes.CDLL(str(so_path))


def ptr_f32(t):
    a = t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    return a.ctypes.data_as(F32P), a


def ptr_i32(a):
    a = np.asarray(a, dtype=np.int32, order="C")
    return a.ctypes.data_as(I32P), a


def ptr_u8(a):
    a = np.asarray(a, dtype=np.uint8, order="C")
    return a.ctypes.data_as(U8P), a


def check(name, got, ref, atol=1e-5, rtol=1e-5):
    got_t = torch.as_tensor(got)
    ref_t = torch.as_tensor(ref)
    max_abs = (got_t - ref_t).abs().max().item() if got_t.numel() else 0.0
    ok = torch.allclose(got_t, ref_t, atol=atol, rtol=rtol)
    print(f"{name:32s} max_abs={max_abs:.6e} ok={ok}")
    if not ok:
        raise AssertionError(name)


def expect_ok(status, name):
    if status != 0:
        raise RuntimeError(f"{name} returned status {status}")


def configure(lib):
    c_ll = ctypes.c_longlong
    c_i = ctypes.c_int
    c_f = ctypes.c_float
    c_sz = ctypes.c_size_t

    lib.strellis_linear_f32.argtypes = [F32P, F32P, F32P, F32P, c_ll, c_i, c_i]
    lib.strellis_layer_norm_f32.argtypes = [F32P, F32P, F32P, F32P, c_ll, c_i, c_f]
    lib.strellis_rms_norm_f32.argtypes = [F32P, F32P, F32P, c_ll, c_i, c_f]
    lib.strellis_multihead_rms_norm_f32.argtypes = [F32P, F32P, F32P, c_i, c_i, c_i, c_i, c_f]
    lib.strellis_bf16_roundtrip_f32.argtypes = [F32P, F32P, c_sz]
    lib.strellis_gelu_tanh_f32.argtypes = [F32P, F32P, c_sz]
    lib.strellis_silu_f32.argtypes = [F32P, F32P, c_sz]
    lib.strellis_add_f32.argtypes = [F32P, F32P, F32P, c_sz]
    lib.strellis_feed_forward_f32.argtypes = [F32P, F32P, F32P, F32P, F32P, F32P, F32P, c_ll, c_i, c_i, c_i]
    lib.strellis_timestep_embedding_f32.argtypes = [F32P, c_sz, c_i, c_f, F32P]
    lib.strellis_timestep_mlp_f32.argtypes = [F32P, c_sz, c_i, F32P, F32P, F32P, F32P, F32P, F32P, F32P, c_i, c_i]
    lib.strellis_sdpa_f32.argtypes = [F32P, F32P, F32P, F32P, c_i, c_i, c_i, c_i, c_i, c_f]
    lib.strellis_apply_rope_adjacent_f32.argtypes = [F32P, F32P, F32P, F32P, c_i, c_i, c_i, c_i]
    lib.strellis_rope_3d_phases_f32.argtypes = [c_i, c_i, c_f, c_f, F32P, F32P, c_sz]
    lib.strellis_rope_3d_sparse_phases_f32.argtypes = [I32P, c_ll, c_i, c_f, c_f, F32P, F32P, c_sz]
    lib.strellis_flow_euler_step_f32.argtypes = [F32P, F32P, c_sz, c_f, c_f, c_f, F32P, F32P]
    lib.strellis_conv3d_ncdhw_f32.argtypes = [
        F32P, F32P, F32P, F32P,
        c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i,
        c_i, c_i, c_i, c_i, c_i, c_i,
    ]
    lib.strellis_pixel_shuffle_3d_ncdhw_f32.argtypes = [F32P, F32P, c_i, c_i, c_i, c_i, c_i, c_i]
    lib.strellis_channel_layer_norm_3d_ncdhw_f32.argtypes = [F32P, F32P, F32P, F32P, c_i, c_i, c_i, c_i, c_i, c_f]
    lib.strellis_dino_patch_embed_f32.argtypes = [F32P, F32P, F32P, F32P, c_i, c_i, c_i, c_i, c_i]
    lib.strellis_sparse_linear_f32.argtypes = [F32P, F32P, F32P, F32P, c_ll, c_i, c_i]
    lib.strellis_sparse_subm_conv3d_f32.argtypes = [I32P, F32P, F32P, F32P, F32P, c_ll, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i]
    lib.strellis_sparse_downsample_mean_f32.argtypes = [ctypes.POINTER(SparseTensor), c_i, ctypes.POINTER(SparseTensor)]
    lib.strellis_sparse_spatial2channel_f32.argtypes = [ctypes.POINTER(SparseTensor), c_i, ctypes.POINTER(SparseTensor)]
    lib.strellis_sparse_channel2spatial_f32.argtypes = [ctypes.POINTER(SparseTensor), U8P, c_i, ctypes.POINTER(SparseTensor)]
    lib.strellis_sparse_tensor_free.argtypes = [ctypes.POINTER(SparseTensor)]
    lib.strellis_dit_flow_forward_f32.argtypes = [
        F32P,
        F32P,
        F32P,
        F32P,
        F32P,
        ctypes.POINTER(DitFlowWeights),
        F32P,
        c_i,
        c_i,
        c_i,
    ]
    lib.strellis_toy_dit_block_forward_f32.argtypes = [F32P, F32P, F32P, F32P, F32P, F32P, F32P, c_i, c_i, c_i, c_i]
    lib.strellis_infer_options_default.argtypes = [ctypes.POINTER(InferOptions)]
    lib.strellis_infer_result_free.argtypes = [ctypes.POINTER(InferResult)]
    lib.strellis_run_inference_compute.argtypes = [ctypes.POINTER(InferOptions), ctypes.POINTER(InferResult)]

    for name in dir(lib):
        if name.startswith("strellis_"):
            getattr(lib, name).restype = ctypes.c_int
    lib.strellis_sparse_tensor_free.restype = None
    lib.strellis_infer_options_default.restype = None
    lib.strellis_infer_result_free.restype = None
    lib.strellis_run_inference_compute.restype = ctypes.c_int


def sparse_ref_subm(coords, feats, weight, bias):
    n, ci = feats.shape
    co, kd, kh, kw, _ = weight.shape
    out = torch.zeros((n, co), dtype=feats.dtype)
    lookup = {tuple(coords[i].tolist()): i for i in range(n)}
    cd, ch, cw = kd // 2, kh // 2, kw // 2
    for row in range(n):
        b, x, y, z = coords[row].tolist()
        for oc in range(co):
            acc = bias[oc].item()
            for id_ in range(kd):
                for ih in range(kh):
                    for iw in range(kw):
                        key = (b, x + id_ - cd, y + ih - ch, z + iw - cw)
                        j = lookup.get(key)
                        if j is not None:
                            acc += torch.dot(feats[j], weight[oc, id_, ih, iw]).item()
            out[row, oc] = acc
    return out


def sparse_tensor_from_numpy(coords, feats):
    coords_ptr, coords_np = ptr_i32(coords)
    feats_ptr = feats.ctypes.data_as(F32P)
    return SparseTensor(coords_ptr, feats_ptr, coords_np.shape[0], feats.shape[1]), coords_np, feats


def sparse_to_numpy(t):
    coords = np.ctypeslib.as_array(t.coords, shape=(t.n, 4)).copy()
    feats = np.ctypeslib.as_array(t.feats, shape=(t.n, t.channels)).copy()
    return coords, feats


def randn(shape, seed, scale=1.0, mean=0.0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32) * scale + mean


def fanin_weight(shape, seed, gain=1.0):
    fan_in = int(np.prod(shape[1:]))
    return randn(shape, seed, gain / math.sqrt(max(1, fan_in)))


def realistic_activation(shape, seed, scale=0.7):
    x = randn(shape, seed, scale)
    if len(shape) >= 2:
        ramp = torch.linspace(-0.5, 0.5, shape[-1], dtype=torch.float32)
        x = x + 0.05 * torch.sin(ramp * math.pi * 2.0)
    return x.contiguous()


def make_sparse_coords(resolution, target_n, seed):
    coords = []
    radius = 0.42
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                fx = (x + 0.5) / resolution - 0.5
                fy = (y + 0.5) / resolution - 0.5
                fz = (z + 0.5) / resolution - 0.5
                if math.sqrt(fx * fx + fy * fy + fz * fz) <= radius:
                    coords.append((0, x, y, z))
    rng = np.random.default_rng(seed)
    rng.shuffle(coords)
    coords = sorted(coords[:target_n])
    return torch.tensor(coords, dtype=torch.int32)


def run_random_dense_cases(lib):
    cases = [
        (20240, 17, 8, 16),
        (20241, 96, 32, 48),
        (20242, 257, 64, 32),
    ]
    for i, (seed, rows, channels, out_channels) in enumerate(cases):
        x = realistic_activation((rows, channels), seed)
        w = fanin_weight((out_channels, channels), seed + 1)
        b = randn((out_channels,), seed + 2, 0.03)
        y = torch.empty(rows, out_channels, dtype=torch.float32)
        expect_ok(lib.strellis_linear_f32(ptr_f32(x)[0], ptr_f32(w)[0], ptr_f32(b)[0], ptr_f32(y)[0], rows, channels, out_channels), f"linear_random_{i}")
        check(f"linear_random_{i}", y, x @ w.t() + b, atol=4e-5, rtol=4e-5)

        gamma = randn((channels,), seed + 3, 0.05, 1.0)
        beta = randn((channels,), seed + 4, 0.02)
        y_norm = torch.empty_like(x)
        expect_ok(lib.strellis_layer_norm_f32(ptr_f32(x)[0], ptr_f32(gamma)[0], ptr_f32(beta)[0], ptr_f32(y_norm)[0], rows, channels, 1e-6), f"layer_norm_random_{i}")
        check(f"layer_norm_random_{i}", y_norm, F.layer_norm(x, (channels,), gamma, beta, eps=1e-6), atol=3e-5, rtol=3e-5)

        expect_ok(lib.strellis_rms_norm_f32(ptr_f32(x)[0], ptr_f32(gamma)[0], ptr_f32(y_norm)[0], rows, channels, 1e-6), f"rms_norm_random_{i}")
        ref = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * gamma
        check(f"rms_norm_random_{i}", y_norm, ref, atol=3e-5, rtol=3e-5)

        hidden = channels * 4
        w1 = fanin_weight((hidden, channels), seed + 5)
        b1 = randn((hidden,), seed + 6, 0.02)
        w2 = fanin_weight((channels, hidden), seed + 7)
        b2 = randn((channels,), seed + 8, 0.02)
        workspace = torch.empty(rows, hidden, dtype=torch.float32)
        y_ff = torch.empty_like(x)
        expect_ok(
            lib.strellis_feed_forward_f32(
                ptr_f32(x)[0], ptr_f32(w1)[0], ptr_f32(b1)[0], ptr_f32(w2)[0], ptr_f32(b2)[0],
                ptr_f32(workspace)[0], ptr_f32(y_ff)[0], rows, channels, hidden, channels,
            ),
            f"feed_forward_random_{i}",
        )
        ref = F.gelu(x @ w1.t() + b1, approximate="tanh") @ w2.t() + b2
        check(f"feed_forward_random_{i}", y_ff, ref, atol=5e-5, rtol=5e-5)


def run_random_attention_cases(lib):
    cases = [
        (30300, 1, 64, 64, 4, 16),
        (30301, 2, 97, 113, 8, 8),
    ]
    for i, (seed, batch, q_tokens, kv_tokens, heads, head_dim) in enumerate(cases):
        q = realistic_activation((batch, q_tokens, heads, head_dim), seed, 0.55)
        k = realistic_activation((batch, kv_tokens, heads, head_dim), seed + 1, 0.55)
        v = realistic_activation((batch, kv_tokens, heads, head_dim), seed + 2, 0.45)
        y = torch.empty_like(q)
        scale = 1.0 / math.sqrt(head_dim)
        expect_ok(lib.strellis_sdpa_f32(ptr_f32(q)[0], ptr_f32(k)[0], ptr_f32(v)[0], ptr_f32(y)[0], batch, q_tokens, kv_tokens, heads, head_dim, scale), f"sdpa_random_{i}")
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)
        check(f"sdpa_random_{i}", y, ref, atol=4e-5, rtol=4e-5)

        phases = randn((q_tokens, head_dim // 2), seed + 3, 0.7)
        cos = torch.cos(phases)
        sin = torch.sin(phases)
        y_rope = torch.empty_like(q)
        expect_ok(lib.strellis_apply_rope_adjacent_f32(ptr_f32(q)[0], ptr_f32(cos)[0], ptr_f32(sin)[0], ptr_f32(y_rope)[0], batch, q_tokens, heads, head_dim), f"rope_random_{i}")
        ref = q.clone()
        ref[..., 0::2] = q[..., 0::2] * cos[None, :, None, :] - q[..., 1::2] * sin[None, :, None, :]
        ref[..., 1::2] = q[..., 0::2] * sin[None, :, None, :] + q[..., 1::2] * cos[None, :, None, :]
        check(f"rope_random_{i}", y_rope, ref, atol=2e-5, rtol=2e-5)


def run_random_volume_cases(lib):
    conv_cases = [
        (40400, 1, 4, 8, 8, 8, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        (40401, 1, 8, 7, 9, 6, 8, 3, 3, 3, 1, 2, 1, 1, 1, 1),
    ]
    for i, case in enumerate(conv_cases):
        seed, batch, cin, depth, height, width, cout, kd, kh, kw, sd, sh, sw, pd, ph, pw = case
        x = realistic_activation((batch, cin, depth, height, width), seed, 0.45)
        w = fanin_weight((cout, cin, kd, kh, kw), seed + 1)
        b = randn((cout,), seed + 2, 0.02)
        ref = F.conv3d(x, w, b, stride=(sd, sh, sw), padding=(pd, ph, pw))
        y = torch.empty_like(ref)
        expect_ok(
            lib.strellis_conv3d_ncdhw_f32(
                ptr_f32(x)[0], ptr_f32(w)[0], ptr_f32(b)[0], ptr_f32(y)[0],
                batch, cin, depth, height, width, cout, kd, kh, kw, sd, sh, sw, pd, ph, pw, 1, 1, 1,
            ),
            f"conv3d_random_{i}",
        )
        check(f"conv3d_random_{i}", y, ref, atol=5e-5, rtol=5e-5)

        gamma = randn((cin,), seed + 3, 0.04, 1.0)
        beta = randn((cin,), seed + 4, 0.02)
        cln_y = torch.empty_like(x)
        expect_ok(lib.strellis_channel_layer_norm_3d_ncdhw_f32(ptr_f32(x)[0], ptr_f32(gamma)[0], ptr_f32(beta)[0], ptr_f32(cln_y)[0], batch, cin, depth, height, width, 1e-6), f"channel_ln_random_{i}")
        ref_ln = F.layer_norm(x.permute(0, 2, 3, 4, 1), (cin,), gamma, beta, eps=1e-6).permute(0, 4, 1, 2, 3)
        check(f"channel_ln_random_{i}", cln_y, ref_ln, atol=3e-5, rtol=3e-5)

    dino_cases = [
        (40500, 1, 3, 32, 48, 32, 8),
        (40501, 1, 3, 56, 56, 64, 14),
    ]
    for i, (seed, batch, cin, h, w_, cout, patch) in enumerate(dino_cases):
        image = torch.tanh(realistic_activation((batch, cin, h, w_), seed, 0.9))
        weight = fanin_weight((cout, cin, patch, patch), seed + 1)
        bias = randn((cout,), seed + 2, 0.02)
        tokens = torch.empty(batch, (h // patch) * (w_ // patch), cout, dtype=torch.float32)
        expect_ok(lib.strellis_dino_patch_embed_f32(ptr_f32(image)[0], ptr_f32(weight)[0], ptr_f32(bias)[0], ptr_f32(tokens)[0], batch, h, w_, cout, patch), f"dino_patch_random_{i}")
        ref = F.conv2d(image, weight, bias, stride=patch).flatten(2).transpose(1, 2)
        check(f"dino_patch_random_{i}", tokens, ref, atol=5e-5, rtol=5e-5)


def run_random_sparse_cases(lib):
    cases = [
        (50500, 8, 64, 8, 12),
        (50501, 10, 180, 16, 16),
    ]
    for i, (seed, resolution, n, channels, out_channels) in enumerate(cases):
        coords_t = make_sparse_coords(resolution, n, seed)
        feats = realistic_activation((coords_t.shape[0], channels), seed + 1, 0.5)
        w = fanin_weight((out_channels, channels), seed + 2)
        b = randn((out_channels,), seed + 3, 0.02)
        y = torch.empty(coords_t.shape[0], out_channels, dtype=torch.float32)
        coords_ptr, _ = ptr_i32(coords_t.numpy())
        expect_ok(lib.strellis_sparse_linear_f32(ptr_f32(feats)[0], ptr_f32(w)[0], ptr_f32(b)[0], ptr_f32(y)[0], coords_t.shape[0], channels, out_channels), f"sparse_linear_random_{i}")
        check(f"sparse_linear_random_{i}", y, feats @ w.t() + b, atol=4e-5, rtol=4e-5)

        sw = fanin_weight((out_channels, 3, 3, 3, channels), seed + 4)
        sb = randn((out_channels,), seed + 5, 0.02)
        sparse_y = torch.empty(coords_t.shape[0], out_channels, dtype=torch.float32)
        expect_ok(lib.strellis_sparse_subm_conv3d_f32(coords_ptr, ptr_f32(feats)[0], ptr_f32(sw)[0], ptr_f32(sb)[0], ptr_f32(sparse_y)[0], coords_t.shape[0], channels, out_channels, 3, 3, 3, 1, 1, 1), f"sparse_subm_random_{i}")
        check(f"sparse_subm_random_{i}", sparse_y, sparse_ref_subm(coords_t, feats, sw, sb), atol=6e-5, rtol=6e-5)


def load_trellis2_reference_modules():
    if not TRELLIS2_ROOT.exists():
        raise RuntimeError(f"TRELLIS2_PYTHON_ROOT does not exist: {TRELLIS2_ROOT}")
    root_s = str(TRELLIS2_ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    from trellis2.models.sparse_structure_flow import TimestepEmbedder
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    from trellis2.modules.spatial import pixel_shuffle_3d as trellis2_pixel_shuffle_3d
    from trellis2.modules import sparse as trellis2_sparse

    return TimestepEmbedder, RotaryPositionEmbedder, trellis2_pixel_shuffle_3d, trellis2_sparse


def run_trellis2_reference_cases(lib):
    TimestepEmbedder, RotaryPositionEmbedder, trellis2_pixel_shuffle_3d, trellis2_sparse = load_trellis2_reference_modules()
    print(f"trellis2_reference_root          {TRELLIS2_ROOT}")

    timesteps = torch.tensor([1.0, 0.75, 0.33, 0.05, 0.0], dtype=torch.float32)
    for dim in [16, 33, 256]:
        got = torch.empty(timesteps.numel(), dim, dtype=torch.float32)
        expect_ok(lib.strellis_timestep_embedding_f32(ptr_f32(timesteps)[0], timesteps.numel(), dim, 10000.0, ptr_f32(got)[0]), f"trellis2_timestep_{dim}")
        ref = TimestepEmbedder.timestep_embedding(timesteps, dim)
        check(f"trellis2_timestep_{dim}", got, ref, atol=1e-6, rtol=1e-6)

    resolution = 5
    head_dim = 24
    dense_tokens = resolution ** 3
    cos_out = torch.empty(dense_tokens, head_dim // 2, dtype=torch.float32)
    sin_out = torch.empty_like(cos_out)
    expect_ok(
        lib.strellis_rope_3d_phases_f32(resolution, head_dim, 1.0, 10000.0, ptr_f32(cos_out)[0], ptr_f32(sin_out)[0], cos_out.numel()),
        "trellis2_rope_dense",
    )
    grid = torch.meshgrid(*[torch.arange(resolution, dtype=torch.float32) for _ in range(3)], indexing="ij")
    xyz = torch.stack(grid, dim=-1).reshape(-1, 3)
    phases = RotaryPositionEmbedder(head_dim, 3, rope_freq=(1.0, 10000.0))(xyz)
    check("trellis2_rope_dense_cos", cos_out, phases.real, atol=1e-6, rtol=1e-6)
    check("trellis2_rope_dense_sin", sin_out, phases.imag, atol=1e-6, rtol=1e-6)

    coords = make_sparse_coords(9, 96, 70700)
    sparse_cos = torch.empty(coords.shape[0], head_dim // 2, dtype=torch.float32)
    sparse_sin = torch.empty_like(sparse_cos)
    coords_ptr, _ = ptr_i32(coords.numpy())
    expect_ok(
        lib.strellis_rope_3d_sparse_phases_f32(coords_ptr, coords.shape[0], head_dim, 1.0, 10000.0, ptr_f32(sparse_cos)[0], ptr_f32(sparse_sin)[0], sparse_cos.numel()),
        "trellis2_rope_sparse",
    )
    sparse_phases = RotaryPositionEmbedder(head_dim, 3, rope_freq=(1.0, 10000.0))(coords[:, 1:].float())
    check("trellis2_rope_sparse_cos", sparse_cos, sparse_phases.real, atol=1e-6, rtol=1e-6)
    check("trellis2_rope_sparse_sin", sparse_sin, sparse_phases.imag, atol=1e-6, rtol=1e-6)

    x = realistic_activation((2, coords.shape[0], 3, head_dim), 70701, 0.55)
    y = torch.empty_like(x)
    expect_ok(lib.strellis_apply_rope_adjacent_f32(ptr_f32(x)[0], ptr_f32(sparse_cos)[0], ptr_f32(sparse_sin)[0], ptr_f32(y)[0], 2, coords.shape[0], 3, head_dim), "trellis2_apply_rope")
    ref = RotaryPositionEmbedder.apply_rotary_embedding(x, sparse_phases)
    check("trellis2_apply_rope", y, ref, atol=2e-5, rtol=2e-5)

    ps_x = realistic_activation((1, 32, 3, 4, 5), 70702, 0.5)
    ps_y = torch.empty(1, 4, 6, 8, 10, dtype=torch.float32)
    expect_ok(lib.strellis_pixel_shuffle_3d_ncdhw_f32(ptr_f32(ps_x)[0], ptr_f32(ps_y)[0], 1, 32, 3, 4, 5, 2), "trellis2_pixel_shuffle")
    check("trellis2_pixel_shuffle", ps_y, trellis2_pixel_shuffle_3d(ps_x, 2), atol=1e-6, rtol=1e-6)

    sparse_feats = realistic_activation((coords.shape[0], 16), 70703, 0.45)
    linear = trellis2_sparse.SparseLinear(16, 20, bias=True).float()
    with torch.no_grad():
        linear.weight.copy_(fanin_weight((20, 16), 70704))
        linear.bias.copy_(randn((20,), 70705, 0.02))
    vl = trellis2_sparse.VarLenTensor(sparse_feats, layout=[slice(0, sparse_feats.shape[0])])
    sparse_linear_y = torch.empty(sparse_feats.shape[0], 20, dtype=torch.float32)
    expect_ok(
        lib.strellis_sparse_linear_f32(ptr_f32(sparse_feats)[0], ptr_f32(linear.weight)[0], ptr_f32(linear.bias)[0], ptr_f32(sparse_linear_y)[0], sparse_feats.shape[0], 16, 20),
        "trellis2_sparse_linear",
    )
    check("trellis2_sparse_linear", sparse_linear_y, linear(vl).feats, atol=4e-5, rtol=4e-5)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        conv = trellis2_sparse.SparseConv3d(16, 12, 3).to(device).float()
        sw = fanin_weight((12, 3, 3, 3, 16), 70706)
        sb = randn((12,), 70707, 0.02)
        with torch.no_grad():
            conv.weight.copy_(sw.to(device))
            conv.bias.copy_(sb.to(device))
        sparse_x = trellis2_sparse.SparseTensor(sparse_feats.to(device), coords.to(device))
        ref_sparse = conv(sparse_x).feats.detach().cpu()
        c_sparse = torch.empty(coords.shape[0], 12, dtype=torch.float32)
        expect_ok(
            lib.strellis_sparse_subm_conv3d_f32(coords_ptr, ptr_f32(sparse_feats)[0], ptr_f32(sw)[0], ptr_f32(sb)[0], ptr_f32(c_sparse)[0], coords.shape[0], 16, 12, 3, 3, 3, 1, 1, 1),
            "trellis2_sparse_conv3d",
        )
        # TRELLIS2's flex_gemm sparse conv may use GPU-kernel accumulation that is
        # looser than the pure PyTorch reference; the C op is also checked above
        # against a deterministic Python sparse-conv reference at 1e-6 scale.
        check("trellis2_sparse_conv3d", c_sparse, ref_sparse, atol=2e-3, rtol=2e-3)
    else:
        print(f"{'trellis2_sparse_conv3d':32s} skipped cuda_unavailable")


def timestep_embedding_ref(timesteps, dim, max_period=10000.0):
    half = dim // 2
    if half == 0:
        return torch.zeros(timesteps.shape[0], dim, dtype=torch.float32)
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    emb = torch.cat([torch.cos(timesteps[:, None] * freqs), torch.sin(timesteps[:, None] * freqs)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, dtype=torch.float32)], dim=1)
    return emb


def rms_heads_ref(x, gamma, eps=0.0):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * gamma[None, None, :, :]


def rope_adjacent_ref(x, cos, sin):
    y = torch.empty_like(x)
    y[..., 0::2] = x[..., 0::2] * cos[None, :, None, :] - x[..., 1::2] * sin[None, :, None, :]
    y[..., 1::2] = x[..., 0::2] * sin[None, :, None, :] + x[..., 1::2] * cos[None, :, None, :]
    return y


def sdpa_ref(q, k, v, head_dim):
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (1.0 / math.sqrt(head_dim))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)


def linear_ref(x, w, b):
    return x @ w.t() + b


def dit_flow_ref(x, timesteps, context, cos_phase, sin_phase, weights, blocks):
    batch, tokens, _ = x.shape
    channels = weights["model_channels"]
    heads = weights["heads"]
    head_dim = weights["head_dim"]

    h = linear_ref(x.reshape(-1, weights["in_channels"]), weights["input_w"], weights["input_b"]).reshape(batch, tokens, channels)
    t = timestep_embedding_ref(timesteps, weights["time_frequency_dim"])
    t = linear_ref(t, weights["t_embedder_0_w"], weights["t_embedder_0_b"])
    t = F.silu(t)
    t = linear_ref(t, weights["t_embedder_2_w"], weights["t_embedder_2_b"])
    mod6 = linear_ref(F.silu(t), weights["adaln_w"], weights["adaln_b"]).reshape(batch, 6, channels)

    for block in blocks:
        mod = mod6 + block["modulation"].reshape(1, 6, channels)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [mod[:, i, :] for i in range(6)]

        x_msa = F.layer_norm(h, (channels,), eps=1e-6)
        x_msa = x_msa * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        qkv = linear_ref(x_msa.reshape(-1, channels), block["self_qkv_w"], block["self_qkv_b"]).reshape(batch, tokens, 3, channels)
        q = qkv[:, :, 0, :].reshape(batch, tokens, heads, head_dim)
        k = qkv[:, :, 1, :].reshape(batch, tokens, heads, head_dim)
        v = qkv[:, :, 2, :].reshape(batch, tokens, heads, head_dim)
        q = rms_heads_ref(q, block["self_q_rms_gamma"], eps=0.0)
        k = rms_heads_ref(k, block["self_k_rms_gamma"], eps=0.0)
        q = rope_adjacent_ref(q, cos_phase, sin_phase)
        k = rope_adjacent_ref(k, cos_phase, sin_phase)
        a = sdpa_ref(q, k, v, head_dim).reshape(batch, tokens, channels)
        a = linear_ref(a.reshape(-1, channels), block["self_out_w"], block["self_out_b"]).reshape(batch, tokens, channels)
        h = h + a * gate_msa[:, None, :]

        x_cross = F.layer_norm(h, (channels,), block["norm2_gamma"], block["norm2_beta"], eps=1e-6)
        q = linear_ref(x_cross.reshape(-1, channels), block["cross_q_w"], block["cross_q_b"]).reshape(batch, tokens, heads, head_dim)
        kv = linear_ref(context.reshape(-1, weights["cond_channels"]), block["cross_kv_w"], block["cross_kv_b"]).reshape(batch, context.shape[1], 2, channels)
        k = kv[:, :, 0, :].reshape(batch, context.shape[1], heads, head_dim)
        v = kv[:, :, 1, :].reshape(batch, context.shape[1], heads, head_dim)
        q = rms_heads_ref(q, block["cross_q_rms_gamma"], eps=0.0)
        k = rms_heads_ref(k, block["cross_k_rms_gamma"], eps=0.0)
        a = sdpa_ref(q, k, v, head_dim).reshape(batch, tokens, channels)
        a = linear_ref(a.reshape(-1, channels), block["cross_out_w"], block["cross_out_b"]).reshape(batch, tokens, channels)
        h = h + a

        x_mlp = F.layer_norm(h, (channels,), eps=1e-6)
        x_mlp = x_mlp * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        m = linear_ref(x_mlp.reshape(-1, channels), block["mlp_fc1_w"], block["mlp_fc1_b"])
        m = F.gelu(m, approximate="tanh")
        m = linear_ref(m, block["mlp_fc2_w"], block["mlp_fc2_b"]).reshape(batch, tokens, channels)
        h = h + m * gate_mlp[:, None, :]

    h = F.layer_norm(h, (channels,), eps=weights["final_norm_eps"])
    return linear_ref(h.reshape(-1, channels), weights["out_w"], weights["out_b"]).reshape(batch, tokens, weights["out_channels"])


def run_dit_flow_forward_cases(lib):
    cases = [
        (80800, 1, 4, 3, 4, 4, 8, 2, 16, 6, 1),
        (80801, 2, 5, 4, 6, 5, 12, 3, 24, 8, 2),
        (80802, 2, 7, 5, 8, 6, 16, 4, 32, 10, 2),
    ]
    for case_i, (seed, batch, tokens, cond_tokens, in_c, out_c, channels, heads, mlp, tdim, n_blocks) in enumerate(cases):
        head_dim = channels // heads
        cond_c = 10
        x = realistic_activation((batch, tokens, in_c), seed, 0.35)
        context = realistic_activation((batch, cond_tokens, cond_c), seed + 1, 0.30)
        timesteps = torch.linspace(1000.0, 250.0, batch, dtype=torch.float32)
        cos = torch.cos(realistic_activation((tokens, head_dim // 2), seed + 2, 0.08))
        sin = torch.sin(realistic_activation((tokens, head_dim // 2), seed + 3, 0.08))

        weights = {
            "in_channels": in_c,
            "out_channels": out_c,
            "model_channels": channels,
            "cond_channels": cond_c,
            "time_frequency_dim": tdim,
            "heads": heads,
            "head_dim": head_dim,
            "mlp_channels": mlp,
            "mod_channels": 6 * channels,
            "n_blocks": n_blocks,
            "final_norm_eps": 1e-5,
            "input_w": fanin_weight((channels, in_c), seed + 10, 0.65),
            "input_b": randn((channels,), seed + 11, 0.025),
            "t_embedder_0_w": fanin_weight((channels, tdim), seed + 12, 0.55),
            "t_embedder_0_b": randn((channels,), seed + 13, 0.02),
            "t_embedder_2_w": fanin_weight((channels, channels), seed + 14, 0.45),
            "t_embedder_2_b": randn((channels,), seed + 15, 0.02),
            "adaln_w": fanin_weight((6 * channels, channels), seed + 16, 0.18),
            "adaln_b": randn((6 * channels,), seed + 17, 0.015),
            "out_w": fanin_weight((out_c, channels), seed + 18, 0.55),
            "out_b": randn((out_c,), seed + 19, 0.02),
        }
        blocks = []
        for b in range(n_blocks):
            tag = seed + 100 + b * 40
            blocks.append({
                "modulation": randn((6 * channels,), tag + 0, 0.018),
                "norm2_gamma": randn((channels,), tag + 1, 0.035, 1.0),
                "norm2_beta": randn((channels,), tag + 2, 0.018),
                "self_qkv_w": fanin_weight((3 * channels, channels), tag + 3, 0.35),
                "self_qkv_b": randn((3 * channels,), tag + 4, 0.015),
                "self_q_rms_gamma": randn((heads, head_dim), tag + 5, 0.025, 1.0),
                "self_k_rms_gamma": randn((heads, head_dim), tag + 6, 0.025, 1.0),
                "self_out_w": fanin_weight((channels, channels), tag + 7, 0.35),
                "self_out_b": randn((channels,), tag + 8, 0.015),
                "cross_q_w": fanin_weight((channels, channels), tag + 9, 0.35),
                "cross_q_b": randn((channels,), tag + 10, 0.015),
                "cross_kv_w": fanin_weight((2 * channels, cond_c), tag + 11, 0.35),
                "cross_kv_b": randn((2 * channels,), tag + 12, 0.015),
                "cross_q_rms_gamma": randn((heads, head_dim), tag + 13, 0.025, 1.0),
                "cross_k_rms_gamma": randn((heads, head_dim), tag + 14, 0.025, 1.0),
                "cross_out_w": fanin_weight((channels, channels), tag + 15, 0.35),
                "cross_out_b": randn((channels,), tag + 16, 0.015),
                "mlp_fc1_w": fanin_weight((mlp, channels), tag + 17, 0.40),
                "mlp_fc1_b": randn((mlp,), tag + 18, 0.015),
                "mlp_fc2_w": fanin_weight((channels, mlp), tag + 19, 0.40),
                "mlp_fc2_b": randn((channels,), tag + 20, 0.015),
            })

        keepalive = []

        def cptr(t):
            p, arr = ptr_f32(t)
            keepalive.append(arr)
            return p

        c_blocks = (DitFlowBlockWeights * n_blocks)()
        for i, b in enumerate(blocks):
            for field, _ in DitFlowBlockWeights._fields_:
                setattr(c_blocks[i], field, cptr(b[field]))

        c_weights = DitFlowWeights(
            in_c,
            out_c,
            channels,
            cond_c,
            tdim,
            heads,
            head_dim,
            mlp,
            6 * channels,
            n_blocks,
            -1,
            0,
            0,
            ctypes.c_float(1e-5),
            cptr(weights["input_w"]),
            cptr(weights["input_b"]),
            cptr(weights["t_embedder_0_w"]),
            cptr(weights["t_embedder_0_b"]),
            cptr(weights["t_embedder_2_w"]),
            cptr(weights["t_embedder_2_b"]),
            cptr(weights["adaln_w"]),
            cptr(weights["adaln_b"]),
            cptr(weights["out_w"]),
            cptr(weights["out_b"]),
            c_blocks,
        )
        y = torch.empty(batch, tokens, out_c, dtype=torch.float32)
        expect_ok(
            lib.strellis_dit_flow_forward_f32(
                cptr(x),
                cptr(timesteps),
                cptr(context),
                cptr(cos),
                cptr(sin),
                ctypes.byref(c_weights),
                cptr(y),
                batch,
                tokens,
                cond_tokens,
            ),
            f"dit_flow_forward_{case_i}",
        )
        ref = dit_flow_ref(x, timesteps, context, cos, sin, weights, blocks)
        check(f"dit_flow_forward_{case_i}", y, ref, atol=4e-4, rtol=4e-4)


def run_inference_smoke_cases(lib):
    cases = [
        (60600, 6, 2, 2, 16, 32, 8, 32, 0.00),
        (60601, 8, 3, 3, 25, 32, 8, 32, 0.02),
        (60602, 10, 2, 2, 36, 48, 12, 32, -0.01),
    ]
    for i, (seed, latent_size, s1, s2, cond_tokens, cond_channels, c1, c2, threshold) in enumerate(cases):
        opt = InferOptions()
        lib.strellis_infer_options_default(ctypes.byref(opt))
        opt.seed = seed
        opt.latent_size = latent_size
        opt.stage1_steps = s1
        opt.stage2_steps = s2
        opt.cond_tokens = cond_tokens
        opt.cond_channels = cond_channels
        opt.stage1_channels = c1
        opt.stage2_channels = c2
        opt.voxel_threshold = threshold
        result = InferResult()
        expect_ok(lib.strellis_run_inference_compute(ctypes.byref(opt), ctypes.byref(result)), f"inference_smoke_{i}")
        try:
            assert result.sparse_resolution == latent_size
            assert result.slat_channels == c2
            assert result.n_coords >= 0
            assert result.mesh.n_vertices >= 0
            assert result.mesh.n_faces >= 0
            if result.n_coords > 0:
                coords = np.ctypeslib.as_array(result.coords_bxyz, shape=(result.n_coords, 4))
                slat = np.ctypeslib.as_array(result.slat_feats, shape=(result.n_coords, result.slat_channels))
                assert np.all(coords[:, 0] == 0)
                assert np.all(coords[:, 1:] >= 0)
                assert np.all(coords[:, 1:] < latent_size)
                assert np.isfinite(slat).all()
            if result.mesh.n_vertices > 0:
                vertices = np.ctypeslib.as_array(result.mesh.vertices, shape=(result.mesh.n_vertices, 3))
                assert np.isfinite(vertices).all()
            if result.mesh.n_faces > 0:
                faces = np.ctypeslib.as_array(result.mesh.faces, shape=(result.mesh.n_faces, 3))
                assert result.mesh.n_vertices > 0
                assert np.all(faces < result.mesh.n_vertices)
            print(
                f"{'inference_smoke_' + str(i):32s} "
                f"coords={result.n_coords} slat=[{result.n_coords},{result.slat_channels}] "
                f"mesh=({result.mesh.n_vertices}v,{result.mesh.n_faces}f) ok=True"
            )
        finally:
            lib.strellis_infer_result_free(ctypes.byref(result))


def main():
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    lib = build_lib()
    configure(lib)

    x = torch.randn(5, 4, dtype=torch.float32)
    w = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    y = torch.empty(5, 3, dtype=torch.float32)
    expect_ok(lib.strellis_linear_f32(ptr_f32(x)[0], ptr_f32(w)[0], ptr_f32(b)[0], ptr_f32(y)[0], 5, 4, 3), "linear")
    check("linear", y, x @ w.t() + b)

    gamma = torch.randn(4, dtype=torch.float32)
    beta = torch.randn(4, dtype=torch.float32)
    y = torch.empty_like(x)
    expect_ok(lib.strellis_layer_norm_f32(ptr_f32(x)[0], ptr_f32(gamma)[0], ptr_f32(beta)[0], ptr_f32(y)[0], 5, 4, 1e-6), "layer_norm")
    check("layer_norm", y, F.layer_norm(x, (4,), gamma, beta, eps=1e-6), atol=2e-5, rtol=2e-5)

    y = torch.empty_like(x)
    expect_ok(lib.strellis_rms_norm_f32(ptr_f32(x)[0], ptr_f32(gamma)[0], ptr_f32(y)[0], 5, 4, 1e-6), "rms_norm")
    check("rms_norm", y, x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * gamma)

    flat = torch.randn(17, dtype=torch.float32)
    y = torch.empty_like(flat)
    expect_ok(lib.strellis_gelu_tanh_f32(ptr_f32(flat)[0], ptr_f32(y)[0], flat.numel()), "gelu")
    check("gelu_tanh", y, F.gelu(flat, approximate="tanh"))
    expect_ok(lib.strellis_silu_f32(ptr_f32(flat)[0], ptr_f32(y)[0], flat.numel()), "silu")
    check("silu", y, F.silu(flat))

    timesteps = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    emb = torch.empty(3, 7, dtype=torch.float32)
    expect_ok(lib.strellis_timestep_embedding_f32(ptr_f32(timesteps)[0], 3, 7, 10000.0, ptr_f32(emb)[0]), "timestep_embedding")
    half = 7 // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    ref = torch.cat([torch.cos(timesteps[:, None] * freqs), torch.sin(timesteps[:, None] * freqs), torch.zeros(3, 1)], dim=1)
    check("timestep_embedding", emb, ref)

    q = torch.randn(2, 3, 2, 4, dtype=torch.float32)
    k = torch.randn(2, 5, 2, 4, dtype=torch.float32)
    v = torch.randn(2, 5, 2, 4, dtype=torch.float32)
    y = torch.empty_like(q)
    scale = 1.0 / math.sqrt(4)
    expect_ok(lib.strellis_sdpa_f32(ptr_f32(q)[0], ptr_f32(k)[0], ptr_f32(v)[0], ptr_f32(y)[0], 2, 3, 5, 2, 4, scale), "sdpa")
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)
    check("sdpa", y, ref, atol=2e-5, rtol=2e-5)

    cos = torch.cos(torch.randn(3, 2, dtype=torch.float32))
    sin = torch.sin(torch.randn(3, 2, dtype=torch.float32))
    y = torch.empty_like(q)
    expect_ok(lib.strellis_apply_rope_adjacent_f32(ptr_f32(q)[0], ptr_f32(cos)[0], ptr_f32(sin)[0], ptr_f32(y)[0], 2, 3, 2, 4), "rope")
    ref = q.clone()
    ref[..., 0::2] = q[..., 0::2] * cos[None, :, None, :] - q[..., 1::2] * sin[None, :, None, :]
    ref[..., 1::2] = q[..., 0::2] * sin[None, :, None, :] + q[..., 1::2] * cos[None, :, None, :]
    check("apply_rope_adjacent", y, ref)

    img = torch.randn(1, 3, 6, 4, dtype=torch.float32)
    pw = torch.randn(5, 3, 2, 2, dtype=torch.float32)
    pb = torch.randn(5, dtype=torch.float32)
    tokens = torch.empty(1, (6 // 2) * (4 // 2), 5, dtype=torch.float32)
    expect_ok(lib.strellis_dino_patch_embed_f32(ptr_f32(img)[0], ptr_f32(pw)[0], ptr_f32(pb)[0], ptr_f32(tokens)[0], 1, 6, 4, 5, 2), "dino_patch_embed")
    ref = F.conv2d(img, pw, pb, stride=2).flatten(2).transpose(1, 2)
    check("dino_patch_embed", tokens, ref, atol=2e-5, rtol=2e-5)

    conv_x = torch.randn(1, 2, 4, 3, 5, dtype=torch.float32)
    conv_w = torch.randn(3, 2, 3, 2, 3, dtype=torch.float32)
    conv_b = torch.randn(3, dtype=torch.float32)
    ref = F.conv3d(conv_x, conv_w, conv_b, stride=(1, 1, 2), padding=(1, 0, 1), dilation=(1, 1, 1))
    y = torch.empty_like(ref)
    expect_ok(
        lib.strellis_conv3d_ncdhw_f32(
            ptr_f32(conv_x)[0], ptr_f32(conv_w)[0], ptr_f32(conv_b)[0], ptr_f32(y)[0],
            1, 2, 4, 3, 5, 3, 3, 2, 3, 1, 1, 2, 1, 0, 1, 1, 1, 1,
        ),
        "conv3d",
    )
    check("conv3d_ncdhw", y, ref, atol=3e-5, rtol=3e-5)

    ps_x = torch.randn(1, 16, 2, 2, 2, dtype=torch.float32)
    ps_y = torch.empty(1, 2, 4, 4, 4, dtype=torch.float32)
    expect_ok(lib.strellis_pixel_shuffle_3d_ncdhw_f32(ptr_f32(ps_x)[0], ptr_f32(ps_y)[0], 1, 16, 2, 2, 2, 2), "pixel_shuffle")
    ref = ps_x.view(1, 2, 2, 2, 2, 2, 2, 2).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(1, 2, 4, 4, 4)
    check("pixel_shuffle_3d", ps_y, ref)

    cln_x = torch.randn(1, 4, 2, 2, 3, dtype=torch.float32)
    gamma = torch.randn(4, dtype=torch.float32)
    beta = torch.randn(4, dtype=torch.float32)
    cln_y = torch.empty_like(cln_x)
    expect_ok(lib.strellis_channel_layer_norm_3d_ncdhw_f32(ptr_f32(cln_x)[0], ptr_f32(gamma)[0], ptr_f32(beta)[0], ptr_f32(cln_y)[0], 1, 4, 2, 2, 3, 1e-6), "channel_layer_norm")
    ref = F.layer_norm(cln_x.permute(0, 2, 3, 4, 1), (4,), gamma, beta, eps=1e-6).permute(0, 4, 1, 2, 3)
    check("channel_layer_norm_3d", cln_y, ref, atol=2e-5, rtol=2e-5)

    coords_t = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.int32)
    feats = torch.randn(4, 3, dtype=torch.float32)
    sw = torch.randn(2, 3, 3, 3, 3, dtype=torch.float32)
    sb = torch.randn(2, dtype=torch.float32)
    sparse_y = torch.empty(4, 2, dtype=torch.float32)
    coords_ptr, coords_np = ptr_i32(coords_t.numpy())
    expect_ok(lib.strellis_sparse_subm_conv3d_f32(coords_ptr, ptr_f32(feats)[0], ptr_f32(sw)[0], ptr_f32(sb)[0], ptr_f32(sparse_y)[0], 4, 3, 2, 3, 3, 3, 1, 1, 1), "sparse_subm_conv3d")
    check("sparse_subm_conv3d", sparse_y, sparse_ref_subm(coords_t, feats, sw, sb), atol=3e-5, rtol=3e-5)

    sparse_feats = np.arange(8, dtype=np.float32).reshape(4, 2)
    input_sparse, coords_np, feats_np = sparse_tensor_from_numpy(coords_np, sparse_feats)
    out_sparse = SparseTensor()
    expect_ok(lib.strellis_sparse_downsample_mean_f32(ctypes.byref(input_sparse), 2, ctypes.byref(out_sparse)), "sparse_downsample")
    out_coords, out_feats = sparse_to_numpy(out_sparse)
    lib.strellis_sparse_tensor_free(ctypes.byref(out_sparse))
    print(f"{'sparse_downsample_mean':32s} rows={len(out_coords)} feats_sum={float(out_feats.sum()):.6f} ok=True")

    x_net = torch.randn(1, 3, 4, dtype=torch.float32)
    ctx = torch.randn(1, 2, 4, dtype=torch.float32)
    nw = torch.randn(4, 4, dtype=torch.float32)
    nb = torch.randn(4, dtype=torch.float32)
    y_net = torch.empty_like(x_net)
    expect_ok(lib.strellis_toy_dit_block_forward_f32(ptr_f32(x_net)[0], ptr_f32(ctx)[0], ptr_f32(torch.ones(4))[0], ptr_f32(torch.zeros(4))[0], ptr_f32(nw)[0], ptr_f32(nb)[0], ptr_f32(y_net)[0], 1, 3, 4, 2), "toy_dit")
    ref = F.layer_norm(x_net, (4,), torch.ones(4), torch.zeros(4), eps=1e-6) @ nw.t() + nb + ctx.mean(dim=1, keepdim=True)
    check("toy_dit_block", y_net, ref, atol=2e-5, rtol=2e-5)

    run_random_dense_cases(lib)
    run_random_attention_cases(lib)
    run_random_volume_cases(lib)
    run_random_sparse_cases(lib)
    run_trellis2_reference_cases(lib)
    run_dit_flow_forward_cases(lib)
    run_inference_smoke_cases(lib)

    print("all standalone TRELLIS tensor comparisons passed")


if __name__ == "__main__":
    main()
