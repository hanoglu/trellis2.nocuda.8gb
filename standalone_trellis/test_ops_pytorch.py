#!/usr/bin/env python3
import ctypes
import math
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parent
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
    run_inference_smoke_cases(lib)

    print("all standalone TRELLIS tensor comparisons passed")


if __name__ == "__main__":
    main()
