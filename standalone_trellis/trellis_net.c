#include "trellis_ops.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef STRELLIS_PI
#define STRELLIS_PI 3.14159265358979323846f
#endif

static const strellis_network_step_plan g_network_plan[] = {
    {
        "image_preprocess",
        "procedural RGB tensor",
        "normalized image [1,3,H,W]",
        "local deterministic tensor generator",
        "standalone compute path implemented; real image IO intentionally omitted"
    },
    {
        "dino_vit_encoder",
        "normalized image",
        "image condition tokens",
        "dino_patch_embed, dino_rope_2d_phases, apply_rope_adjacent, layer_norm",
        "procedural DINO-like conditioning implemented; checkpoint weights pending"
    },
    {
        "sparse_structure_flow",
        "noise latent + timestep + condition tokens",
        "dense sparse-structure latent",
        "timestep_mlp, rope_3d_phases, apply_rope_adjacent, feed_forward, flow_euler_step",
        "CPU reference compute path implemented with procedural parameters"
    },
    {
        "sparse_structure_decoder",
        "structure latent",
        "active voxel coordinates",
        "conv3d_ncdhw, channel_layer_norm_3d, silu",
        "CPU reference compute path implemented"
    },
    {
        "shape_slat_flow",
        "active coords + SLat noise + timestep + condition tokens",
        "denoised SLat features",
        "rms_norm, rope_3d_sparse_phases, apply_rope_adjacent, sparse_linear, sparse_subm_conv3d, flow_euler_step",
        "CPU reference compute path implemented with procedural parameters"
    },
    {
        "shape_decoder",
        "active coords + SLat features",
        "FlexiDualGrid logits",
        "rms_norm, sparse_subm_conv3d, silu, sparse_linear",
        "CPU reference compute path implemented"
    },
    {
        "mesh_extract",
        "active coords + FlexiDualGrid logits",
        "mesh vertices and faces",
        "flexible_dual_grid_mesh",
        "CPU reference voxel-surface extractor implemented"
    },
};

size_t strellis_network_plan_count(void) {
    return sizeof(g_network_plan) / sizeof(g_network_plan[0]);
}

const strellis_network_step_plan * strellis_network_plan_at(size_t index) {
    if (index >= strellis_network_plan_count()) {
        return NULL;
    }
    return &g_network_plan[index];
}

void strellis_print_network_plan(FILE * out) {
    if (out == NULL) {
        out = stdout;
    }
    for (size_t i = 0; i < strellis_network_plan_count(); ++i) {
        const strellis_network_step_plan * p = &g_network_plan[i];
        fprintf(out, "%02zu %-26s %s -> %s\n", i, p->name, p->inputs, p->outputs);
        fprintf(out, "    ops: %s\n", p->required_ops);
        fprintf(out, "    status: %s\n", p->status);
    }
}

static void * calloc_array(size_t count, size_t size) {
    if (size != 0u && count > SIZE_MAX / size) {
        return NULL;
    }
    return calloc(count, size);
}

static int max_int(int a, int b) {
    return a > b ? a : b;
}

static int ceil_sqrt_int(int value) {
    int r = 1;
    while (r < 46340 && r * r < value) {
        ++r;
    }
    return r;
}

static uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float hash_unit_f32(uint32_t seed, uint32_t tag, uint32_t a, uint32_t b) {
    uint32_t h = seed ^ (tag * 0x9e3779b9u) ^ (a * 0x85ebca6bu) ^ (b * 0xc2b2ae35u);
    h = mix_u32(h);
    return ((float) (h & 0x00ffffffu) / 8388607.5f) - 1.0f;
}

static float param_f32(uint32_t seed, uint32_t tag, size_t index, float scale) {
    return hash_unit_f32(seed, tag, (uint32_t) index, (uint32_t) (index >> 32)) * scale;
}

static void fill_param(float * dst, size_t n, uint32_t seed, uint32_t tag, float scale) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = param_f32(seed, tag, i, scale);
    }
}

static size_t net_ncdhw_index(
    int b,
    int c,
    int d,
    int h,
    int w,
    int channels,
    int depth,
    int height,
    int width) {
    return (((((size_t) b * (size_t) channels + (size_t) c) * (size_t) depth + (size_t) d) * (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
}

static float centered_coord(int i, int resolution) {
    return ((float) i + 0.5f) / (float) resolution - 0.5f;
}

static void dense_tokens_to_ncdhw(const float * tokens, float * ncdhw, int resolution, int channels) {
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                const size_t row = ((size_t) x * (size_t) resolution + (size_t) y) * (size_t) resolution + (size_t) z;
                for (int c = 0; c < channels; ++c) {
                    ncdhw[net_ncdhw_index(0, c, x, y, z, channels, resolution, resolution, resolution)] =
                        tokens[row * (size_t) channels + (size_t) c];
                }
            }
        }
    }
}

static strellis_status validate_options(const strellis_infer_options * opt) {
    if (opt == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (opt->latent_size < 2 || opt->latent_size > 32 ||
        opt->stage1_steps <= 0 || opt->stage1_steps > 128 ||
        opt->stage2_steps <= 0 || opt->stage2_steps > 128 ||
        opt->cond_tokens <= 0 || opt->cond_tokens > 4096 ||
        opt->cond_channels < 4 || opt->cond_channels > 512 ||
        opt->stage1_channels < 2 || opt->stage1_channels > 512 ||
        opt->stage2_channels < 2 || opt->stage2_channels > 512 ||
        opt->stage2_rescale_t <= 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((opt->cond_channels & 1) != 0 ||
        (opt->stage1_channels & 1) != 0 ||
        (opt->stage2_channels & 1) != 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return STRELLIS_STATUS_OK;
}

static strellis_status condition_projection(
    const float * condition,
    int tokens,
    int cond_channels,
    int out_channels,
    uint32_t seed,
    uint32_t tag,
    float * out) {
    if (condition == NULL || out == NULL || tokens <= 0 || cond_channels <= 0 || out_channels <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }

    float * mean = (float *) calloc_array((size_t) cond_channels, sizeof(float));
    float * weight = (float *) calloc_array((size_t) out_channels * (size_t) cond_channels, sizeof(float));
    float * bias = (float *) calloc_array((size_t) out_channels, sizeof(float));
    if (mean == NULL || weight == NULL || bias == NULL) {
        free(mean);
        free(weight);
        free(bias);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < cond_channels; ++c) {
            mean[c] += condition[(size_t) t * (size_t) cond_channels + (size_t) c];
        }
    }
    for (int c = 0; c < cond_channels; ++c) {
        mean[c] /= (float) tokens;
    }
    fill_param(weight, (size_t) out_channels * (size_t) cond_channels, seed, tag, 0.16f / sqrtf((float) cond_channels));
    fill_param(bias, (size_t) out_channels, seed, tag + 1u, 0.02f);

    strellis_status status = strellis_linear_f32(mean, weight, bias, out, 1, cond_channels, out_channels);
    free(mean);
    free(weight);
    free(bias);
    return status;
}

static strellis_status build_condition_tokens(const strellis_infer_options * opt, float ** tokens_out) {
    *tokens_out = NULL;
    const int grid = ceil_sqrt_int(opt->cond_tokens);
    const int patch = 2;
    const int image_h = grid * patch;
    const int image_w = grid * patch;
    const int full_tokens = grid * grid;
    const size_t image_n = (size_t) 3u * (size_t) image_h * (size_t) image_w;
    const size_t patch_weight_n = (size_t) opt->cond_channels * 3u * (size_t) patch * (size_t) patch;
    const size_t token_n = (size_t) full_tokens * (size_t) opt->cond_channels;
    const size_t out_n = (size_t) opt->cond_tokens * (size_t) opt->cond_channels;

    float * image = (float *) calloc_array(image_n, sizeof(float));
    float * patch_w = (float *) calloc_array(patch_weight_n, sizeof(float));
    float * patch_b = (float *) calloc_array((size_t) opt->cond_channels, sizeof(float));
    float * tokens = (float *) calloc_array(token_n, sizeof(float));
    float * rotated = (float *) calloc_array(token_n, sizeof(float));
    float * cos_phase = (float *) calloc_array(token_n, sizeof(float));
    float * sin_phase = (float *) calloc_array(token_n, sizeof(float));
    float * out = (float *) calloc_array(out_n, sizeof(float));
    if (image == NULL || patch_w == NULL || patch_b == NULL || tokens == NULL ||
        rotated == NULL || cos_phase == NULL || sin_phase == NULL || out == NULL) {
        free(image);
        free(patch_w);
        free(patch_b);
        free(tokens);
        free(rotated);
        free(cos_phase);
        free(sin_phase);
        free(out);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < image_h; ++y) {
            for (int x = 0; x < image_w; ++x) {
                const float fx = ((float) x + 0.5f) / (float) image_w;
                const float fy = ((float) y + 0.5f) / (float) image_h;
                const float p0 = hash_unit_f32(opt->seed, 11u, (uint32_t) c, 0u) * STRELLIS_PI;
                const float p1 = hash_unit_f32(opt->seed, 12u, (uint32_t) c, 0u) * STRELLIS_PI;
                const float v = 0.55f * sinf((fx * (float) (c + 1) + fy * 0.7f) * STRELLIS_PI * 2.0f + p0) +
                    0.35f * cosf((fy * (float) (c + 2) - fx * 0.3f) * STRELLIS_PI * 2.0f + p1);
                image[((size_t) c * (size_t) image_h + (size_t) y) * (size_t) image_w + (size_t) x] = v;
            }
        }
    }

    fill_param(patch_w, patch_weight_n, opt->seed, 20u, 0.20f);
    fill_param(patch_b, (size_t) opt->cond_channels, opt->seed, 21u, 0.04f);

    strellis_status status = strellis_dino_patch_embed_f32(
        image,
        patch_w,
        patch_b,
        tokens,
        1,
        image_h,
        image_w,
        opt->cond_channels,
        patch);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_dino_rope_2d_phases_f32(
            0,
            grid,
            grid,
            opt->cond_channels,
            1.0f,
            10000.0f,
            cos_phase,
            sin_phase,
            token_n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_apply_rope_adjacent_f32(
            tokens,
            cos_phase,
            sin_phase,
            rotated,
            1,
            full_tokens,
            1,
            opt->cond_channels);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_layer_norm_f32(rotated, NULL, NULL, tokens, full_tokens, opt->cond_channels, 1e-6f);
    }
    if (status == STRELLIS_STATUS_OK) {
        memcpy(out, tokens, out_n * sizeof(float));
        *tokens_out = out;
        out = NULL;
    }

    free(image);
    free(patch_w);
    free(patch_b);
    free(tokens);
    free(rotated);
    free(cos_phase);
    free(sin_phase);
    free(out);
    return status;
}

static strellis_status run_stage1_structure(
    const strellis_infer_options * opt,
    const float * condition,
    int32_t ** coords_out,
    int64_t * n_coords_out) {
    *coords_out = NULL;
    *n_coords_out = 0;

    const int l = opt->latent_size;
    const int c = opt->stage1_channels;
    const int hidden = max_int(c * 2, 16);
    const int time_hidden = max_int(c * 2, 16);
    const int time_dim = 16;
    const size_t tokens = (size_t) l * (size_t) l * (size_t) l;
    const size_t sample_n = tokens * (size_t) c;
    const size_t phase_n = tokens * (size_t) (c / 2);
    const size_t conv_feat_n = sample_n;

    float * latent = (float *) calloc_array(sample_n, sizeof(float));
    float * normed = (float *) calloc_array(sample_n, sizeof(float));
    float * rope = (float *) calloc_array(sample_n, sizeof(float));
    float * pred = (float *) calloc_array(sample_n, sizeof(float));
    float * next = (float *) calloc_array(sample_n, sizeof(float));
    float * x0 = (float *) calloc_array(sample_n, sizeof(float));
    float * workspace = (float *) calloc_array(tokens * (size_t) hidden, sizeof(float));
    float * pairs = (float *) calloc_array((size_t) opt->stage1_steps * 2u, sizeof(float));
    float * cos_phase = (float *) calloc_array(phase_n, sizeof(float));
    float * sin_phase = (float *) calloc_array(phase_n, sizeof(float));
    float * gamma = (float *) calloc_array((size_t) c, sizeof(float));
    float * beta = (float *) calloc_array((size_t) c, sizeof(float));
    float * ff_w1 = (float *) calloc_array((size_t) hidden * (size_t) c, sizeof(float));
    float * ff_b1 = (float *) calloc_array((size_t) hidden, sizeof(float));
    float * ff_w2 = (float *) calloc_array((size_t) c * (size_t) hidden, sizeof(float));
    float * ff_b2 = (float *) calloc_array((size_t) c, sizeof(float));
    float * cond_bias = (float *) calloc_array((size_t) c, sizeof(float));
    float * t_w1 = (float *) calloc_array((size_t) time_hidden * (size_t) time_dim, sizeof(float));
    float * t_b1 = (float *) calloc_array((size_t) time_hidden, sizeof(float));
    float * t_w2 = (float *) calloc_array((size_t) c * (size_t) time_hidden, sizeof(float));
    float * t_b2 = (float *) calloc_array((size_t) c, sizeof(float));
    float * t_emb = (float *) calloc_array((size_t) time_dim, sizeof(float));
    float * t_hidden = (float *) calloc_array((size_t) time_hidden, sizeof(float));
    float * t_vec = (float *) calloc_array((size_t) c, sizeof(float));
    float * ncdhw = (float *) calloc_array(sample_n, sizeof(float));
    float * conv_w = (float *) calloc_array((size_t) c * (size_t) c * 27u, sizeof(float));
    float * conv_b = (float *) calloc_array((size_t) c, sizeof(float));
    float * conv_feat = (float *) calloc_array(conv_feat_n, sizeof(float));
    float * conv_norm = (float *) calloc_array(conv_feat_n, sizeof(float));
    float * cln_gamma = (float *) calloc_array((size_t) c, sizeof(float));
    float * cln_beta = (float *) calloc_array((size_t) c, sizeof(float));
    float * logit_w = (float *) calloc_array((size_t) c, sizeof(float));
    float * logit_b = (float *) calloc_array(1u, sizeof(float));
    float * logits = (float *) calloc_array(tokens, sizeof(float));

    if (latent == NULL || normed == NULL || rope == NULL || pred == NULL || next == NULL || x0 == NULL ||
        workspace == NULL || pairs == NULL || cos_phase == NULL || sin_phase == NULL ||
        gamma == NULL || beta == NULL || ff_w1 == NULL || ff_b1 == NULL || ff_w2 == NULL || ff_b2 == NULL ||
        cond_bias == NULL || t_w1 == NULL || t_b1 == NULL || t_w2 == NULL || t_b2 == NULL ||
        t_emb == NULL || t_hidden == NULL || t_vec == NULL || ncdhw == NULL ||
        conv_w == NULL || conv_b == NULL || conv_feat == NULL || conv_norm == NULL ||
        cln_gamma == NULL || cln_beta == NULL || logit_w == NULL || logit_b == NULL || logits == NULL) {
        free(latent); free(normed); free(rope); free(pred); free(next); free(x0); free(workspace); free(pairs);
        free(cos_phase); free(sin_phase); free(gamma); free(beta); free(ff_w1); free(ff_b1); free(ff_w2); free(ff_b2);
        free(cond_bias); free(t_w1); free(t_b1); free(t_w2); free(t_b2); free(t_emb); free(t_hidden); free(t_vec);
        free(ncdhw); free(conv_w); free(conv_b); free(conv_feat); free(conv_norm); free(cln_gamma); free(cln_beta);
        free(logit_w); free(logit_b); free(logits);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < c; ++i) {
        gamma[i] = 1.0f + param_f32(opt->seed, 30u, (size_t) i, 0.04f);
        beta[i] = param_f32(opt->seed, 31u, (size_t) i, 0.03f);
        cln_gamma[i] = 1.0f;
        cln_beta[i] = 0.0f;
    }
    fill_param(ff_w1, (size_t) hidden * (size_t) c, opt->seed, 32u, 0.10f / sqrtf((float) c));
    fill_param(ff_b1, (size_t) hidden, opt->seed, 33u, 0.02f);
    fill_param(ff_w2, (size_t) c * (size_t) hidden, opt->seed, 34u, 0.10f / sqrtf((float) hidden));
    fill_param(ff_b2, (size_t) c, opt->seed, 35u, 0.02f);
    fill_param(t_w1, (size_t) time_hidden * (size_t) time_dim, opt->seed, 36u, 0.10f);
    fill_param(t_b1, (size_t) time_hidden, opt->seed, 37u, 0.02f);
    fill_param(t_w2, (size_t) c * (size_t) time_hidden, opt->seed, 38u, 0.10f / sqrtf((float) time_hidden));
    fill_param(t_b2, (size_t) c, opt->seed, 39u, 0.02f);
    fill_param(conv_w, (size_t) c * (size_t) c * 27u, opt->seed, 40u, 0.08f / sqrtf((float) (c * 27)));
    fill_param(conv_b, (size_t) c, opt->seed, 41u, 0.01f);
    fill_param(logit_w, (size_t) c, opt->seed, 42u, 0.14f / sqrtf((float) c));
    logit_b[0] = param_f32(opt->seed, 43u, 0u, 0.02f);

    for (int x = 0; x < l; ++x) {
        for (int y = 0; y < l; ++y) {
            for (int z = 0; z < l; ++z) {
                const size_t row = ((size_t) x * (size_t) l + (size_t) y) * (size_t) l + (size_t) z;
                const float fx = centered_coord(x, l);
                const float fy = centered_coord(y, l);
                const float fz = centered_coord(z, l);
                const float dist = sqrtf(fx * fx + fy * fy + fz * fz);
                for (int ch = 0; ch < c; ++ch) {
                    const float noise = hash_unit_f32(opt->seed, 50u + (uint32_t) ch, (uint32_t) row, 0u);
                    const float wave = sinf((fx * (float) (ch + 1) + fy * 0.7f - fz * 0.3f) * STRELLIS_PI * 2.0f);
                    float v = 0.24f * noise + 0.08f * wave;
                    if (ch == 0) {
                        v += 0.34f - dist;
                    }
                    latent[row * (size_t) c + (size_t) ch] = v;
                }
            }
        }
    }

    strellis_status status = condition_projection(condition, opt->cond_tokens, opt->cond_channels, c, opt->seed, 60u, cond_bias);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_rope_3d_phases_f32(l, c, 1.0f, 10000.0f, cos_phase, sin_phase, phase_n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_flow_timestep_pairs_f32(opt->stage1_steps, opt->stage2_rescale_t, pairs, (size_t) opt->stage1_steps * 2u);
    }

    for (int step = 0; status == STRELLIS_STATUS_OK && step < opt->stage1_steps; ++step) {
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        status = strellis_layer_norm_f32(latent, gamma, beta, normed, (int64_t) tokens, c, 1e-6f);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_apply_rope_adjacent_f32(normed, cos_phase, sin_phase, rope, 1, (int) tokens, 1, c);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_feed_forward_f32(rope, ff_w1, ff_b1, ff_w2, ff_b2, workspace, pred, (int64_t) tokens, c, hidden, c);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_timestep_mlp_f32(&t, 1, time_dim, t_w1, t_b1, t_w2, t_b2, t_emb, t_hidden, t_vec, time_hidden, c);
        if (status != STRELLIS_STATUS_OK) break;
        for (size_t row = 0; row < tokens; ++row) {
            for (int ch = 0; ch < c; ++ch) {
                pred[row * (size_t) c + (size_t) ch] += 0.18f * cond_bias[ch] + 0.10f * t_vec[ch];
            }
        }
        status = strellis_flow_euler_step_f32(latent, pred, sample_n, 0.0f, t, t_prev, next, x0);
        if (status != STRELLIS_STATUS_OK) break;
        {
            float * swap = latent;
            latent = next;
            next = swap;
        }
    }

    if (status == STRELLIS_STATUS_OK) {
        dense_tokens_to_ncdhw(latent, ncdhw, l, c);
        status = strellis_conv3d_ncdhw_f32(ncdhw, conv_w, conv_b, conv_feat, 1, c, l, l, l, c, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_channel_layer_norm_3d_ncdhw_f32(conv_feat, cln_gamma, cln_beta, conv_norm, 1, c, l, l, l, 1e-6f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_silu_f32(conv_norm, conv_norm, conv_feat_n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_conv3d_ncdhw_f32(conv_norm, logit_w, logit_b, logits, 1, c, l, l, l, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1);
    }

    int64_t n_active = 0;
    if (status == STRELLIS_STATUS_OK) {
        for (int x = 0; x < l; ++x) {
            for (int y = 0; y < l; ++y) {
                for (int z = 0; z < l; ++z) {
                    const size_t row = ((size_t) x * (size_t) l + (size_t) y) * (size_t) l + (size_t) z;
                    const float fx = centered_coord(x, l);
                    const float fy = centered_coord(y, l);
                    const float fz = centered_coord(z, l);
                    const float dist = sqrtf(fx * fx + fy * fy + fz * fz);
                    const float field = 0.39f - dist + 0.08f * logits[row] + 0.04f * latent[row * (size_t) c];
                    if (field >= opt->voxel_threshold) {
                        ++n_active;
                    }
                }
            }
        }
    }

    int32_t * coords = NULL;
    if (status == STRELLIS_STATUS_OK && n_active > 0) {
        coords = (int32_t *) calloc_array((size_t) n_active * 4u, sizeof(int32_t));
        if (coords == NULL) {
            status = STRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == STRELLIS_STATUS_OK && n_active > 0) {
        int64_t row_out = 0;
        for (int x = 0; x < l; ++x) {
            for (int y = 0; y < l; ++y) {
                for (int z = 0; z < l; ++z) {
                    const size_t row = ((size_t) x * (size_t) l + (size_t) y) * (size_t) l + (size_t) z;
                    const float fx = centered_coord(x, l);
                    const float fy = centered_coord(y, l);
                    const float fz = centered_coord(z, l);
                    const float dist = sqrtf(fx * fx + fy * fy + fz * fz);
                    const float field = 0.39f - dist + 0.08f * logits[row] + 0.04f * latent[row * (size_t) c];
                    if (field >= opt->voxel_threshold) {
                        coords[4 * row_out + 0] = 0;
                        coords[4 * row_out + 1] = x;
                        coords[4 * row_out + 2] = y;
                        coords[4 * row_out + 3] = z;
                        ++row_out;
                    }
                }
            }
        }
    }

    if (status == STRELLIS_STATUS_OK) {
        *coords_out = coords;
        *n_coords_out = n_active;
        coords = NULL;
    }

    free(coords);
    free(latent); free(normed); free(rope); free(pred); free(next); free(x0); free(workspace); free(pairs);
    free(cos_phase); free(sin_phase); free(gamma); free(beta); free(ff_w1); free(ff_b1); free(ff_w2); free(ff_b2);
    free(cond_bias); free(t_w1); free(t_b1); free(t_w2); free(t_b2); free(t_emb); free(t_hidden); free(t_vec);
    free(ncdhw); free(conv_w); free(conv_b); free(conv_feat); free(conv_norm); free(cln_gamma); free(cln_beta);
    free(logit_w); free(logit_b); free(logits);
    return status;
}

static strellis_status run_stage2_slat(
    const strellis_infer_options * opt,
    const float * condition,
    const int32_t * coords,
    int64_t n_coords,
    float ** slat_out,
    strellis_mesh * mesh_out) {
    *slat_out = NULL;
    memset(mesh_out, 0, sizeof(*mesh_out));
    if (n_coords == 0) {
        return STRELLIS_STATUS_OK;
    }
    if (n_coords < 0 || n_coords > INT_MAX || coords == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int l = opt->latent_size;
    const int c = opt->stage2_channels;
    const int hidden = max_int(c * 2, 32);
    const int time_dim = 16;
    const int time_hidden = max_int(c * 2, 32);
    const size_t sample_n = (size_t) n_coords * (size_t) c;
    const size_t phase_n = (size_t) n_coords * (size_t) (c / 2);
    const size_t logits_n = (size_t) n_coords * 7u;

    float * slat = (float *) calloc_array(sample_n, sizeof(float));
    float * normed = (float *) calloc_array(sample_n, sizeof(float));
    float * rope = (float *) calloc_array(sample_n, sizeof(float));
    float * lin_pred = (float *) calloc_array(sample_n, sizeof(float));
    float * conv_pred = (float *) calloc_array(sample_n, sizeof(float));
    float * pred = (float *) calloc_array(sample_n, sizeof(float));
    float * next = (float *) calloc_array(sample_n, sizeof(float));
    float * x0 = (float *) calloc_array(sample_n, sizeof(float));
    float * pairs = (float *) calloc_array((size_t) opt->stage2_steps * 2u, sizeof(float));
    float * cos_phase = (float *) calloc_array(phase_n, sizeof(float));
    float * sin_phase = (float *) calloc_array(phase_n, sizeof(float));
    float * gamma = (float *) calloc_array((size_t) c, sizeof(float));
    float * linear_w = (float *) calloc_array((size_t) c * (size_t) c, sizeof(float));
    float * linear_b = (float *) calloc_array((size_t) c, sizeof(float));
    float * conv_w = (float *) calloc_array((size_t) c * 27u * (size_t) c, sizeof(float));
    float * conv_b = (float *) calloc_array((size_t) c, sizeof(float));
    float * cond_bias = (float *) calloc_array((size_t) c, sizeof(float));
    float * t_w1 = (float *) calloc_array((size_t) time_hidden * (size_t) time_dim, sizeof(float));
    float * t_b1 = (float *) calloc_array((size_t) time_hidden, sizeof(float));
    float * t_w2 = (float *) calloc_array((size_t) c * (size_t) time_hidden, sizeof(float));
    float * t_b2 = (float *) calloc_array((size_t) c, sizeof(float));
    float * t_emb = (float *) calloc_array((size_t) time_dim, sizeof(float));
    float * t_hidden = (float *) calloc_array((size_t) time_hidden, sizeof(float));
    float * t_vec = (float *) calloc_array((size_t) c, sizeof(float));
    float * dec_w = (float *) calloc_array((size_t) c * 27u * (size_t) c, sizeof(float));
    float * dec_b = (float *) calloc_array((size_t) c, sizeof(float));
    float * decoded = (float *) calloc_array(sample_n, sizeof(float));
    float * logit_w = (float *) calloc_array(7u * (size_t) c, sizeof(float));
    float * logit_b = (float *) calloc_array(7u, sizeof(float));
    float * logits = (float *) calloc_array(logits_n, sizeof(float));

    (void) hidden;
    if (slat == NULL || normed == NULL || rope == NULL || lin_pred == NULL || conv_pred == NULL ||
        pred == NULL || next == NULL || x0 == NULL || pairs == NULL || cos_phase == NULL || sin_phase == NULL ||
        gamma == NULL || linear_w == NULL || linear_b == NULL || conv_w == NULL || conv_b == NULL ||
        cond_bias == NULL || t_w1 == NULL || t_b1 == NULL || t_w2 == NULL || t_b2 == NULL ||
        t_emb == NULL || t_hidden == NULL || t_vec == NULL || dec_w == NULL || dec_b == NULL ||
        decoded == NULL || logit_w == NULL || logit_b == NULL || logits == NULL) {
        free(slat); free(normed); free(rope); free(lin_pred); free(conv_pred); free(pred); free(next); free(x0); free(pairs);
        free(cos_phase); free(sin_phase); free(gamma); free(linear_w); free(linear_b); free(conv_w); free(conv_b);
        free(cond_bias); free(t_w1); free(t_b1); free(t_w2); free(t_b2); free(t_emb); free(t_hidden); free(t_vec);
        free(dec_w); free(dec_b); free(decoded); free(logit_w); free(logit_b); free(logits);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int ch = 0; ch < c; ++ch) {
        gamma[ch] = 1.0f + param_f32(opt->seed, 70u, (size_t) ch, 0.04f);
    }
    fill_param(linear_w, (size_t) c * (size_t) c, opt->seed, 71u, 0.10f / sqrtf((float) c));
    fill_param(linear_b, (size_t) c, opt->seed, 72u, 0.02f);
    fill_param(conv_w, (size_t) c * 27u * (size_t) c, opt->seed, 73u, 0.06f / sqrtf((float) (c * 27)));
    fill_param(conv_b, (size_t) c, opt->seed, 74u, 0.01f);
    fill_param(t_w1, (size_t) time_hidden * (size_t) time_dim, opt->seed, 75u, 0.10f);
    fill_param(t_b1, (size_t) time_hidden, opt->seed, 76u, 0.02f);
    fill_param(t_w2, (size_t) c * (size_t) time_hidden, opt->seed, 77u, 0.10f / sqrtf((float) time_hidden));
    fill_param(t_b2, (size_t) c, opt->seed, 78u, 0.02f);
    fill_param(dec_w, (size_t) c * 27u * (size_t) c, opt->seed, 79u, 0.06f / sqrtf((float) (c * 27)));
    fill_param(dec_b, (size_t) c, opt->seed, 80u, 0.01f);
    fill_param(logit_w, 7u * (size_t) c, opt->seed, 81u, 0.12f / sqrtf((float) c));
    fill_param(logit_b, 7u, opt->seed, 82u, 0.02f);

    for (int64_t row = 0; row < n_coords; ++row) {
        const float fx = centered_coord(coords[4 * row + 1], l);
        const float fy = centered_coord(coords[4 * row + 2], l);
        const float fz = centered_coord(coords[4 * row + 3], l);
        const float dist = sqrtf(fx * fx + fy * fy + fz * fz);
        for (int ch = 0; ch < c; ++ch) {
            const float noise = hash_unit_f32(opt->seed, 90u + (uint32_t) ch, (uint32_t) row, 0u);
            const float wave = sinf((fx * 0.9f + fy * (float) (ch + 1) * 0.13f + fz * 0.4f) * STRELLIS_PI * 2.0f);
            float v = 0.18f * noise + 0.06f * wave;
            if (ch == 0) {
                v += 0.42f - dist;
            }
            slat[(size_t) row * (size_t) c + (size_t) ch] = v;
        }
    }

    strellis_status status = condition_projection(condition, opt->cond_tokens, opt->cond_channels, c, opt->seed, 91u, cond_bias);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_rope_3d_sparse_phases_f32(coords, n_coords, c, 1.0f, 10000.0f, cos_phase, sin_phase, phase_n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_flow_timestep_pairs_f32(opt->stage2_steps, opt->stage2_rescale_t, pairs, (size_t) opt->stage2_steps * 2u);
    }

    for (int step = 0; status == STRELLIS_STATUS_OK && step < opt->stage2_steps; ++step) {
        const float t = pairs[2 * step + 0];
        const float t_prev = pairs[2 * step + 1];
        status = strellis_rms_norm_f32(slat, gamma, normed, n_coords, c, 1e-6f);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_apply_rope_adjacent_f32(normed, cos_phase, sin_phase, rope, 1, (int) n_coords, 1, c);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_sparse_linear_f32(rope, linear_w, linear_b, lin_pred, n_coords, c, c);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_sparse_subm_conv3d_f32(coords, normed, conv_w, conv_b, conv_pred, n_coords, c, c, 3, 3, 3, 1, 1, 1);
        if (status != STRELLIS_STATUS_OK) break;
        status = strellis_timestep_mlp_f32(&t, 1, time_dim, t_w1, t_b1, t_w2, t_b2, t_emb, t_hidden, t_vec, time_hidden, c);
        if (status != STRELLIS_STATUS_OK) break;
        for (int64_t row = 0; row < n_coords; ++row) {
            for (int ch = 0; ch < c; ++ch) {
                const size_t idx = (size_t) row * (size_t) c + (size_t) ch;
                pred[idx] = lin_pred[idx] + 0.25f * conv_pred[idx] + 0.16f * cond_bias[ch] + 0.08f * t_vec[ch];
            }
        }
        status = strellis_flow_euler_step_f32(slat, pred, sample_n, 0.0f, t, t_prev, next, x0);
        if (status != STRELLIS_STATUS_OK) break;
        {
            float * swap = slat;
            slat = next;
            next = swap;
        }
    }

    if (status == STRELLIS_STATUS_OK) {
        status = strellis_rms_norm_f32(slat, gamma, normed, n_coords, c, 1e-6f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_sparse_subm_conv3d_f32(coords, normed, dec_w, dec_b, decoded, n_coords, c, c, 3, 3, 3, 1, 1, 1);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_silu_f32(decoded, decoded, sample_n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_sparse_linear_f32(decoded, logit_w, logit_b, logits, n_coords, c, 7);
    }
    if (status == STRELLIS_STATUS_OK) {
        for (int64_t row = 0; row < n_coords; ++row) {
            const float fx = centered_coord(coords[4 * row + 1], l);
            const float fy = centered_coord(coords[4 * row + 2], l);
            const float fz = centered_coord(coords[4 * row + 3], l);
            const float dist = sqrtf(fx * fx + fy * fy + fz * fz);
            logits[(size_t) row * 7u] = 0.24f + (0.43f - dist) + 0.05f * slat[(size_t) row * (size_t) c] +
                0.05f * logits[(size_t) row * 7u];
        }
        status = strellis_flexible_dual_grid_mesh_from_decoder_logits_f32(coords, logits, n_coords, 7, l, mesh_out);
    }

    if (status == STRELLIS_STATUS_OK) {
        *slat_out = slat;
        slat = NULL;
    } else {
        strellis_mesh_free(mesh_out);
    }

    free(slat); free(normed); free(rope); free(lin_pred); free(conv_pred); free(pred); free(next); free(x0); free(pairs);
    free(cos_phase); free(sin_phase); free(gamma); free(linear_w); free(linear_b); free(conv_w); free(conv_b);
    free(cond_bias); free(t_w1); free(t_b1); free(t_w2); free(t_b2); free(t_emb); free(t_hidden); free(t_vec);
    free(dec_w); free(dec_b); free(decoded); free(logit_w); free(logit_b); free(logits);
    return status;
}

void strellis_infer_options_default(strellis_infer_options * options) {
    if (options == NULL) {
        return;
    }
    options->latent_size = 8;
    options->stage1_steps = 4;
    options->stage2_steps = 4;
    options->cond_tokens = 16;
    options->cond_channels = 32;
    options->stage1_channels = 8;
    options->stage2_channels = 32;
    options->voxel_threshold = 0.0f;
    options->stage2_rescale_t = 3.0f;
    options->seed = 1u;
}

void strellis_infer_result_free(strellis_infer_result * result) {
    if (result == NULL) {
        return;
    }
    free(result->coords_bxyz);
    free(result->slat_feats);
    strellis_mesh_free(&result->mesh);
    memset(result, 0, sizeof(*result));
}

strellis_status strellis_run_inference_compute(
    const strellis_infer_options * options,
    strellis_infer_result * result) {
    if (result == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }

    strellis_infer_options opt;
    if (options == NULL) {
        strellis_infer_options_default(&opt);
    } else {
        opt = *options;
    }
    strellis_status status = validate_options(&opt);
    if (status != STRELLIS_STATUS_OK) {
        return status;
    }

    memset(result, 0, sizeof(*result));
    float * condition = NULL;
    int32_t * coords = NULL;
    int64_t n_coords = 0;
    float * slat = NULL;
    strellis_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));

    status = build_condition_tokens(&opt, &condition);
    if (status == STRELLIS_STATUS_OK) {
        status = run_stage1_structure(&opt, condition, &coords, &n_coords);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = run_stage2_slat(&opt, condition, coords, n_coords, &slat, &mesh);
    }

    free(condition);
    if (status != STRELLIS_STATUS_OK) {
        free(coords);
        free(slat);
        strellis_mesh_free(&mesh);
        return status;
    }

    result->coords_bxyz = coords;
    result->n_coords = n_coords;
    result->sparse_resolution = opt.latent_size;
    result->slat_feats = slat;
    result->slat_channels = opt.stage2_channels;
    result->mesh = mesh;
    return STRELLIS_STATUS_OK;
}

static int dit_flow_weights_valid(const strellis_dit_flow_weights * w) {
    if (w == NULL || w->in_channels <= 0 || w->out_channels <= 0 || w->model_channels <= 0 ||
        w->cond_channels <= 0 || w->time_frequency_dim <= 0 || w->heads <= 0 || w->head_dim <= 0 ||
        w->mlp_channels <= 0 || w->mod_channels != 6 * w->model_channels || w->n_blocks < 0 ||
        w->heads * w->head_dim != w->model_channels || w->input_w == NULL || w->input_b == NULL ||
        w->t_embedder_0_w == NULL || w->t_embedder_0_b == NULL ||
        w->t_embedder_2_w == NULL || w->t_embedder_2_b == NULL ||
        w->adaln_w == NULL || w->adaln_b == NULL || w->out_w == NULL || w->out_b == NULL) {
        return 0;
    }
    if (w->n_blocks > 0 && w->blocks == NULL) {
        return 0;
    }
    for (int i = 0; i < w->n_blocks; ++i) {
        const strellis_dit_flow_block_weights * b = &w->blocks[i];
        if (b->modulation == NULL || b->norm2_gamma == NULL || b->norm2_beta == NULL ||
            b->self_qkv_w == NULL || b->self_qkv_b == NULL ||
            b->self_q_rms_gamma == NULL || b->self_k_rms_gamma == NULL ||
            b->self_out_w == NULL || b->self_out_b == NULL ||
            b->cross_q_w == NULL || b->cross_q_b == NULL ||
            b->cross_kv_w == NULL || b->cross_kv_b == NULL ||
            b->cross_q_rms_gamma == NULL || b->cross_k_rms_gamma == NULL ||
            b->cross_out_w == NULL || b->cross_out_b == NULL ||
            b->mlp_fc1_w == NULL || b->mlp_fc1_b == NULL ||
            b->mlp_fc2_w == NULL || b->mlp_fc2_b == NULL) {
            return 0;
        }
    }
    return 1;
}

static void dit_split_heads(
    const float * src,
    float * dst,
    int batch,
    int tokens,
    int total_channels,
    int offset,
    int heads,
    int head_dim) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            const float * row = src + ((size_t) b * (size_t) tokens + (size_t) t) * (size_t) total_channels + (size_t) offset;
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    dst[(((size_t) b * (size_t) tokens + (size_t) t) * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d] =
                        row[(size_t) h * (size_t) head_dim + (size_t) d];
                }
            }
        }
    }
}

static void dit_heads_to_tokens(
    const float * src,
    float * dst,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    const int channels = heads * head_dim;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            float * row = dst + ((size_t) b * (size_t) tokens + (size_t) t) * (size_t) channels;
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    row[(size_t) h * (size_t) head_dim + (size_t) d] =
                        src[(((size_t) b * (size_t) tokens + (size_t) t) * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d];
                }
            }
        }
    }
}

static strellis_status dit_modulated_norm_f32(
    const float * x,
    const float * mod,
    int shift_index,
    int scale_index,
    float * y,
    int batch,
    int tokens,
    int channels) {
    strellis_status status = strellis_layer_norm_f32(x, NULL, NULL, y, (int64_t) batch * (int64_t) tokens, channels, 1e-6f);
    if (status != STRELLIS_STATUS_OK) {
        return status;
    }
    for (int b = 0; b < batch; ++b) {
        const float * shift = mod + ((size_t) b * 6u + (size_t) shift_index) * (size_t) channels;
        const float * scale = mod + ((size_t) b * 6u + (size_t) scale_index) * (size_t) channels;
        for (int t = 0; t < tokens; ++t) {
            float * row = y + ((size_t) b * (size_t) tokens + (size_t) t) * (size_t) channels;
            for (int c = 0; c < channels; ++c) {
                row[c] = row[c] * (1.0f + scale[c]) + shift[c];
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

static void dit_add_gated_residual_f32(
    float * x,
    const float * residual,
    const float * mod,
    int gate_index,
    int batch,
    int tokens,
    int channels) {
    for (int b = 0; b < batch; ++b) {
        const float * gate = mod + ((size_t) b * 6u + (size_t) gate_index) * (size_t) channels;
        for (int t = 0; t < tokens; ++t) {
            const size_t base = ((size_t) b * (size_t) tokens + (size_t) t) * (size_t) channels;
            for (int c = 0; c < channels; ++c) {
                x[base + (size_t) c] += residual[base + (size_t) c] * gate[c];
            }
        }
    }
}

static void dit_add_residual_f32(float * x, const float * residual, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] += residual[i];
    }
}

static strellis_status dit_self_attention_f32(
    const float * x,
    const strellis_dit_flow_block_weights * block,
    const float * cos_phase,
    const float * sin_phase,
    int disable_rope,
    float * y,
    int batch,
    int tokens,
    int channels,
    int heads,
    int head_dim) {
    const size_t rows = (size_t) batch * (size_t) tokens;
    const size_t qkv_n = rows * 3u * (size_t) channels;
    const size_t head_n = rows * (size_t) channels;
    float * qkv = (float *) calloc_array(qkv_n, sizeof(float));
    float * q = (float *) calloc_array(head_n, sizeof(float));
    float * k = (float *) calloc_array(head_n, sizeof(float));
    float * v = (float *) calloc_array(head_n, sizeof(float));
    float * qn = (float *) calloc_array(head_n, sizeof(float));
    float * kn = (float *) calloc_array(head_n, sizeof(float));
    float * qr = (float *) calloc_array(head_n, sizeof(float));
    float * kr = (float *) calloc_array(head_n, sizeof(float));
    float * attn = (float *) calloc_array(head_n, sizeof(float));
    float * tokens_out = (float *) calloc_array(head_n, sizeof(float));
    if (qkv == NULL || q == NULL || k == NULL || v == NULL || qn == NULL || kn == NULL ||
        qr == NULL || kr == NULL || attn == NULL || tokens_out == NULL) {
        free(qkv); free(q); free(k); free(v); free(qn); free(kn); free(qr); free(kr); free(attn); free(tokens_out);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    strellis_status status = strellis_linear_f32(x, block->self_qkv_w, block->self_qkv_b, qkv, (int64_t) rows, channels, 3 * channels);
    if (status == STRELLIS_STATUS_OK) {
        dit_split_heads(qkv, q, batch, tokens, 3 * channels, 0, heads, head_dim);
        dit_split_heads(qkv, k, batch, tokens, 3 * channels, channels, heads, head_dim);
        dit_split_heads(qkv, v, batch, tokens, 3 * channels, 2 * channels, heads, head_dim);
        status = strellis_multihead_rms_norm_f32(q, block->self_q_rms_gamma, qn, batch, tokens, heads, head_dim, 0.0f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_multihead_rms_norm_f32(k, block->self_k_rms_gamma, kn, batch, tokens, heads, head_dim, 0.0f);
    }
    const float * q_attn = qn;
    const float * k_attn = kn;
    if (status == STRELLIS_STATUS_OK && !disable_rope && cos_phase != NULL && sin_phase != NULL) {
        status = strellis_apply_rope_adjacent_f32(qn, cos_phase, sin_phase, qr, batch, tokens, heads, head_dim);
        if (status == STRELLIS_STATUS_OK) {
            status = strellis_apply_rope_adjacent_f32(kn, cos_phase, sin_phase, kr, batch, tokens, heads, head_dim);
        }
        q_attn = qr;
        k_attn = kr;
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_sdpa_f32(q_attn, k_attn, v, attn, batch, tokens, tokens, heads, head_dim, 1.0f / sqrtf((float) head_dim));
    }
    if (status == STRELLIS_STATUS_OK) {
        dit_heads_to_tokens(attn, tokens_out, batch, tokens, heads, head_dim);
        status = strellis_linear_f32(tokens_out, block->self_out_w, block->self_out_b, y, (int64_t) rows, channels, channels);
    }

    free(qkv); free(q); free(k); free(v); free(qn); free(kn); free(qr); free(kr); free(attn); free(tokens_out);
    return status;
}

static strellis_status dit_cross_attention_f32(
    const float * x,
    const float * context,
    const strellis_dit_flow_block_weights * block,
    float * y,
    int batch,
    int tokens,
    int cond_tokens,
    int channels,
    int cond_channels,
    int heads,
    int head_dim) {
    const size_t rows = (size_t) batch * (size_t) tokens;
    const size_t ctx_rows = (size_t) batch * (size_t) cond_tokens;
    const size_t q_linear_n = rows * (size_t) channels;
    const size_t kv_linear_n = ctx_rows * 2u * (size_t) channels;
    const size_t q_head_n = rows * (size_t) channels;
    const size_t kv_head_n = ctx_rows * (size_t) channels;
    float * q_linear = (float *) calloc_array(q_linear_n, sizeof(float));
    float * kv_linear = (float *) calloc_array(kv_linear_n, sizeof(float));
    float * q = (float *) calloc_array(q_head_n, sizeof(float));
    float * k = (float *) calloc_array(kv_head_n, sizeof(float));
    float * v = (float *) calloc_array(kv_head_n, sizeof(float));
    float * qn = (float *) calloc_array(q_head_n, sizeof(float));
    float * kn = (float *) calloc_array(kv_head_n, sizeof(float));
    float * attn = (float *) calloc_array(q_head_n, sizeof(float));
    float * tokens_out = (float *) calloc_array(q_head_n, sizeof(float));
    if (q_linear == NULL || kv_linear == NULL || q == NULL || k == NULL || v == NULL ||
        qn == NULL || kn == NULL || attn == NULL || tokens_out == NULL) {
        free(q_linear); free(kv_linear); free(q); free(k); free(v); free(qn); free(kn); free(attn); free(tokens_out);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    strellis_status status = strellis_linear_f32(x, block->cross_q_w, block->cross_q_b, q_linear, (int64_t) rows, channels, channels);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_linear_f32(context, block->cross_kv_w, block->cross_kv_b, kv_linear, (int64_t) ctx_rows, cond_channels, 2 * channels);
    }
    if (status == STRELLIS_STATUS_OK) {
        dit_split_heads(q_linear, q, batch, tokens, channels, 0, heads, head_dim);
        dit_split_heads(kv_linear, k, batch, cond_tokens, 2 * channels, 0, heads, head_dim);
        dit_split_heads(kv_linear, v, batch, cond_tokens, 2 * channels, channels, heads, head_dim);
        status = strellis_multihead_rms_norm_f32(q, block->cross_q_rms_gamma, qn, batch, tokens, heads, head_dim, 0.0f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_multihead_rms_norm_f32(k, block->cross_k_rms_gamma, kn, batch, cond_tokens, heads, head_dim, 0.0f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_sdpa_f32(qn, kn, v, attn, batch, tokens, cond_tokens, heads, head_dim, 1.0f / sqrtf((float) head_dim));
    }
    if (status == STRELLIS_STATUS_OK) {
        dit_heads_to_tokens(attn, tokens_out, batch, tokens, heads, head_dim);
        status = strellis_linear_f32(tokens_out, block->cross_out_w, block->cross_out_b, y, (int64_t) rows, channels, channels);
    }

    free(q_linear); free(kv_linear); free(q); free(k); free(v); free(qn); free(kn); free(attn); free(tokens_out);
    return status;
}

static strellis_status dit_block_forward_f32(
    float * x,
    const float * mod6,
    const float * context,
    const strellis_dit_flow_block_weights * block,
    const float * cos_phase,
    const float * sin_phase,
    int batch,
    int tokens,
    int cond_tokens,
    const strellis_dit_flow_weights * weights) {
    const int channels = weights->model_channels;
    const size_t n = (size_t) batch * (size_t) tokens * (size_t) channels;
    const size_t mod_n = (size_t) batch * 6u * (size_t) channels;
    float * mod = (float *) calloc_array(mod_n, sizeof(float));
    float * h = (float *) calloc_array(n, sizeof(float));
    float * residual = (float *) calloc_array(n, sizeof(float));
    float * mlp_workspace = (float *) calloc_array((size_t) batch * (size_t) tokens * (size_t) weights->mlp_channels, sizeof(float));
    if (mod == NULL || h == NULL || residual == NULL || mlp_workspace == NULL) {
        free(mod); free(h); free(residual); free(mlp_workspace);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < 6 * channels; ++c) {
            mod[(size_t) b * 6u * (size_t) channels + (size_t) c] =
                mod6[(size_t) b * 6u * (size_t) channels + (size_t) c] + block->modulation[c];
        }
    }
    if (weights->emulate_bf16_blocks) {
        strellis_status bf_status = strellis_bf16_roundtrip_f32(mod, mod, mod_n);
        if (bf_status != STRELLIS_STATUS_OK) {
            free(mod); free(h); free(residual); free(mlp_workspace);
            return bf_status;
        }
    }

    const int debug_parts = weights->debug_block_parts < 0 ? 3 : weights->debug_block_parts;
    strellis_status status = STRELLIS_STATUS_OK;
    if (debug_parts <= 0) {
        goto cleanup;
    }

    status = dit_modulated_norm_f32(x, mod, 0, 1, h, batch, tokens, channels);
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(h, h, n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = dit_self_attention_f32(
            h,
            block,
            cos_phase,
            sin_phase,
            weights->debug_disable_rope,
            residual,
            batch,
            tokens,
            channels,
            weights->heads,
            weights->head_dim);
    }
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(residual, residual, n);
    }
    if (status != STRELLIS_STATUS_OK) {
        goto cleanup;
    }
    dit_add_gated_residual_f32(x, residual, mod, 2, batch, tokens, channels);
    if (weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(x, x, n);
        if (status != STRELLIS_STATUS_OK) goto cleanup;
    }
    if (debug_parts <= 1) {
        goto cleanup;
    }

    status = strellis_layer_norm_f32(x, block->norm2_gamma, block->norm2_beta, h, (int64_t) batch * (int64_t) tokens, channels, 1e-6f);
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(h, h, n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = dit_cross_attention_f32(
            h,
            context,
            block,
            residual,
            batch,
            tokens,
            cond_tokens,
            channels,
            weights->cond_channels,
            weights->heads,
            weights->head_dim);
    }
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(residual, residual, n);
    }
    if (status != STRELLIS_STATUS_OK) {
        goto cleanup;
    }
    dit_add_residual_f32(x, residual, n);
    if (weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(x, x, n);
        if (status != STRELLIS_STATUS_OK) goto cleanup;
    }
    if (debug_parts <= 2) {
        goto cleanup;
    }

    status = dit_modulated_norm_f32(x, mod, 3, 4, h, batch, tokens, channels);
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(h, h, n);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_feed_forward_f32(
            h,
            block->mlp_fc1_w,
            block->mlp_fc1_b,
            block->mlp_fc2_w,
            block->mlp_fc2_b,
            mlp_workspace,
            residual,
            (int64_t) batch * (int64_t) tokens,
            channels,
            weights->mlp_channels,
            channels);
    }
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(residual, residual, n);
    }
    if (status != STRELLIS_STATUS_OK) {
        goto cleanup;
    }
    dit_add_gated_residual_f32(x, residual, mod, 5, batch, tokens, channels);
    if (weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(x, x, n);
    }

cleanup:
    free(mod);
    free(h);
    free(residual);
    free(mlp_workspace);
    return status;
}

strellis_status strellis_dit_flow_forward_f32(
    const float * x,
    const float * timesteps,
    const float * context,
    const float * cos_phase,
    const float * sin_phase,
    const strellis_dit_flow_weights * weights,
    float * y,
    int batch,
    int tokens,
    int cond_tokens) {
    if (x == NULL || timesteps == NULL || context == NULL || weights == NULL || y == NULL ||
        batch <= 0 || tokens <= 0 || cond_tokens <= 0 || !dit_flow_weights_valid(weights) ||
        (cos_phase == NULL) != (sin_phase == NULL)) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!weights->debug_disable_rope && (cos_phase == NULL || sin_phase == NULL)) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int c = weights->model_channels;
    const int rows_i = batch * tokens;
    const size_t rows = (size_t) rows_i;
    const size_t h_n = rows * (size_t) c;
    const size_t mod_n = (size_t) batch * (size_t) weights->mod_channels;
    const size_t ctx_n = (size_t) batch * (size_t) cond_tokens * (size_t) weights->cond_channels;
    float * h = (float *) calloc_array(h_n, sizeof(float));
    float * t_emb_workspace = (float *) calloc_array((size_t) batch * (size_t) weights->time_frequency_dim, sizeof(float));
    float * t_hidden = (float *) calloc_array((size_t) batch * (size_t) c, sizeof(float));
    float * t_emb = (float *) calloc_array((size_t) batch * (size_t) c, sizeof(float));
    float * t_act = (float *) calloc_array((size_t) batch * (size_t) c, sizeof(float));
    float * mod6 = (float *) calloc_array(mod_n, sizeof(float));
    float * final_norm = (float *) calloc_array(h_n, sizeof(float));
    float * context_used = NULL;
    if (h == NULL || t_emb_workspace == NULL || t_hidden == NULL || t_emb == NULL ||
        t_act == NULL || mod6 == NULL || final_norm == NULL) {
        free(h); free(t_emb_workspace); free(t_hidden); free(t_emb); free(t_act); free(mod6); free(final_norm);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const float * context_for_blocks = context;
    if (weights->emulate_bf16_blocks) {
        context_used = (float *) calloc_array(ctx_n, sizeof(float));
        if (context_used == NULL) {
            free(h); free(t_emb_workspace); free(t_hidden); free(t_emb); free(t_act); free(mod6); free(final_norm);
            return STRELLIS_STATUS_OUT_OF_MEMORY;
        }
        strellis_status status_copy = strellis_bf16_roundtrip_f32(context, context_used, ctx_n);
        if (status_copy != STRELLIS_STATUS_OK) {
            free(h); free(t_emb_workspace); free(t_hidden); free(t_emb); free(t_act); free(mod6); free(final_norm); free(context_used);
            return status_copy;
        }
        context_for_blocks = context_used;
    }

    strellis_status status = strellis_linear_f32(x, weights->input_w, weights->input_b, h, (int64_t) rows, weights->in_channels, c);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_timestep_mlp_f32(
            timesteps,
            (size_t) batch,
            weights->time_frequency_dim,
            weights->t_embedder_0_w,
            weights->t_embedder_0_b,
            weights->t_embedder_2_w,
            weights->t_embedder_2_b,
            t_emb_workspace,
            t_hidden,
            t_emb,
            c,
            c);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_silu_f32(t_emb, t_act, (size_t) batch * (size_t) c);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_linear_f32(t_act, weights->adaln_w, weights->adaln_b, mod6, batch, c, weights->mod_channels);
    }
    if (status == STRELLIS_STATUS_OK && weights->emulate_bf16_blocks) {
        status = strellis_bf16_roundtrip_f32(h, h, h_n);
        if (status == STRELLIS_STATUS_OK) {
            status = strellis_bf16_roundtrip_f32(mod6, mod6, mod_n);
        }
    }
    for (int i = 0; status == STRELLIS_STATUS_OK && i < weights->n_blocks; ++i) {
        status = dit_block_forward_f32(
            h,
            mod6,
            context_for_blocks,
            &weights->blocks[i],
            cos_phase,
            sin_phase,
            batch,
            tokens,
            cond_tokens,
            weights);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_layer_norm_f32(
            h,
            NULL,
            NULL,
            final_norm,
            (int64_t) rows,
            c,
            weights->final_norm_eps > 0.0f ? weights->final_norm_eps : 1e-5f);
    }
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_linear_f32(final_norm, weights->out_w, weights->out_b, y, (int64_t) rows, c, weights->out_channels);
    }

    free(h);
    free(t_emb_workspace);
    free(t_hidden);
    free(t_emb);
    free(t_act);
    free(mod6);
    free(final_norm);
    free(context_used);
    return status;
}

strellis_status strellis_toy_dit_block_forward_f32(
    const float * x,
    const float * context,
    const float * norm_gamma,
    const float * norm_beta,
    const float * linear_w,
    const float * linear_b,
    float * y,
    int batch,
    int tokens,
    int channels,
    int context_tokens) {
    if (x == NULL || linear_w == NULL || y == NULL || batch < 0 || tokens <= 0 || channels <= 0 || context_tokens < 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (context_tokens > 0 && context == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }

    const int64_t rows = (int64_t) batch * (int64_t) tokens;
    float * normed = (float *) malloc((size_t) rows * (size_t) channels * sizeof(float));
    if (normed == NULL) {
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    strellis_status status = strellis_layer_norm_f32(
        x,
        norm_gamma,
        norm_beta,
        normed,
        rows,
        channels,
        1e-6f);
    if (status == STRELLIS_STATUS_OK) {
        status = strellis_linear_f32(normed, linear_w, linear_b, y, rows, channels, channels);
    }
    free(normed);
    if (status != STRELLIS_STATUS_OK) {
        return status;
    }

    if (context_tokens > 0) {
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                double mean = 0.0;
                for (int t = 0; t < context_tokens; ++t) {
                    mean += context[((size_t) b * (size_t) context_tokens + (size_t) t) * (size_t) channels + (size_t) c];
                }
                mean /= (double) context_tokens;
                for (int t = 0; t < tokens; ++t) {
                    y[((size_t) b * (size_t) tokens + (size_t) t) * (size_t) channels + (size_t) c] += (float) mean;
                }
            }
        }
    }

    return STRELLIS_STATUS_OK;
}

#ifdef STRELLIS_STANDALONE_CLI
static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [--latent-size N] [--stage1-steps N] [--stage2-steps N]\\n"
        "          [--seed N] [--threshold X] [--cond-tokens N]\\n"
        "          [--cond-channels N] [--stage1-channels N] [--stage2-channels N]\\n"
        "          [--rescale-t X] [--obj path] [--plan]\\n",
        argv0);
}

static int parse_int_value(const char * s, int * out) {
    char * end = NULL;
    long v;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return 0;
    }
    *out = (int) v;
    return 1;
}

static int parse_u32_value(const char * s, uint32_t * out) {
    char * end = NULL;
    unsigned long v;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > 0xfffffffful) {
        return 0;
    }
    *out = (uint32_t) v;
    return 1;
}

static int parse_float_value(const char * s, float * out) {
    char * end = NULL;
    float v;
    errno = 0;
    v = strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    *out = v;
    return 1;
}

static int write_obj(const char * path, const strellis_mesh * mesh) {
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        const float * v = mesh->vertices + (size_t) i * 3u;
        fprintf(f, "v %.9g %.9g %.9g\n", v[0], v[1], v[2]);
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const uint32_t * face = mesh->faces + (size_t) i * 3u;
        fprintf(f, "f %u %u %u\n", face[0] + 1u, face[1] + 1u, face[2] + 1u);
    }
    return fclose(f) == 0;
}

int main(int argc, char ** argv) {
    strellis_infer_options opt;
    strellis_infer_options_default(&opt);
    const char * obj_path = NULL;
    int show_plan = 0;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--plan") == 0) {
            show_plan = 1;
        } else if (strcmp(arg, "--latent-size") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.latent_size)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--stage1-steps") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.stage1_steps)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--stage2-steps") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.stage2_steps)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            if (!parse_u32_value(argv[++i], &opt.seed)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--threshold") == 0 && i + 1 < argc) {
            if (!parse_float_value(argv[++i], &opt.voxel_threshold)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--cond-tokens") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.cond_tokens)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--cond-channels") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.cond_channels)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--stage1-channels") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.stage1_channels)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--stage2-channels") == 0 && i + 1 < argc) {
            if (!parse_int_value(argv[++i], &opt.stage2_channels)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--rescale-t") == 0 && i + 1 < argc) {
            if (!parse_float_value(argv[++i], &opt.stage2_rescale_t)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(arg, "--obj") == 0 && i + 1 < argc) {
            obj_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (show_plan) {
        puts("Standalone TRELLIS operator plan:");
        strellis_print_operator_plan(stdout);
        puts("");
        puts("Standalone TRELLIS network plan:");
        strellis_print_network_plan(stdout);
        puts("");
    }

    strellis_infer_result result;
    memset(&result, 0, sizeof(result));
    strellis_status status = strellis_run_inference_compute(&opt, &result);
    if (status != STRELLIS_STATUS_OK) {
        fprintf(stderr, "strellis_run_inference_compute failed: %s\n", strellis_status_name(status));
        return 1;
    }

    printf("Standalone TRELLIS compute finished\n");
    printf("  weights: procedural deterministic parameters, no external loading\n");
    printf("  latent_size: %d\n", opt.latent_size);
    printf("  stage1_steps: %d\n", opt.stage1_steps);
    printf("  stage2_steps: %d\n", opt.stage2_steps);
    printf("  active_coords: %lld\n", (long long) result.n_coords);
    printf("  sparse_resolution: %d\n", result.sparse_resolution);
    printf("  slat_shape: [%lld,%d]\n", (long long) result.n_coords, result.slat_channels);
    printf("  mesh_vertices: %lld\n", (long long) result.mesh.n_vertices);
    printf("  mesh_faces: %lld\n", (long long) result.mesh.n_faces);

    if (obj_path != NULL) {
        if (!write_obj(obj_path, &result.mesh)) {
            fprintf(stderr, "failed to write OBJ: %s\n", obj_path);
            strellis_infer_result_free(&result);
            return 1;
        }
        printf("  obj: %s\n", obj_path);
    }

    strellis_infer_result_free(&result);
    return 0;
}
#endif

#if defined(STRELLIS_STANDALONE_DEMO) && !defined(STRELLIS_STANDALONE_CLI)
int main(void) {
    puts("Standalone TRELLIS operator plan:");
    strellis_print_operator_plan(stdout);
    puts("");
    puts("Standalone TRELLIS network plan:");
    strellis_print_network_plan(stdout);
    return 0;
}
#endif
