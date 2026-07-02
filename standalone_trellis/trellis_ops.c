#include "trellis_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const strellis_op_plan g_operator_plan[] = {
    {"linear", "Dense affine projection used by DiT, DINO, sparse rows, and decoder MLPs.", "[rows,in] x [out,in] -> [rows,out]", "test_linear", STRELLIS_OP_CPU_REFERENCE},
    {"layer_norm", "Row-wise channel normalization with optional affine scale and bias.", "[rows,channels]", "test_norms", STRELLIS_OP_CPU_REFERENCE},
    {"rms_norm", "Row-wise RMS normalization with optional learned scale.", "[rows,channels]", "test_norms", STRELLIS_OP_CPU_REFERENCE},
    {"multihead_rms_norm", "RMS normalization per attention head for query/key tensors.", "[batch,tokens,heads,head_dim]", "test_norms", STRELLIS_OP_CPU_REFERENCE},
    {"bf16_roundtrip", "Emulates BF16 checkpoint arithmetic for parity tests.", "flat f32 buffer", "test_bf16_roundtrip", STRELLIS_OP_CPU_REFERENCE},
    {"gelu_tanh", "PyTorch GELU approximate='tanh' activation.", "flat f32 buffer", "test_activations", STRELLIS_OP_CPU_REFERENCE},
    {"silu", "SiLU activation used in timestep MLPs and decoder residual blocks.", "flat f32 buffer", "test_activations", STRELLIS_OP_CPU_REFERENCE},
    {"add", "Element-wise residual addition.", "flat f32 buffers", "test_activations", STRELLIS_OP_CPU_REFERENCE},
    {"feed_forward", "Transformer MLP: linear, GELU, linear.", "[rows,channels]", "test_feed_forward", STRELLIS_OP_CPU_REFERENCE},
    {"timestep_embedding", "Sinusoidal flow timestep embedding.", "[n_timesteps] -> [n_timesteps,dim]", "test_flow_math", STRELLIS_OP_CPU_REFERENCE},
    {"timestep_mlp", "Timestep embedding projection MLP.", "[n_timesteps]", "test_timestep_mlp", STRELLIS_OP_CPU_REFERENCE},
    {"sdpa", "Scaled dot-product attention without masking.", "[batch,tokens,heads,head_dim]", "test_sdpa", STRELLIS_OP_CPU_REFERENCE},
    {"apply_rope_adjacent", "Adjacent-pair rotary embedding for attention tensors.", "[batch,tokens,heads,head_dim]", "test_rope", STRELLIS_OP_CPU_REFERENCE},
    {"rope_3d_phases", "Dense 3D voxel rotary phase table.", "[resolution^3,head_dim/2]", "test_rope", STRELLIS_OP_CPU_REFERENCE},
    {"rope_3d_sparse_phases", "Sparse coordinate rotary phase table.", "[n_coords,head_dim/2]", "test_rope", STRELLIS_OP_CPU_REFERENCE},
    {"dino_rope_2d_phases", "DINO patch-token rotary phase table plus special tokens.", "[special+patches,head_dim]", "test_rope", STRELLIS_OP_CPU_REFERENCE},
    {"flow_euler_step", "Euler step and x0 prediction for TRELLIS flow sampling.", "flat f32 buffers", "test_flow_math", STRELLIS_OP_CPU_REFERENCE},
    {"flow_cfg_combine", "Classifier-free guidance blend.", "flat f32 buffers", "test_flow_math", STRELLIS_OP_CPU_REFERENCE},
    {"flow_cfg_rescale_combine", "CFG blend with x0 std rescale.", "[batch,sample_stride]", "test_flow_math", STRELLIS_OP_CPU_REFERENCE},
    {"flow_timestep_pairs", "TRELLIS rescaled flow timestep pair schedule.", "[steps,2]", "test_flow_math", STRELLIS_OP_CPU_REFERENCE},
    {"conv3d_ncdhw", "Dense Conv3D for SparseStructureDecoder.", "x [N,C,D,H,W], w [O,C,KD,KH,KW]", "test_conv3d", STRELLIS_OP_CPU_REFERENCE},
    {"pixel_shuffle_3d", "3D pixel shuffle used by decoder upsample blocks.", "NCDHW", "test_pixel_shuffle_3d", STRELLIS_OP_CPU_REFERENCE},
    {"channel_layer_norm_3d", "Channel-wise volume normalization.", "NCDHW", "test_channel_layer_norm_3d", STRELLIS_OP_CPU_REFERENCE},
    {"dino_patch_embed", "DINO ViT patch embedding convolution.", "image [N,3,H,W] -> tokens [N,P,O]", "test_dino_patch_embed", STRELLIS_OP_CPU_REFERENCE},
    {"sparse_linear", "Row-wise affine projection over active sparse features.", "[n,in] x [out,in] -> [n,out]", "test_sparse_linear", STRELLIS_OP_CPU_REFERENCE},
    {"sparse_downsample_mean", "Sparse coordinate downsample with mean aggregation.", "coords [n,4], feats [n,c]", "test_sparse_coordinate_ops", STRELLIS_OP_CPU_REFERENCE},
    {"sparse_spatial2channel", "Pack sparse subcells into channel slots.", "coords [n,4], feats [n,c]", "test_sparse_coordinate_ops", STRELLIS_OP_CPU_REFERENCE},
    {"sparse_channel2spatial", "Unpack channel slots back into sparse subcells.", "coords [n,4], feats [n,c*factor^3]", "test_sparse_coordinate_ops", STRELLIS_OP_CPU_REFERENCE},
    {"sparse_subm_conv3d", "Submanifold sparse Conv3D over active coordinates.", "coords [n,4], weight [out,kd,kh,kw,in]", "test_sparse_subm_conv3d", STRELLIS_OP_CPU_REFERENCE},
    {"flexible_dual_grid_mesh", "Mesh extraction from FlexiDualGrid decoder logits.", "coords + decoder logits -> mesh", "test_mesh_extract", STRELLIS_OP_CPU_REFERENCE},
    {"modulated_cross_block", "AdaLN modulation, self attention, cross attention, and MLP block.", "network composite", "planned_network_test", STRELLIS_OP_NETWORK_TODO},
    {"dino_vit_encoder", "Full DINOv3 ViT encoder graph.", "network composite", "planned_network_test", STRELLIS_OP_NETWORK_TODO},
    {"sparse_structure_flow", "Stage1 sparse-structure flow transformer.", "network composite", "planned_network_test", STRELLIS_OP_NETWORK_TODO},
    {"shape_slat_flow", "Stage2 SLat flow transformer.", "network composite", "planned_network_test", STRELLIS_OP_NETWORK_TODO},
    {"shape_decoder", "Sparse ConvNeXt shape decoder and FlexiDualGrid head.", "network composite", "planned_network_test", STRELLIS_OP_NETWORK_TODO},
};

const char * strellis_status_name(strellis_status status) {
    switch (status) {
        case STRELLIS_STATUS_OK: return "ok";
        case STRELLIS_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case STRELLIS_STATUS_OUT_OF_MEMORY: return "out of memory";
        case STRELLIS_STATUS_NOT_IMPLEMENTED: return "not implemented";
        default: return "unknown";
    }
}

size_t strellis_operator_plan_count(void) {
    return sizeof(g_operator_plan) / sizeof(g_operator_plan[0]);
}

const strellis_op_plan * strellis_operator_plan_at(size_t index) {
    if (index >= strellis_operator_plan_count()) {
        return NULL;
    }
    return &g_operator_plan[index];
}

void strellis_print_operator_plan(FILE * out) {
    if (out == NULL) {
        out = stdout;
    }
    for (size_t i = 0; i < strellis_operator_plan_count(); ++i) {
        const strellis_op_plan * p = &g_operator_plan[i];
        fprintf(out, "%02zu %-28s %-18s %s\n", i, p->name, p->test_name, p->meaning);
    }
}

void strellis_sparse_tensor_free(strellis_sparse_tensor * tensor) {
    if (tensor == NULL) {
        return;
    }
    free(tensor->coords);
    free(tensor->feats);
    memset(tensor, 0, sizeof(*tensor));
}

void strellis_mesh_free(strellis_mesh * mesh) {
    if (mesh == NULL) {
        return;
    }
    free(mesh->vertices);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

static int valid_ptr4(const void * a, const void * b, const void * c, const void * d) {
    return a != NULL && b != NULL && c != NULL && d != NULL;
}

static size_t ncdhw_index(int b, int c, int d, int h, int w, int channels, int depth, int height, int width) {
    return (((((size_t) b * (size_t) channels + (size_t) c) * (size_t) depth + (size_t) d) * (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
}

static int coord_cmp4(const int32_t * a, const int32_t * b) {
    for (int i = 0; i < 4; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static int64_t find_coord(const int32_t * coords, int64_t n, const int32_t key[4]) {
    for (int64_t i = 0; i < n; ++i) {
        if (coord_cmp4(coords + 4 * i, key) == 0) {
            return i;
        }
    }
    return -1;
}

static int pow_int(int base, int exp) {
    int v = 1;
    for (int i = 0; i < exp; ++i) {
        v *= base;
    }
    return v;
}

static strellis_status alloc_sparse(strellis_sparse_tensor * t, int64_t n, int channels) {
    if (t == NULL || n < 0 || channels < 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(t, 0, sizeof(*t));
    t->n = n;
    t->channels = channels;
    if (n == 0 || channels == 0) {
        return STRELLIS_STATUS_OK;
    }
    t->coords = (int32_t *) calloc((size_t) n * 4u, sizeof(int32_t));
    t->feats = (float *) calloc((size_t) n * (size_t) channels, sizeof(float));
    if (t->coords == NULL || t->feats == NULL) {
        strellis_sparse_tensor_free(t);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return STRELLIS_STATUS_OK;
}

static void sort_sparse(strellis_sparse_tensor * t) {
    if (t == NULL || t->n <= 1 || t->channels <= 0) {
        return;
    }
    float * tmp_feat = (float *) malloc((size_t) t->channels * sizeof(float));
    if (tmp_feat == NULL) {
        return;
    }
    for (int64_t i = 1; i < t->n; ++i) {
        int32_t key_coord[4];
        memcpy(key_coord, &t->coords[4 * i], sizeof(key_coord));
        memcpy(tmp_feat, &t->feats[i * t->channels], (size_t) t->channels * sizeof(float));
        int64_t j = i - 1;
        while (j >= 0 && coord_cmp4(&t->coords[4 * j], key_coord) > 0) {
            memcpy(&t->coords[4 * (j + 1)], &t->coords[4 * j], 4u * sizeof(int32_t));
            memcpy(&t->feats[(j + 1) * t->channels], &t->feats[j * t->channels], (size_t) t->channels * sizeof(float));
            --j;
        }
        memcpy(&t->coords[4 * (j + 1)], key_coord, sizeof(key_coord));
        memcpy(&t->feats[(j + 1) * t->channels], tmp_feat, (size_t) t->channels * sizeof(float));
    }
    free(tmp_feat);
}

strellis_status strellis_linear_f32(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int64_t rows,
    int in_channels,
    int out_channels) {
    if (x == NULL || weight == NULL || y == NULL || rows < 0 || in_channels <= 0 || out_channels <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t r = 0; r < rows; ++r) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float acc = bias == NULL ? 0.0f : bias[oc];
            for (int ic = 0; ic < in_channels; ++ic) {
                acc += x[r * in_channels + ic] * weight[(int64_t) oc * in_channels + ic];
            }
            y[r * out_channels + oc] = acc;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_layer_norm_f32(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int64_t rows,
    int channels,
    float eps) {
    if (x == NULL || y == NULL || rows < 0 || channels <= 0 || eps < 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float * xr = x + r * channels;
        float * yr = y + r * channels;
        double mean = 0.0;
        for (int c = 0; c < channels; ++c) {
            mean += xr[c];
        }
        mean /= (double) channels;
        double var = 0.0;
        for (int c = 0; c < channels; ++c) {
            double d = (double) xr[c] - mean;
            var += d * d;
        }
        var /= (double) channels;
        const float inv = 1.0f / sqrtf((float) var + eps);
        for (int c = 0; c < channels; ++c) {
            float v = ((float) ((double) xr[c] - mean)) * inv;
            if (gamma != NULL) {
                v *= gamma[c];
            }
            if (beta != NULL) {
                v += beta[c];
            }
            yr[c] = v;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_rms_norm_f32(
    const float * x,
    const float * gamma,
    float * y,
    int64_t rows,
    int channels,
    float eps) {
    if (x == NULL || y == NULL || rows < 0 || channels <= 0 || eps < 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float * xr = x + r * channels;
        float * yr = y + r * channels;
        double mean_sq = 0.0;
        for (int c = 0; c < channels; ++c) {
            mean_sq += (double) xr[c] * (double) xr[c];
        }
        mean_sq /= (double) channels;
        const float inv = 1.0f / sqrtf((float) mean_sq + eps);
        for (int c = 0; c < channels; ++c) {
            float v = xr[c] * inv;
            if (gamma != NULL) {
                v *= gamma[c];
            }
            yr[c] = v;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_multihead_rms_norm_f32(
    const float * x,
    const float * gamma,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim,
    float eps) {
    if (x == NULL || gamma == NULL || y == NULL || batch < 0 || tokens <= 0 || heads <= 0 || head_dim <= 0 || eps < 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = (((size_t) b * (size_t) tokens + (size_t) t) * (size_t) heads + (size_t) h) * (size_t) head_dim;
                double mean_sq = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    mean_sq += (double) x[base + d] * (double) x[base + d];
                }
                mean_sq /= (double) head_dim;
                const float inv = 1.0f / sqrtf((float) mean_sq + eps);
                for (int d = 0; d < head_dim; ++d) {
                    y[base + d] = x[base + d] * inv * gamma[h * head_dim + d];
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_bf16_roundtrip_f32(const float * x, float * y, size_t n) {
    if (x == NULL || y == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        union {
            float f;
            uint32_t u;
        } v;
        v.f = x[i];
        uint32_t lsb = (v.u >> 16) & 1u;
        v.u += 0x7fffu + lsb;
        v.u &= 0xffff0000u;
        y[i] = v.f;
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_gelu_tanh_f32(const float * x, float * y, size_t n) {
    if (x == NULL || y == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float k = sqrtf(2.0f / (float) M_PI);
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_silu_f32(const float * x, float * y, size_t n) {
    if (x == NULL || y == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_add_f32(const float * a, const float * b, float * y, size_t n) {
    if (a == NULL || b == NULL || y == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_feed_forward_f32(
    const float * x,
    const float * w1,
    const float * b1,
    const float * w2,
    const float * b2,
    float * workspace,
    float * y,
    int64_t rows,
    int in_channels,
    int hidden_channels,
    int out_channels) {
    if (!valid_ptr4(x, w1, w2, workspace) || y == NULL || rows < 0 || in_channels <= 0 || hidden_channels <= 0 || out_channels <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    strellis_status status = strellis_linear_f32(x, w1, b1, workspace, rows, in_channels, hidden_channels);
    if (status != STRELLIS_STATUS_OK) return status;
    status = strellis_gelu_tanh_f32(workspace, workspace, (size_t) rows * (size_t) hidden_channels);
    if (status != STRELLIS_STATUS_OK) return status;
    return strellis_linear_f32(workspace, w2, b2, y, rows, hidden_channels, out_channels);
}

strellis_status strellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding) {
    if (timesteps == NULL || embedding == NULL || dim <= 0 || max_period <= 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half = dim / 2;
    if (half == 0) {
        for (size_t n = 0; n < n_timesteps; ++n) {
            embedding[n] = 0.0f;
        }
        return STRELLIS_STATUS_OK;
    }
    for (size_t n = 0; n < n_timesteps; ++n) {
        float * out = embedding + n * (size_t) dim;
        for (int i = 0; i < half; ++i) {
            const float freq = expf(-logf(max_period) * (float) i / (float) half);
            const float arg = timesteps[n] * freq;
            out[i] = cosf(arg);
            out[half + i] = sinf(arg);
        }
        if ((dim & 1) != 0) {
            out[dim - 1] = 0.0f;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_timestep_mlp_f32(
    const float * timesteps,
    size_t n_timesteps,
    int frequency_dim,
    const float * w1,
    const float * b1,
    const float * w2,
    const float * b2,
    float * workspace_embedding,
    float * workspace_hidden,
    float * y,
    int hidden_channels,
    int out_channels) {
    if (timesteps == NULL || w1 == NULL || w2 == NULL || workspace_embedding == NULL || workspace_hidden == NULL || y == NULL ||
        frequency_dim <= 0 || hidden_channels <= 0 || out_channels <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    strellis_status status = strellis_timestep_embedding_f32(timesteps, n_timesteps, frequency_dim, 10000.0f, workspace_embedding);
    if (status != STRELLIS_STATUS_OK) return status;
    status = strellis_linear_f32(workspace_embedding, w1, b1, workspace_hidden, (int64_t) n_timesteps, frequency_dim, hidden_channels);
    if (status != STRELLIS_STATUS_OK) return status;
    status = strellis_silu_f32(workspace_hidden, workspace_hidden, n_timesteps * (size_t) hidden_channels);
    if (status != STRELLIS_STATUS_OK) return status;
    return strellis_linear_f32(workspace_hidden, w2, b2, y, (int64_t) n_timesteps, hidden_channels, out_channels);
}

strellis_status strellis_sdpa_f32(
    const float * q,
    const float * k,
    const float * v,
    float * y,
    int batch,
    int q_tokens,
    int kv_tokens,
    int heads,
    int head_dim,
    float scale) {
    if (!valid_ptr4(q, k, v, y) || batch < 0 || q_tokens <= 0 || kv_tokens <= 0 || heads <= 0 || head_dim <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int b = 0; b < batch; ++b) {
        for (int tq = 0; tq < q_tokens; ++tq) {
            for (int h = 0; h < heads; ++h) {
                float max_score = -INFINITY;
                for (int tk = 0; tk < kv_tokens; ++tk) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t qi = (((size_t) b * q_tokens + tq) * heads + h) * head_dim + d;
                        size_t ki = (((size_t) b * kv_tokens + tk) * heads + h) * head_dim + d;
                        dot += (double) q[qi] * (double) k[ki];
                    }
                    float score = (float) dot * scale;
                    if (score > max_score) {
                        max_score = score;
                    }
                }
                double denom = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    size_t yi = (((size_t) b * q_tokens + tq) * heads + h) * head_dim + d;
                    y[yi] = 0.0f;
                }
                for (int tk = 0; tk < kv_tokens; ++tk) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t qi = (((size_t) b * q_tokens + tq) * heads + h) * head_dim + d;
                        size_t ki = (((size_t) b * kv_tokens + tk) * heads + h) * head_dim + d;
                        dot += (double) q[qi] * (double) k[ki];
                    }
                    const float prob_unnorm = expf((float) dot * scale - max_score);
                    denom += prob_unnorm;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t yi = (((size_t) b * q_tokens + tq) * heads + h) * head_dim + d;
                        size_t vi = (((size_t) b * kv_tokens + tk) * heads + h) * head_dim + d;
                        y[yi] += prob_unnorm * v[vi];
                    }
                }
                for (int d = 0; d < head_dim; ++d) {
                    size_t yi = (((size_t) b * q_tokens + tq) * heads + h) * head_dim + d;
                    y[yi] = (float) ((double) y[yi] / denom);
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_apply_rope_adjacent_f32(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim) {
    if (!valid_ptr4(x, cos_phase, sin_phase, y) || batch < 0 || tokens <= 0 || heads <= 0 || head_dim <= 0 || (head_dim & 1) != 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half_dim = head_dim / 2;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = (((size_t) b * tokens + t) * heads + h) * head_dim;
                for (int p = 0; p < half_dim; ++p) {
                    const int d0 = 2 * p;
                    const int d1 = d0 + 1;
                    const float c = cos_phase[(size_t) t * half_dim + p];
                    const float s = sin_phase[(size_t) t * half_dim + p];
                    const float v0 = x[base + d0];
                    const float v1 = x[base + d1];
                    y[base + d0] = v0 * c - v1 * s;
                    y[base + d1] = v0 * s + v1 * c;
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (resolution <= 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f || cos_out == NULL || sin_out == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half = head_dim / 2;
    const int freq_dim = half / 3;
    const size_t tokens = (size_t) resolution * (size_t) resolution * (size_t) resolution;
    if (phase_count < tokens * (size_t) half) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                const size_t token = ((size_t) x * (size_t) resolution + (size_t) y) * (size_t) resolution + (size_t) z;
                const int indices[3] = {x, y, z};
                float * cos_row = cos_out + token * (size_t) half;
                float * sin_row = sin_out + token * (size_t) half;
                int k = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    for (int f = 0; f < freq_dim; ++f) {
                        const float power = (float) f / (float) freq_dim;
                        const float freq = freq_scale / powf(freq_base, power);
                        const float phase = (float) indices[axis] * freq;
                        cos_row[k] = cosf(phase);
                        sin_row[k] = sinf(phase);
                        ++k;
                    }
                }
                while (k < half) {
                    cos_row[k] = 1.0f;
                    sin_row[k] = 0.0f;
                    ++k;
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (coords == NULL || n_coords < 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f || cos_out == NULL || sin_out == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int half = head_dim / 2;
    const int freq_dim = half / 3;
    if (phase_count < (size_t) n_coords * (size_t) half) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t token = 0; token < n_coords; ++token) {
        const int32_t * row = coords + token * 4;
        const int indices[3] = {row[1], row[2], row[3]};
        float * cos_row = cos_out + (size_t) token * (size_t) half;
        float * sin_row = sin_out + (size_t) token * (size_t) half;
        int k = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int f = 0; f < freq_dim; ++f) {
                const float power = (float) f / (float) freq_dim;
                const float freq = freq_scale / powf(freq_base, power);
                const float phase = (float) indices[axis] * freq;
                cos_row[k] = cosf(phase);
                sin_row[k] = sinf(phase);
                ++k;
            }
        }
        while (k < half) {
            cos_row[k] = 1.0f;
            sin_row[k] = 0.0f;
            ++k;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_dino_rope_2d_phases_f32(
    int n_special_tokens,
    int patches_h,
    int patches_w,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count) {
    if (n_special_tokens < 0 || patches_h <= 0 || patches_w <= 0 ||
        head_dim <= 0 || (head_dim & 1) != 0 ||
        freq_scale <= 0.0f || freq_base <= 0.0f ||
        cos_out == NULL || sin_out == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int freq_dim = head_dim / 4;
    const int total_tokens = n_special_tokens + patches_h * patches_w;
    if (freq_dim <= 0 || phase_count < (size_t) total_tokens * (size_t) head_dim) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int token = 0; token < total_tokens; ++token) {
        float * cos_row = cos_out + (size_t) token * (size_t) head_dim;
        float * sin_row = sin_out + (size_t) token * (size_t) head_dim;
        if (token < n_special_tokens) {
            for (int k = 0; k < head_dim; ++k) {
                cos_row[k] = 1.0f;
                sin_row[k] = 0.0f;
            }
            continue;
        }
        const int patch = token - n_special_tokens;
        const int py = patch / patches_w;
        const int px = patch - py * patches_w;
        const float coords2[2] = {
            2.0f * (((float) py + 0.5f) / (float) patches_h) - 1.0f,
            2.0f * (((float) px + 0.5f) / (float) patches_w) - 1.0f,
        };
        int k = 0;
        for (int axis = 0; axis < 2; ++axis) {
            for (int f = 0; f < freq_dim; ++f) {
                const float power = (float) (4 * f) / (float) head_dim;
                const float freq = freq_scale / powf(freq_base, power);
                const float phase = (float) (2.0 * M_PI) * coords2[axis] * freq;
                cos_row[k] = cosf(phase);
                sin_row[k] = sinf(phase);
                ++k;
            }
        }
        for (int i = 0; i < k; ++i) {
            cos_row[k + i] = cos_row[i];
            sin_row[k + i] = sin_row[i];
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0) {
    if (!valid_ptr4(x_t, pred_v, pred_x_prev, pred_x0)) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    for (size_t i = 0; i < n; ++i) {
        pred_x_prev[i] = x_t[i] - (t - t_prev) * pred_v[i];
        pred_x0[i] = (1.0f - sigma_min) * x_t[i] - sigma_t * pred_v[i];
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred) {
    if (pred_pos == NULL || pred_neg == NULL || pred == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        pred[i] = guidance_strength * pred_pos[i] + (1.0f - guidance_strength) * pred_neg[i];
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_flow_cfg_rescale_combine_f32(
    const float * x_t,
    const float * pred_pos,
    const float * pred_neg,
    size_t batch,
    size_t sample_stride,
    float sigma_min,
    float t,
    float guidance_strength,
    float guidance_rescale,
    float * pred) {
    if (x_t == NULL || pred_pos == NULL || pred_neg == NULL || pred == NULL || sample_stride == 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const float sigma_t = sigma_min + (1.0f - sigma_min) * t;
    const float one_minus_sigma_min = 1.0f - sigma_min;
    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_stride;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            pred[j] = guidance_strength * pred_pos[j] + (1.0f - guidance_strength) * pred_neg[j];
        }
        if (guidance_rescale <= 0.0f) {
            continue;
        }
        double mean_pos = 0.0;
        double mean_cfg = 0.0;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const float x0_pos = one_minus_sigma_min * x_t[j] - sigma_t * pred_pos[j];
            const float x0_cfg = one_minus_sigma_min * x_t[j] - sigma_t * pred[j];
            mean_pos += x0_pos;
            mean_cfg += x0_cfg;
        }
        mean_pos /= (double) sample_stride;
        mean_cfg /= (double) sample_stride;
        double var_pos = 0.0;
        double var_cfg = 0.0;
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const double x0_pos = (double) (one_minus_sigma_min * x_t[j] - sigma_t * pred_pos[j]) - mean_pos;
            const double x0_cfg = (double) (one_minus_sigma_min * x_t[j] - sigma_t * pred[j]) - mean_cfg;
            var_pos += x0_pos * x0_pos;
            var_cfg += x0_cfg * x0_cfg;
        }
        const double denom = sample_stride > 1 ? (double) (sample_stride - 1) : (double) sample_stride;
        const double std_pos = sqrt(var_pos / denom);
        const double std_cfg = sqrt(var_cfg / denom);
        if (std_cfg <= 0.0) {
            continue;
        }
        const float ratio = (float) (std_pos / std_cfg);
        for (size_t i = 0; i < sample_stride; ++i) {
            const size_t j = base + i;
            const float x0_cfg = one_minus_sigma_min * x_t[j] - sigma_t * pred[j];
            const float x0_rescaled = x0_cfg * ratio;
            const float x0 = guidance_rescale * x0_rescaled + (1.0f - guidance_rescale) * x0_cfg;
            pred[j] = (one_minus_sigma_min * x_t[j] - x0) / sigma_t;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count) {
    if (steps <= 0 || rescale_t <= 0.0f || pairs == NULL || pair_count < (size_t) steps * 2u) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int i = 0; i < steps; ++i) {
        const float u0 = 1.0f - (float) i / (float) steps;
        const float u1 = 1.0f - (float) (i + 1) / (float) steps;
        pairs[2 * i + 0] = rescale_t * u0 / (1.0f + (rescale_t - 1.0f) * u0);
        pairs[2 * i + 1] = rescale_t * u1 / (1.0f + (rescale_t - 1.0f) * u1);
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_conv3d_ncdhw_f32(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    if (x == NULL || weight == NULL || y == NULL || batch < 0 || in_channels <= 0 || in_d <= 0 || in_h <= 0 || in_w <= 0 ||
        out_channels <= 0 || kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 || stride_d <= 0 || stride_h <= 0 || stride_w <= 0 ||
        dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_d = (in_d + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    const int out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float acc = bias == NULL ? 0.0f : bias[oc];
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kd = 0; kd < kernel_d; ++kd) {
                                const int id = od * stride_d + kd * dilation_d - pad_d;
                                if (id < 0 || id >= in_d) continue;
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    const int ih = oh * stride_h + kh * dilation_h - pad_h;
                                    if (ih < 0 || ih >= in_h) continue;
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        const int iw = ow * stride_w + kw * dilation_w - pad_w;
                                        if (iw < 0 || iw >= in_w) continue;
                                        const size_t xi = ncdhw_index(b, ic, id, ih, iw, in_channels, in_d, in_h, in_w);
                                        const size_t wi = (((((size_t) oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                                        acc += x[xi] * weight[wi];
                                    }
                                }
                            }
                        }
                        y[ncdhw_index(b, oc, od, oh, ow, out_channels, out_d, out_h, out_w)] = acc;
                    }
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_pixel_shuffle_3d_ncdhw_f32(
    const float * x,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale) {
    if (x == NULL || y == NULL || batch < 0 || in_channels <= 0 || in_d <= 0 || in_h <= 0 || in_w <= 0 || scale <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int slots = scale * scale * scale;
    if (in_channels % slots != 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_channels = in_channels / slots;
    const int out_d = in_d * scale;
    const int out_h = in_h * scale;
    const int out_w = in_w * scale;
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < out_d; ++od) {
                const int id = od / scale;
                const int rd = od - id * scale;
                for (int oh = 0; oh < out_h; ++oh) {
                    const int ih = oh / scale;
                    const int rh = oh - ih * scale;
                    for (int ow = 0; ow < out_w; ++ow) {
                        const int iw = ow / scale;
                        const int rw = ow - iw * scale;
                        const int ic = (((oc * scale) + rd) * scale + rh) * scale + rw;
                        y[ncdhw_index(b, oc, od, oh, ow, out_channels, out_d, out_h, out_w)] =
                            x[ncdhw_index(b, ic, id, ih, iw, in_channels, in_d, in_h, in_w)];
                    }
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_channel_layer_norm_3d_ncdhw_f32(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps) {
    if (x == NULL || y == NULL || batch < 0 || channels <= 0 || depth <= 0 || height <= 0 || width <= 0 || eps < 0.0f) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < depth; ++d) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    double mean = 0.0;
                    for (int c = 0; c < channels; ++c) {
                        mean += x[ncdhw_index(b, c, d, h, w, channels, depth, height, width)];
                    }
                    mean /= (double) channels;
                    double var = 0.0;
                    for (int c = 0; c < channels; ++c) {
                        const double delta = (double) x[ncdhw_index(b, c, d, h, w, channels, depth, height, width)] - mean;
                        var += delta * delta;
                    }
                    var /= (double) channels;
                    const float inv = 1.0f / sqrtf((float) var + eps);
                    for (int c = 0; c < channels; ++c) {
                        const size_t idx = ncdhw_index(b, c, d, h, w, channels, depth, height, width);
                        float v = ((float) ((double) x[idx] - mean)) * inv;
                        if (gamma != NULL) v *= gamma[c];
                        if (beta != NULL) v += beta[c];
                        y[idx] = v;
                    }
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_dino_patch_embed_f32(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size) {
    if (image == NULL || weight == NULL || tokens == NULL || batch < 0 || image_h <= 0 || image_w <= 0 || out_channels <= 0 || patch_size <= 0 ||
        image_h % patch_size != 0 || image_w % patch_size != 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int patches_h = image_h / patch_size;
    const int patches_w = image_w / patch_size;
    const int n_patches = patches_h * patches_w;
    for (int b = 0; b < batch; ++b) {
        for (int py = 0; py < patches_h; ++py) {
            for (int px = 0; px < patches_w; ++px) {
                const int token = py * patches_w + px;
                for (int oc = 0; oc < out_channels; ++oc) {
                    float acc = bias == NULL ? 0.0f : bias[oc];
                    for (int ic = 0; ic < 3; ++ic) {
                        for (int ky = 0; ky < patch_size; ++ky) {
                            for (int kx = 0; kx < patch_size; ++kx) {
                                const int iy = py * patch_size + ky;
                                const int ix = px * patch_size + kx;
                                const size_t image_idx = (((size_t) b * 3u + (size_t) ic) * (size_t) image_h + (size_t) iy) * (size_t) image_w + (size_t) ix;
                                const size_t weight_idx = (((size_t) oc * 3u + (size_t) ic) * (size_t) patch_size + (size_t) ky) * (size_t) patch_size + (size_t) kx;
                                acc += image[image_idx] * weight[weight_idx];
                            }
                        }
                    }
                    tokens[((size_t) b * (size_t) n_patches + (size_t) token) * (size_t) out_channels + (size_t) oc] = acc;
                }
            }
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_sparse_linear_f32(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    return strellis_linear_f32(x, weight, bias, y, n, in_channels, out_channels);
}

strellis_status strellis_sparse_downsample_mean_f32(
    const strellis_sparse_tensor * input,
    int factor,
    strellis_sparse_tensor * output) {
    if (input == NULL || output == NULL || factor <= 0 || input->channels <= 0 || input->n < 0 || input->coords == NULL || input->feats == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    strellis_sparse_tensor tmp;
    strellis_status status = alloc_sparse(&tmp, input->n, input->channels);
    if (status != STRELLIS_STATUS_OK) return status;
    int64_t * counts = (int64_t *) calloc((size_t) input->n, sizeof(int64_t));
    if (counts == NULL) {
        strellis_sparse_tensor_free(&tmp);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }
    int64_t groups = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        int32_t key[4] = {
            input->coords[4 * i + 0],
            input->coords[4 * i + 1] / factor,
            input->coords[4 * i + 2] / factor,
            input->coords[4 * i + 3] / factor,
        };
        int64_t idx = find_coord(tmp.coords, groups, key);
        if (idx < 0) {
            idx = groups++;
            memcpy(&tmp.coords[4 * idx], key, sizeof(key));
        }
        counts[idx] += 1;
        for (int c = 0; c < input->channels; ++c) {
            tmp.feats[idx * input->channels + c] += input->feats[i * input->channels + c];
        }
    }
    for (int64_t i = 0; i < groups; ++i) {
        for (int c = 0; c < input->channels; ++c) {
            tmp.feats[i * input->channels + c] /= (float) counts[i];
        }
    }
    free(counts);
    strellis_sparse_tensor shrunk;
    status = alloc_sparse(&shrunk, groups, input->channels);
    if (status != STRELLIS_STATUS_OK) {
        strellis_sparse_tensor_free(&tmp);
        return status;
    }
    memcpy(shrunk.coords, tmp.coords, (size_t) groups * 4u * sizeof(int32_t));
    memcpy(shrunk.feats, tmp.feats, (size_t) groups * (size_t) input->channels * sizeof(float));
    strellis_sparse_tensor_free(&tmp);
    sort_sparse(&shrunk);
    *output = shrunk;
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_sparse_spatial2channel_f32(
    const strellis_sparse_tensor * input,
    int factor,
    strellis_sparse_tensor * output) {
    if (input == NULL || output == NULL || factor <= 0 || input->channels <= 0 || input->n < 0 || input->coords == NULL || input->feats == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int slots = pow_int(factor, 3);
    strellis_sparse_tensor tmp;
    strellis_status status = alloc_sparse(&tmp, input->n, input->channels * slots);
    if (status != STRELLIS_STATUS_OK) return status;
    int64_t groups = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        int32_t key[4] = {
            input->coords[4 * i + 0],
            input->coords[4 * i + 1] / factor,
            input->coords[4 * i + 2] / factor,
            input->coords[4 * i + 3] / factor,
        };
        int64_t idx = find_coord(tmp.coords, groups, key);
        if (idx < 0) {
            idx = groups++;
            memcpy(&tmp.coords[4 * idx], key, sizeof(key));
        }
        int sx = input->coords[4 * i + 1] % factor;
        int sy = input->coords[4 * i + 2] % factor;
        int sz = input->coords[4 * i + 3] % factor;
        if (sx < 0) sx += factor;
        if (sy < 0) sy += factor;
        if (sz < 0) sz += factor;
        const int subidx = sx + sy * factor + sz * factor * factor;
        for (int c = 0; c < input->channels; ++c) {
            tmp.feats[idx * tmp.channels + (int64_t) subidx * input->channels + c] = input->feats[i * input->channels + c];
        }
    }
    strellis_sparse_tensor shrunk;
    status = alloc_sparse(&shrunk, groups, input->channels * slots);
    if (status != STRELLIS_STATUS_OK) {
        strellis_sparse_tensor_free(&tmp);
        return status;
    }
    memcpy(shrunk.coords, tmp.coords, (size_t) groups * 4u * sizeof(int32_t));
    memcpy(shrunk.feats, tmp.feats, (size_t) groups * (size_t) tmp.channels * sizeof(float));
    strellis_sparse_tensor_free(&tmp);
    sort_sparse(&shrunk);
    *output = shrunk;
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_sparse_channel2spatial_f32(
    const strellis_sparse_tensor * input,
    const uint8_t * subdivision,
    int factor,
    strellis_sparse_tensor * output) {
    if (input == NULL || output == NULL || subdivision == NULL || factor <= 0 ||
        input->channels <= 0 || input->n < 0 || input->coords == NULL || input->feats == NULL) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int slots = pow_int(factor, 3);
    if (input->channels % slots != 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int out_channels = input->channels / slots;
    int64_t n_out = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        for (int s = 0; s < slots; ++s) {
            if (subdivision[i * slots + s]) {
                ++n_out;
            }
        }
    }
    strellis_sparse_tensor out;
    strellis_status status = alloc_sparse(&out, n_out, out_channels);
    if (status != STRELLIS_STATUS_OK) return status;
    int64_t row = 0;
    for (int64_t i = 0; i < input->n; ++i) {
        for (int s = 0; s < slots; ++s) {
            if (!subdivision[i * slots + s]) {
                continue;
            }
            const int sx = s % factor;
            const int sy = (s / factor) % factor;
            const int sz = (s / (factor * factor)) % factor;
            out.coords[4 * row + 0] = input->coords[4 * i + 0];
            out.coords[4 * row + 1] = input->coords[4 * i + 1] * factor + sx;
            out.coords[4 * row + 2] = input->coords[4 * i + 2] * factor + sy;
            out.coords[4 * row + 3] = input->coords[4 * i + 3] * factor + sz;
            for (int c = 0; c < out_channels; ++c) {
                out.feats[row * out_channels + c] = input->feats[i * input->channels + (int64_t) s * out_channels + c];
            }
            ++row;
        }
    }
    sort_sparse(&out);
    *output = out;
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_sparse_subm_conv3d_f32(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    const float * bias,
    float * out,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w) {
    if (coords == NULL || feats == NULL || weight == NULL || out == NULL || n < 0 || in_channels <= 0 || out_channels <= 0 ||
        kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 || dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const int center_d = kernel_d / 2;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * center = coords + 4 * row;
        for (int oc = 0; oc < out_channels; ++oc) {
            float acc = bias == NULL ? 0.0f : bias[oc];
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int nx = center[1] + (kd - center_d) * dilation_d;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    const int ny = center[2] + (kh - center_h) * dilation_h;
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int nz = center[3] + (kw - center_w) * dilation_w;
                        const int32_t key[4] = {center[0], nx, ny, nz};
                        const int64_t in_row = find_coord(coords, n, key);
                        if (in_row < 0) {
                            continue;
                        }
                        const int64_t w_base = ((((int64_t) oc * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * in_channels;
                        const int64_t f_base = in_row * in_channels;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            acc += feats[f_base + ic] * weight[w_base + ic];
                        }
                    }
                }
            }
            out[row * out_channels + oc] = acc;
        }
    }
    return STRELLIS_STATUS_OK;
}

strellis_status strellis_flexible_dual_grid_mesh_from_decoder_logits_f32(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    strellis_mesh * mesh_out) {
    if (coords == NULL || feats == NULL || mesh_out == NULL || n < 0 || channels <= 0 || resolution <= 0) {
        return STRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));

    const int dirs[6][3] = {
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1},
    };
    int64_t face_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (feats[i * channels] < 0.0f) {
            continue;
        }
        const int32_t * c = coords + 4 * i;
        for (int d = 0; d < 6; ++d) {
            const int32_t key[4] = {
                c[0],
                c[1] + dirs[d][0],
                c[2] + dirs[d][1],
                c[3] + dirs[d][2],
            };
            int64_t j = find_coord(coords, n, key);
            if (j < 0 || feats[j * channels] < 0.0f) {
                ++face_count;
            }
        }
    }
    if (face_count == 0) {
        return STRELLIS_STATUS_OK;
    }

    const int64_t vertex_count = face_count * 4;
    const int64_t tri_count = face_count * 2;
    float * vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    uint32_t * faces = (uint32_t *) malloc((size_t) tri_count * 3u * sizeof(uint32_t));
    if (vertices == NULL || faces == NULL) {
        free(vertices);
        free(faces);
        return STRELLIS_STATUS_OUT_OF_MEMORY;
    }

    const int quads[6][4] = {
        {0, 4, 7, 3},
        {1, 2, 6, 5},
        {0, 1, 5, 4},
        {3, 7, 6, 2},
        {0, 3, 2, 1},
        {4, 5, 6, 7},
    };
    int64_t vrow = 0;
    int64_t frow = 0;
    const float inv_res = 1.0f / (float) resolution;
    for (int64_t i = 0; i < n; ++i) {
        if (feats[i * channels] < 0.0f) {
            continue;
        }
        const int32_t * c = coords + 4 * i;
        const float x0 = (float) c[1] * inv_res - 0.5f;
        const float y0 = (float) c[2] * inv_res - 0.5f;
        const float z0 = (float) c[3] * inv_res - 0.5f;
        const float x1 = (float) (c[1] + 1) * inv_res - 0.5f;
        const float y1 = (float) (c[2] + 1) * inv_res - 0.5f;
        const float z1 = (float) (c[3] + 1) * inv_res - 0.5f;
        const float corners[8][3] = {
            {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
            {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1},
        };
        for (int d = 0; d < 6; ++d) {
            const int32_t key[4] = {
                c[0],
                c[1] + dirs[d][0],
                c[2] + dirs[d][1],
                c[3] + dirs[d][2],
            };
            int64_t j = find_coord(coords, n, key);
            if (j >= 0 && feats[j * channels] >= 0.0f) {
                continue;
            }
            const uint32_t base = (uint32_t) vrow;
            for (int q = 0; q < 4; ++q) {
                const int ci = quads[d][q];
                vertices[(size_t) vrow * 3u + 0u] = corners[ci][0];
                vertices[(size_t) vrow * 3u + 1u] = corners[ci][1];
                vertices[(size_t) vrow * 3u + 2u] = corners[ci][2];
                ++vrow;
            }
            faces[(size_t) frow * 3u + 0u] = base + 0u;
            faces[(size_t) frow * 3u + 1u] = base + 1u;
            faces[(size_t) frow * 3u + 2u] = base + 2u;
            ++frow;
            faces[(size_t) frow * 3u + 0u] = base + 0u;
            faces[(size_t) frow * 3u + 1u] = base + 2u;
            faces[(size_t) frow * 3u + 2u] = base + 3u;
            ++frow;
        }
    }

    mesh_out->vertices = vertices;
    mesh_out->faces = faces;
    mesh_out->n_vertices = vrow;
    mesh_out->n_faces = frow;
    return STRELLIS_STATUS_OK;
}
