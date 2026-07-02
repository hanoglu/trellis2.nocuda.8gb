#ifndef STANDALONE_TRELLIS_OPS_H
#define STANDALONE_TRELLIS_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRELLIS_MAX_DIMS 6

typedef enum strellis_status {
    STRELLIS_STATUS_OK = 0,
    STRELLIS_STATUS_INVALID_ARGUMENT = 1,
    STRELLIS_STATUS_OUT_OF_MEMORY = 2,
    STRELLIS_STATUS_NOT_IMPLEMENTED = 3
} strellis_status;

typedef enum strellis_op_state {
    STRELLIS_OP_PLANNED = 0,
    STRELLIS_OP_CPU_REFERENCE = 1,
    STRELLIS_OP_ACCELERATOR_TODO = 2,
    STRELLIS_OP_NETWORK_TODO = 3
} strellis_op_state;

typedef struct strellis_op_plan {
    const char * name;
    const char * meaning;
    const char * layout;
    const char * test_name;
    strellis_op_state state;
} strellis_op_plan;

typedef struct strellis_sparse_tensor {
    int32_t * coords; /* [n, 4], rows are batch,x,y,z */
    float * feats;    /* [n, channels] */
    int64_t n;
    int channels;
} strellis_sparse_tensor;

typedef struct strellis_mesh {
    float * vertices;   /* [n_vertices, 3] */
    uint32_t * faces;   /* [n_faces, 3] */
    int64_t n_vertices;
    int64_t n_faces;
} strellis_mesh;

typedef struct strellis_network_step_plan {
    const char * name;
    const char * inputs;
    const char * outputs;
    const char * required_ops;
    const char * status;
} strellis_network_step_plan;

typedef struct strellis_infer_options {
    int latent_size;       /* Stage1 dense grid edge. Keep small for CPU reference. */
    int stage1_steps;
    int stage2_steps;
    int cond_tokens;
    int cond_channels;
    int stage1_channels;
    int stage2_channels;
    float voxel_threshold;
    float stage2_rescale_t;
    uint32_t seed;
} strellis_infer_options;

typedef struct strellis_infer_result {
    int32_t * coords_bxyz;
    int64_t n_coords;
    int sparse_resolution;
    float * slat_feats;
    int slat_channels;
    strellis_mesh mesh;
} strellis_infer_result;

typedef struct strellis_dit_flow_block_weights {
    const float * modulation;       /* [6 * model_channels] */
    const float * norm2_gamma;      /* [model_channels] */
    const float * norm2_beta;       /* [model_channels] */

    const float * self_qkv_w;       /* [3 * model_channels, model_channels] */
    const float * self_qkv_b;       /* [3 * model_channels] */
    const float * self_q_rms_gamma; /* [heads, head_dim] */
    const float * self_k_rms_gamma; /* [heads, head_dim] */
    const float * self_out_w;       /* [model_channels, model_channels] */
    const float * self_out_b;       /* [model_channels] */

    const float * cross_q_w;        /* [model_channels, model_channels] */
    const float * cross_q_b;        /* [model_channels] */
    const float * cross_kv_w;       /* [2 * model_channels, cond_channels] */
    const float * cross_kv_b;       /* [2 * model_channels] */
    const float * cross_q_rms_gamma;/* [heads, head_dim] */
    const float * cross_k_rms_gamma;/* [heads, head_dim] */
    const float * cross_out_w;      /* [model_channels, model_channels] */
    const float * cross_out_b;      /* [model_channels] */

    const float * mlp_fc1_w;        /* [mlp_channels, model_channels] */
    const float * mlp_fc1_b;        /* [mlp_channels] */
    const float * mlp_fc2_w;        /* [model_channels, mlp_channels] */
    const float * mlp_fc2_b;        /* [model_channels] */
} strellis_dit_flow_block_weights;

typedef struct strellis_dit_flow_weights {
    int in_channels;
    int out_channels;
    int model_channels;
    int cond_channels;
    int time_frequency_dim;
    int heads;
    int head_dim;
    int mlp_channels;
    int mod_channels;
    int n_blocks;
    int debug_block_parts;  /* -1 runs full block, otherwise 0..3 like the ggml debug path. */
    int debug_disable_rope;
    int emulate_bf16_blocks;
    float final_norm_eps;

    const float * input_w;       /* [model_channels, in_channels] */
    const float * input_b;       /* [model_channels] */
    const float * t_embedder_0_w;/* [model_channels, time_frequency_dim] */
    const float * t_embedder_0_b;/* [model_channels] */
    const float * t_embedder_2_w;/* [model_channels, model_channels] */
    const float * t_embedder_2_b;/* [model_channels] */
    const float * adaln_w;       /* [6 * model_channels, model_channels] */
    const float * adaln_b;       /* [6 * model_channels] */
    const float * out_w;         /* [out_channels, model_channels] */
    const float * out_b;         /* [out_channels] */
    const strellis_dit_flow_block_weights * blocks;
} strellis_dit_flow_weights;

const char * strellis_status_name(strellis_status status);
size_t strellis_operator_plan_count(void);
const strellis_op_plan * strellis_operator_plan_at(size_t index);
void strellis_print_operator_plan(FILE * out);

size_t strellis_network_plan_count(void);
const strellis_network_step_plan * strellis_network_plan_at(size_t index);
void strellis_print_network_plan(FILE * out);

void strellis_sparse_tensor_free(strellis_sparse_tensor * tensor);
void strellis_mesh_free(strellis_mesh * mesh);
void strellis_infer_options_default(strellis_infer_options * options);
void strellis_infer_result_free(strellis_infer_result * result);
strellis_status strellis_run_inference_compute(
    const strellis_infer_options * options,
    strellis_infer_result * result);

/* TRELLIS DiT flow forward without ggml: x/context are [batch,tokens,channels]. */
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
    int cond_tokens);

/* Dense affine projection: y[row,out] = bias[out] + sum_i x[row,i] * weight[out,i]. */
strellis_status strellis_linear_f32(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int64_t rows,
    int in_channels,
    int out_channels);

/* Channel-wise layer normalization over each row. */
strellis_status strellis_layer_norm_f32(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int64_t rows,
    int channels,
    float eps);

/* RMS normalization over each row. */
strellis_status strellis_rms_norm_f32(
    const float * x,
    const float * gamma,
    float * y,
    int64_t rows,
    int channels,
    float eps);

/* Per-head RMS norm for [batch,tokens,heads,head_dim] attention tensors. */
strellis_status strellis_multihead_rms_norm_f32(
    const float * x,
    const float * gamma,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim,
    float eps);

/* BF16 roundtrip emulation for checkpoint-parity tests. */
strellis_status strellis_bf16_roundtrip_f32(const float * x, float * y, size_t n);

/* GELU with PyTorch approximate="tanh" semantics. */
strellis_status strellis_gelu_tanh_f32(const float * x, float * y, size_t n);

/* SiLU activation: y = x * sigmoid(x). */
strellis_status strellis_silu_f32(const float * x, float * y, size_t n);

/* Element-wise add for flat buffers. */
strellis_status strellis_add_f32(const float * a, const float * b, float * y, size_t n);

/* Transformer feed-forward MLP: linear, GELU, then linear. */
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
    int out_channels);

/* Sinusoidal timestep embedding used before the timestep MLP. */
strellis_status strellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding);

/* Timestep embedding MLP: sinusoidal embedding, linear, SiLU, then linear. */
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
    int out_channels);

/* Scaled dot-product attention for q/k/v in [batch,tokens,heads,head_dim]. */
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
    float scale);

/* Applies adjacent-pair RoPE to [batch,tokens,heads,head_dim]. */
strellis_status strellis_apply_rope_adjacent_f32(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int batch,
    int tokens,
    int heads,
    int head_dim);

/* Dense 3D rotary phase table for resolution^3 voxel tokens. */
strellis_status strellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* Sparse 3D rotary phase table for active [batch,x,y,z] coordinates. */
strellis_status strellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* 2D rotary phase table for DINO image patch tokens plus special tokens. */
strellis_status strellis_dino_rope_2d_phases_f32(
    int n_special_tokens,
    int patches_h,
    int patches_w,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* Euler update used by flow samplers to step from t to t_prev. */
strellis_status strellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0);

/* Classifier-free guidance blend of positive and negative predictions. */
strellis_status strellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred);

/* CFG blend with x0 standard-deviation rescaling. */
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
    float * pred);

/* Builds flow sampler timestep pairs after TRELLIS rescaling. */
strellis_status strellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count);

/* Dense Conv3D in contiguous NCDHW layout, weight [out,in,kd,kh,kw]. */
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
    int dilation_w);

/* 3D pixel shuffle for NCDHW tensors. */
strellis_status strellis_pixel_shuffle_3d_ncdhw_f32(
    const float * x,
    float * y,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale);

/* Channel-wise layer norm over NCDHW volumes. */
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
    float eps);

/* DINO patch embedding: image [batch,3,H,W], weight [out,3,p,p], tokens [batch,n_patches,out]. */
strellis_status strellis_dino_patch_embed_f32(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size);

/* Row-wise sparse linear projection for active sparse features. */
strellis_status strellis_sparse_linear_f32(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int64_t n,
    int in_channels,
    int out_channels);

/* Sparse downsample by averaging features that fall into the same coarser cell. */
strellis_status strellis_sparse_downsample_mean_f32(
    const strellis_sparse_tensor * input,
    int factor,
    strellis_sparse_tensor * output);

/* Packs spatial neighborhoods into channel features for sparse tensors. */
strellis_status strellis_sparse_spatial2channel_f32(
    const strellis_sparse_tensor * input,
    int factor,
    strellis_sparse_tensor * output);

/* Expands channel-packed sparse features back into spatial coordinates. */
strellis_status strellis_sparse_channel2spatial_f32(
    const strellis_sparse_tensor * input,
    const uint8_t * subdivision,
    int factor,
    strellis_sparse_tensor * output);

/* Brute-force submanifold sparse Conv3D over active [batch,x,y,z] coordinates. */
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
    int dilation_w);

/* Extracts a mesh from FlexiDualGrid decoder logits. Planned for this standalone path. */
strellis_status strellis_flexible_dual_grid_mesh_from_decoder_logits_f32(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    strellis_mesh * mesh_out);

/* Network dry-run helper proving network code can call standalone ops. */
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
    int context_tokens);

#ifdef __cplusplus
}
#endif

#endif /* STANDALONE_TRELLIS_OPS_H */
