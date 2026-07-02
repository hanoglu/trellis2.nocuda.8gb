#ifndef TRELLIS2_C_OPS_H
#define TRELLIS2_C_OPS_H

#include "trellis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dense affine projection: applies weight * x plus an optional bias in ggml layout. */
struct ggml_tensor * trellis_ggml_linear(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias);

/* Channel-wise layer normalization with optional affine scale and bias. */
struct ggml_tensor * trellis_ggml_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps);

/* RMS normalization with an optional learned scale. */
struct ggml_tensor * trellis_ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

/* BF16 emulation helper used to compare C/ggml math with BF16 checkpoint behavior. */
struct ggml_tensor * trellis_ggml_bf16_roundtrip(
    struct ggml_context * ctx,
    struct ggml_tensor * x);

/* Selects flash-attention or explicit attention for trellis_ggml_sdpa. */
void trellis_ggml_set_flash_attn_enabled(int enabled);

/* Per-head RMS normalization for attention query/key tensors. */
struct ggml_tensor * trellis_ggml_multihead_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma,
    float eps);

/* Transformer feed-forward MLP: linear, GELU, then linear. */
struct ggml_tensor * trellis_ggml_feed_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

/* Timestep embedding MLP used by flow models to condition denoising steps. */
struct ggml_tensor * trellis_ggml_timestep_mlp(
    struct ggml_context * ctx,
    struct ggml_tensor * timesteps,
    int frequency_dim,
    struct ggml_tensor * w1,
    struct ggml_tensor * b1,
    struct ggml_tensor * w2,
    struct ggml_tensor * b2);

/* Scaled dot-product attention over query, key, and value tensors. */
struct ggml_tensor * trellis_ggml_sdpa(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    float scale);

/* Applies adjacent-pair rotary position embedding to attention heads. */
struct ggml_tensor * trellis_ggml_apply_rope_adjacent(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

/* Multi-head self-attention without rotary position embedding. */
struct ggml_tensor * trellis_ggml_self_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

/* Multi-head self-attention with rotary position embedding. */
struct ggml_tensor * trellis_ggml_self_attention_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int n_heads,
    struct ggml_tensor * qkv_w,
    struct ggml_tensor * qkv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

/* Cross-attention that lets latent tokens attend to conditioning tokens. */
struct ggml_tensor * trellis_ggml_cross_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * context,
    int n_heads,
    struct ggml_tensor * q_w,
    struct ggml_tensor * q_b,
    struct ggml_tensor * kv_w,
    struct ggml_tensor * kv_b,
    struct ggml_tensor * q_rms_gamma,
    struct ggml_tensor * k_rms_gamma,
    struct ggml_tensor * out_w,
    struct ggml_tensor * out_b);

/* Euler update used by flow samplers to step from t to t_prev. */
void trellis_flow_euler_step_f32(
    const float * x_t,
    const float * pred_v,
    size_t n,
    float sigma_min,
    float t,
    float t_prev,
    float * pred_x_prev,
    float * pred_x0);

/* Classifier-free guidance blend of positive and negative predictions. */
void trellis_flow_cfg_combine_f32(
    const float * pred_pos,
    const float * pred_neg,
    size_t n,
    float guidance_strength,
    float * pred);

/* Classifier-free guidance blend with x0 standard-deviation rescaling. */
void trellis_flow_cfg_rescale_combine_f32(
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
trellis_status trellis_flow_timestep_pairs_f32(
    int steps,
    float rescale_t,
    float * pairs,
    size_t pair_count);

/* Sinusoidal timestep embedding used before the timestep MLP. */
void trellis_timestep_embedding_f32(
    const float * timesteps,
    size_t n_timesteps,
    int dim,
    float max_period,
    float * embedding);

/* Dense 3D rotary phase table for voxel-grid latent tokens. */
trellis_status trellis_rope_3d_phases_f32(
    int resolution,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* Sparse 3D rotary phase table for active voxel coordinates. */
trellis_status trellis_rope_3d_sparse_phases_f32(
    const int32_t * coords,
    int64_t n_coords,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* 2D rotary phase table for DINO image patch tokens. */
trellis_status trellis_dino_rope_2d_phases_f32(
    int n_special_tokens,
    int patches_h,
    int patches_w,
    int head_dim,
    float freq_scale,
    float freq_base,
    float * cos_out,
    float * sin_out,
    size_t phase_count);

/* Dense 3D convolution in contiguous NCDHW layout. */
trellis_status trellis_cuda_conv3d_f32(
    const float * x_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * y_dev,
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

/* Host wrapper for the dense 3D convolution CUDA kernel. */
trellis_status trellis_cuda_conv3d_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
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

/* 3D pixel shuffle that expands channels into spatial resolution. */
trellis_status trellis_cuda_pixel_shuffle_3d_f32(
    const float * x_dev,
    float * y_dev,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale);

/* Host wrapper for the 3D pixel shuffle CUDA kernel. */
trellis_status trellis_cuda_pixel_shuffle_3d_f32_host(
    const float * x,
    float * y,
    int device,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int scale);

/* Channel-wise layer norm over 3D feature volumes. */
trellis_status trellis_cuda_channel_layer_norm_3d_f32(
    const float * x_dev,
    const float * gamma_dev,
    const float * beta_dev,
    float * y_dev,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps);

/* Host wrapper for channel-wise 3D layer normalization. */
trellis_status trellis_cuda_channel_layer_norm_3d_f32_host(
    const float * x,
    const float * gamma,
    const float * beta,
    float * y,
    int device,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    float eps);

/* SiLU activation for flat float buffers. */
trellis_status trellis_cuda_silu_f32(
    const float * x_dev,
    float * y_dev,
    size_t n);

/* Host wrapper for the SiLU activation CUDA kernel. */
trellis_status trellis_cuda_silu_f32_host(
    const float * x,
    float * y,
    int device,
    size_t n);

/* Element-wise addition for flat float buffers. */
trellis_status trellis_cuda_add_f32(
    const float * a_dev,
    const float * b_dev,
    float * y_dev,
    size_t n);

/* Host wrapper for the element-wise addition CUDA kernel. */
trellis_status trellis_cuda_add_f32_host(
    const float * a,
    const float * b,
    float * y,
    int device,
    size_t n);

/* Row-wise sparse linear projection for active sparse features. */
trellis_status trellis_cuda_sparse_linear_f32_host(
    const float * x,
    const float * weight,
    const float * bias,
    float * y,
    int device,
    int64_t n,
    int in_channels,
    int out_channels);

/* DINO patch embedding convolution over normalized image pixels. */
trellis_status trellis_cuda_dino_patch_embed_f32(
    const float * image_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * tokens_dev,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size);

/* Host wrapper for the DINO patch embedding CUDA kernel. */
trellis_status trellis_cuda_dino_patch_embed_f32_host(
    const float * image,
    const float * weight,
    const float * bias,
    float * tokens,
    int device,
    int batch,
    int image_h,
    int image_w,
    int out_channels,
    int patch_size);

/* Submanifold sparse 3D convolution over active BXYZ coordinates. */
trellis_status trellis_cuda_sparse_subm_conv3d_f32(
    const int32_t * coords_dev,
    const float * feats_dev,
    const float * weight_dev,
    const float * bias_dev,
    float * out_dev,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

/* Host wrapper for submanifold sparse 3D convolution. */
trellis_status trellis_cuda_sparse_subm_conv3d_f32_host(
    const int32_t * coords,
    const float * feats,
    const float * weight,
    const float * bias,
    float * out,
    int device,
    int64_t n,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int dilation_d,
    int dilation_h,
    int dilation_w);

/* CUDA rotary position embedding for attention tensors. */
trellis_status trellis_cuda_apply_rope_f32(
    const float * x_dev,
    const float * cos_dev,
    const float * sin_dev,
    float * y_dev,
    int batch,
    int tokens,
    int heads,
    int head_dim);

/* Host wrapper for CUDA rotary position embedding. */
trellis_status trellis_cuda_apply_rope_f32_host(
    const float * x,
    const float * cos_phase,
    const float * sin_phase,
    float * y,
    int device,
    int batch,
    int tokens,
    int heads,
    int head_dim);

typedef struct trellis_ggml_modulated_cross_block_params {
    struct ggml_tensor * block_modulation; /* [6 * channels] */
    struct ggml_tensor * norm2_gamma;      /* [channels] */
    struct ggml_tensor * norm2_beta;       /* [channels] */

    struct ggml_tensor * self_qkv_w;
    struct ggml_tensor * self_qkv_b;
    struct ggml_tensor * self_q_rms_gamma;
    struct ggml_tensor * self_k_rms_gamma;
    struct ggml_tensor * self_out_w;
    struct ggml_tensor * self_out_b;

    struct ggml_tensor * cross_q_w;
    struct ggml_tensor * cross_q_b;
    struct ggml_tensor * cross_kv_w;
    struct ggml_tensor * cross_kv_b;
    struct ggml_tensor * cross_q_rms_gamma;
    struct ggml_tensor * cross_k_rms_gamma;
    struct ggml_tensor * cross_out_w;
    struct ggml_tensor * cross_out_b;

    struct ggml_tensor * mlp_fc1_w;
    struct ggml_tensor * mlp_fc1_b;
    struct ggml_tensor * mlp_fc2_w;
    struct ggml_tensor * mlp_fc2_b;
    int debug_parts;
    int emulate_bf16;
} trellis_ggml_modulated_cross_block_params;

/* DiT block combining AdaLN modulation, self-attention, cross-attention, and MLP. */
struct ggml_tensor * trellis_ggml_modulated_cross_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params);

/* RoPE-enabled DiT block combining modulation, attention, and MLP. */
struct ggml_tensor * trellis_ggml_modulated_cross_block_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * mod6,
    struct ggml_tensor * context,
    int n_heads,
    const trellis_ggml_modulated_cross_block_params * params,
    struct ggml_tensor * cos_phase,
    struct ggml_tensor * sin_phase);

/* Sparse downsample by averaging features that fall into the same coarser cell. */
trellis_status trellis_sparse_downsample_mean_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

/* Packs spatial neighborhoods into channel features for sparse tensors. */
trellis_status trellis_sparse_spatial2channel_host(
    const trellis_sparse_tensor_host * input,
    int factor,
    trellis_sparse_tensor_host * output);

/* Expands channel-packed sparse features back into spatial coordinates. */
trellis_status trellis_sparse_channel2spatial_host(
    const trellis_sparse_tensor_host * input,
    const uint8_t * subdivision,
    int factor,
    trellis_sparse_tensor_host * output);

/* Extracts a mesh from FlexiDualGrid decoder logits. */
trellis_status trellis_flexible_dual_grid_mesh_from_decoder_logits_host(
    const int32_t * coords,
    const float * feats,
    int64_t n,
    int channels,
    int resolution,
    trellis_mesh_host * mesh_out);

#ifdef __cplusplus
}
#endif

#endif
