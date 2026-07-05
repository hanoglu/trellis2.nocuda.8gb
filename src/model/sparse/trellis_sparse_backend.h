#ifndef TRELLIS2_C_SRC_MODEL_SPARSE_TRELLIS_SPARSE_BACKEND_H
#define TRELLIS2_C_SRC_MODEL_SPARSE_TRELLIS_SPARSE_BACKEND_H

#include "trellis.h"

#include <stddef.h>
#include <stdint.h>

typedef struct trellis_sparse_buffer trellis_sparse_buffer;
typedef struct trellis_sparse_rulebook trellis_sparse_rulebook;
typedef struct trellis_sparse_backend trellis_sparse_backend;

typedef struct trellis_sparse_backend_ops {
    void (*destroy)(trellis_sparse_backend * backend);

    trellis_status (*alloc_f32)(trellis_sparse_backend * backend, size_t count, trellis_sparse_buffer ** out);
    trellis_status (*upload_f32)(trellis_sparse_backend * backend, const float * src, size_t count, trellis_sparse_buffer ** out);
    trellis_status (*upload_i32)(trellis_sparse_backend * backend, const int32_t * src, size_t count, trellis_sparse_buffer ** out);
    trellis_status (*download_f32)(trellis_sparse_backend * backend, const trellis_sparse_buffer * src, float * dst, size_t count);
    void (*free_buffer)(trellis_sparse_backend * backend, trellis_sparse_buffer * buffer);

    trellis_status (*linear)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const float * weight,
        const float * bias,
        trellis_sparse_buffer * y,
        int64_t n,
        int in_channels,
        int out_channels);

    trellis_status (*linear_silu)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const float * weight,
        const float * bias,
        trellis_sparse_buffer * y,
        int64_t n,
        int in_channels,
        int out_channels);

    trellis_status (*row_norm)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const float * gamma,
        const float * beta,
        trellis_sparse_buffer * y,
        int64_t n,
        int channels,
        float eps);

    trellis_status (*row_norm_silu)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const float * gamma,
        const float * beta,
        trellis_sparse_buffer * y,
        int64_t n,
        int channels,
        float eps);

    trellis_status (*silu_inplace)(trellis_sparse_backend * backend, trellis_sparse_buffer * x, size_t count);
    trellis_status (*add)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * a,
        const trellis_sparse_buffer * b,
        trellis_sparse_buffer * y,
        size_t count);

    trellis_status (*build_rulebook)(
        trellis_sparse_backend * backend,
        const int32_t * coords_bxyz,
        int64_t n,
        trellis_sparse_rulebook ** out);

    trellis_status (*sparse_conv3d)(
        trellis_sparse_backend * backend,
        const trellis_sparse_rulebook * rulebook,
        const trellis_sparse_buffer * feats,
        const float * weight,
        const float * bias,
        trellis_sparse_buffer * out,
        int64_t n,
        int in_channels,
        int out_channels);

    void (*free_rulebook)(trellis_sparse_backend * backend, trellis_sparse_rulebook * rulebook);

    trellis_status (*c2s_gather)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const int32_t * parent,
        const int32_t * subidx,
        trellis_sparse_buffer * y,
        int64_t n_out,
        int out_channels);

    trellis_status (*skip_repeat)(
        trellis_sparse_backend * backend,
        const trellis_sparse_buffer * x,
        const int32_t * parent,
        const int32_t * subidx,
        trellis_sparse_buffer * y,
        int64_t n_out,
        int in_channels,
        int out_channels);

    trellis_status (*build_c2s_map)(
        trellis_sparse_backend * backend,
        const int32_t * coords_bxyz,
        const trellis_sparse_buffer * logits,
        int64_t n,
        int32_t ** coords_out,
        int32_t ** parent_out,
        int32_t ** subidx_out,
        int64_t * n_out);
} trellis_sparse_backend_ops;

struct trellis_sparse_backend {
    const trellis_sparse_backend_ops * ops;
    trellis_sparse_backend_kind kind;
    int device;
};

trellis_status trellis_sparse_cpu_backend_create(trellis_sparse_backend ** out);
trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out);

void trellis_sparse_backend_destroy(trellis_sparse_backend * backend);

#endif
