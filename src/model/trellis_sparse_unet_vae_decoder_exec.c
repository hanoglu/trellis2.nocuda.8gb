#include "trellis.h"
#include "sparse/trellis_sparse_backend.h"

#include <stdlib.h>
#include <string.h>

typedef struct trellis_host_c2s_map {
    int32_t * coords;
    int32_t * parent;
    int32_t * subidx;
    int64_t n;
} trellis_host_c2s_map;

static void host_c2s_map_free(trellis_host_c2s_map * map) {
    if (map == NULL) {
        return;
    }
    free(map->coords);
    free(map->parent);
    free(map->subidx);
    memset(map, 0, sizeof(*map));
}

static trellis_status host_c2s_map_copy_to_guide(
    const trellis_host_c2s_map * map,
    trellis_sparse_c2s_guide_level * guide) {
    if (map == NULL || guide == NULL || map->n <= 0 ||
        map->coords == NULL || map->parent == NULL || map->subidx == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(guide, 0, sizeof(*guide));
    guide->coords_bxyz = (int32_t *) malloc((size_t) map->n * 4u * sizeof(int32_t));
    guide->parent = (int32_t *) malloc((size_t) map->n * sizeof(int32_t));
    guide->subidx = (int32_t *) malloc((size_t) map->n * sizeof(int32_t));
    if (guide->coords_bxyz == NULL || guide->parent == NULL || guide->subidx == NULL) {
        free(guide->coords_bxyz);
        free(guide->parent);
        free(guide->subidx);
        memset(guide, 0, sizeof(*guide));
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    guide->n_coords = map->n;
    memcpy(guide->coords_bxyz, map->coords, (size_t) map->n * 4u * sizeof(int32_t));
    memcpy(guide->parent, map->parent, (size_t) map->n * sizeof(int32_t));
    memcpy(guide->subidx, map->subidx, (size_t) map->n * sizeof(int32_t));
    return TRELLIS_STATUS_OK;
}

static trellis_status host_c2s_map_from_guide(
    const trellis_sparse_c2s_guide_level * guide,
    trellis_host_c2s_map * map) {
    if (guide == NULL || map == NULL || guide->n_coords <= 0 ||
        guide->coords_bxyz == NULL || guide->parent == NULL || guide->subidx == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(map, 0, sizeof(*map));
    map->coords = (int32_t *) malloc((size_t) guide->n_coords * 4u * sizeof(int32_t));
    map->parent = (int32_t *) malloc((size_t) guide->n_coords * sizeof(int32_t));
    map->subidx = (int32_t *) malloc((size_t) guide->n_coords * sizeof(int32_t));
    if (map->coords == NULL || map->parent == NULL || map->subidx == NULL) {
        host_c2s_map_free(map);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    map->n = guide->n_coords;
    memcpy(map->coords, guide->coords_bxyz, (size_t) map->n * 4u * sizeof(int32_t));
    memcpy(map->parent, guide->parent, (size_t) map->n * sizeof(int32_t));
    memcpy(map->subidx, guide->subidx, (size_t) map->n * sizeof(int32_t));
    return TRELLIS_STATUS_OK;
}

static trellis_status host_c2s_map_from_logits(
    const int32_t * coords,
    const float * logits,
    int64_t n,
    trellis_host_c2s_map * map) {
    if (coords == NULL || logits == NULL || map == NULL || n <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(map, 0, sizeof(*map));
    int64_t m = 0;
    for (int64_t row = 0; row < n; ++row) {
        for (int s = 0; s < 8; ++s) {
            if (logits[row * 8 + s] > 0.0f) {
                ++m;
            }
        }
    }
    if (m <= 0) {
        return TRELLIS_STATUS_ERROR;
    }
    map->coords = (int32_t *) malloc((size_t) m * 4u * sizeof(int32_t));
    map->parent = (int32_t *) malloc((size_t) m * sizeof(int32_t));
    map->subidx = (int32_t *) malloc((size_t) m * sizeof(int32_t));
    if (map->coords == NULL || map->parent == NULL || map->subidx == NULL) {
        host_c2s_map_free(map);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    int64_t dst = 0;
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * c = coords + row * 4;
        for (int s = 0; s < 8; ++s) {
            if (logits[row * 8 + s] <= 0.0f) {
                continue;
            }
            map->coords[4 * dst + 0] = c[0];
            map->coords[4 * dst + 1] = c[1] * 2 + (s & 1);
            map->coords[4 * dst + 2] = c[2] * 2 + ((s >> 1) & 1);
            map->coords[4 * dst + 3] = c[3] * 2 + ((s >> 2) & 1);
            map->parent[dst] = (int32_t) row;
            map->subidx[dst] = (int32_t) s;
            ++dst;
        }
    }
    map->n = m;
    return TRELLIS_STATUS_OK;
}

static trellis_status sparse_backend_create(
    trellis_sparse_backend_kind kind,
    int device,
    trellis_sparse_backend ** backend_out) {
    if (backend_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *backend_out = NULL;
    switch (kind) {
        case TRELLIS_SPARSE_BACKEND_CPU:
            return trellis_sparse_cpu_backend_create(backend_out);
        case TRELLIS_SPARSE_BACKEND_VULKAN:
            return trellis_sparse_vulkan_backend_create(device, backend_out);
        case TRELLIS_SPARSE_BACKEND_CUDA:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        default:
            return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
}

static trellis_status sparse_convnext_block_exec(
    trellis_sparse_backend * backend,
    const trellis_sparse_unet_vae_decoder_convnext_block_weights * block,
    const trellis_sparse_rulebook * rulebook,
    trellis_sparse_buffer * h,
    int64_t n,
    trellis_sparse_buffer ** out_h) {
    if (backend == NULL || backend->ops == NULL || block == NULL || rulebook == NULL ||
        h == NULL || out_h == NULL || n <= 0 || block->channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out_h = NULL;
    const trellis_sparse_backend_ops * ops = backend->ops;
    const int c = block->channels;
    trellis_sparse_buffer * conv = NULL;
    trellis_sparse_buffer * norm = NULL;
    trellis_sparse_buffer * mlp1 = NULL;
    trellis_sparse_buffer * mlp2 = NULL;
    trellis_sparse_buffer * out = NULL;
    trellis_status status = ops->alloc_f32(backend, (size_t) n * (size_t) c, &conv);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) c, &norm);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) c * 4u, &mlp1);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) c, &mlp2);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) c, &out);
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(backend, rulebook, h, block->conv_w, block->conv_b, conv, n, c, c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(backend, conv, block->norm_gamma, block->norm_beta, norm, n, c, 1e-6f);
    }
    if (status == TRELLIS_STATUS_OK && ops->linear_silu != NULL) {
        status = ops->linear_silu(backend, norm, block->mlp0_w, block->mlp0_b, mlp1, n, c, 4 * c);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(backend, norm, block->mlp0_w, block->mlp0_b, mlp1, n, c, 4 * c);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(backend, mlp1, (size_t) n * (size_t) c * 4u);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(backend, mlp1, block->mlp2_w, block->mlp2_b, mlp2, n, 4 * c, c);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->add(backend, h, mlp2, out, (size_t) n * (size_t) c);
    }
    ops->free_buffer(backend, conv);
    ops->free_buffer(backend, norm);
    ops->free_buffer(backend, mlp1);
    ops->free_buffer(backend, mlp2);
    if (status == TRELLIS_STATUS_OK) {
        *out_h = out;
    } else {
        ops->free_buffer(backend, out);
    }
    return status;
}

static trellis_status sparse_c2s_block_exec(
    trellis_sparse_backend * backend,
    const trellis_sparse_unet_vae_decoder_c2s_block_weights * block,
    const int32_t * coords,
    const trellis_sparse_rulebook * cur_rulebook,
    trellis_sparse_buffer * h,
    int64_t n,
    const trellis_sparse_c2s_guide_level * guide_in,
    trellis_sparse_c2s_guide_level * guide_out,
    int32_t ** next_coords_out,
    trellis_sparse_buffer ** next_h_out,
    trellis_sparse_rulebook ** next_rulebook_out,
    int64_t * next_n_out) {
    if (backend == NULL || backend->ops == NULL || block == NULL || coords == NULL ||
        cur_rulebook == NULL || h == NULL || next_coords_out == NULL || next_h_out == NULL ||
        next_rulebook_out == NULL || next_n_out == NULL || n <= 0 ||
        block->in_channels <= 0 || block->out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *next_coords_out = NULL;
    *next_h_out = NULL;
    *next_rulebook_out = NULL;
    *next_n_out = 0;
    const trellis_sparse_backend_ops * ops = backend->ops;
    const int ci = block->in_channels;
    const int co = block->out_channels;
    trellis_host_c2s_map map;
    memset(&map, 0, sizeof(map));
    trellis_sparse_rulebook * next_rulebook = NULL;
    trellis_sparse_buffer * subdiv = NULL;
    trellis_sparse_buffer * norm1 = NULL;
    trellis_sparse_buffer * conv1 = NULL;
    trellis_sparse_buffer * h_up = NULL;
    trellis_sparse_buffer * skip = NULL;
    trellis_sparse_buffer * norm2 = NULL;
    trellis_sparse_buffer * conv2 = NULL;
    trellis_sparse_buffer * out = NULL;
    float * subdiv_host = NULL;

    trellis_status status = TRELLIS_STATUS_OK;
    if (guide_out != NULL) {
        memset(guide_out, 0, sizeof(*guide_out));
    }
    if (guide_in != NULL) {
        status = host_c2s_map_from_guide(guide_in, &map);
    } else {
        if (block->to_subdiv_w == NULL || block->to_subdiv_b == NULL) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->alloc_f32(backend, (size_t) n * 8u, &subdiv);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->linear(backend, h, block->to_subdiv_w, block->to_subdiv_b, subdiv, n, ci, 8);
        }
        if (status == TRELLIS_STATUS_OK && ops->build_c2s_map != NULL) {
            status = ops->build_c2s_map(
                backend,
                coords,
                subdiv,
                n,
                &map.coords,
                &map.parent,
                &map.subidx,
                &map.n);
        } else if (status == TRELLIS_STATUS_OK) {
            subdiv_host = (float *) malloc((size_t) n * 8u * sizeof(float));
            if (subdiv_host == NULL) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
            }
            if (status == TRELLIS_STATUS_OK) {
                status = ops->download_f32(backend, subdiv, subdiv_host, (size_t) n * 8u);
            }
            if (status == TRELLIS_STATUS_OK) {
                status = host_c2s_map_from_logits(coords, subdiv_host, n, &map);
            }
        }
    }
    if (status == TRELLIS_STATUS_OK && guide_out != NULL) {
        status = host_c2s_map_copy_to_guide(&map, guide_out);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->build_rulebook(backend, map.coords, map.n, &next_rulebook);
    }
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) ci, &norm1);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) n * (size_t) co * 8u, &conv1);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) map.n * (size_t) co, &h_up);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) map.n * (size_t) co, &skip);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) map.n * (size_t) co, &norm2);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) map.n * (size_t) co, &conv2);
    if (status == TRELLIS_STATUS_OK) status = ops->alloc_f32(backend, (size_t) map.n * (size_t) co, &out);
    if (status == TRELLIS_STATUS_OK && ops->row_norm_silu != NULL) {
        status = ops->row_norm_silu(backend, h, block->norm1_gamma, block->norm1_beta, norm1, n, ci, 1e-6f);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(backend, h, block->norm1_gamma, block->norm1_beta, norm1, n, ci, 1e-6f);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(backend, norm1, (size_t) n * (size_t) ci);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(backend, cur_rulebook, norm1, block->conv1_w, block->conv1_b, conv1, n, ci, 8 * co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->c2s_gather(backend, conv1, map.parent, map.subidx, h_up, map.n, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->skip_repeat(backend, h, map.parent, map.subidx, skip, map.n, ci, co);
    }
    if (status == TRELLIS_STATUS_OK && ops->row_norm_silu != NULL) {
        status = ops->row_norm_silu(backend, h_up, NULL, NULL, norm2, map.n, co, 1e-6f);
    } else if (status == TRELLIS_STATUS_OK) {
        status = ops->row_norm(backend, h_up, NULL, NULL, norm2, map.n, co, 1e-6f);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->silu_inplace(backend, norm2, (size_t) map.n * (size_t) co);
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->sparse_conv3d(backend, next_rulebook, norm2, block->conv2_w, block->conv2_b, conv2, map.n, co, co);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->add(backend, conv2, skip, out, (size_t) map.n * (size_t) co);
    }

    ops->free_buffer(backend, subdiv);
    ops->free_buffer(backend, norm1);
    ops->free_buffer(backend, conv1);
    ops->free_buffer(backend, h_up);
    ops->free_buffer(backend, skip);
    ops->free_buffer(backend, norm2);
    ops->free_buffer(backend, conv2);
    free(subdiv_host);
    if (status == TRELLIS_STATUS_OK) {
        *next_coords_out = map.coords;
        map.coords = NULL;
        *next_h_out = out;
        out = NULL;
        *next_rulebook_out = next_rulebook;
        next_rulebook = NULL;
        *next_n_out = map.n;
    }
    ops->free_buffer(backend, out);
    if (next_rulebook != NULL) {
        ops->free_rulebook(backend, next_rulebook);
    }
    host_c2s_map_free(&map);
    return status;
}

trellis_status trellis_sparse_unet_vae_decoder_forward_backend_f32_host(
    const trellis_sparse_unet_vae_decoder_weights * weights,
    const int32_t * coords,
    const float * feats,
    int64_t n,
    const trellis_sparse_unet_vae_decoder_forward_options * options,
    int32_t ** coords_out,
    float ** feats_out,
    int64_t * n_out,
    int * channels_out) {
    if (weights == NULL || coords == NULL || feats == NULL || options == NULL ||
        coords_out == NULL || feats_out == NULL || n_out == NULL || channels_out == NULL ||
        n <= 0 || weights->latent_channels <= 0 || weights->out_channels <= 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *feats_out = NULL;
    *n_out = 0;
    *channels_out = 0;
    if (options->return_subs != NULL) {
        memset(options->return_subs, 0, sizeof(*options->return_subs));
    }
    if (options->backend_kind == TRELLIS_SPARSE_BACKEND_CUDA) {
        return trellis_sparse_unet_vae_decoder_forward_f32_host(
            weights,
            coords,
            feats,
            n,
            options->device,
            options->max_levels,
            options->guide_subs,
            options->return_subs,
            coords_out,
            feats_out,
            n_out,
            channels_out);
    }

    const int levels = weights->levels <= 0 ? TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS : weights->levels;
    const int levels_to_run = options->max_levels <= 0 || options->max_levels > levels ? levels : options->max_levels;
    if (levels_to_run <= 0 || levels_to_run > TRELLIS_SPARSE_UNET_VAE_DECODER_LEVELS) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (!weights->pred_subdiv && options->guide_subs == NULL && levels_to_run > 1) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_sparse_backend * backend = NULL;
    trellis_sparse_buffer * in = NULL;
    trellis_sparse_buffer * cur_h = NULL;
    trellis_sparse_rulebook * cur_rulebook = NULL;
    int32_t * cur_coords = NULL;
    float * host_feats = NULL;
    int32_t * host_coords = NULL;

    trellis_status status = sparse_backend_create(options->backend_kind, options->device, &backend);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    const trellis_sparse_backend_ops * ops = backend->ops;
    cur_coords = (int32_t *) malloc((size_t) n * 4u * sizeof(int32_t));
    if (cur_coords == NULL) {
        status = TRELLIS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(cur_coords, coords, (size_t) n * 4u * sizeof(int32_t));
    status = ops->upload_f32(backend, feats, (size_t) n * (size_t) weights->latent_channels, &in);
    if (status == TRELLIS_STATUS_OK) {
        status = ops->alloc_f32(backend, (size_t) n * (size_t) weights->channels[0], &cur_h);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = ops->linear(backend, in, weights->from_latent_w, weights->from_latent_b, cur_h, n, weights->latent_channels, weights->channels[0]);
    }
    ops->free_buffer(backend, in);
    in = NULL;
    int64_t cur_n = n;
    int cur_channels = weights->channels[0];
    if (status == TRELLIS_STATUS_OK) {
        status = ops->build_rulebook(backend, cur_coords, cur_n, &cur_rulebook);
    }

    for (int level = 0; status == TRELLIS_STATUS_OK && level < levels_to_run; ++level) {
        cur_channels = weights->channels[level];
        for (int block = 0; status == TRELLIS_STATUS_OK && block < weights->blocks_per_level[level]; ++block) {
            trellis_sparse_buffer * next_h = NULL;
            status = sparse_convnext_block_exec(
                backend,
                &weights->blocks[level][block],
                cur_rulebook,
                cur_h,
                cur_n,
                &next_h);
            if (status == TRELLIS_STATUS_OK) {
                ops->free_buffer(backend, cur_h);
                cur_h = next_h;
            }
        }
        if (status != TRELLIS_STATUS_OK || level >= levels_to_run - 1 || level >= levels - 1) {
            break;
        }

        const trellis_sparse_c2s_guide_level * guide_in = NULL;
        trellis_sparse_c2s_guide_level * guide_out = NULL;
        if (!weights->pred_subdiv) {
            if (options->guide_subs == NULL || options->guide_subs->n_levels <= level) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            }
            guide_in = &options->guide_subs->levels[level];
        } else if (options->return_subs != NULL) {
            guide_out = &options->return_subs->levels[level];
        }
        int32_t * next_coords = NULL;
        trellis_sparse_buffer * next_h = NULL;
        trellis_sparse_rulebook * next_rulebook = NULL;
        int64_t next_n = 0;
        status = sparse_c2s_block_exec(
            backend,
            &weights->up_blocks[level],
            cur_coords,
            cur_rulebook,
            cur_h,
            cur_n,
            guide_in,
            guide_out,
            &next_coords,
            &next_h,
            &next_rulebook,
            &next_n);
        if (status == TRELLIS_STATUS_OK) {
            free(cur_coords);
            ops->free_buffer(backend, cur_h);
            ops->free_rulebook(backend, cur_rulebook);
            cur_coords = next_coords;
            cur_h = next_h;
            cur_rulebook = next_rulebook;
            cur_n = next_n;
            cur_channels = weights->channels[level + 1];
            if (options->return_subs != NULL && weights->pred_subdiv && options->return_subs->n_levels < level + 1) {
                options->return_subs->n_levels = level + 1;
            }
        } else {
            free(next_coords);
            ops->free_buffer(backend, next_h);
            if (next_rulebook != NULL) {
                ops->free_rulebook(backend, next_rulebook);
            }
        }
    }

    if (status == TRELLIS_STATUS_OK && levels_to_run == levels) {
        TRELLIS_INFO(
            "SparseUnetVaeDecoder: output head tokens=%lld channels=%d out=%d",
            (long long) cur_n,
            cur_channels,
            weights->out_channels);
        trellis_sparse_buffer * norm = NULL;
        trellis_sparse_buffer * out = NULL;
        status = ops->alloc_f32(backend, (size_t) cur_n * (size_t) cur_channels, &norm);
        if (status == TRELLIS_STATUS_OK) {
            status = ops->alloc_f32(backend, (size_t) cur_n * (size_t) weights->out_channels, &out);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->row_norm(backend, cur_h, NULL, NULL, norm, cur_n, cur_channels, 1e-5f);
        }
        if (status == TRELLIS_STATUS_OK) {
            status = ops->linear(backend, norm, weights->output_w, weights->output_b, out, cur_n, cur_channels, weights->out_channels);
        }
        ops->free_buffer(backend, norm);
        if (status == TRELLIS_STATUS_OK) {
            ops->free_buffer(backend, cur_h);
            cur_h = out;
            out = NULL;
            cur_channels = weights->out_channels;
        }
        ops->free_buffer(backend, out);
    }

    if (status == TRELLIS_STATUS_OK) {
        host_coords = (int32_t *) malloc((size_t) cur_n * 4u * sizeof(int32_t));
        host_feats = (float *) malloc((size_t) cur_n * (size_t) cur_channels * sizeof(float));
        if (host_coords == NULL || host_feats == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(host_coords, cur_coords, (size_t) cur_n * 4u * sizeof(int32_t));
        status = ops->download_f32(backend, cur_h, host_feats, (size_t) cur_n * (size_t) cur_channels);
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_out = host_coords;
        *feats_out = host_feats;
        *n_out = cur_n;
        *channels_out = cur_channels;
        host_coords = NULL;
        host_feats = NULL;
    }

cleanup:
    free(host_coords);
    free(host_feats);
    if (cur_rulebook != NULL && backend != NULL && backend->ops != NULL) {
        backend->ops->free_rulebook(backend, cur_rulebook);
    }
    if (cur_h != NULL && backend != NULL && backend->ops != NULL) {
        backend->ops->free_buffer(backend, cur_h);
    }
    free(cur_coords);
    trellis_sparse_backend_destroy(backend);
    if (status != TRELLIS_STATUS_OK && options->return_subs != NULL) {
        trellis_sparse_c2s_guides_free(options->return_subs);
    }
    return status;
}
