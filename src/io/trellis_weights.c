#include "trellis.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char * trellis_strdup(const char * s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char * out = (char *) malloc(n + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n + 1);
    return out;
}

static trellis_status store_append(trellis_tensor_store * store, const char * name, struct ggml_tensor * tensor) {
    if (store == NULL || name == NULL || tensor == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (store->n_entries == store->capacity) {
        size_t next_capacity = store->capacity == 0 ? 128 : store->capacity * 2;
        trellis_tensor_store_entry * next = (trellis_tensor_store_entry *) realloc(
            store->entries,
            next_capacity * sizeof(trellis_tensor_store_entry));
        if (next == NULL) {
            return TRELLIS_STATUS_OUT_OF_MEMORY;
        }
        store->entries = next;
        store->capacity = next_capacity;
    }
    char * owned_name = trellis_strdup(name);
    if (owned_name == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    store->entries[store->n_entries].name = owned_name;
    store->entries[store->n_entries].tensor = tensor;
    store->n_entries += 1;
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tensor_store_init(
    trellis_tensor_store * store,
    size_t graph_tensors,
    size_t tensor_data_bytes) {
    if (store == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(store, 0, sizeof(*store));
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * graph_tensors + tensor_data_bytes + 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    store->ctx = ggml_init(params);
    if (store->ctx == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return TRELLIS_STATUS_OK;
}

void trellis_tensor_store_free(trellis_tensor_store * store) {
    if (store == NULL) {
        return;
    }
    if (store->buffer != NULL) {
        ggml_backend_buffer_free(store->buffer);
    }
    for (size_t i = 0; i < store->n_entries; ++i) {
        free(store->entries[i].name);
    }
    free(store->entries);
    if (store->ctx != NULL) {
        ggml_free(store->ctx);
    }
    memset(store, 0, sizeof(*store));
}

const struct ggml_tensor * trellis_tensor_store_get_const(
    const trellis_tensor_store * store,
    const char * name) {
    if (store == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < store->n_entries; ++i) {
        if (strcmp(store->entries[i].name, name) == 0) {
            return store->entries[i].tensor;
        }
    }
    return NULL;
}

struct ggml_tensor * trellis_tensor_store_get(
    trellis_tensor_store * store,
    const char * name) {
    return (struct ggml_tensor *) trellis_tensor_store_get_const(store, name);
}

static int is_linear_weight_name(const char * name) {
    if (name == NULL) {
        return 0;
    }
    size_t n = strlen(name);
    return n >= 7 && strcmp(name + n - 7, ".weight") == 0;
}

static enum ggml_type ggml_type_for_meta(const trellis_safetensor_meta * meta, bool preserve_16bit_weights) {
    if (!preserve_16bit_weights || meta == NULL) {
        return GGML_TYPE_F32;
    }
    switch (meta->dtype) {
        case TRELLIS_DTYPE_F16:
            return GGML_TYPE_F16;
        case TRELLIS_DTYPE_BF16:
            return GGML_TYPE_BF16;
        default:
            return GGML_TYPE_F32;
    }
}

static trellis_status make_tensor_for_meta(
    trellis_tensor_store * store,
    const trellis_safetensor_meta * meta,
    bool transpose_linear_weights,
    bool preserve_16bit_weights,
    struct ggml_tensor ** out) {
    if (store == NULL || meta == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;

    int64_t ne[TRELLIS_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    if (transpose_linear_weights && meta->n_dims == 2 && is_linear_weight_name(meta->name)) {
        ne[0] = meta->shape[1];
        ne[1] = meta->shape[0];
    } else {
        for (int i = 0; i < meta->n_dims; ++i) {
            ne[i] = meta->shape[meta->n_dims - 1 - i];
        }
    }

    struct ggml_tensor * tensor = NULL;
    const enum ggml_type tensor_type = ggml_type_for_meta(meta, preserve_16bit_weights);
    switch (meta->n_dims) {
        case 1:
            tensor = ggml_new_tensor_1d(store->ctx, tensor_type, ne[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(store->ctx, tensor_type, ne[0], ne[1]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(store->ctx, tensor_type, ne[0], ne[1], ne[2]);
            break;
        case 4:
            tensor = ggml_new_tensor_4d(store->ctx, tensor_type, ne[0], ne[1], ne[2], ne[3]);
            break;
        case 5:
            tensor = ggml_new_tensor_4d(store->ctx, tensor_type, ne[0], ne[1], ne[2], ne[3] * ne[4]);
            break;
        default:
            return TRELLIS_STATUS_NOT_IMPLEMENTED;
    }
    if (tensor == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    ggml_set_name(tensor, meta->name);
    *out = tensor;
    return TRELLIS_STATUS_OK;
}

static trellis_status set_tensor_raw_data_from_meta(
    const trellis_safetensors * st,
    const trellis_safetensor_meta * meta,
    struct ggml_tensor * tensor) {
    if (st == NULL || meta == NULL || tensor == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint64_t n = trellis_safetensor_nelements(meta);
    size_t elem_size = trellis_dtype_size(meta->dtype);
    if (elem_size == 0 || n > UINT64_MAX / (uint64_t) elem_size ||
        meta->data_end < meta->data_begin ||
        meta->data_end - meta->data_begin != n * (uint64_t) elem_size) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const uint64_t bytes64 = n * (uint64_t) elem_size;
    if (bytes64 > (uint64_t) SIZE_MAX || bytes64 != (uint64_t) ggml_nbytes(tensor)) {
        return TRELLIS_STATUS_PARSE_ERROR;
    }
    const size_t bytes = (size_t) bytes64;
    void * tmp = malloc(bytes == 0 ? 1 : bytes);
    if (tmp == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    FILE * f = fopen(st->path, "rb");
    if (f == NULL) {
        free(tmp);
        return TRELLIS_STATUS_IO_ERROR;
    }
    if (fseek(f, (long) (st->data_base_offset + meta->data_begin), SEEK_SET) != 0) {
        fclose(f);
        free(tmp);
        return TRELLIS_STATUS_IO_ERROR;
    }
    trellis_status status = TRELLIS_STATUS_OK;
    if (bytes != 0 && fread(tmp, 1, bytes, f) != bytes) {
        status = TRELLIS_STATUS_IO_ERROR;
    }
    fclose(f);
    if (status == TRELLIS_STATUS_OK) {
        ggml_backend_tensor_set(tensor, tmp, 0, bytes);
    }
    free(tmp);
    return status;
}

static trellis_status set_tensor_data_from_meta(
    const trellis_safetensors * st,
    const trellis_safetensor_meta * meta,
    struct ggml_tensor * tensor,
    bool transpose_linear_weights) {
    if (st == NULL || meta == NULL || tensor == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if ((tensor->type == GGML_TYPE_F16 && meta->dtype == TRELLIS_DTYPE_F16) ||
        (tensor->type == GGML_TYPE_BF16 && meta->dtype == TRELLIS_DTYPE_BF16)) {
        /* Shape reversal handles safetensors row-major layout; the bytes stay contiguous as-is. */
        (void) transpose_linear_weights;
        return set_tensor_raw_data_from_meta(st, meta, tensor);
    }
    uint64_t n64 = trellis_safetensor_nelements(meta);
    if (n64 > (uint64_t) SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    size_t n = (size_t) n64;
    float * tmp = (float *) malloc(n * sizeof(float));
    if (tmp == NULL && n != 0) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = trellis_safetensors_read_f32(st, meta, tmp, n);
    if (status != TRELLIS_STATUS_OK) {
        free(tmp);
        return status;
    }

    (void) transpose_linear_weights;
    ggml_backend_tensor_set(tensor, tmp, 0, ggml_nbytes(tensor));
    free(tmp);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tensor_store_load_safetensors_f32(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors) {
    return trellis_tensor_store_load_safetensors_f32_ex(
        store,
        backend,
        safetensors_path,
        transpose_linear_weights,
        loaded_tensors,
        NULL,
        NULL);
}

trellis_status trellis_tensor_store_load_safetensors(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors) {
    return trellis_tensor_store_load_safetensors_ex(
        store,
        backend,
        safetensors_path,
        transpose_linear_weights,
        loaded_tensors,
        NULL,
        NULL);
}

static uint64_t safetensors_total_data_bytes(const trellis_safetensors * st) {
    if (st == NULL) {
        return 0;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < st->n_tensors; ++i) {
        if (st->tensors[i].data_end >= st->tensors[i].data_begin) {
            total += st->tensors[i].data_end - st->tensors[i].data_begin;
        }
    }
    return total;
}

static void emit_load_progress(
    const trellis_safetensors * st,
    const char * tensor_name,
    size_t tensor_index,
    uint64_t bytes_loaded,
    uint64_t total_bytes,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data) {
    if (progress_callback == NULL) {
        return;
    }
    trellis_tensor_store_load_progress progress;
    progress.path = st == NULL ? NULL : st->path;
    progress.tensor_name = tensor_name;
    progress.tensor_index = tensor_index;
    progress.tensor_count = st == NULL ? 0 : st->n_tensors;
    progress.bytes_loaded = bytes_loaded;
    progress.total_bytes = total_bytes;
    progress_callback(&progress, progress_user_data);
}

static trellis_status trellis_tensor_store_load_safetensors_impl(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    bool preserve_16bit_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data) {
    if (store == NULL || store->ctx == NULL || backend == NULL || backend->backend == NULL || safetensors_path == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    if (store->n_entries != 0 || store->buffer != NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }

    trellis_safetensors st;
    trellis_status status = trellis_safetensors_open(safetensors_path, &st);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    const uint64_t total_bytes = safetensors_total_data_bytes(&st);
    emit_load_progress(&st, NULL, 0, 0, total_bytes, progress_callback, progress_user_data);

    for (size_t i = 0; i < st.n_tensors; ++i) {
        struct ggml_tensor * tensor = NULL;
        status = make_tensor_for_meta(store, &st.tensors[i], transpose_linear_weights, preserve_16bit_weights, &tensor);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
        status = store_append(store, st.tensors[i].name, tensor);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
    }

    store->buffer = ggml_backend_alloc_ctx_tensors(store->ctx, backend->backend);
    if (store->buffer == NULL) {
        trellis_safetensors_close(&st);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }

    uint64_t bytes_loaded = 0;
    for (size_t i = 0; i < st.n_tensors; ++i) {
        struct ggml_tensor * tensor = store->entries[i].tensor;
        status = set_tensor_data_from_meta(&st, &st.tensors[i], tensor, transpose_linear_weights);
        if (status != TRELLIS_STATUS_OK) {
            trellis_safetensors_close(&st);
            return status;
        }
        if (st.tensors[i].data_end >= st.tensors[i].data_begin) {
            bytes_loaded += st.tensors[i].data_end - st.tensors[i].data_begin;
        }
        emit_load_progress(
            &st,
            st.tensors[i].name,
            i + 1,
            bytes_loaded,
            total_bytes,
            progress_callback,
            progress_user_data);
    }

    if (loaded_tensors != NULL) {
        *loaded_tensors = st.n_tensors;
    }
    trellis_safetensors_close(&st);
    return TRELLIS_STATUS_OK;
}

trellis_status trellis_tensor_store_load_safetensors_f32_ex(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data) {
    return trellis_tensor_store_load_safetensors_impl(
        store,
        backend,
        safetensors_path,
        transpose_linear_weights,
        false,
        loaded_tensors,
        progress_callback,
        progress_user_data);
}

trellis_status trellis_tensor_store_load_safetensors_ex(
    trellis_tensor_store * store,
    const trellis_backend_context * backend,
    const char * safetensors_path,
    bool transpose_linear_weights,
    size_t * loaded_tensors,
    trellis_tensor_store_load_progress_callback progress_callback,
    void * progress_user_data) {
    return trellis_tensor_store_load_safetensors_impl(
        store,
        backend,
        safetensors_path,
        transpose_linear_weights,
        true,
        loaded_tensors,
        progress_callback,
        progress_user_data);
}
