#include "trellis_ggml_layers.h"

#include "ggml.h"

#include <stdio.h>
#include <string.h>

static int g_failures;

static int tensor_graph_contains_op(const struct ggml_tensor * tensor, enum ggml_op op) {
    if (tensor == NULL) return 0;
    if (tensor->op == op) return 1;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor_graph_contains_op(tensor->src[i], op)) return 1;
    }
    return 0;
}

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        goto cleanup; \
    } \
} while (0)

static void test_explicit_policy_isolation(void) {
    struct ggml_init_params params = {
        .mem_size = 2u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * context = ggml_init(params);
    if (context == NULL) {
        fprintf(stderr, "failed to create ggml test context\n");
        ++g_failures;
        return;
    }

    struct ggml_tensor * q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 4, 3, 2, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(context, GGML_TYPE_F32, 4, 5, 2, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(context, GGML_TYPE_F32, 4, 5, 2, 1);
    trellis_ggml_attention_policy explicit_policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    trellis_ggml_attention_policy flash_policy = TRELLIS_GGML_ATTENTION_POLICY_INIT;
    flash_policy.mode = TRELLIS_GGML_ATTENTION_MODE_FLASH;

    CHECK_TRUE(trellis_ggml_attention_policy_is_valid(&explicit_policy));
    CHECK_TRUE(trellis_ggml_attention_policy_is_valid(&flash_policy));

    trellis_ggml_set_flash_attn_enabled(1);
    struct ggml_tensor * explicit_output = trellis_ggml_sdpa_with_policy(
        context, q, k, v, 0.5f, &explicit_policy);
    CHECK_TRUE(explicit_output != NULL);
    CHECK_TRUE(explicit_output->op == GGML_OP_MUL_MAT);

    trellis_ggml_set_flash_attn_enabled(0);
    struct ggml_tensor * flash_output = trellis_ggml_sdpa_with_policy(
        context, q, k, v, 0.5f, &flash_policy);
    CHECK_TRUE(flash_output != NULL);
    CHECK_TRUE(tensor_graph_contains_op(flash_output, GGML_OP_FLASH_ATTN_EXT));

    trellis_ggml_set_flash_attn_enabled(1);
    struct ggml_tensor * legacy_flash = trellis_ggml_sdpa(context, q, k, v, 0.5f);
    CHECK_TRUE(legacy_flash != NULL);
    CHECK_TRUE(tensor_graph_contains_op(legacy_flash, GGML_OP_FLASH_ATTN_EXT));

    trellis_ggml_attention_policy invalid_policy = explicit_policy;
    invalid_policy.struct_size = 0;
    CHECK_TRUE(!trellis_ggml_attention_policy_is_valid(&invalid_policy));
    CHECK_TRUE(trellis_ggml_sdpa_with_policy(
        context, q, k, v, 0.5f, &invalid_policy) == NULL);

cleanup:
    trellis_ggml_set_flash_attn_enabled(0);
    ggml_free(context);
}

int main(void) {
    test_explicit_policy_isolation();
    if (g_failures != 0) {
        fprintf(stderr, "%d attention policy test(s) failed\n", g_failures);
        return 1;
    }
    puts("attention policy tests passed");
    return 0;
}
