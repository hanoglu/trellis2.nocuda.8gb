#include "trellis_sparse_backend.h"

#ifdef TRELLIS_HAS_SPARSE_VULKAN
#include "trellis_sparse_vk_spv.h"
#include "trellis_sparse_vk_mat_spv.h"

#include <vulkan/vulkan.h>

#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

enum {
    VK_OP_LINEAR = 1,
    VK_OP_ROW_NORM = 2,
    VK_OP_SILU = 3,
    VK_OP_ADD = 4,
    VK_OP_SPARSE_CONV = 5,
    VK_OP_C2S_GATHER = 6,
    VK_OP_SKIP_REPEAT = 7,
    VK_OP_FILL_BIAS = 8,
    VK_OP_RULEBOOK_HASH_INSERT = 9,
    VK_OP_RULEBOOK_FILL = 10,
    VK_OP_C2S_COUNT = 11,
    VK_OP_C2S_FILL = 12,
    VK_OP_ROW_NORM_SILU = 13,
};

enum {
    VK_MAT_OP_LINEAR = 1,
    VK_MAT_OP_SPARSE_CONV_OFFSET = 2,
    VK_MAT_OP_LINEAR_SILU = 3,
};

enum {
    VK_BINDING_COUNT = 10,
};

typedef struct sparse_vk_push {
    uint32_t op;
    uint32_t n;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t total;
    uint32_t table_mask;
    uint32_t flags;
    float eps;
} sparse_vk_push;

struct trellis_sparse_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t bytes;
    int device_local;
    struct trellis_sparse_buffer * next_free;
};

struct trellis_sparse_rulebook {
    int32_t * coords_host;
    int64_t n;
    uint32_t table_mask;
    int64_t offset_starts[27];
    int64_t offset_counts[27];
    int64_t total_pairs;
    trellis_sparse_buffer * coords;
    trellis_sparse_buffer * hash_keys;
    trellis_sparse_buffer * hash_values;
    trellis_sparse_buffer * src_rows;
    trellis_sparse_buffer * dst_rows;
};

typedef struct sparse_vk_weight {
    const float * ptr;
    size_t count;
    trellis_sparse_buffer * buffer;
    struct sparse_vk_weight * next;
} sparse_vk_weight;

typedef struct sparse_vk_descriptor_node {
    VkDescriptorSet set;
    struct sparse_vk_descriptor_node * next;
} sparse_vk_descriptor_node;

typedef struct sparse_vk_command_node {
    VkCommandBuffer command;
    struct sparse_vk_command_node * next;
} sparse_vk_command_node;

typedef struct trellis_sparse_vk_backend {
    trellis_sparse_backend base;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    uint32_t max_workgroup_count[3];
    VkCommandPool command_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkPipeline mat_pipeline;
    VkDescriptorPool descriptor_pool;
    trellis_sparse_buffer * dummy;
    sparse_vk_weight * weights;
    trellis_sparse_buffer * free_buffers;
    sparse_vk_descriptor_node * free_descriptors;
    sparse_vk_command_node * free_commands;
    int destroying;
} trellis_sparse_vk_backend;

static uint32_t vk_hash_u64(uint32_t lo, uint32_t hi) {
    uint32_t x = lo ^ (hi * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void vk_pack_coord4(int b, int x, int y, int z, uint32_t * lo, uint32_t * hi) {
    *hi = (((uint32_t) b & 0xffffu) << 16) | ((uint32_t) x & 0xffffu);
    *lo = (((uint32_t) y & 0xffffu) << 16) | ((uint32_t) z & 0xffffu);
    *lo += 1u;
    if (*lo == 0u) {
        *hi += 1u;
    }
}

static int64_t vk_next_power_of_two(int64_t x) {
    int64_t v = 1;
    while (v < x && v < INT64_MAX / 2) {
        v <<= 1;
    }
    return v;
}

static int32_t vk_find_coord_host(
    const uint32_t * keys,
    const int32_t * values,
    uint32_t table_mask,
    int b,
    int x,
    int y,
    int z) {
    uint32_t lo, hi;
    vk_pack_coord4(b, x, y, z, &lo, &hi);
    uint32_t slot = vk_hash_u64(lo, hi) & table_mask;
    for (uint32_t probe = 0; probe <= table_mask; ++probe) {
        const uint32_t key_lo = keys[2u * slot + 0u];
        const uint32_t key_hi = keys[2u * slot + 1u];
        if (key_lo == lo && key_hi == hi) {
            return values[slot];
        }
        if (key_lo == 0u && key_hi == 0u) {
            return -1;
        }
        slot = (slot + 1u) & table_mask;
    }
    return -1;
}

static trellis_status vk_status(VkResult result) {
    return result == VK_SUCCESS ? TRELLIS_STATUS_OK : TRELLIS_STATUS_ERROR;
}

static uint32_t find_memory_type(
    trellis_sparse_vk_backend * vk,
    uint32_t type_bits,
    VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 &&
            (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static trellis_status vk_alloc_buffer(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_flags,
    int map_memory,
    trellis_sparse_buffer ** out) {
    if (vk == NULL || out == NULL || bytes == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_buffer * b = (trellis_sparse_buffer *) calloc(1, sizeof(*b));
    if (b == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = (VkDeviceSize) bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(vk->device, &buffer_info, NULL, &b->buffer);
    if (result != VK_SUCCESS) {
        free(b);
        return vk_status(result);
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, b->buffer, &req);
    uint32_t memory_type = find_memory_type(vk, req.memoryTypeBits, memory_flags);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(vk->device, b->buffer, NULL);
        free(b);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk->device, &alloc_info, NULL, &b->memory);
    if (result == VK_SUCCESS) {
        result = vkBindBufferMemory(vk->device, b->buffer, b->memory, 0);
    }
    if (result == VK_SUCCESS && map_memory) {
        result = vkMapMemory(vk->device, b->memory, 0, req.size, 0, &b->mapped);
    }
    if (result != VK_SUCCESS) {
        if (b->mapped != NULL) vkUnmapMemory(vk->device, b->memory);
        if (b->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, b->memory, NULL);
        vkDestroyBuffer(vk->device, b->buffer, NULL);
        free(b);
        return vk_status(result);
    }
    b->bytes = bytes;
    b->device_local = (memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
        (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0;
    if (b->mapped != NULL) {
        memset(b->mapped, 0, bytes);
    }
    *out = b;
    return TRELLIS_STATUS_OK;
}

static void vk_destroy_buffer_raw(trellis_sparse_vk_backend * vk, trellis_sparse_buffer * buffer) {
    if (vk == NULL || buffer == NULL) {
        return;
    }
    if (buffer->mapped != NULL) vkUnmapMemory(vk->device, buffer->memory);
    if (buffer->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, buffer->memory, NULL);
    if (buffer->buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk->device, buffer->buffer, NULL);
    free(buffer);
}

static trellis_sparse_buffer * vk_take_pooled_buffer(trellis_sparse_vk_backend * vk, size_t bytes) {
    if (vk == NULL) {
        return NULL;
    }
    trellis_sparse_buffer ** link = &vk->free_buffers;
    while (*link != NULL) {
        trellis_sparse_buffer * b = *link;
        if (b->bytes >= bytes) {
            *link = b->next_free;
            b->next_free = NULL;
            return b;
        }
        link = &b->next_free;
    }
    return NULL;
}

static trellis_status vk_alloc_bytes(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    trellis_sparse_buffer ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_buffer * pooled = vk_take_pooled_buffer(vk, bytes);
    if (pooled != NULL) {
        *out = pooled;
        return TRELLIS_STATUS_OK;
    }
    return vk_alloc_buffer(
        vk,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        0,
        out);
}

static trellis_status vk_alloc_staging(
    trellis_sparse_vk_backend * vk,
    size_t bytes,
    VkBufferUsageFlags usage,
    trellis_sparse_buffer ** out) {
    return vk_alloc_buffer(
        vk,
        bytes,
        usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        1,
        out);
}

static void vk_free_buffer(trellis_sparse_backend * backend, trellis_sparse_buffer * buffer) {
    if (backend == NULL || buffer == NULL) {
        return;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (!vk->destroying && buffer->device_local && buffer->mapped == NULL && buffer != vk->dummy) {
        buffer->next_free = vk->free_buffers;
        vk->free_buffers = buffer;
        return;
    }
    if (vk->device != VK_NULL_HANDLE && !vk->destroying) {
        vkDeviceWaitIdle(vk->device);
    }
    vk_destroy_buffer_raw(vk, buffer);
}

static trellis_status vk_copy_buffer(
    trellis_sparse_vk_backend * vk,
    trellis_sparse_buffer * src,
    trellis_sparse_buffer * dst,
    size_t bytes) {
    if (vk == NULL || src == NULL || dst == NULL || bytes == 0 ||
        src->bytes < bytes || dst->bytes < bytes) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkCommandBuffer command;
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, &command);
    if (result != VK_SUCCESS) {
        return vk_status(result);
    }
    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command, &begin);
    if (result == VK_SUCCESS) {
        VkBufferMemoryBarrier before[2];
        memset(before, 0, sizeof(before));
        before[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        before[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[0].buffer = src->buffer;
        before[0].offset = 0;
        before[0].size = (VkDeviceSize) bytes;
        before[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        before[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        before[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        before[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before[1].buffer = dst->buffer;
        before[1].offset = 0;
        before[1].size = (VkDeviceSize) bytes;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            NULL,
            2,
            before,
            0,
            NULL);
        VkBufferCopy region;
        memset(&region, 0, sizeof(region));
        region.size = (VkDeviceSize) bytes;
        vkCmdCopyBuffer(command, src->buffer, dst->buffer, 1, &region);
        VkBufferMemoryBarrier after;
        memset(&after, 0, sizeof(after));
        after.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.buffer = dst->buffer;
        after.offset = 0;
        after.size = (VkDeviceSize) bytes;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            NULL,
            1,
            &after,
            0,
            NULL);
        result = vkEndCommandBuffer(command);
    }
    if (result == VK_SUCCESS) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
    }
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(vk->queue);
    }
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &command);
    return vk_status(result);
}

static trellis_status vk_upload_bytes(
    trellis_sparse_vk_backend * vk,
    const void * src,
    size_t bytes,
    trellis_sparse_buffer ** out) {
    if (out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    const size_t alloc_bytes = bytes == 0 ? 4 : bytes;
    trellis_status status = vk_alloc_bytes(vk, alloc_bytes, out);
    if (status == TRELLIS_STATUS_OK && bytes != 0 && src != NULL) {
        trellis_sparse_buffer * staging = NULL;
        status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging);
        if (status == TRELLIS_STATUS_OK) {
            memcpy(staging->mapped, src, bytes);
            status = vk_copy_buffer(vk, staging, *out, bytes);
        }
        vk_free_buffer(&vk->base, staging);
        if (status != TRELLIS_STATUS_OK) {
            vk_free_buffer(&vk->base, *out);
            *out = NULL;
        }
    }
    return status;
}

static trellis_status vk_alloc_f32(trellis_sparse_backend * backend, size_t count, trellis_sparse_buffer ** out) {
    if (count > SIZE_MAX / sizeof(float)) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    return vk_alloc_bytes((trellis_sparse_vk_backend *) backend, count * sizeof(float), out);
}

static trellis_status vk_upload_f32(trellis_sparse_backend * backend, const float * src, size_t count, trellis_sparse_buffer ** out) {
    if (src == NULL && count != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_upload_bytes((trellis_sparse_vk_backend *) backend, src, count * sizeof(float), out);
}

static trellis_status vk_upload_i32(trellis_sparse_backend * backend, const int32_t * src, size_t count, trellis_sparse_buffer ** out) {
    if (src == NULL && count != 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_upload_bytes((trellis_sparse_vk_backend *) backend, src, count * sizeof(int32_t), out);
}

static trellis_status vk_download_f32(trellis_sparse_backend * backend, const trellis_sparse_buffer * src, float * dst, size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk == NULL || src == NULL || dst == NULL || src->bytes < count * sizeof(float)) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t bytes = count * sizeof(float);
    trellis_sparse_buffer * staging = NULL;
    trellis_status status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_copy_buffer(vk, (trellis_sparse_buffer *) src, staging, bytes);
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(dst, staging->mapped, bytes);
    }
    vk_free_buffer(backend, staging);
    return status;
}

static trellis_status vk_download_bytes(
    trellis_sparse_vk_backend * vk,
    const trellis_sparse_buffer * src,
    void * dst,
    size_t bytes) {
    if (vk == NULL || src == NULL || dst == NULL || bytes == 0 || src->bytes < bytes) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_buffer * staging = NULL;
    trellis_status status = vk_alloc_staging(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_copy_buffer(vk, (trellis_sparse_buffer *) src, staging, bytes);
    }
    if (status == TRELLIS_STATUS_OK) {
        memcpy(dst, staging->mapped, bytes);
    }
    vk_free_buffer(&vk->base, staging);
    return status;
}

static trellis_status vk_get_weight(
    trellis_sparse_vk_backend * vk,
    const float * ptr,
    size_t count,
    trellis_sparse_buffer ** out) {
    if (vk == NULL || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = vk->dummy;
    if (ptr == NULL || count == 0) {
        return TRELLIS_STATUS_OK;
    }
    for (sparse_vk_weight * w = vk->weights; w != NULL; w = w->next) {
        if (w->ptr == ptr && w->count == count) {
            *out = w->buffer;
            return TRELLIS_STATUS_OK;
        }
    }
    sparse_vk_weight * node = (sparse_vk_weight *) calloc(1, sizeof(*node));
    if (node == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    trellis_status status = vk_upload_bytes(vk, ptr, count * sizeof(float), &node->buffer);
    if (status != TRELLIS_STATUS_OK) {
        free(node);
        return status;
    }
    node->ptr = ptr;
    node->count = count;
    node->next = vk->weights;
    vk->weights = node;
    *out = node->buffer;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_acquire_descriptor_set(
    trellis_sparse_vk_backend * vk,
    VkDescriptorSet * descriptor_set) {
    if (vk == NULL || descriptor_set == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_set = VK_NULL_HANDLE;
    sparse_vk_descriptor_node * node = vk->free_descriptors;
    if (node != NULL) {
        vk->free_descriptors = node->next;
        *descriptor_set = node->set;
        free(node);
        return TRELLIS_STATUS_OK;
    }
    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = vk->descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &vk->descriptor_set_layout;
    VkResult result = vkAllocateDescriptorSets(vk->device, &set_alloc, descriptor_set);
    return vk_status(result);
}

static void vk_release_descriptor_set(trellis_sparse_vk_backend * vk, VkDescriptorSet descriptor_set) {
    if (vk == NULL || descriptor_set == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_descriptor_node * node = (sparse_vk_descriptor_node *) malloc(sizeof(*node));
    if (node == NULL) {
        vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &descriptor_set);
        return;
    }
    node->set = descriptor_set;
    node->next = vk->free_descriptors;
    vk->free_descriptors = node;
}

static trellis_status vk_acquire_command_buffer(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer * command) {
    if (vk == NULL || command == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *command = VK_NULL_HANDLE;
    sparse_vk_command_node * node = vk->free_commands;
    if (node != NULL) {
        vk->free_commands = node->next;
        *command = node->command;
        free(node);
        VkResult result = vkResetCommandBuffer(*command, 0);
        return vk_status(result);
    }
    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, command);
    return vk_status(result);
}

static void vk_release_command_buffer(trellis_sparse_vk_backend * vk, VkCommandBuffer command) {
    if (vk == NULL || command == VK_NULL_HANDLE || vk->destroying) {
        return;
    }
    sparse_vk_command_node * node = (sparse_vk_command_node *) malloc(sizeof(*node));
    if (node == NULL) {
        vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &command);
        return;
    }
    node->command = command;
    node->next = vk->free_commands;
    vk->free_commands = node;
}

static trellis_status vk_record_dispatch(
    trellis_sparse_vk_backend * vk,
    VkCommandBuffer command,
    VkPipeline pipeline,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z,
    VkDescriptorSet * descriptor_set_out) {
    if (vk == NULL || command == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || push == NULL ||
        descriptor_set_out == NULL || groups_x == 0 || groups_y == 0 || groups_z == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_set_out = VK_NULL_HANDLE;
    trellis_sparse_buffer * buffers[VK_BINDING_COUNT] = {
        f0 != NULL ? f0 : vk->dummy,
        f1 != NULL ? f1 : vk->dummy,
        f2 != NULL ? f2 : vk->dummy,
        f3 != NULL ? f3 : vk->dummy,
        i0 != NULL ? i0 : vk->dummy,
        i1 != NULL ? i1 : vk->dummy,
        u0 != NULL ? u0 : vk->dummy,
        i2 != NULL ? i2 : vk->dummy,
        i3 != NULL ? i3 : vk->dummy,
        i4 != NULL ? i4 : vk->dummy,
    };
    VkDescriptorSet descriptor_set;
    trellis_status status = vk_acquire_descriptor_set(vk, &descriptor_set);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }

    VkDescriptorBufferInfo infos[VK_BINDING_COUNT];
    VkWriteDescriptorSet writes[VK_BINDING_COUNT];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < VK_BINDING_COUNT; ++i) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, VK_BINDING_COUNT, writes, 0, NULL);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(command, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatch(command, groups_x, groups_y, groups_z);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        command,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &barrier,
        0,
        NULL,
        0,
        NULL);
    *descriptor_set_out = descriptor_set;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_dispatch_pipeline(
    trellis_sparse_vk_backend * vk,
    VkPipeline pipeline,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    trellis_sparse_buffer * i2,
    trellis_sparse_buffer * i3,
    trellis_sparse_buffer * i4,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z) {
    if (vk == NULL || pipeline == VK_NULL_HANDLE || push == NULL ||
        groups_x == 0 || groups_y == 0 || groups_z == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkCommandBuffer command;
    trellis_status status = vk_acquire_command_buffer(vk, &command);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult result = vkBeginCommandBuffer(command, &begin);
    if (result == VK_SUCCESS) {
        status = vk_record_dispatch(
            vk,
            command,
            pipeline,
            push,
            f0,
            f1,
            f2,
            f3,
            i0,
            i1,
            u0,
            i2,
            i3,
            i4,
            groups_x,
            groups_y,
            groups_z,
            &descriptor_set);
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
        } else {
            result = VK_ERROR_UNKNOWN;
        }
    }
    if (result == VK_SUCCESS) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
    }
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(vk->queue);
    }
    vk_release_command_buffer(vk, command);
    vk_release_descriptor_set(vk, descriptor_set);
    return status == TRELLIS_STATUS_OK ? vk_status(result) : status;
}

static trellis_status vk_dispatch(
    trellis_sparse_vk_backend * vk,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    trellis_sparse_buffer * u0,
    uint32_t work_items) {
    if (work_items == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return vk_dispatch_pipeline(
        vk,
        vk->pipeline,
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        u0,
        NULL,
        NULL,
        NULL,
        (work_items + 127u) / 128u,
        1,
        1);
}

static trellis_status vk_mat_dispatch_dims(
    const trellis_sparse_vk_backend * vk,
    uint32_t cols,
    uint32_t rows,
    uint32_t * groups_x,
    uint32_t * groups_y,
    uint32_t * groups_z) {
    if (vk == NULL || cols == 0 || rows == 0 ||
        groups_x == NULL || groups_y == NULL || groups_z == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *groups_x = (cols + 15u) / 16u;
    const uint32_t row_tiles = (rows + 15u) / 16u;
    const uint32_t max_y = vk->max_workgroup_count[1] == 0 ? 65535u : vk->max_workgroup_count[1];
    const uint32_t max_z = vk->max_workgroup_count[2] == 0 ? 65535u : vk->max_workgroup_count[2];
    *groups_y = row_tiles < max_y ? row_tiles : max_y;
    *groups_z = (row_tiles + *groups_y - 1u) / *groups_y;
    if (*groups_x == 0 || *groups_y == 0 || *groups_z == 0 ||
        *groups_x > vk->max_workgroup_count[0] || *groups_z > max_z) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_dispatch_mat(
    trellis_sparse_vk_backend * vk,
    const sparse_vk_push * push,
    trellis_sparse_buffer * f0,
    trellis_sparse_buffer * f1,
    trellis_sparse_buffer * f2,
    trellis_sparse_buffer * f3,
    trellis_sparse_buffer * i0,
    trellis_sparse_buffer * i1,
    uint32_t cols,
    uint32_t rows) {
    if (vk == NULL || cols == 0 || rows == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    trellis_status status = vk_mat_dispatch_dims(vk, cols, rows, &groups_x, &groups_y, &groups_z);
    if (status != TRELLIS_STATUS_OK) return status;
    return vk_dispatch_pipeline(
        vk,
        vk->mat_pipeline,
        push,
        f0,
        f1,
        f2,
        f3,
        i0,
        i1,
        NULL,
        NULL,
        NULL,
        NULL,
        groups_x,
        groups_y,
        groups_z);
}

static trellis_status vk_linear(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (n <= 0 || n > UINT32_MAX || in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, weight, (size_t) in_channels * (size_t) out_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    sparse_vk_push push = {
        VK_MAT_OP_LINEAR, (uint32_t) n, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    return status == TRELLIS_STATUS_OK ?
        vk_dispatch_mat(vk, &push, (trellis_sparse_buffer *) x, w, b, y, NULL, NULL, (uint32_t) out_channels, (uint32_t) n) :
        status;
}

static trellis_status vk_linear_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * y,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (n <= 0 || n > UINT32_MAX || in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, weight, (size_t) in_channels * (size_t) out_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    sparse_vk_push push = {
        VK_MAT_OP_LINEAR_SILU, (uint32_t) n, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    return status == TRELLIS_STATUS_OK ?
        vk_dispatch_mat(vk, &push, (trellis_sparse_buffer *) x, w, b, y, NULL, NULL, (uint32_t) out_channels, (uint32_t) n) :
        status;
}

static trellis_status vk_row_norm(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * g = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, gamma, (size_t) channels, &g);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, beta, (size_t) channels, &b);
    sparse_vk_push push = {
        VK_OP_ROW_NORM, (uint32_t) n, (uint32_t) channels, (uint32_t) channels,
        (uint32_t) n, 0, (gamma != NULL ? 1u : 0u) | (beta != NULL ? 2u : 0u), eps,
    };
    return status == TRELLIS_STATUS_OK ? vk_dispatch(vk, &push, (trellis_sparse_buffer *) x, g, b, y, NULL, NULL, NULL, (uint32_t) n) : status;
}

static trellis_status vk_row_norm_silu(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const float * gamma,
    const float * beta,
    trellis_sparse_buffer * y,
    int64_t n,
    int channels,
    float eps) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * g = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, gamma, (size_t) channels, &g);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, beta, (size_t) channels, &b);
    sparse_vk_push push = {
        VK_OP_ROW_NORM_SILU, (uint32_t) n, (uint32_t) channels, (uint32_t) channels,
        (uint32_t) n, 0, (gamma != NULL ? 1u : 0u) | (beta != NULL ? 2u : 0u), eps,
    };
    return status == TRELLIS_STATUS_OK ? vk_dispatch(vk, &push, (trellis_sparse_buffer *) x, g, b, y, NULL, NULL, NULL, (uint32_t) n) : status;
}

static trellis_status vk_silu_inplace(trellis_sparse_backend * backend, trellis_sparse_buffer * x, size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    sparse_vk_push push = { VK_OP_SILU, 0, 0, 0, (uint32_t) count, 0, 0, 0.0f };
    return vk_dispatch(vk, &push, NULL, NULL, NULL, x, NULL, NULL, NULL, (uint32_t) count);
}

static trellis_status vk_add(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * a,
    const trellis_sparse_buffer * b,
    trellis_sparse_buffer * y,
    size_t count) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    sparse_vk_push push = { VK_OP_ADD, 0, 0, 0, (uint32_t) count, 0, 0, 0.0f };
    return vk_dispatch(vk, &push, (trellis_sparse_buffer *) a, (trellis_sparse_buffer *) b, NULL, y, NULL, NULL, NULL, (uint32_t) count);
}

static trellis_status vk_build_rulebook(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    int64_t n,
    trellis_sparse_rulebook ** out) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (coords_bxyz == NULL || n <= 0 || out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (n > UINT32_MAX || n > INT32_MAX || n > UINT32_MAX / 27 || n > INT64_MAX / 27) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t row = 0; row < n; ++row) {
        const int32_t * c = coords_bxyz + row * 4;
        if (c[0] < 0 || c[0] > 3 ||
            c[1] < 0 || c[1] > 1023 ||
            c[2] < 0 || c[2] > 1023 ||
            c[3] < 0 || c[3] > 1023) {
            return TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    trellis_sparse_rulebook * r = (trellis_sparse_rulebook *) calloc(1, sizeof(*r));
    if (r == NULL) {
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    const int64_t table_size = vk_next_power_of_two(n * 2);
    if (table_size <= 0 || table_size > UINT32_MAX || 27ull * (uint64_t) n > SIZE_MAX / sizeof(int32_t)) {
        free(r);
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    const size_t keys_count = (size_t) table_size + 32u;
    uint32_t * zero_keys = (uint32_t *) calloc(keys_count, sizeof(uint32_t));
    if (zero_keys == NULL) {
        free(r);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    r->n = n;
    r->table_mask = (uint32_t) (table_size - 1);

    trellis_status status = vk_upload_i32(backend, coords_bxyz, (size_t) n * 4u, &r->coords);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_bytes(vk, zero_keys, keys_count * sizeof(uint32_t), &r->hash_keys);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) table_size * sizeof(int32_t), &r->hash_values);
    }
    const size_t rulebook_slots = (size_t) n * 27u;
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, rulebook_slots * sizeof(int32_t), &r->src_rows);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, rulebook_slots * sizeof(int32_t), &r->dst_rows);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        uint32_t descriptor_count = 0;
        status = vk_acquire_command_buffer(vk, &command);
        VkResult result = VK_SUCCESS;
        if (status == TRELLIS_STATUS_OK) {
            VkCommandBufferBeginInfo begin;
            memset(&begin, 0, sizeof(begin));
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            result = vkBeginCommandBuffer(command, &begin);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push insert_push = {
                VK_OP_RULEBOOK_HASH_INSERT, (uint32_t) n, 0, 0, (uint32_t) n, r->table_mask, 0, 0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipeline,
                &insert_push,
                NULL,
                NULL,
                NULL,
                NULL,
                r->coords,
                r->hash_values,
                r->hash_keys,
                NULL,
                NULL,
                NULL,
                ((uint32_t) n + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (status == TRELLIS_STATUS_OK) {
            sparse_vk_push fill_push = {
                VK_OP_RULEBOOK_FILL,
                (uint32_t) n,
                0,
                0,
                (uint32_t) ((uint64_t) n * 27u),
                r->table_mask,
                0,
                0.0f,
            };
            status = vk_record_dispatch(
                vk,
                command,
                vk->pipeline,
                &fill_push,
                NULL,
                NULL,
                NULL,
                NULL,
                r->coords,
                r->hash_values,
                r->hash_keys,
                r->src_rows,
                r->dst_rows,
                NULL,
                (fill_push.total + 127u) / 128u,
                1,
                1,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkEndCommandBuffer(command);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            VkSubmitInfo submit;
            memset(&submit, 0, sizeof(submit));
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
            status = vk_status(result);
        }
        if (status == TRELLIS_STATUS_OK) {
            result = vkQueueWaitIdle(vk->queue);
            status = vk_status(result);
        }
        if (command != VK_NULL_HANDLE) {
            vk_release_command_buffer(vk, command);
        }
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            vk_release_descriptor_set(vk, descriptor_sets[i]);
        }
    }
    uint32_t counts[27];
    memset(counts, 0, sizeof(counts));
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, r->hash_keys, counts, sizeof(counts));
    }
    if (status == TRELLIS_STATUS_OK) {
        for (int offset = 0; offset < 27; ++offset) {
            r->offset_starts[offset] = (int64_t) offset * n;
            r->offset_counts[offset] = (int64_t) counts[offset];
            r->total_pairs += (int64_t) counts[offset];
        }
        if (r->total_pairs <= 0 || r->total_pairs > INT32_MAX) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
        }
    }
    free(zero_keys);
    if (status != TRELLIS_STATUS_OK) {
        if (r->coords != NULL) vk_free_buffer(backend, r->coords);
        if (r->hash_keys != NULL) vk_free_buffer(backend, r->hash_keys);
        if (r->hash_values != NULL) vk_free_buffer(backend, r->hash_values);
        if (r->src_rows != NULL) vk_free_buffer(backend, r->src_rows);
        if (r->dst_rows != NULL) vk_free_buffer(backend, r->dst_rows);
        free(r->coords_host);
        free(r);
        return status;
    }
    *out = r;
    return TRELLIS_STATUS_OK;
}

static trellis_status vk_sparse_conv3d(
    trellis_sparse_backend * backend,
    const trellis_sparse_rulebook * rulebook,
    const trellis_sparse_buffer * feats,
    const float * weight,
    const float * bias,
    trellis_sparse_buffer * out,
    int64_t n,
    int in_channels,
    int out_channels) {
    if (rulebook == NULL || feats == NULL || weight == NULL || out == NULL || n <= 0 || n > UINT32_MAX ||
        in_channels <= 0 || out_channels <= 0 ||
        (uint64_t) n * (uint64_t) out_channels > UINT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * w = NULL;
    trellis_sparse_buffer * b = NULL;
    trellis_status status = vk_get_weight(vk, weight, (size_t) out_channels * 27u * (size_t) in_channels, &w);
    if (status == TRELLIS_STATUS_OK) status = vk_get_weight(vk, bias, (size_t) out_channels, &b);
    sparse_vk_push fill_push = {
        VK_OP_FILL_BIAS, (uint32_t) n, 0, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n * (uint64_t) out_channels), 0, bias != NULL ? 1u : 0u, 0.0f,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_sets[28];
    uint32_t descriptor_count = 0;
    memset(descriptor_sets, 0, sizeof(descriptor_sets));
    if (status == TRELLIS_STATUS_OK) {
        status = vk_acquire_command_buffer(vk, &command);
    }
    VkResult result = VK_SUCCESS;
    if (status == TRELLIS_STATUS_OK) {
        VkCommandBufferBeginInfo begin;
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        result = vkBeginCommandBuffer(command, &begin);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_record_dispatch(
            vk,
            command,
            vk->pipeline,
            &fill_push,
            NULL,
            NULL,
            b,
            out,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            (fill_push.total + 127u) / 128u,
            1,
            1,
            &descriptor_sets[descriptor_count]);
        if (status == TRELLIS_STATUS_OK) ++descriptor_count;
    }
    for (int offset = 0; status == TRELLIS_STATUS_OK && offset < 27; ++offset) {
        const int64_t pair_count = rulebook->offset_counts[offset];
        const int64_t pair_start = rulebook->offset_starts[offset];
        if (pair_count <= 0) {
            continue;
        }
        if (pair_count > UINT32_MAX || pair_start > UINT32_MAX ||
            (uint64_t) pair_count * (uint64_t) out_channels > UINT32_MAX) {
            status = TRELLIS_STATUS_INVALID_ARGUMENT;
            break;
        }
        sparse_vk_push conv_push = {
            VK_MAT_OP_SPARSE_CONV_OFFSET,
            (uint32_t) pair_count,
            (uint32_t) in_channels,
            (uint32_t) out_channels,
            (uint32_t) ((uint64_t) pair_count * (uint64_t) out_channels),
            (uint32_t) offset,
            (uint32_t) pair_start,
            0.0f,
        };
        uint32_t groups_x = 0;
        uint32_t groups_y = 0;
        uint32_t groups_z = 0;
        status = vk_mat_dispatch_dims(
            vk,
            (uint32_t) out_channels,
            (uint32_t) pair_count,
            &groups_x,
            &groups_y,
            &groups_z);
        if (status == TRELLIS_STATUS_OK) {
            status = vk_record_dispatch(
                vk,
                command,
                vk->mat_pipeline,
                &conv_push,
                (trellis_sparse_buffer *) feats,
                w,
                NULL,
                out,
                rulebook->src_rows,
                rulebook->dst_rows,
                NULL,
                NULL,
                NULL,
                NULL,
                groups_x,
                groups_y,
                groups_z,
                &descriptor_sets[descriptor_count]);
            if (status == TRELLIS_STATUS_OK) ++descriptor_count;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        result = vkEndCommandBuffer(command);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE);
        status = vk_status(result);
    }
    if (status == TRELLIS_STATUS_OK) {
        result = vkQueueWaitIdle(vk->queue);
        status = vk_status(result);
    }
    if (command != VK_NULL_HANDLE) {
        vk_release_command_buffer(vk, command);
    }
    for (uint32_t i = 0; i < descriptor_count; ++i) {
        vk_release_descriptor_set(vk, descriptor_sets[i]);
    }
    return status;
}

static void vk_free_rulebook(trellis_sparse_backend * backend, trellis_sparse_rulebook * rulebook) {
    if (rulebook == NULL) {
        return;
    }
    vk_free_buffer(backend, rulebook->coords);
    vk_free_buffer(backend, rulebook->hash_keys);
    vk_free_buffer(backend, rulebook->hash_values);
    vk_free_buffer(backend, rulebook->src_rows);
    vk_free_buffer(backend, rulebook->dst_rows);
    free(rulebook->coords_host);
    free(rulebook);
}

static trellis_status vk_c2s_gather(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * parent_buf = NULL;
    trellis_sparse_buffer * subidx_buf = NULL;
    trellis_status status = vk_upload_i32(backend, parent, (size_t) n_out, &parent_buf);
    if (status == TRELLIS_STATUS_OK) status = vk_upload_i32(backend, subidx, (size_t) n_out, &subidx_buf);
    sparse_vk_push push = {
        VK_OP_C2S_GATHER, (uint32_t) n_out, 0, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    if (status == TRELLIS_STATUS_OK) {
        status = vk_dispatch(vk, &push, (trellis_sparse_buffer *) x, NULL, NULL, y, parent_buf, subidx_buf, NULL, push.total);
    }
    vk_free_buffer(backend, parent_buf);
    vk_free_buffer(backend, subidx_buf);
    return status;
}

static trellis_status vk_skip_repeat(
    trellis_sparse_backend * backend,
    const trellis_sparse_buffer * x,
    const int32_t * parent,
    const int32_t * subidx,
    trellis_sparse_buffer * y,
    int64_t n_out,
    int in_channels,
    int out_channels) {
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * parent_buf = NULL;
    trellis_sparse_buffer * subidx_buf = NULL;
    trellis_status status = vk_upload_i32(backend, parent, (size_t) n_out, &parent_buf);
    if (status == TRELLIS_STATUS_OK) status = vk_upload_i32(backend, subidx, (size_t) n_out, &subidx_buf);
    sparse_vk_push push = {
        VK_OP_SKIP_REPEAT, (uint32_t) n_out, (uint32_t) in_channels, (uint32_t) out_channels,
        (uint32_t) ((uint64_t) n_out * (uint64_t) out_channels), 0, 0, 0.0f,
    };
    if (status == TRELLIS_STATUS_OK) {
        status = vk_dispatch(vk, &push, (trellis_sparse_buffer *) x, NULL, NULL, y, parent_buf, subidx_buf, NULL, push.total);
    }
    vk_free_buffer(backend, parent_buf);
    vk_free_buffer(backend, subidx_buf);
    return status;
}

static trellis_status vk_build_c2s_map(
    trellis_sparse_backend * backend,
    const int32_t * coords_bxyz,
    const trellis_sparse_buffer * logits,
    int64_t n,
    int32_t ** coords_out,
    int32_t ** parent_out,
    int32_t ** subidx_out,
    int64_t * n_out) {
    if (backend == NULL || coords_bxyz == NULL || logits == NULL ||
        coords_out == NULL || parent_out == NULL || subidx_out == NULL || n_out == NULL ||
        n <= 0 || n > UINT32_MAX / 8 || n > INT32_MAX) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *coords_out = NULL;
    *parent_out = NULL;
    *subidx_out = NULL;
    *n_out = 0;
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    trellis_sparse_buffer * coords_dev = NULL;
    trellis_sparse_buffer * counts_dev = NULL;
    trellis_sparse_buffer * prefix_dev = NULL;
    trellis_sparse_buffer * out_coords_dev = NULL;
    trellis_sparse_buffer * parent_dev = NULL;
    trellis_sparse_buffer * subidx_dev = NULL;
    int32_t * counts = NULL;
    int32_t * prefix = NULL;
    int32_t * out_coords = NULL;
    int32_t * out_parent = NULL;
    int32_t * out_subidx = NULL;

    trellis_status status = vk_upload_i32(backend, coords_bxyz, (size_t) n * 4u, &coords_dev);
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) n * sizeof(int32_t), &counts_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push count_push = {
            VK_OP_C2S_COUNT, (uint32_t) n, 0, 0, (uint32_t) n, 0, 0, 0.0f,
        };
        status = vk_dispatch_pipeline(
            vk,
            vk->pipeline,
            &count_push,
            (trellis_sparse_buffer *) logits,
            NULL,
            NULL,
            NULL,
            NULL,
            counts_dev,
            NULL,
            NULL,
            NULL,
            NULL,
            ((uint32_t) n + 127u) / 128u,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK) {
        counts = (int32_t *) malloc((size_t) n * sizeof(int32_t));
        prefix = (int32_t *) malloc((size_t) n * sizeof(int32_t));
        if (counts == NULL || prefix == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, counts_dev, counts, (size_t) n * sizeof(int32_t));
    }
    int64_t m = 0;
    if (status == TRELLIS_STATUS_OK) {
        for (int64_t row = 0; row < n; ++row) {
            if (counts[row] < 0 || counts[row] > 8 || m > INT32_MAX - counts[row]) {
                status = TRELLIS_STATUS_INVALID_ARGUMENT;
                break;
            }
            prefix[row] = (int32_t) m;
            m += counts[row];
        }
        if (status == TRELLIS_STATUS_OK && m <= 0) {
            status = TRELLIS_STATUS_ERROR;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_upload_i32(backend, prefix, (size_t) n, &prefix_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * 4u * sizeof(int32_t), &out_coords_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &parent_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_alloc_bytes(vk, (size_t) m * sizeof(int32_t), &subidx_dev);
    }
    if (status == TRELLIS_STATUS_OK) {
        sparse_vk_push fill_push = {
            VK_OP_C2S_FILL,
            (uint32_t) n,
            0,
            0,
            (uint32_t) ((uint64_t) n * 8u),
            0,
            0,
            0.0f,
        };
        status = vk_dispatch_pipeline(
            vk,
            vk->pipeline,
            &fill_push,
            (trellis_sparse_buffer *) logits,
            NULL,
            NULL,
            NULL,
            coords_dev,
            prefix_dev,
            NULL,
            out_coords_dev,
            parent_dev,
            subidx_dev,
            (fill_push.total + 127u) / 128u,
            1,
            1);
    }
    if (status == TRELLIS_STATUS_OK) {
        out_coords = (int32_t *) malloc((size_t) m * 4u * sizeof(int32_t));
        out_parent = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        out_subidx = (int32_t *) malloc((size_t) m * sizeof(int32_t));
        if (out_coords == NULL || out_parent == NULL || out_subidx == NULL) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, out_coords_dev, out_coords, (size_t) m * 4u * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, parent_dev, out_parent, (size_t) m * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        status = vk_download_bytes(vk, subidx_dev, out_subidx, (size_t) m * sizeof(int32_t));
    }
    if (status == TRELLIS_STATUS_OK) {
        *coords_out = out_coords;
        *parent_out = out_parent;
        *subidx_out = out_subidx;
        *n_out = m;
        out_coords = NULL;
        out_parent = NULL;
        out_subidx = NULL;
    }
    free(out_coords);
    free(out_parent);
    free(out_subidx);
    free(prefix);
    free(counts);
    vk_free_buffer(backend, subidx_dev);
    vk_free_buffer(backend, parent_dev);
    vk_free_buffer(backend, out_coords_dev);
    vk_free_buffer(backend, prefix_dev);
    vk_free_buffer(backend, counts_dev);
    vk_free_buffer(backend, coords_dev);
    return status;
}

static void vk_destroy(trellis_sparse_backend * backend) {
    if (backend == NULL) {
        return;
    }
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) backend;
    if (vk->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk->device);
    }
    vk->destroying = 1;
    sparse_vk_weight * w = vk->weights;
    while (w != NULL) {
        sparse_vk_weight * next = w->next;
        vk_free_buffer(backend, w->buffer);
        free(w);
        w = next;
    }
    vk_free_buffer(backend, vk->dummy);
    while (vk->free_buffers != NULL) {
        trellis_sparse_buffer * next = vk->free_buffers->next_free;
        vk_destroy_buffer_raw(vk, vk->free_buffers);
        vk->free_buffers = next;
    }
    while (vk->free_commands != NULL) {
        sparse_vk_command_node * next = vk->free_commands->next;
        vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &vk->free_commands->command);
        free(vk->free_commands);
        vk->free_commands = next;
    }
    while (vk->free_descriptors != NULL) {
        sparse_vk_descriptor_node * next = vk->free_descriptors->next;
        vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &vk->free_descriptors->set);
        free(vk->free_descriptors);
        vk->free_descriptors = next;
    }
    if (vk->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->pipeline, NULL);
    if (vk->mat_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->mat_pipeline, NULL);
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    free(vk);
}

static const trellis_sparse_backend_ops g_vk_ops = {
    vk_destroy,
    vk_alloc_f32,
    vk_upload_f32,
    vk_upload_i32,
    vk_download_f32,
    vk_free_buffer,
    vk_linear,
    vk_linear_silu,
    vk_row_norm,
    vk_row_norm_silu,
    vk_silu_inplace,
    vk_add,
    vk_build_rulebook,
    vk_sparse_conv3d,
    vk_free_rulebook,
    vk_c2s_gather,
    vk_skip_repeat,
    vk_build_c2s_map,
};

static trellis_status vk_create_pipeline_from_spv(
    trellis_sparse_vk_backend * vk,
    const unsigned char * spv,
    unsigned int spv_len,
    VkPipeline * pipeline_out) {
    if (vk == NULL || spv == NULL || spv_len == 0 || pipeline_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_len;
    shader_info.pCode = (const uint32_t *) spv;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(vk->device, &shader_info, NULL, &shader);
    if (result != VK_SUCCESS) return vk_status(result);

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = vk->pipeline_layout;
    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline_out);
    vkDestroyShaderModule(vk->device, shader, NULL);
    return vk_status(result);
}

static trellis_status vk_create_pipelines(trellis_sparse_vk_backend * vk) {
    trellis_status status = vk_create_pipeline_from_spv(
        vk,
        trellis_sparse_vk_spv,
        trellis_sparse_vk_spv_len,
        &vk->pipeline);
    if (status != TRELLIS_STATUS_OK) {
        return status;
    }
    return vk_create_pipeline_from_spv(
        vk,
        trellis_sparse_vk_mat_spv,
        trellis_sparse_vk_mat_spv_len,
        &vk->mat_pipeline);
}

trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out) {
    if (out == NULL || device < 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    trellis_sparse_vk_backend * vk = (trellis_sparse_vk_backend *) calloc(1, sizeof(*vk));
    if (vk == NULL) return TRELLIS_STATUS_OUT_OF_MEMORY;
    vk->base.ops = &g_vk_ops;
    vk->base.kind = TRELLIS_SPARSE_BACKEND_VULKAN;
    vk->base.device = device;

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "trellis2.c sparse";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    VkResult result = vkCreateInstance(&instance_info, NULL, &vk->instance);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    uint32_t physical_count = 0;
    vkEnumeratePhysicalDevices(vk->instance, &physical_count, NULL);
    if (physical_count == 0 || (uint32_t) device >= physical_count) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }
    VkPhysicalDevice * physical = (VkPhysicalDevice *) malloc((size_t) physical_count * sizeof(VkPhysicalDevice));
    if (physical == NULL) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    vkEnumeratePhysicalDevices(vk->instance, &physical_count, physical);
    vk->physical_device = physical[device];
    free(physical);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk->physical_device, &props);
    vk->max_workgroup_count[0] = props.limits.maxComputeWorkGroupCount[0];
    vk->max_workgroup_count[1] = props.limits.maxComputeWorkGroupCount[1];
    vk->max_workgroup_count[2] = props.limits.maxComputeWorkGroupCount[2];

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &family_count, NULL);
    VkQueueFamilyProperties * families = (VkQueueFamilyProperties *) malloc((size_t) family_count * sizeof(*families));
    if (families == NULL) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_OUT_OF_MEMORY;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &family_count, families);
    vk->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            vk->queue_family = i;
            break;
        }
    }
    free(families);
    if (vk->queue_family == UINT32_MAX) {
        vk_destroy(&vk->base);
        return TRELLIS_STATUS_CUDA_UNAVAILABLE;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk->queue_family;
    result = vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }

    VkDescriptorSetLayoutBinding bindings[VK_BINDING_COUNT];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < VK_BINDING_COUNT; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = VK_BINDING_COUNT;
    layout_info.pBindings = bindings;
    result = vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->descriptor_set_layout);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(sparse_vk_push);
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    result = vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipeline_layout);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    trellis_status status = vk_create_pipelines(vk);
    if (status != TRELLIS_STATUS_OK) {
        vk_destroy(&vk->base);
        return status;
    }

    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = VK_BINDING_COUNT * 4096;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 4096;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    result = vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool);
    if (result != VK_SUCCESS) {
        vk_destroy(&vk->base);
        return vk_status(result);
    }
    status = vk_alloc_bytes(vk, 4, &vk->dummy);
    if (status != TRELLIS_STATUS_OK) {
        vk_destroy(&vk->base);
        return status;
    }
    *out = &vk->base;
    return TRELLIS_STATUS_OK;
}

#else

trellis_status trellis_sparse_vulkan_backend_create(int device, trellis_sparse_backend ** out) {
    (void) device;
    if (out == 0) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    *out = 0;
    return TRELLIS_STATUS_NOT_IMPLEMENTED;
}

#endif
