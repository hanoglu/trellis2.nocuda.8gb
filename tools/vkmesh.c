#include <vulkan/vulkan.h>

#include "vkmesh_spv.h"

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct vkmesh_mesh {
    float * vertices;
    int32_t * faces;
    int64_t n_vertices;
    int64_t n_faces;
    int64_t vertex_capacity;
    int64_t face_capacity;
} vkmesh_mesh;

typedef struct vkmesh_edge {
    uint32_t min_v;
    uint32_t max_v;
    uint32_t a;
    uint32_t b;
    uint32_t face;
} vkmesh_edge;

typedef struct vkmesh_boundary_edge {
    int32_t a;
    int32_t b;
    uint32_t min_v;
    uint32_t max_v;
} vkmesh_boundary_edge;

typedef struct vkmesh_vk_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t bytes;
} vkmesh_vk_buffer;

typedef struct vkmesh_vk {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorPool descriptor_pool;
} vkmesh_vk;

typedef struct vkmesh_push {
    uint32_t op;
    uint32_t n;
    uint32_t aux0;
    uint32_t aux1;
    float eps;
} vkmesh_push;

enum {
    VKMESH_OP_EXPAND_EDGES = 1,
    VKMESH_OP_DEGENERATE_FLAGS = 2,
};

static void mesh_free(vkmesh_mesh * mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

static int mesh_reserve_vertices(vkmesh_mesh * mesh, int64_t need) {
    if (need <= mesh->vertex_capacity) return 1;
    int64_t cap = mesh->vertex_capacity > 0 ? mesh->vertex_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    float * next = (float *) realloc(mesh->vertices, (size_t) cap * 3u * sizeof(float));
    if (next == NULL) return 0;
    mesh->vertices = next;
    mesh->vertex_capacity = cap;
    return 1;
}

static int mesh_reserve_faces(vkmesh_mesh * mesh, int64_t need) {
    if (need <= mesh->face_capacity) return 1;
    int64_t cap = mesh->face_capacity > 0 ? mesh->face_capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) return 0;
        cap *= 2;
    }
    int32_t * next = (int32_t *) realloc(mesh->faces, (size_t) cap * 3u * sizeof(int32_t));
    if (next == NULL) return 0;
    mesh->faces = next;
    mesh->face_capacity = cap;
    return 1;
}

static int mesh_add_vertex(vkmesh_mesh * mesh, float x, float y, float z) {
    if (!mesh_reserve_vertices(mesh, mesh->n_vertices + 1)) return 0;
    float * v = mesh->vertices + (size_t) mesh->n_vertices * 3u;
    v[0] = x;
    v[1] = y;
    v[2] = z;
    ++mesh->n_vertices;
    return 1;
}

static int mesh_add_face(vkmesh_mesh * mesh, int32_t a, int32_t b, int32_t c) {
    if (a < 0 || b < 0 || c < 0 ||
        a >= mesh->n_vertices || b >= mesh->n_vertices || c >= mesh->n_vertices) {
        return 0;
    }
    if (!mesh_reserve_faces(mesh, mesh->n_faces + 1)) return 0;
    int32_t * f = mesh->faces + (size_t) mesh->n_faces * 3u;
    f[0] = a;
    f[1] = b;
    f[2] = c;
    ++mesh->n_faces;
    return 1;
}

static int parse_obj_index(const char * token, int64_t n_vertices, int32_t * out) {
    char * end = NULL;
    long idx = strtol(token, &end, 10);
    if (end == token || idx == 0) return 0;
    int64_t resolved = idx > 0 ? (int64_t) idx - 1 : n_vertices + (int64_t) idx;
    if (resolved < 0 || resolved >= n_vertices || resolved > INT32_MAX) return 0;
    *out = (int32_t) resolved;
    return 1;
}

static int load_obj(const char * path, vkmesh_mesh * mesh) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s\n", path);
        return 0;
    }
    char line[8192];
    int ok = 1;
    int64_t line_no = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        char * p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (p[0] == 'v' && isspace((unsigned char) p[1])) {
            float x, y, z;
            if (sscanf(p + 1, "%f %f %f", &x, &y, &z) != 3 || !mesh_add_vertex(mesh, x, y, z)) {
                fprintf(stderr, "vkmesh: bad vertex at %s:%" PRId64 "\n", path, line_no);
                ok = 0;
                break;
            }
        } else if (p[0] == 'f' && isspace((unsigned char) p[1])) {
            int32_t ids[256];
            int n = 0;
            char * q = p + 1;
            while (*q != '\0') {
                while (*q != '\0' && isspace((unsigned char) *q)) ++q;
                if (*q == '\0' || *q == '#') break;
                if (n >= (int) (sizeof(ids) / sizeof(ids[0]))) {
                    fprintf(stderr, "vkmesh: face has too many vertices at %s:%" PRId64 "\n", path, line_no);
                    ok = 0;
                    break;
                }
                if (!parse_obj_index(q, mesh->n_vertices, &ids[n++])) {
                    fprintf(stderr, "vkmesh: bad face index at %s:%" PRId64 "\n", path, line_no);
                    ok = 0;
                    break;
                }
                while (*q != '\0' && !isspace((unsigned char) *q)) ++q;
            }
            if (!ok) break;
            if (n < 3) {
                fprintf(stderr, "vkmesh: face has fewer than 3 vertices at %s:%" PRId64 "\n", path, line_no);
                ok = 0;
                break;
            }
            for (int i = 1; i + 1 < n; ++i) {
                if (!mesh_add_face(mesh, ids[0], ids[i], ids[i + 1])) {
                    fprintf(stderr, "vkmesh: failed to append face at %s:%" PRId64 "\n", path, line_no);
                    ok = 0;
                    break;
                }
            }
            if (!ok) break;
        }
    }
    fclose(f);
    if (ok && (mesh->n_vertices <= 0 || mesh->n_faces <= 0)) {
        fprintf(stderr, "vkmesh: %s did not contain a usable triangle mesh\n", path);
        ok = 0;
    }
    return ok;
}

static int write_obj(const char * path, const vkmesh_mesh * mesh) {
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s for writing\n", path);
        return 0;
    }
    fprintf(f, "# vkmesh output\n");
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        const float * v = mesh->vertices + (size_t) i * 3u;
        fprintf(f, "v %.9g %.9g %.9g\n", v[0], v[1], v[2]);
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * face = mesh->faces + (size_t) i * 3u;
        fprintf(f, "f %d %d %d\n", face[0] + 1, face[1] + 1, face[2] + 1);
    }
    fclose(f);
    return 1;
}

static uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t type_bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static void vk_buffer_destroy(vkmesh_vk * vk, vkmesh_vk_buffer * b);

static int vk_buffer_create(vkmesh_vk * vk, size_t bytes, vkmesh_vk_buffer * out) {
    memset(out, 0, sizeof(*out));
    out->bytes = bytes;
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = (VkDeviceSize) bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(vk->device, &buffer_info, NULL, &out->buffer);
    if (result != VK_SUCCESS) return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, out->buffer, &req);
    uint32_t memory_type = find_memory_type(
        vk->physical_device,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(vk->device, out->buffer, NULL);
        memset(out, 0, sizeof(*out));
        return 0;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(vk->device, &alloc_info, NULL, &out->memory);
    if (result == VK_SUCCESS) result = vkBindBufferMemory(vk->device, out->buffer, out->memory, 0);
    if (result == VK_SUCCESS) result = vkMapMemory(vk->device, out->memory, 0, req.size, 0, &out->mapped);
    if (result != VK_SUCCESS) {
        vk_buffer_destroy(vk, out);
        return 0;
    }
    memset(out->mapped, 0, bytes);
    return 1;
}

static void vk_buffer_destroy(vkmesh_vk * vk, vkmesh_vk_buffer * b) {
    if (b == NULL) return;
    if (b->mapped != NULL) vkUnmapMemory(vk->device, b->memory);
    if (b->memory != VK_NULL_HANDLE) vkFreeMemory(vk->device, b->memory, NULL);
    if (b->buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk->device, b->buffer, NULL);
    memset(b, 0, sizeof(*b));
}

static void vkmesh_vk_destroy(vkmesh_vk * vk) {
    if (vk == NULL) return;
    if (vk->device != VK_NULL_HANDLE) vkDeviceWaitIdle(vk->device);
    if (vk->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    if (vk->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->pipeline, NULL);
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

static int vkmesh_vk_init(vkmesh_vk * vk) {
    memset(vk, 0, sizeof(*vk));
    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkmesh";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    if (vkCreateInstance(&instance_info, NULL, &vk->instance) != VK_SUCCESS) {
        fprintf(stderr, "vkmesh: failed to create Vulkan instance\n");
        return 0;
    }

    uint32_t physical_count = 0;
    if (vkEnumeratePhysicalDevices(vk->instance, &physical_count, NULL) != VK_SUCCESS || physical_count == 0) {
        fprintf(stderr, "vkmesh: no Vulkan physical devices\n");
        return 0;
    }
    VkPhysicalDevice * physical = (VkPhysicalDevice *) malloc((size_t) physical_count * sizeof(*physical));
    if (physical == NULL) return 0;
    if (vkEnumeratePhysicalDevices(vk->instance, &physical_count, physical) != VK_SUCCESS) {
        free(physical);
        return 0;
    }
    int found = 0;
    for (uint32_t i = 0; i < physical_count && !found; ++i) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical[i], &family_count, NULL);
        VkQueueFamilyProperties * families =
            (VkQueueFamilyProperties *) malloc((size_t) family_count * sizeof(*families));
        if (families == NULL) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(physical[i], &family_count, families);
        for (uint32_t q = 0; q < family_count; ++q) {
            if ((families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                vk->physical_device = physical[i];
                vk->queue_family = q;
                found = 1;
                break;
            }
        }
        free(families);
    }
    free(physical);
    if (!found) {
        fprintf(stderr, "vkmesh: no Vulkan compute queue\n");
        return 0;
    }

    float priority = 1.0f;
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
    if (vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device) != VK_SUCCESS) {
        fprintf(stderr, "vkmesh: failed to create Vulkan device\n");
        return 0;
    }
    vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk->queue_family;
    if (vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool) != VK_SUCCESS) return 0;

    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 4;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->descriptor_set_layout) != VK_SUCCESS) return 0;

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(vkmesh_push);
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk->descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    if (vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipeline_layout) != VK_SUCCESS) return 0;

    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = vkmesh_spv_len;
    shader_info.pCode = (const uint32_t *) vkmesh_spv;
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk->device, &shader_info, NULL, &shader) != VK_SUCCESS) return 0;

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
    VkResult result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &vk->pipeline);
    vkDestroyShaderModule(vk->device, shader, NULL);
    if (result != VK_SUCCESS) return 0;

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool) != VK_SUCCESS) return 0;
    return 1;
}

static int vkmesh_dispatch(
    vkmesh_vk * vk,
    vkmesh_vk_buffer buffers[4],
    const vkmesh_push * push,
    uint32_t groups_x) {
    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = vk->descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &vk->descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(vk->device, &set_alloc, &descriptor_set) != VK_SUCCESS) return 0;

    VkDescriptorBufferInfo infos[4];
    VkWriteDescriptorSet writes[4];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        infos[i].buffer = buffers[i].buffer;
        infos[i].offset = 0;
        infos[i].range = (VkDeviceSize) buffers[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, 4, writes, 0, NULL);

    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vk->device, &cmd_alloc, &command) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return 0;
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(command, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatch(command, groups_x, 1, 1);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) return 0;

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) return 0;
    vkQueueWaitIdle(vk->queue);
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &command);
    vkFreeDescriptorSets(vk->device, vk->descriptor_pool, 1, &descriptor_set);
    return 1;
}

static int expand_edges_vulkan(const vkmesh_mesh * mesh, vkmesh_edge ** edges_out, int64_t * edge_count_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;

    vkmesh_vk vk;
    if (!vkmesh_vk_init(&vk)) {
        vkmesh_vk_destroy(&vk);
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t edge_count = (size_t) mesh->n_faces * 3u;
    const size_t edges_bytes = edge_count * sizeof(vkmesh_edge);
    int ok = 0;
    if (!vk_buffer_create(&vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(&vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(&vk, edges_bytes, &buffers[2]) ||
        !vk_buffer_create(&vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.op = VKMESH_OP_EXPAND_EDGES;
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(&vk, buffers, &push, groups)) goto cleanup;

    vkmesh_edge * edges = (vkmesh_edge *) malloc(edges_bytes);
    if (edges == NULL) goto cleanup;
    memcpy(edges, buffers[2].mapped, edges_bytes);
    *edges_out = edges;
    *edge_count_out = (int64_t) edge_count;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(&vk, &buffers[i]);
    vkmesh_vk_destroy(&vk);
    return ok;
}

static int edge_compare(const void * a, const void * b) {
    const vkmesh_edge * ea = (const vkmesh_edge *) a;
    const vkmesh_edge * eb = (const vkmesh_edge *) b;
    if (ea->min_v != eb->min_v) return ea->min_v < eb->min_v ? -1 : 1;
    if (ea->max_v != eb->max_v) return ea->max_v < eb->max_v ? -1 : 1;
    return 0;
}

static int uf_find(int * parent, int x) {
    int r = x;
    while (parent[r] != r) r = parent[r];
    while (parent[x] != x) {
        int p = parent[x];
        parent[x] = r;
        x = p;
    }
    return r;
}

static void uf_union(int * parent, int a, int b) {
    int ra = uf_find(parent, a);
    int rb = uf_find(parent, b);
    if (ra == rb) return;
    if (ra < rb) parent[rb] = ra;
    else parent[ra] = rb;
}

static double edge_length(const vkmesh_mesh * mesh, int32_t a, int32_t b) {
    const float * va = mesh->vertices + (size_t) a * 3u;
    const float * vb = mesh->vertices + (size_t) b * 3u;
    double dx = (double) va[0] - (double) vb[0];
    double dy = (double) va[1] - (double) vb[1];
    double dz = (double) va[2] - (double) vb[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static int vkmesh_fill_holes(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    vkmesh_edge * edges = NULL;
    int64_t edge_count = 0;
    if (!expand_edges_vulkan(mesh, &edges, &edge_count)) {
        fprintf(stderr, "vkmesh: Vulkan edge expansion failed\n");
        return 0;
    }
    qsort(edges, (size_t) edge_count, sizeof(edges[0]), edge_compare);

    vkmesh_boundary_edge * boundary = NULL;
    int64_t boundary_count = 0;
    int64_t boundary_capacity = 0;
    for (int64_t i = 0; i < edge_count;) {
        int64_t j = i + 1;
        while (j < edge_count && edges[j].min_v == edges[i].min_v && edges[j].max_v == edges[i].max_v) ++j;
        if (j - i == 1) {
            if (boundary_count == boundary_capacity) {
                int64_t next_cap = boundary_capacity > 0 ? boundary_capacity * 2 : 1024;
                vkmesh_boundary_edge * next =
                    (vkmesh_boundary_edge *) realloc(boundary, (size_t) next_cap * sizeof(*boundary));
                if (next == NULL) {
                    free(edges);
                    free(boundary);
                    return 0;
                }
                boundary = next;
                boundary_capacity = next_cap;
            }
            boundary[boundary_count].a = (int32_t) edges[i].a;
            boundary[boundary_count].b = (int32_t) edges[i].b;
            boundary[boundary_count].min_v = edges[i].min_v;
            boundary[boundary_count].max_v = edges[i].max_v;
            ++boundary_count;
        }
        i = j;
    }
    free(edges);
    if (boundary_count == 0) {
        free(boundary);
        return 1;
    }

    int * degree = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * parent = (int *) malloc((size_t) boundary_count * sizeof(int));
    if (degree == NULL || offset == NULL || cursor == NULL || parent == NULL) {
        free(boundary);
        free(degree);
        free(offset);
        free(cursor);
        free(parent);
        return 0;
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        ++degree[boundary[i].a];
        ++degree[boundary[i].b];
        parent[i] = (int) i;
    }
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        offset[v + 1] = offset[v] + degree[v];
    }
    int * adj = (int *) malloc((size_t) offset[mesh->n_vertices] * sizeof(int));
    if (adj == NULL) {
        free(boundary);
        free(degree);
        free(offset);
        free(cursor);
        free(parent);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_vertices + 1u) * sizeof(int));
    for (int64_t i = 0; i < boundary_count; ++i) {
        adj[cursor[boundary[i].a]++] = (int) i;
        adj[cursor[boundary[i].b]++] = (int) i;
    }
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        int begin = offset[v];
        int end = offset[v + 1];
        for (int k = begin + 1; k < end; ++k) {
            uf_union(parent, adj[begin], adj[k]);
        }
    }

    uint8_t * valid = (uint8_t *) calloc((size_t) boundary_count, sizeof(uint8_t));
    int * comp_edges = (int *) calloc((size_t) boundary_count, sizeof(int));
    double * comp_perim = (double *) calloc((size_t) boundary_count, sizeof(double));
    double * comp_mid = (double *) calloc((size_t) boundary_count * 3u, sizeof(double));
    int32_t * comp_new_vertex = (int32_t *) malloc((size_t) boundary_count * sizeof(int32_t));
    if (valid == NULL || comp_edges == NULL || comp_perim == NULL || comp_mid == NULL || comp_new_vertex == NULL) {
        free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
        free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
        return 0;
    }
    memset(comp_new_vertex, 0xff, (size_t) boundary_count * sizeof(int32_t));
    for (int64_t i = 0; i < boundary_count; ++i) valid[uf_find(parent, (int) i)] = 1u;
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        if (degree[v] == 0) continue;
        int root = uf_find(parent, adj[offset[v]]);
        if (degree[v] != 2) valid[root] = 0u;
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        int root = uf_find(parent, (int) i);
        const float * a = mesh->vertices + (size_t) boundary[i].a * 3u;
        const float * b = mesh->vertices + (size_t) boundary[i].b * 3u;
        ++comp_edges[root];
        comp_perim[root] += edge_length(mesh, boundary[i].a, boundary[i].b);
        comp_mid[(size_t) root * 3u + 0u] += ((double) a[0] + (double) b[0]) * 0.5;
        comp_mid[(size_t) root * 3u + 1u] += ((double) a[1] + (double) b[1]) * 0.5;
        comp_mid[(size_t) root * 3u + 2u] += ((double) a[2] + (double) b[2]) * 0.5;
    }
    for (int64_t root = 0; root < boundary_count; ++root) {
        if (!valid[root] || comp_edges[root] <= 0 || comp_perim[root] >= (double) max_hole_perimeter) continue;
        double inv = 1.0 / (double) comp_edges[root];
        if (!mesh_add_vertex(
                mesh,
                (float) (comp_mid[(size_t) root * 3u + 0u] * inv),
                (float) (comp_mid[(size_t) root * 3u + 1u] * inv),
                (float) (comp_mid[(size_t) root * 3u + 2u] * inv))) {
            free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
            free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
            return 0;
        }
        comp_new_vertex[root] = (int32_t) (mesh->n_vertices - 1);
        ++(*filled_loops);
    }
    for (int64_t i = 0; i < boundary_count; ++i) {
        int root = uf_find(parent, (int) i);
        int32_t center = comp_new_vertex[root];
        if (center < 0) continue;
        if (!mesh_add_face(mesh, boundary[i].b, boundary[i].a, center)) {
            free(boundary); free(degree); free(offset); free(cursor); free(parent); free(adj);
            free(valid); free(comp_edges); free(comp_perim); free(comp_mid); free(comp_new_vertex);
            return 0;
        }
        ++(*added_faces);
    }

    free(boundary);
    free(degree);
    free(offset);
    free(cursor);
    free(parent);
    free(adj);
    free(valid);
    free(comp_edges);
    free(comp_perim);
    free(comp_mid);
    free(comp_new_vertex);
    return 1;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --input in.obj --output out.obj [options]\n"
        "\n"
        "Implemented stage:\n"
        "  --fill-holes             Fill small manifold boundary loops (default)\n"
        "  --no-fill-holes          Only load and write the mesh\n"
        "  --max-hole-perimeter X   Default 0.03, matching CuMesh/TRELLIS.2\n"
        "\n"
        "Planned stages, in order: remove_duplicate_faces, remove_degenerate_faces,\n"
        "repair_non_manifold_edges, remove_small_connected_components,\n"
        "unify_face_orientations, simplify, uv_unwrap, cuBVH, remesh_narrow_band_dc.\n",
        argv0);
}

int main(int argc, char ** argv) {
    const char * input = NULL;
    const char * output = NULL;
    float max_hole_perimeter = 3e-2f;
    int fill_holes = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--max-hole-perimeter") == 0 && i + 1 < argc) {
            max_hole_perimeter = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--fill-holes") == 0) {
            fill_holes = 1;
        } else if (strcmp(argv[i], "--no-fill-holes") == 0) {
            fill_holes = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "vkmesh: unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (input == NULL || output == NULL) {
        print_usage(argv[0]);
        return 2;
    }

    vkmesh_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (!load_obj(input, &mesh)) {
        mesh_free(&mesh);
        return 1;
    }
    fprintf(stderr, "vkmesh: loaded %" PRId64 " vertices, %" PRId64 " faces\n", mesh.n_vertices, mesh.n_faces);
    if (fill_holes) {
        int filled = 0;
        int added_faces = 0;
        int64_t before_v = mesh.n_vertices;
        int64_t before_f = mesh.n_faces;
        if (!vkmesh_fill_holes(&mesh, max_hole_perimeter, &filled, &added_faces)) {
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr,
            "vkmesh: fill_holes loops=%d added_vertices=%" PRId64 " added_faces=%d faces=%" PRId64 "->%" PRId64 "\n",
            filled,
            mesh.n_vertices - before_v,
            added_faces,
            before_f,
            mesh.n_faces);
    }
    if (!write_obj(output, &mesh)) {
        mesh_free(&mesh);
        return 1;
    }
    fprintf(stderr, "vkmesh: wrote %s (%" PRId64 " vertices, %" PRId64 " faces)\n", output, mesh.n_vertices, mesh.n_faces);
    mesh_free(&mesh);
    return 0;
}
