#include <vulkan/vulkan.h>

#include "vkmesh_accumulate_hole_components_spv.h"
#include "vkmesh_apply_orientation_flips_spv.h"
#include "vkmesh_assign_vertex_map_spv.h"
#include "vkmesh_assign_corner_vertices_spv.h"
#include "vkmesh_boundary_degree_owner_spv.h"
#include "vkmesh_compact_boundary_edges_spv.h"
#include "vkmesh_compact_face_pairs_spv.h"
#include "vkmesh_compact_faces_spv.h"
#include "vkmesh_compact_vertices_spv.h"
#include "vkmesh_compact_unique_simplify_edges_spv.h"
#include "vkmesh_copy_u32_spv.h"
#include "vkmesh_degenerate_faces_spv.h"
#include "vkmesh_expand_edges_spv.h"
#include "vkmesh_face_keys_spv.h"
#include "vkmesh_fill_u32_spv.h"
#include "vkmesh_component_area_spv.h"
#include "vkmesh_compress_parents_spv.h"
#include "vkmesh_init_u32_sequence_spv.h"
#include "vkmesh_init_orientation_state_spv.h"
#include "vkmesh_mark_boundary_edges_spv.h"
#include "vkmesh_mark_component_keep_spv.h"
#include "vkmesh_mark_duplicate_faces_spv.h"
#include "vkmesh_mark_hole_faces_spv.h"
#include "vkmesh_mark_hole_roots_spv.h"
#include "vkmesh_remap_faces_spv.h"
#include "vkmesh_scan_u32_stride_spv.h"
#include "vkmesh_seed_vertex_offsets_spv.h"
#include "vkmesh_sort_edges_spv.h"
#include "vkmesh_sort_face_keys_spv.h"
#include "vkmesh_simplify_best_edge_spv.h"
#include "vkmesh_simplify_collapse_edges_spv.h"
#include "vkmesh_simplify_edge_cost_spv.h"
#include "vkmesh_simplify_propagate_cost_spv.h"
#include "vkmesh_union_face_edges_spv.h"
#include "vkmesh_union_boundary_edges_spv.h"
#include "vkmesh_union_corner_edges_spv.h"
#include "vkmesh_union_orientation_edges_spv.h"
#include "vkmesh_unsigned_distance_spv.h"
#include "vkmesh_vertex_face_adjacency_spv.h"
#include "vkmesh_vertex_face_degree_spv.h"
#include "vkmesh_vertex_qem_spv.h"
#include "vkmesh_compress_orientation_state_spv.h"
#include "vkmesh_write_hole_faces_spv.h"
#include "vkmesh_write_hole_vertices_spv.h"
#include "vkmesh_write_repaired_mesh_spv.h"
#include "xatlas_c.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct vkmesh_mesh {
    float * vertices;
    float * uvs;
    int32_t * faces;
    int64_t n_vertices;
    int64_t n_faces;
    int64_t vertex_capacity;
    int64_t face_capacity;
    int has_uvs;
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

typedef struct vkmesh_face_key {
    int32_t v0;
    int32_t v1;
    int32_t v2;
    int32_t face;
} vkmesh_face_key;

typedef struct vkmesh_face_pair {
    int32_t f0;
    int32_t f1;
    uint32_t v0;
    uint32_t v1;
} vkmesh_face_pair;

typedef struct vkmesh_qem {
    double m[10];
} vkmesh_qem;

typedef struct vkmesh_simplify_edge {
    int32_t v0;
    int32_t v1;
    double cost;
    float pos[3];
} vkmesh_simplify_edge;

typedef struct vkmesh_bvh_tri {
    float bmin[3];
    float bmax[3];
    float centroid[3];
    uint32_t face;
} vkmesh_bvh_tri;

typedef struct vkmesh_bvh_node {
    float bmin[3];
    float bmax[3];
    uint32_t left;
    uint32_t meta;
} vkmesh_bvh_node;

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
    VkPipeline pipelines[45];
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkCommandBuffer command_buffer;
} vkmesh_vk;

typedef struct vkmesh_device_mesh {
    vkmesh_vk * vk;
    vkmesh_vk_buffer vertices;
    vkmesh_vk_buffer faces;
    uint32_t n_vertices;
    uint32_t n_faces;
} vkmesh_device_mesh;

typedef struct vkmesh_push {
    uint32_t n;
    uint32_t aux0;
    uint32_t aux1;
    uint32_t aux2;
    float eps;
    float rel_eps;
} vkmesh_push;

typedef enum vkmesh_pipeline_kind {
    VKMESH_PIPE_EXPAND_EDGES = 0,
    VKMESH_PIPE_DEGENERATE_FACES = 1,
    VKMESH_PIPE_FACE_KEYS = 2,
    VKMESH_PIPE_SORT_EDGES = 3,
    VKMESH_PIPE_SORT_FACE_KEYS = 4,
    VKMESH_PIPE_MARK_DUPLICATE_FACES = 5,
    VKMESH_PIPE_MARK_BOUNDARY_EDGES = 6,
    VKMESH_PIPE_BOUNDARY_DEGREE_OWNER = 7,
    VKMESH_PIPE_COMPACT_BOUNDARY_EDGES = 8,
    VKMESH_PIPE_COMPACT_FACE_PAIRS = 9,
    VKMESH_PIPE_COMPACT_FACES = 10,
    VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES = 11,
    VKMESH_PIPE_COPY_U32 = 12,
    VKMESH_PIPE_FILL_U32 = 13,
    VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS = 14,
    VKMESH_PIPE_APPLY_ORIENTATION_FLIPS = 15,
    VKMESH_PIPE_ASSIGN_VERTEX_MAP = 16,
    VKMESH_PIPE_REMAP_FACES = 17,
    VKMESH_PIPE_COMPACT_VERTICES = 18,
    VKMESH_PIPE_INIT_U32_SEQUENCE = 19,
    VKMESH_PIPE_INIT_ORIENTATION_STATE = 20,
    VKMESH_PIPE_SEED_VERTEX_OFFSETS = 21,
    VKMESH_PIPE_SCAN_U32_STRIDE = 22,
    VKMESH_PIPE_UNION_FACE_EDGES = 23,
    VKMESH_PIPE_UNION_BOUNDARY_EDGES = 24,
    VKMESH_PIPE_UNION_CORNER_EDGES = 25,
    VKMESH_PIPE_UNION_ORIENTATION_EDGES = 26,
    VKMESH_PIPE_COMPRESS_PARENTS = 27,
    VKMESH_PIPE_COMPRESS_ORIENTATION_STATE = 28,
    VKMESH_PIPE_VERTEX_FACE_DEGREE = 29,
    VKMESH_PIPE_VERTEX_FACE_ADJACENCY = 30,
    VKMESH_PIPE_VERTEX_QEM = 31,
    VKMESH_PIPE_SIMPLIFY_EDGE_COST = 32,
    VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST = 33,
    VKMESH_PIPE_SIMPLIFY_BEST_EDGE = 34,
    VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES = 35,
    VKMESH_PIPE_ASSIGN_CORNER_VERTICES = 36,
    VKMESH_PIPE_WRITE_REPAIRED_MESH = 37,
    VKMESH_PIPE_COMPONENT_AREA = 38,
    VKMESH_PIPE_MARK_COMPONENT_KEEP = 39,
    VKMESH_PIPE_MARK_HOLE_ROOTS = 40,
    VKMESH_PIPE_MARK_HOLE_FACES = 41,
    VKMESH_PIPE_WRITE_HOLE_VERTICES = 42,
    VKMESH_PIPE_WRITE_HOLE_FACES = 43,
    VKMESH_PIPE_UNSIGNED_DISTANCE = 44,
    VKMESH_PIPE_COUNT = 45,
} vkmesh_pipeline_kind;

typedef struct vkmesh_shader_blob {
    const unsigned char * data;
    unsigned int len;
    const char * name;
} vkmesh_shader_blob;

static const vkmesh_shader_blob vkmesh_shaders[VKMESH_PIPE_COUNT] = {
    { vkmesh_expand_edges_spv, vkmesh_expand_edges_spv_len, "expand_edges" },
    { vkmesh_degenerate_faces_spv, vkmesh_degenerate_faces_spv_len, "degenerate_faces" },
    { vkmesh_face_keys_spv, vkmesh_face_keys_spv_len, "face_keys" },
    { vkmesh_sort_edges_spv, vkmesh_sort_edges_spv_len, "sort_edges" },
    { vkmesh_sort_face_keys_spv, vkmesh_sort_face_keys_spv_len, "sort_face_keys" },
    { vkmesh_mark_duplicate_faces_spv, vkmesh_mark_duplicate_faces_spv_len, "mark_duplicate_faces" },
    { vkmesh_mark_boundary_edges_spv, vkmesh_mark_boundary_edges_spv_len, "mark_boundary_edges" },
    { vkmesh_boundary_degree_owner_spv, vkmesh_boundary_degree_owner_spv_len, "boundary_degree_owner" },
    { vkmesh_compact_boundary_edges_spv, vkmesh_compact_boundary_edges_spv_len, "compact_boundary_edges" },
    { vkmesh_compact_face_pairs_spv, vkmesh_compact_face_pairs_spv_len, "compact_face_pairs" },
    { vkmesh_compact_faces_spv, vkmesh_compact_faces_spv_len, "compact_faces" },
    { vkmesh_compact_unique_simplify_edges_spv, vkmesh_compact_unique_simplify_edges_spv_len, "compact_unique_simplify_edges" },
    { vkmesh_copy_u32_spv, vkmesh_copy_u32_spv_len, "copy_u32" },
    { vkmesh_fill_u32_spv, vkmesh_fill_u32_spv_len, "fill_u32" },
    { vkmesh_accumulate_hole_components_spv, vkmesh_accumulate_hole_components_spv_len, "accumulate_hole_components" },
    { vkmesh_apply_orientation_flips_spv, vkmesh_apply_orientation_flips_spv_len, "apply_orientation_flips" },
    { vkmesh_assign_vertex_map_spv, vkmesh_assign_vertex_map_spv_len, "assign_vertex_map" },
    { vkmesh_remap_faces_spv, vkmesh_remap_faces_spv_len, "remap_faces" },
    { vkmesh_compact_vertices_spv, vkmesh_compact_vertices_spv_len, "compact_vertices" },
    { vkmesh_init_u32_sequence_spv, vkmesh_init_u32_sequence_spv_len, "init_u32_sequence" },
    { vkmesh_init_orientation_state_spv, vkmesh_init_orientation_state_spv_len, "init_orientation_state" },
    { vkmesh_seed_vertex_offsets_spv, vkmesh_seed_vertex_offsets_spv_len, "seed_vertex_offsets" },
    { vkmesh_scan_u32_stride_spv, vkmesh_scan_u32_stride_spv_len, "scan_u32_stride" },
    { vkmesh_union_face_edges_spv, vkmesh_union_face_edges_spv_len, "union_face_edges" },
    { vkmesh_union_boundary_edges_spv, vkmesh_union_boundary_edges_spv_len, "union_boundary_edges" },
    { vkmesh_union_corner_edges_spv, vkmesh_union_corner_edges_spv_len, "union_corner_edges" },
    { vkmesh_union_orientation_edges_spv, vkmesh_union_orientation_edges_spv_len, "union_orientation_edges" },
    { vkmesh_compress_parents_spv, vkmesh_compress_parents_spv_len, "compress_parents" },
    { vkmesh_compress_orientation_state_spv, vkmesh_compress_orientation_state_spv_len, "compress_orientation_state" },
    { vkmesh_vertex_face_degree_spv, vkmesh_vertex_face_degree_spv_len, "vertex_face_degree" },
    { vkmesh_vertex_face_adjacency_spv, vkmesh_vertex_face_adjacency_spv_len, "vertex_face_adjacency" },
    { vkmesh_vertex_qem_spv, vkmesh_vertex_qem_spv_len, "vertex_qem" },
    { vkmesh_simplify_edge_cost_spv, vkmesh_simplify_edge_cost_spv_len, "simplify_edge_cost" },
    { vkmesh_simplify_propagate_cost_spv, vkmesh_simplify_propagate_cost_spv_len, "simplify_propagate_cost" },
    { vkmesh_simplify_best_edge_spv, vkmesh_simplify_best_edge_spv_len, "simplify_best_edge" },
    { vkmesh_simplify_collapse_edges_spv, vkmesh_simplify_collapse_edges_spv_len, "simplify_collapse_edges" },
    { vkmesh_assign_corner_vertices_spv, vkmesh_assign_corner_vertices_spv_len, "assign_corner_vertices" },
    { vkmesh_write_repaired_mesh_spv, vkmesh_write_repaired_mesh_spv_len, "write_repaired_mesh" },
    { vkmesh_component_area_spv, vkmesh_component_area_spv_len, "component_area" },
    { vkmesh_mark_component_keep_spv, vkmesh_mark_component_keep_spv_len, "mark_component_keep" },
    { vkmesh_mark_hole_roots_spv, vkmesh_mark_hole_roots_spv_len, "mark_hole_roots" },
    { vkmesh_mark_hole_faces_spv, vkmesh_mark_hole_faces_spv_len, "mark_hole_faces" },
    { vkmesh_write_hole_vertices_spv, vkmesh_write_hole_vertices_spv_len, "write_hole_vertices" },
    { vkmesh_write_hole_faces_spv, vkmesh_write_hole_faces_spv_len, "write_hole_faces" },
    { vkmesh_unsigned_distance_spv, vkmesh_unsigned_distance_spv_len, "unsigned_distance" },
};

static vkmesh_vk * g_active_vkmesh_vk = NULL;

static void mesh_free(vkmesh_mesh * mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->uvs);
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
    if (mesh->uvs != NULL) {
        float * next_uvs = (float *) realloc(mesh->uvs, (size_t) cap * 2u * sizeof(float));
        if (next_uvs == NULL) return 0;
        mesh->uvs = next_uvs;
    }
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
    if (mesh->uvs != NULL) {
        float * uv = mesh->uvs + (size_t) mesh->n_vertices * 2u;
        uv[0] = 0.0f;
        uv[1] = 0.0f;
    }
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

static int meshbin_checked_count(uint64_t n, uint64_t channels, size_t * out) {
    if (out == NULL || channels == 0 || n > (uint64_t) SIZE_MAX / channels) {
        return 0;
    }
    *out = (size_t) n * (size_t) channels;
    return 1;
}

static int load_meshbin(const char * path, vkmesh_mesh * mesh) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open meshbin %s: %s\n", path, strerror(errno));
        return 0;
    }

    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    int ok = 1;
    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t uv_count = 0;
    vkmesh_mesh out;
    memset(&out, 0, sizeof(out));
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_vertices, sizeof(n_vertices), 1, f) != 1 ||
        fread(&n_faces, sizeof(n_faces), 1, f) != 1 ||
        fread(&flags, sizeof(flags), 1, f) != 1 ||
        fread(&reserved, sizeof(reserved), 1, f) != 1 ||
        memcmp(magic, "TRLMESH1", 8) != 0 ||
        n_vertices == 0 || n_faces == 0 ||
        n_vertices > (uint64_t) INT64_MAX ||
        n_faces > (uint64_t) INT64_MAX ||
        !meshbin_checked_count(n_vertices, 3u, &vertex_count) ||
        !meshbin_checked_count(n_faces, 3u, &face_count) ||
        vertex_count > SIZE_MAX / sizeof(float) ||
        face_count > SIZE_MAX / sizeof(int32_t) ||
        ((flags & 1u) != 0 && (!meshbin_checked_count(n_vertices, 2u, &uv_count) ||
                               uv_count > SIZE_MAX / sizeof(float)))) {
        ok = 0;
    }
    (void) reserved;

    if (ok) {
        out.vertices = (float *) malloc(vertex_count * sizeof(float));
        out.faces = (int32_t *) malloc(face_count * sizeof(int32_t));
        if ((flags & 1u) != 0) {
            out.uvs = (float *) malloc(uv_count * sizeof(float));
        }
        out.n_vertices = (int64_t) n_vertices;
        out.n_faces = (int64_t) n_faces;
        out.vertex_capacity = out.n_vertices;
        out.face_capacity = out.n_faces;
        out.has_uvs = ((flags & 1u) != 0 && out.uvs != NULL) ? 1 : 0;
        if (out.vertices == NULL || out.faces == NULL || (((flags & 1u) != 0) && out.uvs == NULL)) {
            ok = 0;
        } else if (fread(out.vertices, sizeof(float), vertex_count, f) != vertex_count ||
                   fread(out.faces, sizeof(int32_t), face_count, f) != face_count ||
                   (((flags & 1u) != 0) && fread(out.uvs, sizeof(float), uv_count, f) != uv_count)) {
            ok = 0;
        }
    }
    if (ok) {
        for (size_t i = 0; i < face_count; ++i) {
            if (out.faces[i] < 0 || (uint64_t) out.faces[i] >= n_vertices) {
                ok = 0;
                break;
            }
        }
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "vkmesh: failed to read meshbin %s\n", path);
        mesh_free(&out);
        return 0;
    }
    *mesh = out;
    return 1;
}

static int write_meshbin(const char * path, const vkmesh_mesh * mesh) {
    if (path == NULL || path[0] == '\0' ||
        mesh == NULL || mesh->vertices == NULL || mesh->faces == NULL ||
        mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        fprintf(stderr, "vkmesh: cannot write empty meshbin %s\n", path != NULL ? path : "(null)");
        return 0;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s for writing: %s\n", path, strerror(errno));
        return 0;
    }

    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint64_t n_vertices = (uint64_t) mesh->n_vertices;
    const uint64_t n_faces = (uint64_t) mesh->n_faces;
    const uint32_t flags = (mesh->has_uvs && mesh->uvs != NULL) ? 1u : 0u;
    const uint32_t reserved = 0;
    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t uv_count = 0;
    int ok =
        meshbin_checked_count(n_vertices, 3u, &vertex_count) &&
        meshbin_checked_count(n_faces, 3u, &face_count) &&
        vertex_count <= SIZE_MAX / sizeof(float) &&
        face_count <= SIZE_MAX / sizeof(int32_t) &&
        (flags == 0u || (meshbin_checked_count(n_vertices, 2u, &uv_count) &&
                         uv_count <= SIZE_MAX / sizeof(float)));
    if (ok) {
        ok = fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
             fwrite(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
             fwrite(&n_faces, sizeof(n_faces), 1, f) == 1 &&
             fwrite(&flags, sizeof(flags), 1, f) == 1 &&
             fwrite(&reserved, sizeof(reserved), 1, f) == 1 &&
             fwrite(mesh->vertices, sizeof(float), vertex_count, f) == vertex_count &&
             fwrite(mesh->faces, sizeof(int32_t), face_count, f) == face_count &&
             (flags == 0u || fwrite(mesh->uvs, sizeof(float), uv_count, f) == uv_count);
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "vkmesh: failed to write meshbin %s\n", path);
        return 0;
    }
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
static int vkmesh_vk_init(vkmesh_vk * vk);

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
    for (uint32_t i = 0; i < VKMESH_PIPE_COUNT; ++i) {
        if (vk->pipelines[i] != VK_NULL_HANDLE) vkDestroyPipeline(vk->device, vk->pipelines[i], NULL);
    }
    if (vk->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE) vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

static int vkmesh_acquire_vk(vkmesh_vk ** vk_out, vkmesh_vk * local_vk, int * owns_vk) {
    if (vk_out == NULL || local_vk == NULL || owns_vk == NULL) return 0;
    if (g_active_vkmesh_vk != NULL) {
        *vk_out = g_active_vkmesh_vk;
        *owns_vk = 0;
        return 1;
    }
    if (!vkmesh_vk_init(local_vk)) {
        vkmesh_vk_destroy(local_vk);
        return 0;
    }
    *vk_out = local_vk;
    *owns_vk = 1;
    return 1;
}

static int vkmesh_create_compute_pipeline(
    vkmesh_vk * vk,
    const vkmesh_shader_blob * blob,
    VkPipeline * pipeline) {
    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = blob->len;
    shader_info.pCode = (const uint32_t *) blob->data;
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk->device, &shader_info, NULL, &shader) != VK_SUCCESS) {
        fprintf(stderr, "vkmesh: failed to create shader module for %s\n", blob->name);
        return 0;
    }

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
    VkResult result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
    vkDestroyShaderModule(vk->device, shader, NULL);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkmesh: failed to create compute pipeline for %s\n", blob->name);
        return 0;
    }
    return 1;
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

    for (uint32_t i = 0; i < VKMESH_PIPE_COUNT; ++i) {
        if (!vkmesh_create_compute_pipeline(vk, &vkmesh_shaders[i], &vk->pipelines[i])) return 0;
    }

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk->device, &descriptor_pool_info, NULL, &vk->descriptor_pool) != VK_SUCCESS) return 0;

    VkDescriptorSetAllocateInfo set_alloc;
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = vk->descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &vk->descriptor_set_layout;
    if (vkAllocateDescriptorSets(vk->device, &set_alloc, &vk->descriptor_set) != VK_SUCCESS) return 0;

    VkCommandBufferAllocateInfo cmd_alloc;
    memset(&cmd_alloc, 0, sizeof(cmd_alloc));
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(vk->device, &cmd_alloc, &vk->command_buffer) != VK_SUCCESS) return 0;
    return 1;
}

static void vkmesh_update_descriptor_set(vkmesh_vk * vk, vkmesh_vk_buffer buffers[4]) {
    VkDescriptorBufferInfo infos[4];
    VkWriteDescriptorSet writes[4];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        infos[i].buffer = buffers[i].buffer;
        infos[i].offset = 0;
        infos[i].range = (VkDeviceSize) buffers[i].bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = vk->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, 4, writes, 0, NULL);
}

static int vkmesh_dispatch(
    vkmesh_vk * vk,
    vkmesh_pipeline_kind pipeline_kind,
    vkmesh_vk_buffer buffers[4],
    const vkmesh_push * push,
    uint32_t groups_x) {
    vkmesh_update_descriptor_set(vk, buffers);
    if (vkResetCommandBuffer(vk->command_buffer, 0) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(vk->command_buffer, &begin) != VK_SUCCESS) return 0;
    vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[pipeline_kind]);
    vkCmdBindDescriptorSets(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &vk->descriptor_set, 0, NULL);
    vkCmdPushConstants(vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);
    vkCmdDispatch(vk->command_buffer, groups_x, 1, 1);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        vk->command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(vk->command_buffer) != VK_SUCCESS) return 0;

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk->command_buffer;
    if (vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) return 0;
    vkQueueWaitIdle(vk->queue);
    return 1;
}

static size_t next_power_of_two_size(size_t n) {
    size_t p = 1;
    while (p < n) {
        if (p > SIZE_MAX / 2) return 0;
        p *= 2;
    }
    return p;
}

static int vkmesh_sort_records_vulkan(
    vkmesh_vk * vk,
    vkmesh_vk_buffer * records,
    vkmesh_vk_buffer * dummy,
    size_t record_count,
    vkmesh_pipeline_kind pipeline_kind) {
    if (record_count <= 1) return 1;
    if (record_count > UINT32_MAX) return 0;

    vkmesh_vk_buffer buffers[4];
    buffers[0] = *records;
    buffers[1] = *dummy;
    buffers[2] = *dummy;
    buffers[3] = *dummy;
    vkmesh_update_descriptor_set(vk, buffers);
    if (vkResetCommandBuffer(vk->command_buffer, 0) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(vk->command_buffer, &begin) != VK_SUCCESS) return 0;

    vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipelines[pipeline_kind]);
    vkCmdBindDescriptorSets(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_layout, 0, 1, &vk->descriptor_set, 0, NULL);
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    uint32_t groups = (uint32_t) ((record_count + 255u) / 256u);
    for (size_t k = 2u; k <= record_count; k <<= 1u) {
        for (size_t j = k >> 1u; j > 0u; j >>= 1u) {
            vkmesh_push push;
            memset(&push, 0, sizeof(push));
            push.n = (uint32_t) record_count;
            push.aux0 = (uint32_t) k;
            push.aux1 = (uint32_t) j;
            vkCmdPushConstants(vk->command_buffer, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(vk->command_buffer, groups, 1, 1);
            vkCmdPipelineBarrier(
                vk->command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }
    if (vkEndCommandBuffer(vk->command_buffer) != VK_SUCCESS) return 0;

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk->command_buffer;
    if (vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) return 0;
    vkQueueWaitIdle(vk->queue);
    return 1;
}

static int expand_edges_vulkan(const vkmesh_mesh * mesh, vkmesh_edge ** edges_out, int64_t * edge_count_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    int ok = 0;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t edge_count = (size_t) mesh->n_faces * 3u;
    const size_t edge_sort_count = next_power_of_two_size(edge_count);
    if (edge_sort_count == 0 || edge_sort_count > UINT32_MAX) goto cleanup;
    const size_t edges_bytes = edge_sort_count * sizeof(vkmesh_edge);
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, edges_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);
    vkmesh_edge * mapped_edges = (vkmesh_edge *) buffers[2].mapped;
    for (size_t i = edge_count; i < edge_sort_count; ++i) {
        mapped_edges[i].min_v = UINT32_MAX;
        mapped_edges[i].max_v = UINT32_MAX;
        mapped_edges[i].a = UINT32_MAX;
        mapped_edges[i].b = UINT32_MAX;
        mapped_edges[i].face = UINT32_MAX;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_EXPAND_EDGES, buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &buffers[2], &buffers[3], edge_sort_count, VKMESH_PIPE_SORT_EDGES)) goto cleanup;

    vkmesh_edge * edges = (vkmesh_edge *) malloc(edge_count * sizeof(vkmesh_edge));
    if (edges == NULL) goto cleanup;
    memcpy(edges, buffers[2].mapped, edge_count * sizeof(vkmesh_edge));
    *edges_out = edges;
    *edge_count_out = (int64_t) edge_count;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int expand_edges_device(
    vkmesh_vk * vk,
    const vkmesh_mesh * mesh,
    vkmesh_vk_buffer * faces_buffer,
    vkmesh_vk_buffer * vertices_buffer,
    vkmesh_vk_buffer * edges_buffer_out,
    vkmesh_vk_buffer * dummy_buffer_out,
    size_t * edge_count_out,
    size_t * edge_sort_count_out) {
    memset(edges_buffer_out, 0, sizeof(*edges_buffer_out));
    memset(dummy_buffer_out, 0, sizeof(*dummy_buffer_out));
    *edge_count_out = 0;
    *edge_sort_count_out = 0;
    if (vk == NULL || mesh == NULL || faces_buffer == NULL || vertices_buffer == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) {
        return 0;
    }

    const size_t edge_count = (size_t) mesh->n_faces * 3u;
    const size_t edge_sort_count = next_power_of_two_size(edge_count);
    if (edge_sort_count == 0 || edge_sort_count > UINT32_MAX) return 0;
    const size_t edges_bytes = edge_sort_count * sizeof(vkmesh_edge);
    if (!vk_buffer_create(vk, edges_bytes, edges_buffer_out) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), dummy_buffer_out)) {
        vk_buffer_destroy(vk, edges_buffer_out);
        vk_buffer_destroy(vk, dummy_buffer_out);
        return 0;
    }

    vkmesh_edge * mapped_edges = (vkmesh_edge *) edges_buffer_out->mapped;
    for (size_t i = edge_count; i < edge_sort_count; ++i) {
        mapped_edges[i].min_v = UINT32_MAX;
        mapped_edges[i].max_v = UINT32_MAX;
        mapped_edges[i].a = UINT32_MAX;
        mapped_edges[i].b = UINT32_MAX;
        mapped_edges[i].face = UINT32_MAX;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = *faces_buffer;
    buffers[1] = *vertices_buffer;
    buffers[2] = *edges_buffer_out;
    buffers[3] = *dummy_buffer_out;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_EXPAND_EDGES, buffers, &push, groups) ||
        !vkmesh_sort_records_vulkan(vk, edges_buffer_out, dummy_buffer_out, edge_sort_count, VKMESH_PIPE_SORT_EDGES)) {
        vk_buffer_destroy(vk, edges_buffer_out);
        vk_buffer_destroy(vk, dummy_buffer_out);
        return 0;
    }
    *edge_count_out = edge_count;
    *edge_sort_count_out = edge_sort_count;
    return 1;
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

static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static double face_area(const vkmesh_mesh * mesh, int64_t face_id) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    const float * a = mesh->vertices + (size_t) f[0] * 3u;
    const float * b = mesh->vertices + (size_t) f[1] * 3u;
    const float * c = mesh->vertices + (size_t) f[2] * 3u;
    double abx = (double) b[0] - (double) a[0];
    double aby = (double) b[1] - (double) a[1];
    double abz = (double) b[2] - (double) a[2];
    double acx = (double) c[0] - (double) a[0];
    double acy = (double) c[1] - (double) a[1];
    double acz = (double) c[2] - (double) a[2];
    double cx = aby * acz - abz * acy;
    double cy = abz * acx - abx * acz;
    double cz = abx * acy - aby * acx;
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
}

static int mesh_remove_unreferenced_vertices(vkmesh_mesh * mesh) {
    if (mesh->n_vertices <= 0) return 1;
    uint8_t * referenced = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    int32_t * map = (int32_t *) malloc((size_t) mesh->n_vertices * sizeof(int32_t));
    if (referenced == NULL || map == NULL) {
        free(referenced);
        free(map);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        referenced[f[0]] = 1u;
        referenced[f[1]] = 1u;
        referenced[f[2]] = 1u;
    }
    int64_t new_count = 0;
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        if (referenced[i]) map[i] = (int32_t) new_count++;
        else map[i] = -1;
    }

    float * new_vertices = NULL;
    float * new_uvs = NULL;
    if (new_count > 0) {
        new_vertices = (float *) malloc((size_t) new_count * 3u * sizeof(float));
        if (new_vertices == NULL) {
            free(referenced);
            free(map);
            return 0;
        }
        if (mesh->has_uvs && mesh->uvs != NULL) {
            new_uvs = (float *) malloc((size_t) new_count * 2u * sizeof(float));
            if (new_uvs == NULL) {
                free(new_vertices);
                free(referenced);
                free(map);
                return 0;
            }
        }
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) {
        int32_t dst = map[i];
        if (dst < 0) continue;
        memcpy(new_vertices + (size_t) dst * 3u, mesh->vertices + (size_t) i * 3u, 3u * sizeof(float));
        if (new_uvs != NULL) {
            memcpy(new_uvs + (size_t) dst * 2u, mesh->uvs + (size_t) i * 2u, 2u * sizeof(float));
        }
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        int32_t * f = mesh->faces + (size_t) i * 3u;
        f[0] = map[f[0]];
        f[1] = map[f[1]];
        f[2] = map[f[2]];
    }
    free(mesh->vertices);
    free(mesh->uvs);
    mesh->vertices = new_vertices;
    mesh->uvs = new_uvs;
    mesh->n_vertices = new_count;
    mesh->vertex_capacity = new_count;
    if (new_uvs == NULL) mesh->has_uvs = 0;
    free(referenced);
    free(map);
    return 1;
}

static int mesh_remove_faces_by_mask(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces) {
    int64_t keep_count = 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if (keep_mask[i]) ++keep_count;
    }
    int32_t * new_faces = NULL;
    if (keep_count > 0) {
        new_faces = (int32_t *) malloc((size_t) keep_count * 3u * sizeof(int32_t));
        if (new_faces == NULL) return 0;
    }
    int64_t dst = 0;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if (!keep_mask[i]) continue;
        memcpy(new_faces + (size_t) dst * 3u, mesh->faces + (size_t) i * 3u, 3u * sizeof(int32_t));
        ++dst;
    }
    if (removed_faces != NULL) *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->faces);
    mesh->faces = new_faces;
    mesh->n_faces = keep_count;
    mesh->face_capacity = keep_count;
    return mesh_remove_unreferenced_vertices(mesh);
}

static int mesh_remove_faces_by_mask_vulkan(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces);

static void vkmesh_device_mesh_destroy(vkmesh_device_mesh * dm) {
    if (dm == NULL || dm->vk == NULL) return;
    vk_buffer_destroy(dm->vk, &dm->vertices);
    vk_buffer_destroy(dm->vk, &dm->faces);
    memset(dm, 0, sizeof(*dm));
}

static int vkmesh_device_mesh_upload(vkmesh_vk * vk, const vkmesh_mesh * mesh, vkmesh_device_mesh * dm) {
    if (vk == NULL || mesh == NULL || dm == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    memset(dm, 0, sizeof(*dm));
    dm->vk = vk;
    dm->n_vertices = (uint32_t) mesh->n_vertices;
    dm->n_faces = (uint32_t) mesh->n_faces;
    const size_t vertices_bytes = (size_t) dm->n_vertices * 3u * sizeof(float);
    const size_t faces_bytes = (size_t) dm->n_faces * 3u * sizeof(int32_t);
    if (!vk_buffer_create(vk, vertices_bytes, &dm->vertices) ||
        !vk_buffer_create(vk, faces_bytes, &dm->faces)) {
        vkmesh_device_mesh_destroy(dm);
        return 0;
    }
    memcpy(dm->vertices.mapped, mesh->vertices, vertices_bytes);
    memcpy(dm->faces.mapped, mesh->faces, faces_bytes);
    return 1;
}

static int vkmesh_device_mesh_download(vkmesh_device_mesh * dm, vkmesh_mesh * mesh) {
    if (dm == NULL || mesh == NULL || dm->vk == NULL ||
        dm->n_vertices == 0u || dm->n_faces == 0u) {
        return 0;
    }
    float * vertices = (float *) malloc((size_t) dm->n_vertices * 3u * sizeof(float));
    int32_t * faces = (int32_t *) malloc((size_t) dm->n_faces * 3u * sizeof(int32_t));
    if (vertices == NULL || faces == NULL) {
        free(vertices);
        free(faces);
        return 0;
    }
    memcpy(vertices, dm->vertices.mapped, (size_t) dm->n_vertices * 3u * sizeof(float));
    memcpy(faces, dm->faces.mapped, (size_t) dm->n_faces * 3u * sizeof(int32_t));
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = vertices;
    mesh->uvs = NULL;
    mesh->faces = faces;
    mesh->n_vertices = (int64_t) dm->n_vertices;
    mesh->n_faces = (int64_t) dm->n_faces;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 0;
    return 1;
}

static int compact_device_mesh_from_flags(
    vkmesh_device_mesh * dm,
    vkmesh_vk_buffer * flags_buffer,
    uint32_t flags_offset,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || flags_buffer == NULL ||
        dm->n_faces == 0u || dm->n_vertices == 0u) {
        return 0;
    }
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer out_faces;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer vertex_map;
    vkmesh_vk_buffer out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&counter, 0, sizeof(counter));
    memset(&vertex_map, 0, sizeof(vertex_map));
    memset(&out_vertices, 0, sizeof(out_vertices));

    const size_t faces_bytes = (size_t) dm->n_faces * 3u * sizeof(int32_t);
    const size_t vertex_map_bytes = (size_t) dm->n_vertices * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) dm->n_vertices * 3u * sizeof(float);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &out_faces) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter) ||
        !vk_buffer_create(vk, vertex_map_bytes, &vertex_map) ||
        !vk_buffer_create(vk, vertices_bytes, &out_vertices)) {
        goto cleanup;
    }
    memset(vertex_map.mapped, 0xff, vertex_map_bytes);

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = dm->faces;
    compact_buffers[1] = *flags_buffer;
    compact_buffers[2] = out_faces;
    compact_buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.aux0 = flags_offset;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACES, compact_buffers, &push, groups)) goto cleanup;

    uint32_t keep_count = ((const uint32_t *) counter.mapped)[0];
    if (keep_count == 0u || keep_count > dm->n_faces) goto cleanup;
    if (removed_faces != NULL) *removed_faces = dm->n_faces - keep_count;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = out_faces;
    assign_buffers[1] = vertex_map;
    assign_buffers[2] = counter;
    assign_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_VERTEX_MAP, assign_buffers, &push, groups)) goto cleanup;
    uint32_t vertex_count = ((const uint32_t *) counter.mapped)[0];
    if (vertex_count == 0u || vertex_count > dm->n_vertices) goto cleanup;

    vkmesh_vk_buffer remap_buffers[4];
    remap_buffers[0] = out_faces;
    remap_buffers[1] = vertex_map;
    remap_buffers[2] = counter;
    remap_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_REMAP_FACES, remap_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer vertex_buffers[4];
    vertex_buffers[0] = dm->vertices;
    vertex_buffers[1] = vertex_map;
    vertex_buffers[2] = out_vertices;
    vertex_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_vertices;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_VERTICES, vertex_buffers, &push, groups)) goto cleanup;

    vk_buffer_destroy(vk, &dm->faces);
    vk_buffer_destroy(vk, &dm->vertices);
    dm->faces = out_faces;
    dm->vertices = out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&out_vertices, 0, sizeof(out_vertices));
    dm->n_faces = keep_count;
    dm->n_vertices = vertex_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &out_faces);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &vertex_map);
    vk_buffer_destroy(vk, &out_vertices);
    return ok;
}

static int compact_mesh_from_device_flags(
    vkmesh_vk * vk,
    const vkmesh_mesh * mesh,
    vkmesh_vk_buffer * faces_buffer,
    vkmesh_vk_buffer * vertices_buffer,
    vkmesh_vk_buffer * flags_buffer,
    uint32_t flags_offset,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (vk == NULL || mesh == NULL || faces_buffer == NULL || vertices_buffer == NULL || flags_buffer == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk_buffer out_faces;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer vertex_map;
    vkmesh_vk_buffer out_vertices;
    memset(&out_faces, 0, sizeof(out_faces));
    memset(&counter, 0, sizeof(counter));
    memset(&vertex_map, 0, sizeof(vertex_map));
    memset(&out_vertices, 0, sizeof(out_vertices));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertex_map_bytes = (size_t) mesh->n_vertices * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &out_faces) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter) ||
        !vk_buffer_create(vk, vertex_map_bytes, &vertex_map) ||
        !vk_buffer_create(vk, vertices_bytes, &out_vertices)) {
        goto cleanup;
    }
    memset(vertex_map.mapped, 0xff, vertex_map_bytes);

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = *faces_buffer;
    compact_buffers[1] = *flags_buffer;
    compact_buffers[2] = out_faces;
    compact_buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.aux0 = flags_offset;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACES, compact_buffers, &push, groups)) goto cleanup;

    uint32_t keep_count_u32 = ((const uint32_t *) counter.mapped)[0];
    if (keep_count_u32 == 0) {
        ok = 1;
        goto cleanup;
    }

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = out_faces;
    assign_buffers[1] = vertex_map;
    assign_buffers[2] = counter;
    assign_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count_u32 * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_VERTEX_MAP, assign_buffers, &push, groups)) goto cleanup;

    uint32_t vertex_count_u32 = ((const uint32_t *) counter.mapped)[0];

    vkmesh_vk_buffer remap_buffers[4];
    remap_buffers[0] = out_faces;
    remap_buffers[1] = vertex_map;
    remap_buffers[2] = counter;
    remap_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = keep_count_u32 * 3u;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_REMAP_FACES, remap_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer vertex_buffers[4];
    vertex_buffers[0] = *vertices_buffer;
    vertex_buffers[1] = vertex_map;
    vertex_buffers[2] = out_vertices;
    vertex_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_vertices;
    groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_VERTICES, vertex_buffers, &push, groups)) goto cleanup;

    int32_t * compacted_faces = (int32_t *) malloc((size_t) keep_count_u32 * 3u * sizeof(int32_t));
    float * compacted_vertices = (float *) malloc((size_t) vertex_count_u32 * 3u * sizeof(float));
    if (compacted_faces == NULL || compacted_vertices == NULL) {
        free(compacted_faces);
        free(compacted_vertices);
        goto cleanup;
    }
    memcpy(compacted_faces, out_faces.mapped, (size_t) keep_count_u32 * 3u * sizeof(int32_t));
    memcpy(compacted_vertices, out_vertices.mapped, (size_t) vertex_count_u32 * 3u * sizeof(float));
    *vertices_out = compacted_vertices;
    *vertex_count_out = (int64_t) vertex_count_u32;
    *faces_out = compacted_faces;
    *face_count_out = (int64_t) keep_count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &out_faces);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &vertex_map);
    vk_buffer_destroy(vk, &out_vertices);
    return ok;
}

static int mesh_remove_faces_by_mask_vulkan(vkmesh_mesh * mesh, const uint8_t * keep_mask, int * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0;
    if (mesh == NULL || keep_mask == NULL ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer flags_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&flags_buffer, 0, sizeof(flags_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, flags_bytes, &flags_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);
    uint32_t * flags = (uint32_t *) flags_buffer.mapped;
    for (int64_t i = 0; i < mesh->n_faces; ++i) flags[i] = keep_mask[i] ? 1u : 0u;

    float * new_vertices = NULL;
    int64_t new_vertex_count = 0;
    int32_t * new_faces = NULL;
    int64_t new_face_count = 0;
    if (!compact_mesh_from_device_flags(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &flags_buffer,
            0u,
            &new_vertices,
            &new_vertex_count,
            &new_faces,
            &new_face_count)) {
        goto cleanup;
    }

    if (removed_faces != NULL) *removed_faces = (int) (mesh->n_faces - new_face_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = new_face_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = new_face_count;
    mesh->has_uvs = 0;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &flags_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_remove_duplicate_faces(vkmesh_device_mesh * dm, uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces <= 1u) return 1;
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer keys;
    vkmesh_vk_buffer flags;
    vkmesh_vk_buffer dummy;
    memset(&keys, 0, sizeof(keys));
    memset(&flags, 0, sizeof(flags));
    memset(&dummy, 0, sizeof(dummy));
    int ok = 0;
    size_t sort_count = next_power_of_two_size((size_t) dm->n_faces);
    if (sort_count == 0 || sort_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, sort_count * sizeof(vkmesh_face_key), &keys) ||
        !vk_buffer_create(vk, (size_t) dm->n_faces * sizeof(uint32_t), &flags) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy)) {
        goto cleanup;
    }
    vkmesh_face_key * mapped = (vkmesh_face_key *) keys.mapped;
    for (size_t i = (size_t) dm->n_faces; i < sort_count; ++i) {
        mapped[i].v0 = -1;
        mapped[i].v1 = -1;
        mapped[i].v2 = -1;
        mapped[i].face = -1;
    }

    vkmesh_vk_buffer key_buffers[4];
    key_buffers[0] = dm->faces;
    key_buffers[1] = dummy;
    key_buffers[2] = keys;
    key_buffers[3] = flags;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FACE_KEYS, key_buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &keys, &dummy, sort_count, VKMESH_PIPE_SORT_FACE_KEYS)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = keys;
    mark_buffers[1] = dummy;
    mark_buffers[2] = flags;
    mark_buffers[3] = dummy;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) sort_count;
    push.aux0 = dm->n_faces;
    groups = ((uint32_t) sort_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_DUPLICATE_FACES, mark_buffers, &push, groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &keys);
    vk_buffer_destroy(vk, &flags);
    vk_buffer_destroy(vk, &dummy);
    return ok;
}

static int vkmesh_device_remove_degenerate_faces(
    vkmesh_device_mesh * dm,
    float abs_thresh,
    float rel_thresh,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u) return 0;
    vkmesh_vk * vk = dm->vk;
    vkmesh_vk_buffer flags;
    vkmesh_vk_buffer dummy;
    memset(&flags, 0, sizeof(flags));
    memset(&dummy, 0, sizeof(dummy));
    int ok = 0;
    if (!vk_buffer_create(vk, (size_t) dm->n_faces * sizeof(uint32_t), &flags) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy)) {
        goto cleanup;
    }
    vkmesh_vk_buffer buffers[4];
    buffers[0] = dm->faces;
    buffers[1] = dm->vertices;
    buffers[2] = flags;
    buffers[3] = dummy;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.eps = abs_thresh;
    push.rel_eps = rel_thresh;
    uint32_t groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_DEGENERATE_FACES, buffers, &push, groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &flags);
    vk_buffer_destroy(vk, &dummy);
    return ok;
}

static int vkmesh_log_duplicate_degenerate_device_cluster(
    vkmesh_mesh * mesh,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t duplicate_removed = 0u;
    uint32_t degenerate_removed = 0u;
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_remove_duplicate_faces(&dm, &duplicate_removed)) goto cleanup;
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, duplicate_removed, dm.n_faces);
    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(&dm, degenerate_abs, degenerate_rel, &degenerate_removed)) goto cleanup;
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, degenerate_removed, dm.n_faces);
    }
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_remove_small_connected_components(
    vkmesh_device_mesh * dm,
    float min_area,
    uint32_t * removed_faces) {
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u) return 0;
    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer area_bits;
    vkmesh_vk_buffer keep_flags;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&area_bits, 0, sizeof(area_bits));
    memset(&keep_flags, 0, sizeof(keep_flags));

    const size_t parents_bytes = (size_t) dm->n_faces * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;
    if (!vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, parents_bytes, &area_bits) ||
        !vk_buffer_create(vk, parents_bytes, &keep_flags)) {
        goto cleanup;
    }
    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &edge_dummy, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    uint32_t face_groups = (dm->n_faces + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = parents;
    union_buffers[2] = changed;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 32; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = dm->n_faces;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_FACE_EDGES, union_buffers, &push, edge_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = dm->n_faces;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, face_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    memset(area_bits.mapped, 0, parents_bytes);
    vkmesh_vk_buffer area_buffers[4];
    area_buffers[0] = dm->faces;
    area_buffers[1] = dm->vertices;
    area_buffers[2] = parents;
    area_buffers[3] = area_bits;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPONENT_AREA, area_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = parents;
    mark_buffers[1] = area_bits;
    mark_buffers[2] = keep_flags;
    mark_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = dm->n_faces;
    push.eps = min_area;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_COMPONENT_KEEP, mark_buffers, &push, face_groups)) goto cleanup;
    if (!compact_device_mesh_from_flags(dm, &keep_flags, 0u, removed_faces)) goto cleanup;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &area_bits);
    vk_buffer_destroy(vk, &keep_flags);
    return ok;
}

static int vkmesh_device_repair_non_manifold_edges(
    vkmesh_device_mesh * dm,
    uint32_t * old_vertices,
    uint32_t * new_vertices) {
    if (old_vertices != NULL) *old_vertices = dm != NULL ? dm->n_vertices : 0u;
    if (new_vertices != NULL) *new_vertices = dm != NULL ? dm->n_vertices : 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer repair_data;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&repair_data, 0, sizeof(repair_data));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t face_count = dm->n_faces;
    const uint32_t corner_count = face_count * 3u;
    const uint32_t root_map_words = corner_count;
    const uint32_t counter_word = root_map_words;
    const uint32_t out_vertices_word = counter_word + 1u;
    const uint32_t out_faces_word = out_vertices_word + corner_count * 3u;
    const size_t repair_words = (size_t) out_faces_word + (size_t) corner_count;
    if (repair_words > UINT32_MAX) return 0;

    const size_t parents_bytes = (size_t) corner_count * sizeof(uint32_t);
    const size_t repair_bytes = repair_words * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, repair_bytes, &repair_data)) {
        goto cleanup;
    }

    if (!expand_edges_device(
            vk,
            &view,
            &dm->faces,
            &dm->vertices,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    uint32_t corner_groups = (corner_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = dm->faces;
    union_buffers[2] = parents;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 128; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_CORNER_EDGES, union_buffers, &push, edge_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = corner_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, corner_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    uint32_t * repair_words_u32 = (uint32_t *) repair_data.mapped;
    memset(repair_words_u32, 0xff, (size_t) root_map_words * sizeof(uint32_t));
    repair_words_u32[counter_word] = 0u;

    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = parents;
    assign_buffers[1] = repair_data;
    assign_buffers[2] = changed;
    assign_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_CORNER_VERTICES, assign_buffers, &push, corner_groups)) goto cleanup;

    uint32_t vertex_count = repair_words_u32[counter_word];
    if (vertex_count == 0u || vertex_count > corner_count) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) face_count * 3u * sizeof(int32_t), &out_faces)) {
        goto cleanup;
    }

    vkmesh_vk_buffer write_buffers[4];
    write_buffers[0] = dm->faces;
    write_buffers[1] = dm->vertices;
    write_buffers[2] = parents;
    write_buffers[3] = repair_data;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    push.aux1 = out_vertices_word;
    push.aux2 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_REPAIRED_MESH, write_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = repair_data;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = changed;
    copy_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    push.aux0 = out_vertices_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    push.aux0 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vk_buffer_destroy(vk, &dm->vertices);
    vk_buffer_destroy(vk, &dm->faces);
    dm->vertices = out_vertices;
    dm->faces = out_faces;
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));
    dm->n_vertices = vertex_count;
    dm->n_faces = face_count;
    if (new_vertices != NULL) *new_vertices = vertex_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &repair_data);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    return ok;
}

static int vkmesh_device_unify_face_orientations(vkmesh_device_mesh * dm, uint32_t * flipped_faces) {
    if (flipped_faces != NULL) *flipped_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u || dm->n_faces > UINT32_MAX / 2u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer state;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer counter;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&state, 0, sizeof(state));
    memset(&changed, 0, sizeof(changed));
    memset(&counter, 0, sizeof(counter));

    const uint32_t face_count = dm->n_faces;
    const size_t state_bytes = (size_t) face_count * sizeof(uint32_t);
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, state_bytes, &state) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }

    if (!expand_edges_device(
            vk,
            &view,
            &dm->faces,
            &dm->vertices,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = state;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_ORIENTATION_STATE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = dm->faces;
    union_buffers[2] = state;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = state;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    int converged = 0;
    for (int iter = 0; iter < 512; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_ORIENTATION_EDGES, union_buffers, &push, edge_groups)) goto cleanup;

        memset(&push, 0, sizeof(push));
        push.n = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_ORIENTATION_STATE, compress_buffers, &push, face_groups)) goto cleanup;

        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer apply_buffers[4];
    apply_buffers[0] = dm->faces;
    apply_buffers[1] = state;
    apply_buffers[2] = counter;
    apply_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_APPLY_ORIENTATION_FLIPS, apply_buffers, &push, face_groups)) goto cleanup;

    if (flipped_faces != NULL) *flipped_faces = ((const uint32_t *) counter.mapped)[0];
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &state);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &counter);
    return ok;
}

static int compute_duplicate_compacted_mesh_vulkan(
    const vkmesh_mesh * mesh,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }
    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    int ok = 0;
    size_t sort_count = next_power_of_two_size((size_t) mesh->n_faces);
    if (sort_count == 0 || sort_count > UINT32_MAX) goto cleanup;
    size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    size_t keys_bytes = sort_count * sizeof(vkmesh_face_key);
    size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, keys_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);
    vkmesh_face_key * mapped = (vkmesh_face_key *) buffers[2].mapped;
    for (size_t i = (size_t) mesh->n_faces; i < sort_count; ++i) {
        mapped[i].v0 = -1;
        mapped[i].v1 = -1;
        mapped[i].v2 = -1;
        mapped[i].face = -1;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FACE_KEYS, buffers, &push, groups)) goto cleanup;
    if (!vkmesh_sort_records_vulkan(vk, &buffers[2], &buffers[1], sort_count, VKMESH_PIPE_SORT_FACE_KEYS)) goto cleanup;

    vkmesh_vk_buffer mark_buffers[4];
    mark_buffers[0] = buffers[2];
    mark_buffers[1] = buffers[1];
    mark_buffers[2] = buffers[3];
    mark_buffers[3] = buffers[1];
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) sort_count;
    push.aux0 = (uint32_t) mesh->n_faces;
    groups = (uint32_t) ((sort_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_DUPLICATE_FACES, mark_buffers, &push, groups)) goto cleanup;

    if (!compact_mesh_from_device_flags(
            vk, mesh, &buffers[0], &buffers[1], &buffers[3], 0u,
            vertices_out, vertex_count_out, faces_out, face_count_out)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_remove_duplicate_faces(vkmesh_mesh * mesh, int * removed_faces) {
    *removed_faces = 0;
    if (mesh->n_faces <= 1) return 1;
    float * new_vertices = NULL;
    int32_t * new_faces = NULL;
    int64_t new_vertex_count = 0;
    int64_t keep_count = 0;
    if (!compute_duplicate_compacted_mesh_vulkan(mesh, &new_vertices, &new_vertex_count, &new_faces, &keep_count)) return 0;
    *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = keep_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = keep_count;
    mesh->has_uvs = 0;
    return 1;
}

static int compute_degenerate_compacted_mesh_vulkan(
    const vkmesh_mesh * mesh,
    float abs_thresh,
    float rel_thresh,
    float ** vertices_out,
    int64_t * vertex_count_out,
    int32_t ** faces_out,
    int64_t * face_count_out) {
    *vertices_out = NULL;
    *vertex_count_out = 0;
    *faces_out = NULL;
    *face_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.eps = abs_thresh;
    push.rel_eps = rel_thresh;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_DEGENERATE_FACES, buffers, &push, groups)) goto cleanup;

    if (!compact_mesh_from_device_flags(
            vk, mesh, &buffers[0], &buffers[1], &buffers[2], 0u,
            vertices_out, vertex_count_out, faces_out, face_count_out)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_remove_degenerate_faces(vkmesh_mesh * mesh, float abs_thresh, float rel_thresh, int * removed_faces) {
    *removed_faces = 0;
    float * new_vertices = NULL;
    int32_t * new_faces = NULL;
    int64_t new_vertex_count = 0;
    int64_t keep_count = 0;
    if (!compute_degenerate_compacted_mesh_vulkan(mesh, abs_thresh, rel_thresh, &new_vertices, &new_vertex_count, &new_faces, &keep_count)) {
        fprintf(stderr, "vkmesh: Vulkan degenerate-face pass failed\n");
        return 0;
    }
    *removed_faces = (int) (mesh->n_faces - keep_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = keep_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = keep_count;
    mesh->has_uvs = 0;
    return 1;
}

static int get_sorted_edges(const vkmesh_mesh * mesh, vkmesh_edge ** edges_out, int64_t * edge_count_out) {
    if (!expand_edges_vulkan(mesh, edges_out, edge_count_out)) return 0;
    return 1;
}

static int mark_boundary_edges_vulkan(const vkmesh_edge * edges, int64_t edge_count, uint32_t ** flags_out) {
    *flags_out = NULL;
    if (edge_count <= 0 || edge_count > UINT32_MAX) return 0;
    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }
    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t edges_bytes = (size_t) edge_count * sizeof(vkmesh_edge);
    const size_t flags_bytes = (size_t) edge_count * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, edges_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[1]) ||
        !vk_buffer_create(vk, flags_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, edges, edges_bytes);
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t groups = (uint32_t) ((edge_count + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_BOUNDARY_EDGES, buffers, &push, groups)) goto cleanup;
    uint32_t * flags = (uint32_t *) malloc(flags_bytes);
    if (flags == NULL) goto cleanup;
    memcpy(flags, buffers[2].mapped, flags_bytes);
    *flags_out = flags;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_boundary_edges_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *boundary_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) mesh->n_faces * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = boundary_buffer;
    buffers[2] = counter_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, buffers, &push, groups)) goto cleanup;

    uint32_t count_u32 = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) count_u32 > max_boundary_count) goto cleanup;
    if (count_u32 > 0u) {
        vkmesh_boundary_edge * boundary = (vkmesh_boundary_edge *) malloc((size_t) count_u32 * sizeof(*boundary));
        if (boundary == NULL) goto cleanup;
        memcpy(boundary, boundary_buffer.mapped, (size_t) count_u32 * sizeof(*boundary));
        *boundary_out = boundary;
    }
    *boundary_count_out = (int64_t) count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_boundary_edges_fallback(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *boundary_count_out = 0;
    vkmesh_edge * edges = NULL;
    int64_t edge_count = 0;
    if (!expand_edges_vulkan(mesh, &edges, &edge_count)) {
        fprintf(stderr, "vkmesh: Vulkan edge expansion failed\n");
        return 0;
    }
    uint32_t * boundary_flags = NULL;
    if (!mark_boundary_edges_vulkan(edges, edge_count, &boundary_flags)) {
        fprintf(stderr, "vkmesh: Vulkan boundary-edge marking failed\n");
        free(edges);
        return 0;
    }

    vkmesh_boundary_edge * boundary = NULL;
    int64_t boundary_count = 0;
    int64_t boundary_capacity = 0;
    for (int64_t i = 0; i < edge_count; ++i) {
        if (!boundary_flags[i]) continue;
        if (boundary_count == boundary_capacity) {
            int64_t next_cap = boundary_capacity > 0 ? boundary_capacity * 2 : 1024;
            vkmesh_boundary_edge * next =
                (vkmesh_boundary_edge *) realloc(boundary, (size_t) next_cap * sizeof(*boundary));
            if (next == NULL) {
                free(edges);
                free(boundary_flags);
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

    free(edges);
    free(boundary_flags);
    *boundary_out = boundary;
    *boundary_count_out = boundary_count;
    return 1;
}

static int get_boundary_edges(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    int64_t * boundary_count_out) {
    if (get_boundary_edges_vulkan(mesh, boundary_out, boundary_count_out)) return 1;
    return get_boundary_edges_fallback(mesh, boundary_out, boundary_count_out);
}

static int get_manifold_face_pairs_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    *pairs_out = NULL;
    *pair_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer pairs_buffer;
    vkmesh_vk_buffer counter_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&pairs_buffer, 0, sizeof(pairs_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_pair_count = ((size_t) mesh->n_faces * 3u) / 2u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_pair_count * sizeof(vkmesh_face_pair), &pairs_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = pairs_buffer;
    buffers[2] = counter_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    push.aux0 = (uint32_t) mesh->n_faces;
    uint32_t groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_FACE_PAIRS, buffers, &push, groups)) goto cleanup;

    uint32_t count_u32 = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) count_u32 > max_pair_count) goto cleanup;
    if (count_u32 > 0u) {
        vkmesh_face_pair * pairs = (vkmesh_face_pair *) malloc((size_t) count_u32 * sizeof(*pairs));
        if (pairs == NULL) goto cleanup;
        memcpy(pairs, pairs_buffer.mapped, (size_t) count_u32 * sizeof(*pairs));
        *pairs_out = pairs;
    }
    *pair_count_out = (int64_t) count_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &pairs_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_manifold_face_pairs_cpu(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    *pairs_out = NULL;
    *pair_count_out = 0;
    vkmesh_edge * edges = NULL;
    int64_t edge_count = 0;
    if (!get_sorted_edges(mesh, &edges, &edge_count)) return 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t count = 0;
    int64_t capacity = 0;
    for (int64_t i = 0; i < edge_count;) {
        int64_t j = i + 1;
        while (j < edge_count && edges[j].min_v == edges[i].min_v && edges[j].max_v == edges[i].max_v) ++j;
        if (j - i == 2) {
            if (count == capacity) {
                int64_t next_cap = capacity > 0 ? capacity * 2 : 1024;
                vkmesh_face_pair * next = (vkmesh_face_pair *) realloc(pairs, (size_t) next_cap * sizeof(*pairs));
                if (next == NULL) {
                    free(edges);
                    free(pairs);
                    return 0;
                }
                pairs = next;
                capacity = next_cap;
            }
            pairs[count].f0 = (int32_t) edges[i].face;
            pairs[count].f1 = (int32_t) edges[i + 1].face;
            pairs[count].v0 = edges[i].min_v;
            pairs[count].v1 = edges[i].max_v;
            ++count;
        }
        i = j;
    }
    free(edges);
    *pairs_out = pairs;
    *pair_count_out = count;
    return 1;
}

static int get_manifold_face_pairs(
    const vkmesh_mesh * mesh,
    vkmesh_face_pair ** pairs_out,
    int64_t * pair_count_out) {
    if (get_manifold_face_pairs_vulkan(mesh, pairs_out, pair_count_out)) return 1;
    return get_manifold_face_pairs_cpu(mesh, pairs_out, pair_count_out);
}

static int face_local_index(const vkmesh_mesh * mesh, int32_t face_id, int32_t vertex_id) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    if (f[0] == vertex_id) return 0;
    if (f[1] == vertex_id) return 1;
    if (f[2] == vertex_id) return 2;
    return -1;
}

static int vkmesh_repair_non_manifold_edges_vulkan(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer parents;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer repair_data;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&parents, 0, sizeof(parents));
    memset(&changed, 0, sizeof(changed));
    memset(&repair_data, 0, sizeof(repair_data));

    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const uint32_t corner_count = face_count * 3u;
    const uint32_t root_map_words = corner_count;
    const uint32_t counter_word = root_map_words;
    const uint32_t out_vertices_word = counter_word + 1u;
    const uint32_t out_faces_word = out_vertices_word + corner_count * 3u;
    const size_t repair_words = (size_t) out_faces_word + (size_t) corner_count;
    if (repair_words > UINT32_MAX) {
        if (owns_vk) vkmesh_vk_destroy(vk);
        return 0;
    }

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t parents_bytes = (size_t) corner_count * sizeof(uint32_t);
    const size_t repair_bytes = repair_words * sizeof(uint32_t);
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, parents_bytes, &parents) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, repair_bytes, &repair_data)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = parents;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    uint32_t corner_groups = (corner_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, corner_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = faces_buffer;
    union_buffers[2] = parents;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = parents;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    int converged = 0;
    for (int iter = 0; iter < 128; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_CORNER_EDGES, union_buffers, &push, edge_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = corner_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, corner_groups)) goto cleanup;
        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    uint32_t * repair_words_u32 = (uint32_t *) repair_data.mapped;
    memset(repair_words_u32, 0xff, (size_t) root_map_words * sizeof(uint32_t));
    repair_words_u32[counter_word] = 0u;

    vkmesh_vk_buffer assign_buffers[4];
    assign_buffers[0] = parents;
    assign_buffers[1] = repair_data;
    assign_buffers[2] = changed;
    assign_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ASSIGN_CORNER_VERTICES, assign_buffers, &push, corner_groups)) goto cleanup;

    uint32_t vertex_count = repair_words_u32[counter_word];
    if (vertex_count == 0u || vertex_count > corner_count) goto cleanup;

    vkmesh_vk_buffer write_buffers[4];
    write_buffers[0] = faces_buffer;
    write_buffers[1] = vertices_buffer;
    write_buffers[2] = parents;
    write_buffers[3] = repair_data;
    memset(&push, 0, sizeof(push));
    push.n = corner_count;
    push.aux1 = out_vertices_word;
    push.aux2 = out_faces_word;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_REPAIRED_MESH, write_buffers, &push, corner_groups)) goto cleanup;

    float * out_vertices = (float *) malloc((size_t) vertex_count * 3u * sizeof(float));
    int32_t * out_faces = (int32_t *) malloc((size_t) face_count * 3u * sizeof(int32_t));
    if (out_vertices == NULL || out_faces == NULL) {
        free(out_vertices);
        free(out_faces);
        goto cleanup;
    }
    memcpy(out_vertices, repair_words_u32 + out_vertices_word, (size_t) vertex_count * 3u * sizeof(float));
    memcpy(out_faces, repair_words_u32 + out_faces_word, (size_t) face_count * 3u * sizeof(int32_t));

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = out_vertices;
    mesh->uvs = NULL;
    mesh->faces = out_faces;
    mesh->n_vertices = (int64_t) vertex_count;
    mesh->n_faces = (int64_t) face_count;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 0;
    *new_vertices = (int) mesh->n_vertices;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &parents);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &repair_data);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_repair_non_manifold_edges_cpu(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX / 3) return 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;

    int64_t corner_count = mesh->n_faces * 3;
    int * parent = (int *) malloc((size_t) corner_count * sizeof(int));
    int32_t * root_to_vertex = (int32_t *) malloc((size_t) corner_count * sizeof(int32_t));
    int32_t * corner_to_vertex = (int32_t *) malloc((size_t) corner_count * sizeof(int32_t));
    if (parent == NULL || root_to_vertex == NULL || corner_to_vertex == NULL) {
        free(pairs);
        free(parent);
        free(root_to_vertex);
        free(corner_to_vertex);
        return 0;
    }
    for (int64_t i = 0; i < corner_count; ++i) {
        parent[i] = (int) i;
        root_to_vertex[i] = -1;
    }
    for (int64_t i = 0; i < pair_count; ++i) {
        int a0 = face_local_index(mesh, pairs[i].f0, (int32_t) pairs[i].v0);
        int a1 = face_local_index(mesh, pairs[i].f0, (int32_t) pairs[i].v1);
        int b0 = face_local_index(mesh, pairs[i].f1, (int32_t) pairs[i].v0);
        int b1 = face_local_index(mesh, pairs[i].f1, (int32_t) pairs[i].v1);
        if (a0 >= 0 && b0 >= 0) uf_union(parent, pairs[i].f0 * 3 + a0, pairs[i].f1 * 3 + b0);
        if (a1 >= 0 && b1 >= 0) uf_union(parent, pairs[i].f0 * 3 + a1, pairs[i].f1 * 3 + b1);
    }

    int64_t out_v_count = 0;
    for (int64_t corner = 0; corner < corner_count; ++corner) {
        int root = uf_find(parent, (int) corner);
        if (root_to_vertex[root] < 0) root_to_vertex[root] = (int32_t) out_v_count++;
        corner_to_vertex[corner] = root_to_vertex[root];
    }

    float * out_vertices = (float *) malloc((size_t) out_v_count * 3u * sizeof(float));
    float * out_uvs = NULL;
    if (out_vertices == NULL) {
        free(pairs); free(parent); free(root_to_vertex); free(corner_to_vertex);
        return 0;
    }
    if (mesh->has_uvs && mesh->uvs != NULL) {
        out_uvs = (float *) malloc((size_t) out_v_count * 2u * sizeof(float));
        if (out_uvs == NULL) {
            free(out_vertices);
            free(pairs); free(parent); free(root_to_vertex); free(corner_to_vertex);
            return 0;
        }
    }
    memset(root_to_vertex, 0xff, (size_t) corner_count * sizeof(int32_t));
    for (int64_t corner = 0; corner < corner_count; ++corner) {
        int root = uf_find(parent, (int) corner);
        int32_t dst = corner_to_vertex[corner];
        if (root_to_vertex[root] >= 0) continue;
        int64_t face_id = corner / 3;
        int local = (int) (corner % 3);
        int32_t src_v = mesh->faces[(size_t) face_id * 3u + (size_t) local];
        memcpy(out_vertices + (size_t) dst * 3u, mesh->vertices + (size_t) src_v * 3u, 3u * sizeof(float));
        if (out_uvs != NULL) {
            memcpy(out_uvs + (size_t) dst * 2u, mesh->uvs + (size_t) src_v * 2u, 2u * sizeof(float));
        }
        root_to_vertex[root] = dst;
    }
    for (int64_t face_id = 0; face_id < mesh->n_faces; ++face_id) {
        int32_t * f = mesh->faces + (size_t) face_id * 3u;
        f[0] = corner_to_vertex[face_id * 3 + 0];
        f[1] = corner_to_vertex[face_id * 3 + 1];
        f[2] = corner_to_vertex[face_id * 3 + 2];
    }

    free(mesh->vertices);
    free(mesh->uvs);
    mesh->vertices = out_vertices;
    mesh->uvs = out_uvs;
    mesh->n_vertices = out_v_count;
    mesh->vertex_capacity = out_v_count;
    if (out_uvs == NULL) mesh->has_uvs = 0;
    *new_vertices = (int) mesh->n_vertices;
    free(pairs);
    free(parent);
    free(root_to_vertex);
    free(corner_to_vertex);
    return 1;
}

static int vkmesh_repair_non_manifold_edges(vkmesh_mesh * mesh, int * old_vertices, int * new_vertices) {
    if (vkmesh_repair_non_manifold_edges_vulkan(mesh, old_vertices, new_vertices)) return 1;
    return vkmesh_repair_non_manifold_edges_cpu(mesh, old_vertices, new_vertices);
}

static int vkmesh_remove_small_connected_components(vkmesh_mesh * mesh, float min_area, int * removed_faces) {
    *removed_faces = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > INT32_MAX) return 0;
    if (mesh->n_vertices > 0 && mesh->n_vertices <= INT32_MAX) {
        vkmesh_vk local_vk;
        vkmesh_vk * vk = NULL;
        int owns_vk = 0;
        if (vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
            vkmesh_device_mesh dm;
            memset(&dm, 0, sizeof(dm));
            uint32_t removed_u32 = 0u;
            int ok = 0;
            if (vkmesh_device_mesh_upload(vk, mesh, &dm) &&
                vkmesh_device_remove_small_connected_components(&dm, min_area, &removed_u32) &&
                vkmesh_device_mesh_download(&dm, mesh)) {
                *removed_faces = (int) removed_u32;
                ok = 1;
            }
            vkmesh_device_mesh_destroy(&dm);
            if (owns_vk) vkmesh_vk_destroy(vk);
            if (ok) return 1;
            fprintf(stderr, "vkmesh: device connected-components pass failed, falling back\n");
        }
    }
    if (mesh->n_vertices > 0 && mesh->n_vertices <= INT32_MAX) {
        vkmesh_vk local_vk;
        vkmesh_vk * vk = NULL;
        int owns_vk = 0;
        if (vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
            vkmesh_vk_buffer faces_buffer;
            vkmesh_vk_buffer vertices_buffer;
            vkmesh_vk_buffer edges_buffer;
            vkmesh_vk_buffer edge_dummy;
            vkmesh_vk_buffer parents;
            vkmesh_vk_buffer changed;
            vkmesh_vk_buffer area_bits;
            vkmesh_vk_buffer keep_flags;
            memset(&faces_buffer, 0, sizeof(faces_buffer));
            memset(&vertices_buffer, 0, sizeof(vertices_buffer));
            memset(&edges_buffer, 0, sizeof(edges_buffer));
            memset(&edge_dummy, 0, sizeof(edge_dummy));
            memset(&parents, 0, sizeof(parents));
            memset(&changed, 0, sizeof(changed));
            memset(&area_bits, 0, sizeof(area_bits));
            memset(&keep_flags, 0, sizeof(keep_flags));

            const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
            const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
            const size_t parents_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
            const size_t flags_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
            size_t edge_count = 0;
            size_t edge_sort_count = 0;
            int ok = 0;
            if (vk_buffer_create(vk, faces_bytes, &faces_buffer) &&
                vk_buffer_create(vk, vertices_bytes, &vertices_buffer) &&
                vk_buffer_create(vk, parents_bytes, &parents) &&
                vk_buffer_create(vk, sizeof(uint32_t), &changed) &&
                vk_buffer_create(vk, parents_bytes, &area_bits) &&
                vk_buffer_create(vk, flags_bytes, &keep_flags)) {
                memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
                memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);
                if (expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &edge_dummy, &edge_count, &edge_sort_count)) {
                    vkmesh_vk_buffer init_buffers[4];
                    init_buffers[0] = parents;
                    init_buffers[1] = changed;
                    init_buffers[2] = changed;
                    init_buffers[3] = changed;
                    vkmesh_push push;
                    memset(&push, 0, sizeof(push));
                    push.n = (uint32_t) mesh->n_faces;
                    uint32_t groups = (push.n + 127u) / 128u;
                    if (vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, groups)) {
                        vkmesh_vk_buffer union_buffers[4];
                        union_buffers[0] = edges_buffer;
                        union_buffers[1] = parents;
                        union_buffers[2] = changed;
                        union_buffers[3] = changed;
                        vkmesh_vk_buffer compress_buffers[4];
                        compress_buffers[0] = parents;
                        compress_buffers[1] = changed;
                        compress_buffers[2] = changed;
                        compress_buffers[3] = changed;
                        uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
                        uint32_t face_groups = (uint32_t) ((mesh->n_faces + 127) / 128);
                        int converged = 0;
                        for (int iter = 0; iter < 32; ++iter) {
                            ((uint32_t *) changed.mapped)[0] = 0u;
                            memset(&push, 0, sizeof(push));
                            push.n = (uint32_t) edge_count;
                            push.aux0 = (uint32_t) mesh->n_faces;
                            if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_FACE_EDGES, union_buffers, &push, edge_groups)) break;
                            memset(&push, 0, sizeof(push));
                            push.n = (uint32_t) mesh->n_faces;
                            if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, face_groups)) break;
                            if (((const uint32_t *) changed.mapped)[0] == 0u) {
                                converged = 1;
                                break;
                            }
                        }
                        if (converged) {
                            memset(area_bits.mapped, 0, parents_bytes);
                            vkmesh_vk_buffer area_buffers[4];
                            area_buffers[0] = faces_buffer;
                            area_buffers[1] = vertices_buffer;
                            area_buffers[2] = parents;
                            area_buffers[3] = area_bits;
                            memset(&push, 0, sizeof(push));
                            push.n = (uint32_t) mesh->n_faces;
                            if (vkmesh_dispatch(vk, VKMESH_PIPE_COMPONENT_AREA, area_buffers, &push, face_groups)) {
                                vkmesh_vk_buffer mark_buffers[4];
                                mark_buffers[0] = parents;
                                mark_buffers[1] = area_bits;
                                mark_buffers[2] = keep_flags;
                                mark_buffers[3] = changed;
                                memset(&push, 0, sizeof(push));
                                push.n = (uint32_t) mesh->n_faces;
                                push.eps = min_area;
                                if (vkmesh_dispatch(vk, VKMESH_PIPE_MARK_COMPONENT_KEEP, mark_buffers, &push, face_groups)) {
                                    float * new_vertices = NULL;
                                    int64_t new_vertex_count = 0;
                                    int32_t * new_faces = NULL;
                                    int64_t new_face_count = 0;
                                    if (compact_mesh_from_device_flags(
                                            vk,
                                            mesh,
                                            &faces_buffer,
                                            &vertices_buffer,
                                            &keep_flags,
                                            0u,
                                            &new_vertices,
                                            &new_vertex_count,
                                            &new_faces,
                                            &new_face_count)) {
                                        *removed_faces = (int) (mesh->n_faces - new_face_count);
                                        free(mesh->vertices);
                                        free(mesh->uvs);
                                        free(mesh->faces);
                                        mesh->vertices = new_vertices;
                                        mesh->uvs = NULL;
                                        mesh->faces = new_faces;
                                        mesh->n_vertices = new_vertex_count;
                                        mesh->n_faces = new_face_count;
                                        mesh->vertex_capacity = new_vertex_count;
                                        mesh->face_capacity = new_face_count;
                                        mesh->has_uvs = 0;
                                        ok = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            vk_buffer_destroy(vk, &faces_buffer);
            vk_buffer_destroy(vk, &vertices_buffer);
            vk_buffer_destroy(vk, &edges_buffer);
            vk_buffer_destroy(vk, &edge_dummy);
            vk_buffer_destroy(vk, &parents);
            vk_buffer_destroy(vk, &changed);
            vk_buffer_destroy(vk, &area_bits);
            vk_buffer_destroy(vk, &keep_flags);
            if (owns_vk) vkmesh_vk_destroy(vk);
            if (ok) return 1;
            fprintf(stderr, "vkmesh: Vulkan connected-components pass failed, falling back to CPU\n");
        }
    }

    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;
    int * parent = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    double * comp_area = (double *) calloc((size_t) mesh->n_faces, sizeof(double));
    uint8_t * keep = (uint8_t *) malloc((size_t) mesh->n_faces);
    if (parent == NULL || comp_area == NULL || keep == NULL) {
        free(pairs); free(parent); free(comp_area); free(keep);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) parent[i] = (int) i;
    for (int64_t i = 0; i < pair_count; ++i) uf_union(parent, pairs[i].f0, pairs[i].f1);
    for (int64_t i = 0; i < mesh->n_faces; ++i) comp_area[uf_find(parent, (int) i)] += face_area(mesh, i);
    for (int64_t i = 0; i < mesh->n_faces; ++i) keep[i] = comp_area[uf_find(parent, (int) i)] >= (double) min_area ? 1u : 0u;
    int ok = mesh_remove_faces_by_mask(mesh, keep, removed_faces);
    free(pairs);
    free(parent);
    free(comp_area);
    free(keep);
    return ok;
}

static int shared_edge_needs_flip(const vkmesh_mesh * mesh, const vkmesh_face_pair * pair) {
    const int32_t * f0 = mesh->faces + (size_t) pair->f0 * 3u;
    const int32_t * f1 = mesh->faces + (size_t) pair->f1 * 3u;
    int idx0[2];
    int idx1[2];
    int found = 0;
    for (int i = 0; i < 3 && found < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (f0[i] == f1[j]) {
                idx0[found] = i;
                idx1[found] = j;
                ++found;
                break;
            }
        }
    }
    if (found != 2) return 0;
    int dir0 = (idx0[1] - idx0[0] + 3) % 3;
    int dir1 = (idx1[1] - idx1[0] + 3) % 3;
    return dir0 == dir1;
}

static int vkmesh_unify_face_orientations_vulkan(vkmesh_mesh * mesh, int * flipped_faces) {
    *flipped_faces = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_faces > UINT32_MAX / 2u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer state;
    vkmesh_vk_buffer changed;
    vkmesh_vk_buffer counter;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&state, 0, sizeof(state));
    memset(&changed, 0, sizeof(changed));
    memset(&counter, 0, sizeof(counter));

    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t state_bytes = (size_t) mesh->n_faces * sizeof(uint32_t);
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, state_bytes, &state) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &changed) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = state;
    init_buffers[1] = changed;
    init_buffers[2] = changed;
    init_buffers[3] = changed;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_ORIENTATION_STATE, init_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = edges_buffer;
    union_buffers[1] = faces_buffer;
    union_buffers[2] = state;
    union_buffers[3] = changed;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = state;
    compress_buffers[1] = changed;
    compress_buffers[2] = changed;
    compress_buffers[3] = changed;
    int converged = 0;
    for (int iter = 0; iter < 512; ++iter) {
        ((uint32_t *) changed.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = (uint32_t) edge_count;
        push.aux0 = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_ORIENTATION_EDGES, union_buffers, &push, edge_groups)) goto cleanup;

        memset(&push, 0, sizeof(push));
        push.n = face_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_ORIENTATION_STATE, compress_buffers, &push, face_groups)) goto cleanup;

        if (((const uint32_t *) changed.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    ((uint32_t *) counter.mapped)[0] = 0u;
    vkmesh_vk_buffer apply_buffers[4];
    apply_buffers[0] = faces_buffer;
    apply_buffers[1] = state;
    apply_buffers[2] = counter;
    apply_buffers[3] = changed;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_APPLY_ORIENTATION_FLIPS, apply_buffers, &push, face_groups)) goto cleanup;

    memcpy(mesh->faces, faces_buffer.mapped, faces_bytes);
    *flipped_faces = (int) ((const uint32_t *) counter.mapped)[0];
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &state);
    vk_buffer_destroy(vk, &changed);
    vk_buffer_destroy(vk, &counter);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_unify_face_orientations_cpu(vkmesh_mesh * mesh, int * flipped_faces) {
    *flipped_faces = 0;
    vkmesh_face_pair * pairs = NULL;
    int64_t pair_count = 0;
    if (!get_manifold_face_pairs(mesh, &pairs, &pair_count)) return 0;
    int * degree = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_faces + 1u, sizeof(int));
    int * adjacency = NULL;
    int8_t * flip = (int8_t *) malloc((size_t) mesh->n_faces);
    if (degree == NULL || offset == NULL || cursor == NULL || flip == NULL) {
        free(pairs); free(degree); free(offset); free(cursor); free(flip);
        return 0;
    }
    memset(flip, -1, (size_t) mesh->n_faces);
    for (int64_t i = 0; i < pair_count; ++i) {
        ++degree[pairs[i].f0];
        ++degree[pairs[i].f1];
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) offset[i + 1] = offset[i] + degree[i];
    adjacency = (int *) malloc((size_t) offset[mesh->n_faces] * sizeof(int));
    if (adjacency == NULL && offset[mesh->n_faces] > 0) {
        free(pairs); free(degree); free(offset); free(cursor); free(flip);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_faces + 1u) * sizeof(int));
    for (int64_t i = 0; i < pair_count; ++i) {
        adjacency[cursor[pairs[i].f0]++] = (int) i;
        adjacency[cursor[pairs[i].f1]++] = (int) i;
    }
    int * queue = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    if (queue == NULL) {
        free(pairs); free(degree); free(offset); free(cursor); free(adjacency); free(flip);
        return 0;
    }
    for (int64_t seed = 0; seed < mesh->n_faces; ++seed) {
        if (flip[seed] >= 0) continue;
        int begin = 0;
        int end = 0;
        queue[end++] = (int) seed;
        flip[seed] = 0;
        while (begin < end) {
            int f = queue[begin++];
            for (int it = offset[f]; it < offset[f + 1]; ++it) {
                int pair_id = adjacency[it];
                vkmesh_face_pair * pair = &pairs[pair_id];
                int other = pair->f0 == f ? pair->f1 : pair->f0;
                int need = shared_edge_needs_flip(mesh, pair);
                int next_flip = flip[f] ^ need;
                if (flip[other] < 0) {
                    flip[other] = (int8_t) next_flip;
                    queue[end++] = other;
                }
            }
        }
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        if ((flip[i] & 1) == 0) continue;
        int32_t * f = mesh->faces + (size_t) i * 3u;
        int32_t t = f[1];
        f[1] = f[2];
        f[2] = t;
        ++(*flipped_faces);
    }
    free(pairs);
    free(degree);
    free(offset);
    free(cursor);
    free(adjacency);
    free(flip);
    free(queue);
    return 1;
}

static int vkmesh_unify_face_orientations(vkmesh_mesh * mesh, int * flipped_faces) {
    if (vkmesh_unify_face_orientations_vulkan(mesh, flipped_faces)) return 1;
    return vkmesh_unify_face_orientations_cpu(mesh, flipped_faces);
}

static int build_vertex_face_adjacency_vulkan(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    *offset_out = NULL;
    *adj_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices >= UINT32_MAX ||
        (uint64_t) mesh->n_faces * 3ull > INT_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer dummy_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t offset_count = vertex_count + 1u;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t degree_bytes = (size_t) mesh->n_vertices * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = faces_buffer;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = dummy_buffer;
    degree_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    uint32_t groups = (face_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = dummy_buffer;
    seed_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer * scan_in = &offset_a_buffer;
    vkmesh_vk_buffer * scan_out = &offset_b_buffer;
    for (uint64_t stride = 1u; stride < (uint64_t) offset_count; stride <<= 1u) {
        vkmesh_vk_buffer scan_buffers[4];
        scan_buffers[0] = *scan_in;
        scan_buffers[1] = *scan_out;
        scan_buffers[2] = dummy_buffer;
        scan_buffers[3] = dummy_buffer;
        memset(&push, 0, sizeof(push));
        push.n = offset_count;
        push.aux0 = (uint32_t) stride;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_SCAN_U32_STRIDE, scan_buffers, &push, offset_groups)) goto cleanup;
        vkmesh_vk_buffer * tmp = scan_in;
        scan_in = scan_out;
        scan_out = tmp;
    }

    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(*offset));
    if (offset == NULL) goto cleanup;
    const uint32_t * offset_u32 = (const uint32_t *) scan_in->mapped;
    int64_t total_adj = (int64_t) offset_u32[vertex_count];
    if (total_adj > INT_MAX) {
        free(offset);
        goto cleanup;
    }
    for (int64_t i = 0; i <= mesh->n_vertices; ++i) {
        if (offset_u32[i] > INT_MAX) {
            free(offset);
            goto cleanup;
        }
        offset[i] = (int) offset_u32[i];
    }

    int * adj = NULL;
    if (total_adj > 0) {
        adj = (int *) malloc((size_t) total_adj * sizeof(*adj));
        if (adj == NULL) {
            free(offset);
            goto cleanup;
        }
    }

    const size_t adj_bytes = (size_t) (total_adj > 0 ? total_adj : 1) * sizeof(uint32_t);
    if (!vk_buffer_create(vk, adj_bytes, &adjacency_buffer)) {
        free(offset);
        free(adj);
        goto cleanup;
    }

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = faces_buffer;
    adj_buffers[1] = *scan_in;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = (uint32_t) total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, groups)) {
        free(offset);
        free(adj);
        goto cleanup;
    }
    if (total_adj > 0) memcpy(adj, adjacency_buffer.mapped, (size_t) total_adj * sizeof(*adj));

    *offset_out = offset;
    *adj_out = adj;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int build_vertex_face_adjacency_cpu(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    *offset_out = NULL;
    *adj_out = NULL;
    int * degree = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * offset = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    int * cursor = (int *) calloc((size_t) mesh->n_vertices + 1u, sizeof(int));
    if (degree == NULL || offset == NULL || cursor == NULL) {
        free(degree); free(offset); free(cursor);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        ++degree[f[0]];
        ++degree[f[1]];
        ++degree[f[2]];
    }
    for (int64_t i = 0; i < mesh->n_vertices; ++i) offset[i + 1] = offset[i] + degree[i];
    int * adj = (int *) malloc((size_t) offset[mesh->n_vertices] * sizeof(int));
    if (adj == NULL && offset[mesh->n_vertices] > 0) {
        free(degree); free(offset); free(cursor);
        return 0;
    }
    memcpy(cursor, offset, ((size_t) mesh->n_vertices + 1u) * sizeof(int));
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        adj[cursor[f[0]]++] = (int) i;
        adj[cursor[f[1]]++] = (int) i;
        adj[cursor[f[2]]++] = (int) i;
    }
    free(degree);
    free(cursor);
    *offset_out = offset;
    *adj_out = adj;
    return 1;
}

static int build_vertex_face_adjacency(const vkmesh_mesh * mesh, int ** offset_out, int ** adj_out) {
    if (build_vertex_face_adjacency_vulkan(mesh, offset_out, adj_out)) return 1;
    return build_vertex_face_adjacency_cpu(mesh, offset_out, adj_out);
}

static int get_unique_simplify_edges_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    *vertex_boundary_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer unique_buffer;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&unique_buffer, 0, sizeof(unique_buffer));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t boundary_bytes = (size_t) mesh->n_vertices * sizeof(uint32_t);
    int ok = 0;
    size_t raw_count = 0;
    size_t raw_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, boundary_bytes, &boundary_flags) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &raw_count,
            &raw_sort_count)) {
        goto cleanup;
    }
    if (!vk_buffer_create(vk, raw_count * 2u * sizeof(uint32_t), &unique_buffer)) goto cleanup;

    vkmesh_vk_buffer buffers[4];
    buffers[0] = edges_buffer;
    buffers[1] = unique_buffer;
    buffers[2] = boundary_flags;
    buffers[3] = counter;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_count;
    push.aux0 = (uint32_t) mesh->n_vertices;
    uint32_t groups = (uint32_t) ((raw_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, buffers, &push, groups)) goto cleanup;

    uint32_t unique_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_count > raw_count) goto cleanup;
    vkmesh_simplify_edge * edges = (vkmesh_simplify_edge *) malloc((size_t) unique_count * sizeof(*edges));
    uint8_t * boundary = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(*boundary));
    if ((edges == NULL && unique_count > 0u) || boundary == NULL) {
        free(edges);
        free(boundary);
        goto cleanup;
    }
    const uint32_t * unique_u32 = (const uint32_t *) unique_buffer.mapped;
    for (uint32_t i = 0; i < unique_count; ++i) {
        edges[i].v0 = (int32_t) unique_u32[(size_t) i * 2u + 0u];
        edges[i].v1 = (int32_t) unique_u32[(size_t) i * 2u + 1u];
        edges[i].cost = INFINITY;
        edges[i].pos[0] = 0.0f;
        edges[i].pos[1] = 0.0f;
        edges[i].pos[2] = 0.0f;
    }
    const uint32_t * boundary_u32 = (const uint32_t *) boundary_flags.mapped;
    for (int64_t i = 0; i < mesh->n_vertices; ++i) boundary[i] = boundary_u32[i] != 0u ? 1u : 0u;

    *edges_out = edges;
    *edge_count_out = (int64_t) unique_count;
    *vertex_boundary_out = boundary;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &unique_buffer);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int get_unique_simplify_edges_cpu(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    *edges_out = NULL;
    *edge_count_out = 0;
    *vertex_boundary_out = NULL;
    vkmesh_edge * raw = NULL;
    int64_t raw_count = 0;
    if (!get_sorted_edges(mesh, &raw, &raw_count)) return 0;
    vkmesh_simplify_edge * edges = (vkmesh_simplify_edge *) malloc((size_t) raw_count * sizeof(*edges));
    uint8_t * boundary = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    if (edges == NULL || boundary == NULL) {
        free(raw); free(edges); free(boundary);
        return 0;
    }
    int64_t count = 0;
    for (int64_t i = 0; i < raw_count;) {
        int64_t j = i + 1;
        while (j < raw_count && raw[j].min_v == raw[i].min_v && raw[j].max_v == raw[i].max_v) ++j;
        edges[count].v0 = (int32_t) raw[i].min_v;
        edges[count].v1 = (int32_t) raw[i].max_v;
        edges[count].cost = INFINITY;
        edges[count].pos[0] = 0.0f;
        edges[count].pos[1] = 0.0f;
        edges[count].pos[2] = 0.0f;
        if (j - i == 1) {
            boundary[raw[i].min_v] = 1u;
            boundary[raw[i].max_v] = 1u;
        }
        ++count;
        i = j;
    }
    free(raw);
    *edges_out = edges;
    *edge_count_out = count;
    *vertex_boundary_out = boundary;
    return 1;
}

static int get_unique_simplify_edges(
    const vkmesh_mesh * mesh,
    vkmesh_simplify_edge ** edges_out,
    int64_t * edge_count_out,
    uint8_t ** vertex_boundary_out) {
    if (get_unique_simplify_edges_vulkan(mesh, edges_out, edge_count_out, vertex_boundary_out)) return 1;
    return get_unique_simplify_edges_cpu(mesh, edges_out, edge_count_out, vertex_boundary_out);
}

static void qem_add_plane(vkmesh_qem * q, double a, double b, double c, double d) {
    q->m[0] += a * a;
    q->m[1] += a * b;
    q->m[2] += a * c;
    q->m[3] += a * d;
    q->m[4] += b * b;
    q->m[5] += b * c;
    q->m[6] += b * d;
    q->m[7] += c * c;
    q->m[8] += c * d;
    q->m[9] += d * d;
}

static double qem_eval(const vkmesh_qem * q, const float p[3]) {
    double x = p[0], y = p[1], z = p[2], w = 1.0;
    return
        q->m[0] * x * x + 2.0 * q->m[1] * x * y + 2.0 * q->m[2] * x * z + 2.0 * q->m[3] * x * w +
        q->m[4] * y * y + 2.0 * q->m[5] * y * z + 2.0 * q->m[6] * y * w +
        q->m[7] * z * z + 2.0 * q->m[8] * z * w +
        q->m[9] * w * w;
}

static int build_vertex_qems_vulkan(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    *qems_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer qem_buffer;
    vkmesh_vk_buffer dummy_buffer;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&qem_buffer, 0, sizeof(qem_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));

    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t qem_bytes = (size_t) mesh->n_vertices * 10u * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, qem_bytes, &qem_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &dummy_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    vkmesh_vk_buffer buffers[4];
    buffers[0] = faces_buffer;
    buffers[1] = vertices_buffer;
    buffers[2] = qem_buffer;
    buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) mesh->n_faces;
    push.aux0 = (uint32_t) mesh->n_vertices;
    uint32_t groups = (uint32_t) ((mesh->n_faces + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, buffers, &push, groups)) goto cleanup;

    vkmesh_qem * qems = (vkmesh_qem *) calloc((size_t) mesh->n_vertices, sizeof(*qems));
    if (qems == NULL) goto cleanup;
    const float * qem_f = (const float *) qem_buffer.mapped;
    for (int64_t v = 0; v < mesh->n_vertices; ++v) {
        for (uint32_t k = 0; k < 10u; ++k) {
            qems[v].m[k] = (double) qem_f[(size_t) v * 10u + k];
        }
    }
    *qems_out = qems;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &qem_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int build_vertex_qems_cpu(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    *qems_out = (vkmesh_qem *) calloc((size_t) mesh->n_vertices, sizeof(vkmesh_qem));
    if (*qems_out == NULL) return 0;
    vkmesh_qem * qems = *qems_out;
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        const float * a = mesh->vertices + (size_t) f[0] * 3u;
        const float * b = mesh->vertices + (size_t) f[1] * 3u;
        const float * c = mesh->vertices + (size_t) f[2] * 3u;
        double abx = (double) b[0] - (double) a[0];
        double aby = (double) b[1] - (double) a[1];
        double abz = (double) b[2] - (double) a[2];
        double acx = (double) c[0] - (double) a[0];
        double acy = (double) c[1] - (double) a[1];
        double acz = (double) c[2] - (double) a[2];
        double nx = aby * acz - abz * acy;
        double ny = abz * acx - abx * acz;
        double nz = abx * acy - aby * acx;
        double len = sqrt(nx * nx + ny * ny + nz * nz);
        if (len <= 1e-30) continue;
        nx /= len; ny /= len; nz /= len;
        double d = -(nx * a[0] + ny * a[1] + nz * a[2]);
        qem_add_plane(&qems[f[0]], nx, ny, nz, d);
        qem_add_plane(&qems[f[1]], nx, ny, nz, d);
        qem_add_plane(&qems[f[2]], nx, ny, nz, d);
    }
    return 1;
}

static int build_vertex_qems(const vkmesh_mesh * mesh, vkmesh_qem ** qems_out) {
    return build_vertex_qems_cpu(mesh, qems_out);
}

static int face_contains_vertex(const int32_t * f, int32_t v) {
    return f[0] == v || f[1] == v || f[2] == v;
}

static int process_incident_for_simplify(
    const vkmesh_mesh * mesh,
    int32_t face_id,
    int32_t keep_v,
    int32_t other_v,
    const float new_pos[3],
    double * skinny_cost,
    int * num_tri) {
    const int32_t * f = mesh->faces + (size_t) face_id * 3u;
    if (face_contains_vertex(f, other_v)) return 1;
    const float * p0 = mesh->vertices + (size_t) f[0] * 3u;
    const float * p1 = mesh->vertices + (size_t) f[1] * 3u;
    const float * p2 = mesh->vertices + (size_t) f[2] * 3u;
    double old_e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    double old_e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    double old_n[3] = {
        old_e1[1] * old_e2[2] - old_e1[2] * old_e2[1],
        old_e1[2] * old_e2[0] - old_e1[0] * old_e2[2],
        old_e1[0] * old_e2[1] - old_e1[1] * old_e2[0],
    };
    double np[3][3];
    const float * src[3] = { p0, p1, p2 };
    for (int i = 0; i < 3; ++i) {
        if (f[i] == keep_v) {
            np[i][0] = new_pos[0];
            np[i][1] = new_pos[1];
            np[i][2] = new_pos[2];
        } else {
            np[i][0] = src[i][0];
            np[i][1] = src[i][1];
            np[i][2] = src[i][2];
        }
    }
    double new_e1[3] = { np[1][0] - np[0][0], np[1][1] - np[0][1], np[1][2] - np[0][2] };
    double new_e2[3] = { np[2][0] - np[0][0], np[2][1] - np[0][1], np[2][2] - np[0][2] };
    double new_n[3] = {
        new_e1[1] * new_e2[2] - new_e1[2] * new_e2[1],
        new_e1[2] * new_e2[0] - new_e1[0] * new_e2[2],
        new_e1[0] * new_e2[1] - new_e1[1] * new_e2[0],
    };
    double dot_n = old_n[0] * new_n[0] + old_n[1] * new_n[1] + old_n[2] * new_n[2];
    if (dot_n < 0.0) return 0;

    double e0[3] = { np[2][0] - np[1][0], np[2][1] - np[1][1], np[2][2] - np[1][2] };
    double area2 = sqrt(new_n[0] * new_n[0] + new_n[1] * new_n[1] + new_n[2] * new_n[2]);
    double len0 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2];
    double len1 = new_e1[0] * new_e1[0] + new_e1[1] * new_e1[1] + new_e1[2] * new_e1[2];
    double len2 = new_e2[0] * new_e2[0] + new_e2[1] * new_e2[1] + new_e2[2] * new_e2[2];
    double denom = len0 + len1 + len2;
    if (denom < 1e-24) denom = 1e-24;
    double shape = 2.0 * sqrt(3.0) * area2 / denom;
    if (shape < 0.0) shape = 0.0;
    if (shape > 1.0) shape = 1.0;
    *skinny_cost += 1.0 - shape;
    ++(*num_tri);
    return 1;
}

static double simplify_edge_cost(
    const vkmesh_mesh * mesh,
    const int * v2f_offset,
    const int * v2f,
    const uint8_t * boundary,
    const vkmesh_qem * qems,
    float lambda_edge_length,
    float lambda_skinny,
    vkmesh_simplify_edge * edge) {
    const float * v0 = mesh->vertices + (size_t) edge->v0 * 3u;
    const float * v1 = mesh->vertices + (size_t) edge->v1 * 3u;
    float w0 = 0.5f;
    if (boundary[edge->v0] && !boundary[edge->v1]) w0 = 1.0f;
    else if (!boundary[edge->v0] && boundary[edge->v1]) w0 = 0.0f;
    edge->pos[0] = v0[0] * w0 + v1[0] * (1.0f - w0);
    edge->pos[1] = v0[1] * w0 + v1[1] * (1.0f - w0);
    edge->pos[2] = v0[2] * w0 + v1[2] * (1.0f - w0);

    vkmesh_qem q = qems[edge->v0];
    for (int i = 0; i < 10; ++i) q.m[i] += qems[edge->v1].m[i];
    double cost = qem_eval(&q, edge->pos);
    double dx = (double) v1[0] - (double) v0[0];
    double dy = (double) v1[1] - (double) v0[1];
    double dz = (double) v1[2] - (double) v0[2];
    double len2 = dx * dx + dy * dy + dz * dz;
    cost += (double) lambda_edge_length * len2;

    double skinny = 0.0;
    int num_tri = 0;
    for (int i = v2f_offset[edge->v0]; i < v2f_offset[edge->v0 + 1]; ++i) {
        if (!process_incident_for_simplify(mesh, v2f[i], edge->v0, edge->v1, edge->pos, &skinny, &num_tri)) return INFINITY;
    }
    for (int i = v2f_offset[edge->v1]; i < v2f_offset[edge->v1 + 1]; ++i) {
        if (!process_incident_for_simplify(mesh, v2f[i], edge->v1, edge->v0, edge->pos, &skinny, &num_tri)) return INFINITY;
    }
    if (num_tri > 0) skinny /= (double) num_tri;
    cost += (double) lambda_skinny * skinny * len2;
    return cost;
}

static int propagate_edge_cost_to_faces(
    const vkmesh_mesh * mesh,
    const int * v2f_offset,
    const int * v2f,
    const vkmesh_simplify_edge * edges,
    int64_t edge_count,
    int ** best_edge_out) {
    int * best_edge = (int *) malloc((size_t) mesh->n_faces * sizeof(int));
    double * best_cost = (double *) malloc((size_t) mesh->n_faces * sizeof(double));
    if (best_edge == NULL || best_cost == NULL) {
        free(best_edge); free(best_cost);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        best_edge[i] = -1;
        best_cost[i] = INFINITY;
    }
    for (int64_t e = 0; e < edge_count; ++e) {
        if (!isfinite(edges[e].cost)) continue;
        int32_t verts[2] = { edges[e].v0, edges[e].v1 };
        for (int side = 0; side < 2; ++side) {
            int32_t v = verts[side];
            for (int i = v2f_offset[v]; i < v2f_offset[v + 1]; ++i) {
                int f = v2f[i];
                if (edges[e].cost < best_cost[f] || (edges[e].cost == best_cost[f] && e < best_edge[f])) {
                    best_cost[f] = edges[e].cost;
                    best_edge[f] = (int) e;
                }
            }
        }
    }
    free(best_cost);
    *best_edge_out = best_edge;
    return 1;
}

static int vkmesh_simplify_step_vulkan(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    *collapsed_edges = 0;
    *removed_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0) return 1;
    if (mesh->n_faces > UINT32_MAX || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer unique_edges;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer scratch;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&unique_edges, 0, sizeof(unique_edges));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));
    memset(&scratch, 0, sizeof(scratch));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const uint32_t offset_count = vertex_count + 1u;
    const size_t faces_bytes = (size_t) face_count * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) vertex_count * 3u * sizeof(float);
    const size_t degree_bytes = (size_t) vertex_count * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &counter)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t vertex_groups = (vertex_count + 127u) / 128u;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = faces_buffer;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = counter;
    degree_buffers[3] = counter;
    push.n = face_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = counter;
    seed_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer * scan_in = &offset_a_buffer;
    vkmesh_vk_buffer * scan_out = &offset_b_buffer;
    for (uint64_t stride = 1u; stride < (uint64_t) offset_count; stride <<= 1u) {
        vkmesh_vk_buffer scan_buffers[4];
        scan_buffers[0] = *scan_in;
        scan_buffers[1] = *scan_out;
        scan_buffers[2] = counter;
        scan_buffers[3] = counter;
        memset(&push, 0, sizeof(push));
        push.n = offset_count;
        push.aux0 = (uint32_t) stride;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_SCAN_U32_STRIDE, scan_buffers, &push, offset_groups)) goto cleanup;
        vkmesh_vk_buffer * tmp = scan_in;
        scan_in = scan_out;
        scan_out = tmp;
    }

    const uint32_t total_adj = ((const uint32_t *) scan_in->mapped)[vertex_count];
    if (total_adj != face_count * 3u) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) total_adj * sizeof(uint32_t), &adjacency_buffer)) goto cleanup;

    vkmesh_vk_buffer cursor_copy_buffers[4];
    cursor_copy_buffers[0] = *scan_in;
    cursor_copy_buffers[1] = *scan_out;
    cursor_copy_buffers[2] = counter;
    cursor_copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, cursor_copy_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = faces_buffer;
    adj_buffers[1] = *scan_out;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, face_groups)) goto cleanup;

    size_t raw_edge_count = 0;
    size_t raw_edge_sort_count = 0;
    if (!expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &edge_dummy, &raw_edge_count, &raw_edge_sort_count)) {
        goto cleanup;
    }
    if (raw_edge_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, raw_edge_count * 2u * sizeof(uint32_t), &unique_edges) ||
        !vk_buffer_create(vk, (size_t) vertex_count * sizeof(uint32_t), &boundary_flags)) {
        goto cleanup;
    }
    ((uint32_t *) counter.mapped)[0] = 0u;

    vkmesh_vk_buffer unique_buffers[4];
    unique_buffers[0] = edges_buffer;
    unique_buffers[1] = unique_edges;
    unique_buffers[2] = boundary_flags;
    unique_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = 0u;
    uint32_t edge_groups = (uint32_t) ((raw_edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, unique_buffers, &push, edge_groups)) goto cleanup;
    const uint32_t unique_edge_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_edge_count > raw_edge_count) goto cleanup;

    const uint64_t qem_base = 0u;
    const uint64_t offset_base = (uint64_t) vertex_count * 10ull;
    const uint64_t adjacency_base = offset_base + (uint64_t) offset_count;
    const uint64_t boundary_base = adjacency_base + (uint64_t) total_adj;
    const uint64_t cost_base = boundary_base + (uint64_t) vertex_count;
    const uint64_t best_cost_base = cost_base + (uint64_t) unique_edge_count;
    const uint64_t best_edge_base = best_cost_base + (uint64_t) face_count;
    const uint64_t face_keep_base = best_edge_base + (uint64_t) face_count;
    const uint64_t counter_base = face_keep_base + (uint64_t) face_count;
    const uint64_t scratch_count = counter_base + 1ull;
    (void) qem_base;
    if (scratch_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) scratch_count * sizeof(uint32_t), &scratch)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = *scan_in;
    copy_buffers[1] = scratch;
    copy_buffers[2] = counter;
    copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) offset_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, offset_groups)) goto cleanup;

    copy_buffers[0] = adjacency_buffer;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = total_adj;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) adjacency_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (total_adj + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = boundary_flags;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) boundary_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer fill_buffers[4];
    fill_buffers[0] = scratch;
    fill_buffers[1] = counter;
    fill_buffers[2] = counter;
    fill_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 0x7f800000u;
    push.aux1 = (uint32_t) best_cost_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = UINT32_MAX;
    push.aux1 = (uint32_t) best_edge_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 1u;
    push.aux1 = (uint32_t) face_keep_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = 1u;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) counter_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, 1u)) goto cleanup;

    vkmesh_vk_buffer qem_buffers[4];
    qem_buffers[0] = faces_buffer;
    qem_buffers[1] = vertices_buffer;
    qem_buffers[2] = scratch;
    qem_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, qem_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer cost_buffers[4];
    cost_buffers[0] = unique_edges;
    cost_buffers[1] = vertices_buffer;
    cost_buffers[2] = faces_buffer;
    cost_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = lambda_edge_length;
    push.rel_eps = lambda_skinny;
    uint32_t unique_groups = (unique_edge_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_EDGE_COST, cost_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer propagate_buffers[4];
    propagate_buffers[0] = unique_edges;
    propagate_buffers[1] = counter;
    propagate_buffers[2] = counter;
    propagate_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST, propagate_buffers, &push, unique_groups)) goto cleanup;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_BEST_EDGE, propagate_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer collapse_buffers[4];
    collapse_buffers[0] = unique_edges;
    collapse_buffers[1] = vertices_buffer;
    collapse_buffers[2] = faces_buffer;
    collapse_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = threshold;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES, collapse_buffers, &push, unique_groups)) goto cleanup;
    *collapsed_edges = (int) ((const uint32_t *) scratch.mapped)[counter_base];

    float * new_vertices = NULL;
    int64_t new_vertex_count = 0;
    int32_t * new_faces = NULL;
    int64_t new_face_count = 0;
    if (!compact_mesh_from_device_flags(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &scratch,
            (uint32_t) face_keep_base,
            &new_vertices,
            &new_vertex_count,
            &new_faces,
            &new_face_count)) {
        goto cleanup;
    }
    *removed_faces = (int) (mesh->n_faces - new_face_count);
    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = new_vertex_count;
    mesh->n_faces = new_face_count;
    mesh->vertex_capacity = new_vertex_count;
    mesh->face_capacity = new_face_count;
    mesh->has_uvs = 0;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &unique_edges);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &scratch);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_simplify_step(
    vkmesh_device_mesh * dm,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    uint32_t * collapsed_edges,
    uint32_t * removed_faces) {
    if (collapsed_edges != NULL) *collapsed_edges = 0u;
    if (removed_faces != NULL) *removed_faces = 0u;
    if (dm == NULL || dm->vk == NULL) return 0;
    if (dm->n_faces == 0u) return 1;
    if (dm->n_vertices == 0u || dm->n_faces > UINT32_MAX / 3u) return 0;

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer degree_buffer;
    vkmesh_vk_buffer offset_a_buffer;
    vkmesh_vk_buffer offset_b_buffer;
    vkmesh_vk_buffer adjacency_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer edge_dummy;
    vkmesh_vk_buffer unique_edges;
    vkmesh_vk_buffer boundary_flags;
    vkmesh_vk_buffer counter;
    vkmesh_vk_buffer scratch;
    memset(&degree_buffer, 0, sizeof(degree_buffer));
    memset(&offset_a_buffer, 0, sizeof(offset_a_buffer));
    memset(&offset_b_buffer, 0, sizeof(offset_b_buffer));
    memset(&adjacency_buffer, 0, sizeof(adjacency_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&edge_dummy, 0, sizeof(edge_dummy));
    memset(&unique_edges, 0, sizeof(unique_edges));
    memset(&boundary_flags, 0, sizeof(boundary_flags));
    memset(&counter, 0, sizeof(counter));
    memset(&scratch, 0, sizeof(scratch));

    const uint32_t vertex_count = dm->n_vertices;
    const uint32_t face_count = dm->n_faces;
    const uint32_t offset_count = vertex_count + 1u;
    const size_t degree_bytes = (size_t) vertex_count * sizeof(uint32_t);
    const size_t offset_bytes = (size_t) offset_count * sizeof(uint32_t);
    int ok = 0;

    if (!vk_buffer_create(vk, degree_bytes, &degree_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_a_buffer) ||
        !vk_buffer_create(vk, offset_bytes, &offset_b_buffer) ||
        !vk_buffer_create(vk, 4u * sizeof(uint32_t), &counter)) {
        goto cleanup;
    }

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    uint32_t face_groups = (face_count + 127u) / 128u;
    uint32_t vertex_groups = (vertex_count + 127u) / 128u;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = dm->faces;
    degree_buffers[1] = degree_buffer;
    degree_buffers[2] = counter;
    degree_buffers[3] = counter;
    push.n = face_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_DEGREE, degree_buffers, &push, face_groups)) goto cleanup;

    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = degree_buffer;
    seed_buffers[1] = offset_a_buffer;
    seed_buffers[2] = counter;
    seed_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    uint32_t offset_groups = (offset_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer * scan_in = &offset_a_buffer;
    vkmesh_vk_buffer * scan_out = &offset_b_buffer;
    for (uint64_t stride = 1u; stride < (uint64_t) offset_count; stride <<= 1u) {
        vkmesh_vk_buffer scan_buffers[4];
        scan_buffers[0] = *scan_in;
        scan_buffers[1] = *scan_out;
        scan_buffers[2] = counter;
        scan_buffers[3] = counter;
        memset(&push, 0, sizeof(push));
        push.n = offset_count;
        push.aux0 = (uint32_t) stride;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_SCAN_U32_STRIDE, scan_buffers, &push, offset_groups)) goto cleanup;
        vkmesh_vk_buffer * tmp = scan_in;
        scan_in = scan_out;
        scan_out = tmp;
    }

    const uint32_t total_adj = ((const uint32_t *) scan_in->mapped)[vertex_count];
    if (total_adj != face_count * 3u) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) total_adj * sizeof(uint32_t), &adjacency_buffer)) goto cleanup;

    vkmesh_vk_buffer cursor_copy_buffers[4];
    cursor_copy_buffers[0] = *scan_in;
    cursor_copy_buffers[1] = *scan_out;
    cursor_copy_buffers[2] = counter;
    cursor_copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, cursor_copy_buffers, &push, offset_groups)) goto cleanup;

    vkmesh_vk_buffer adj_buffers[4];
    adj_buffers[0] = dm->faces;
    adj_buffers[1] = *scan_out;
    adj_buffers[2] = adjacency_buffer;
    adj_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = vertex_count;
    push.aux1 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_FACE_ADJACENCY, adj_buffers, &push, face_groups)) goto cleanup;

    size_t raw_edge_count = 0;
    size_t raw_edge_sort_count = 0;
    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &edge_dummy, &raw_edge_count, &raw_edge_sort_count)) {
        goto cleanup;
    }
    if (raw_edge_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, raw_edge_count * 2u * sizeof(uint32_t), &unique_edges) ||
        !vk_buffer_create(vk, (size_t) vertex_count * sizeof(uint32_t), &boundary_flags)) {
        goto cleanup;
    }
    ((uint32_t *) counter.mapped)[0] = 0u;

    vkmesh_vk_buffer unique_buffers[4];
    unique_buffers[0] = edges_buffer;
    unique_buffers[1] = unique_edges;
    unique_buffers[2] = boundary_flags;
    unique_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) raw_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = 0u;
    uint32_t edge_groups = (uint32_t) ((raw_edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_UNIQUE_SIMPLIFY_EDGES, unique_buffers, &push, edge_groups)) goto cleanup;
    const uint32_t unique_edge_count = ((const uint32_t *) counter.mapped)[0];
    if ((size_t) unique_edge_count > raw_edge_count) goto cleanup;

    const uint64_t qem_base = 0u;
    const uint64_t offset_base = (uint64_t) vertex_count * 10ull;
    const uint64_t adjacency_base = offset_base + (uint64_t) offset_count;
    const uint64_t boundary_base = adjacency_base + (uint64_t) total_adj;
    const uint64_t cost_base = boundary_base + (uint64_t) vertex_count;
    const uint64_t best_cost_base = cost_base + (uint64_t) unique_edge_count;
    const uint64_t best_edge_base = best_cost_base + (uint64_t) face_count;
    const uint64_t face_keep_base = best_edge_base + (uint64_t) face_count;
    const uint64_t counter_base = face_keep_base + (uint64_t) face_count;
    const uint64_t scratch_count = counter_base + 1ull;
    (void) qem_base;
    if (scratch_count > UINT32_MAX) goto cleanup;
    if (!vk_buffer_create(vk, (size_t) scratch_count * sizeof(uint32_t), &scratch)) goto cleanup;

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = *scan_in;
    copy_buffers[1] = scratch;
    copy_buffers[2] = counter;
    copy_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = offset_count;
    push.aux1 = (uint32_t) offset_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, offset_groups)) goto cleanup;

    copy_buffers[0] = adjacency_buffer;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = total_adj;
    push.aux1 = (uint32_t) adjacency_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (total_adj + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = boundary_flags;
    copy_buffers[1] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux1 = (uint32_t) boundary_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer fill_buffers[4];
    fill_buffers[0] = scratch;
    fill_buffers[1] = counter;
    fill_buffers[2] = counter;
    fill_buffers[3] = counter;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 0x7f800000u;
    push.aux1 = (uint32_t) best_cost_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = UINT32_MAX;
    push.aux1 = (uint32_t) best_edge_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = face_count;
    push.aux0 = 1u;
    push.aux1 = (uint32_t) face_keep_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, face_groups)) goto cleanup;
    memset(&push, 0, sizeof(push));
    push.n = 1u;
    push.aux0 = 0u;
    push.aux1 = (uint32_t) counter_base;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_FILL_U32, fill_buffers, &push, 1u)) goto cleanup;

    vkmesh_vk_buffer qem_buffers[4];
    qem_buffers[0] = dm->faces;
    qem_buffers[1] = dm->vertices;
    qem_buffers[2] = scratch;
    qem_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_VERTEX_QEM, qem_buffers, &push, vertex_groups)) goto cleanup;

    vkmesh_vk_buffer cost_buffers[4];
    cost_buffers[0] = unique_edges;
    cost_buffers[1] = dm->vertices;
    cost_buffers[2] = dm->faces;
    cost_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = lambda_edge_length;
    push.rel_eps = lambda_skinny;
    uint32_t unique_groups = (unique_edge_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_EDGE_COST, cost_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer propagate_buffers[4];
    propagate_buffers[0] = unique_edges;
    propagate_buffers[1] = counter;
    propagate_buffers[2] = counter;
    propagate_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_PROPAGATE_COST, propagate_buffers, &push, unique_groups)) goto cleanup;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_BEST_EDGE, propagate_buffers, &push, unique_groups)) goto cleanup;

    vkmesh_vk_buffer collapse_buffers[4];
    collapse_buffers[0] = unique_edges;
    collapse_buffers[1] = dm->vertices;
    collapse_buffers[2] = dm->faces;
    collapse_buffers[3] = scratch;
    memset(&push, 0, sizeof(push));
    push.n = unique_edge_count;
    push.aux0 = vertex_count;
    push.aux1 = face_count;
    push.aux2 = total_adj;
    push.eps = threshold;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SIMPLIFY_COLLAPSE_EDGES, collapse_buffers, &push, unique_groups)) goto cleanup;

    uint32_t collapsed_u32 = ((const uint32_t *) scratch.mapped)[counter_base];
    uint32_t removed_u32 = 0u;
    if (!compact_device_mesh_from_flags(dm, &scratch, (uint32_t) face_keep_base, &removed_u32)) goto cleanup;
    if (collapsed_edges != NULL) *collapsed_edges = collapsed_u32;
    if (removed_faces != NULL) *removed_faces = removed_u32;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &degree_buffer);
    vk_buffer_destroy(vk, &offset_a_buffer);
    vk_buffer_destroy(vk, &offset_b_buffer);
    vk_buffer_destroy(vk, &adjacency_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &edge_dummy);
    vk_buffer_destroy(vk, &unique_edges);
    vk_buffer_destroy(vk, &boundary_flags);
    vk_buffer_destroy(vk, &counter);
    vk_buffer_destroy(vk, &scratch);
    return ok;
}

static int vkmesh_device_simplify(
    vkmesh_device_mesh * dm,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    if (dm == NULL || dm->vk == NULL || dm->n_vertices == 0u || dm->n_faces == 0u) {
        return 0;
    }
    if (target_faces <= 0) target_faces = 1;

    uint32_t total_collapsed_u32 = 0u;
    uint32_t total_removed_u32 = 0u;

    int step = 0;
    while (dm->n_faces > (uint32_t) target_faces && (max_steps <= 0 || step < max_steps)) {
        uint32_t before = dm->n_faces;
        uint32_t collapsed = 0u;
        uint32_t removed = 0u;
        fprintf(stderr,
            "vkmesh: simplify step=%d begin faces=%u target=%d threshold=%.9g\n",
            step + 1,
            dm->n_faces,
            target_faces,
            threshold);
        fflush(stderr);
        if (!vkmesh_device_simplify_step(dm, lambda_edge_length, lambda_skinny, threshold, &collapsed, &removed)) return 0;
        if (total_collapsed_u32 > UINT32_MAX - collapsed || total_removed_u32 > UINT32_MAX - removed) return 0;
        total_collapsed_u32 += collapsed;
        total_removed_u32 += removed;
        ++step;
        fprintf(stderr,
            "vkmesh: simplify step=%d end collapsed=%u removed_faces=%u faces=%u\n",
            step,
            collapsed,
            removed,
            dm->n_faces);
        fflush(stderr);
        if (dm->n_faces <= (uint32_t) target_faces) break;
        if (collapsed == 0u || removed == 0u || removed * 100u < before) threshold *= 10.0f;
    }

    if (total_collapsed_u32 > (uint32_t) INT_MAX || total_removed_u32 > (uint32_t) INT_MAX) return 0;
    *total_collapsed = (int) total_collapsed_u32;
    *total_removed = (int) total_removed_u32;
    return 1;
}

static int vkmesh_simplify_device_vulkan(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    if (mesh == NULL || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    if (target_faces <= 0) target_faces = 1;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_simplify(
            &dm,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            threshold,
            max_steps,
            total_collapsed,
            total_removed)) {
        goto cleanup;
    }
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_simplify_step_cpu(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    *collapsed_edges = 0;
    *removed_faces = 0;
    if (mesh->n_faces <= 0) return 1;
    int * v2f_offset = NULL;
    int * v2f = NULL;
    vkmesh_simplify_edge * edges = NULL;
    int64_t edge_count = 0;
    uint8_t * boundary = NULL;
    vkmesh_qem * qems = NULL;
    int * best_edge = NULL;
    uint8_t * selected_vertex = NULL;
    uint8_t * selected_edge = NULL;
    uint8_t * keep_face = NULL;
    int ok = 0;

    if (!build_vertex_face_adjacency(mesh, &v2f_offset, &v2f)) goto cleanup;
    if (!get_unique_simplify_edges(mesh, &edges, &edge_count, &boundary)) goto cleanup;
    if (!build_vertex_qems(mesh, &qems)) goto cleanup;
    for (int64_t e = 0; e < edge_count; ++e) {
        edges[e].cost = simplify_edge_cost(mesh, v2f_offset, v2f, boundary, qems, lambda_edge_length, lambda_skinny, &edges[e]);
    }
    if (!propagate_edge_cost_to_faces(mesh, v2f_offset, v2f, edges, edge_count, &best_edge)) goto cleanup;
    selected_vertex = (uint8_t *) calloc((size_t) mesh->n_vertices, sizeof(uint8_t));
    selected_edge = (uint8_t *) calloc((size_t) edge_count, sizeof(uint8_t));
    keep_face = (uint8_t *) malloc((size_t) mesh->n_faces);
    if (selected_vertex == NULL || selected_edge == NULL || keep_face == NULL) goto cleanup;
    memset(keep_face, 1, (size_t) mesh->n_faces);

    for (int64_t e = 0; e < edge_count; ++e) {
        if (edges[e].cost > (double) threshold) continue;
        int32_t verts[2] = { edges[e].v0, edges[e].v1 };
        int can_collapse = !selected_vertex[verts[0]] && !selected_vertex[verts[1]];
        for (int side = 0; side < 2 && can_collapse; ++side) {
            int32_t v = verts[side];
            for (int i = v2f_offset[v]; i < v2f_offset[v + 1]; ++i) {
                if (best_edge[v2f[i]] != (int) e) {
                    can_collapse = 0;
                    break;
                }
            }
        }
        if (!can_collapse) continue;
        selected_edge[e] = 1u;
        selected_vertex[verts[0]] = 1u;
        selected_vertex[verts[1]] = 1u;
        ++(*collapsed_edges);
    }

    for (int64_t e = 0; e < edge_count; ++e) {
        if (!selected_edge[e]) continue;
        int32_t v0 = edges[e].v0;
        int32_t v1 = edges[e].v1;
        memcpy(mesh->vertices + (size_t) v0 * 3u, edges[e].pos, 3u * sizeof(float));
        for (int i = v2f_offset[v0]; i < v2f_offset[v0 + 1]; ++i) {
            int face_id = v2f[i];
            int32_t * f = mesh->faces + (size_t) face_id * 3u;
            if (face_contains_vertex(f, v1)) keep_face[face_id] = 0u;
        }
        for (int i = v2f_offset[v1]; i < v2f_offset[v1 + 1]; ++i) {
            int face_id = v2f[i];
            int32_t * f = mesh->faces + (size_t) face_id * 3u;
            if (f[0] == v1) f[0] = v0;
            if (f[1] == v1) f[1] = v0;
            if (f[2] == v1) f[2] = v0;
        }
    }
    if (!mesh_remove_faces_by_mask_vulkan(mesh, keep_face, removed_faces) &&
        !mesh_remove_faces_by_mask(mesh, keep_face, removed_faces)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(v2f_offset);
    free(v2f);
    free(edges);
    free(boundary);
    free(qems);
    free(best_edge);
    free(selected_vertex);
    free(selected_edge);
    free(keep_face);
    return ok;
}

static int vkmesh_simplify_step(
    vkmesh_mesh * mesh,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int * collapsed_edges,
    int * removed_faces) {
    if (vkmesh_simplify_step_vulkan(mesh, lambda_edge_length, lambda_skinny, threshold, collapsed_edges, removed_faces)) {
        return 1;
    }
    return vkmesh_simplify_step_cpu(mesh, lambda_edge_length, lambda_skinny, threshold, collapsed_edges, removed_faces);
}

static int vkmesh_simplify(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float threshold,
    int max_steps,
    int * total_collapsed,
    int * total_removed) {
    *total_collapsed = 0;
    *total_removed = 0;
    if (target_faces <= 0) target_faces = 1;
    if (vkmesh_simplify_device_vulkan(
            mesh,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            threshold,
            max_steps,
            total_collapsed,
            total_removed)) {
        return 1;
    }

    int step = 0;
    while (mesh->n_faces > target_faces && (max_steps <= 0 || step < max_steps)) {
        int before = (int) mesh->n_faces;
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr,
            "vkmesh: simplify step=%d begin faces=%" PRId64 " target=%d threshold=%.9g\n",
            step + 1,
            mesh->n_faces,
            target_faces,
            threshold);
        fflush(stderr);
        if (!vkmesh_simplify_step(mesh, lambda_edge_length, lambda_skinny, threshold, &collapsed, &removed)) return 0;
        *total_collapsed += collapsed;
        *total_removed += removed;
        ++step;
        fprintf(stderr,
            "vkmesh: simplify step=%d end collapsed=%d removed_faces=%d faces=%" PRId64 "\n",
            step,
            collapsed,
            removed,
            mesh->n_faces);
        fflush(stderr);
        if (mesh->n_faces <= target_faces) break;
        if (collapsed == 0 || removed == 0 || removed * 100 < before) threshold *= 10.0f;
    }
    return 1;
}

static int vkmesh_uv_unwrap(vkmesh_mesh * mesh, int texture_size, int * old_vertices, int * new_vertices, int * chart_count) {
    *old_vertices = (int) mesh->n_vertices;
    *new_vertices = (int) mesh->n_vertices;
    *chart_count = 0;
    if (mesh->n_vertices <= 0 || mesh->n_faces <= 0 ||
        mesh->n_vertices > UINT32_MAX || mesh->n_faces > UINT32_MAX / 3u) {
        return 0;
    }
    uint32_t * indices = (uint32_t *) malloc((size_t) mesh->n_faces * 3u * sizeof(uint32_t));
    if (indices == NULL) return 0;
    for (int64_t i = 0; i < mesh->n_faces * 3; ++i) indices[i] = (uint32_t) mesh->faces[i];

    /* CuMesh accelerates chart clustering on GPU, then hands CPU chart meshes to xatlas. */
    xatlasSetPrint(NULL, false);
    xatlasAtlas * atlas = xatlasCreate();
    if (atlas == NULL) {
        free(indices);
        return 0;
    }
    xatlasMeshDecl decl;
    xatlasMeshDeclInit(&decl);
    decl.vertexCount = (uint32_t) mesh->n_vertices;
    decl.vertexPositionData = mesh->vertices;
    decl.vertexPositionStride = 3u * sizeof(float);
    decl.indexCount = (uint32_t) mesh->n_faces * 3u;
    decl.indexData = indices;
    decl.indexFormat = XATLAS_INDEX_FORMAT_UINT32;
    xatlasAddMeshError add_error = xatlasAddMesh(atlas, &decl, 1);
    if (add_error != XATLAS_ADD_MESH_ERROR_SUCCESS) {
        fprintf(stderr, "vkmesh: xatlas AddMesh failed: %s\n", xatlasAddMeshErrorString(add_error));
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    xatlasPackOptions pack_options;
    xatlasPackOptionsInit(&pack_options);
    pack_options.resolution = texture_size > 0 ? (uint32_t) texture_size : 1024u;
    pack_options.padding = 4;
    pack_options.createImage = false;
    xatlasGenerate(atlas, NULL, &pack_options);
    if (atlas->meshCount != 1 || atlas->meshes == NULL || atlas->meshes[0].vertexCount == 0) {
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    const xatlasMesh * xm = &atlas->meshes[0];
    float * out_vertices = (float *) malloc((size_t) xm->vertexCount * 3u * sizeof(float));
    float * out_uvs = (float *) malloc((size_t) xm->vertexCount * 2u * sizeof(float));
    int32_t * out_faces = (int32_t *) malloc((size_t) xm->indexCount * sizeof(int32_t));
    if (out_vertices == NULL || out_uvs == NULL || out_faces == NULL) {
        free(out_vertices); free(out_uvs); free(out_faces);
        xatlasDestroy(atlas);
        free(indices);
        return 0;
    }
    float atlas_w = atlas->width > 0 ? (float) atlas->width : (float) pack_options.resolution;
    float atlas_h = atlas->height > 0 ? (float) atlas->height : (float) pack_options.resolution;
    for (uint32_t i = 0; i < xm->vertexCount; ++i) {
        const xatlasVertex * v = &xm->vertexArray[i];
        uint32_t src = v->xref;
        memcpy(out_vertices + (size_t) i * 3u, mesh->vertices + (size_t) src * 3u, 3u * sizeof(float));
        out_uvs[(size_t) i * 2u + 0u] = clamp01(v->uv[0] / atlas_w);
        out_uvs[(size_t) i * 2u + 1u] = clamp01(1.0f - v->uv[1] / atlas_h);
    }
    for (uint32_t i = 0; i < xm->indexCount; ++i) out_faces[i] = (int32_t) xm->indexArray[i];

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = out_vertices;
    mesh->uvs = out_uvs;
    mesh->faces = out_faces;
    mesh->n_vertices = (int64_t) xm->vertexCount;
    mesh->n_faces = (int64_t) xm->indexCount / 3;
    mesh->vertex_capacity = mesh->n_vertices;
    mesh->face_capacity = mesh->n_faces;
    mesh->has_uvs = 1;
    *new_vertices = (int) mesh->n_vertices;
    *chart_count = (int) atlas->chartCount;
    xatlasDestroy(atlas);
    free(indices);
    return 1;
}

static void bvh_bounds_init(float bmin[3], float bmax[3]) {
    bmin[0] = bmin[1] = bmin[2] = FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;
}

static void bvh_bounds_add(float bmin[3], float bmax[3], const float pmin[3], const float pmax[3]) {
    for (int i = 0; i < 3; ++i) {
        if (pmin[i] < bmin[i]) bmin[i] = pmin[i];
        if (pmax[i] > bmax[i]) bmax[i] = pmax[i];
    }
}

static uint32_t bvh_build_recursive(
    vkmesh_bvh_tri * tris,
    int start,
    int count,
    vkmesh_bvh_node * nodes,
    uint32_t * node_count) {
    uint32_t node_id = (*node_count)++;
    vkmesh_bvh_node * node = &nodes[node_id];
    bvh_bounds_init(node->bmin, node->bmax);
    for (int i = start; i < start + count; ++i) {
        bvh_bounds_add(node->bmin, node->bmax, tris[i].bmin, tris[i].bmax);
    }
    if (count <= 4) {
        node->left = (uint32_t) start;
        node->meta = 0x80000000u | (uint32_t) count;
        return node_id;
    }

    float cmin[3];
    float cmax[3];
    bvh_bounds_init(cmin, cmax);
    for (int i = start; i < start + count; ++i) {
        bvh_bounds_add(cmin, cmax, tris[i].centroid, tris[i].centroid);
    }
    float ex[3] = { cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2] };
    int axis = 0;
    if (ex[1] > ex[axis]) axis = 1;
    if (ex[2] > ex[axis]) axis = 2;
    float split = 0.5f * (cmin[axis] + cmax[axis]);
    int mid = start;
    for (int i = start; i < start + count; ++i) {
        if (tris[i].centroid[axis] < split) {
            vkmesh_bvh_tri tmp = tris[mid];
            tris[mid] = tris[i];
            tris[i] = tmp;
            ++mid;
        }
    }
    int left_count = mid - start;
    if (left_count <= 0 || left_count >= count) left_count = count / 2;
    uint32_t left = bvh_build_recursive(tris, start, left_count, nodes, node_count);
    uint32_t right = bvh_build_recursive(tris, start + left_count, count - left_count, nodes, node_count);
    node->left = left;
    node->meta = right;
    return node_id;
}

static int vkmesh_build_bvh(
    const vkmesh_mesh * mesh,
    vkmesh_bvh_node ** nodes_out,
    uint32_t * node_count_out,
    uint32_t ** tri_indices_out) {
    *nodes_out = NULL;
    *node_count_out = 0;
    *tri_indices_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 2u) return 0;
    vkmesh_bvh_tri * tris = (vkmesh_bvh_tri *) malloc((size_t) mesh->n_faces * sizeof(*tris));
    vkmesh_bvh_node * nodes = (vkmesh_bvh_node *) malloc((size_t) mesh->n_faces * 2u * sizeof(*nodes));
    uint32_t * tri_indices = (uint32_t *) malloc((size_t) mesh->n_faces * sizeof(uint32_t));
    if (tris == NULL || nodes == NULL || tri_indices == NULL) {
        free(tris); free(nodes); free(tri_indices);
        return 0;
    }
    for (int64_t i = 0; i < mesh->n_faces; ++i) {
        const int32_t * f = mesh->faces + (size_t) i * 3u;
        bvh_bounds_init(tris[i].bmin, tris[i].bmax);
        float sum[3] = { 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < 3; ++k) {
            const float * v = mesh->vertices + (size_t) f[k] * 3u;
            bvh_bounds_add(tris[i].bmin, tris[i].bmax, v, v);
            sum[0] += v[0];
            sum[1] += v[1];
            sum[2] += v[2];
        }
        tris[i].centroid[0] = sum[0] / 3.0f;
        tris[i].centroid[1] = sum[1] / 3.0f;
        tris[i].centroid[2] = sum[2] / 3.0f;
        tris[i].face = (uint32_t) i;
    }
    uint32_t node_count = 0;
    (void) bvh_build_recursive(tris, 0, (int) mesh->n_faces, nodes, &node_count);
    for (int64_t i = 0; i < mesh->n_faces; ++i) tri_indices[i] = tris[i].face;
    free(tris);
    *nodes_out = nodes;
    *node_count_out = node_count;
    *tri_indices_out = tri_indices;
    return 1;
}

static uint32_t u32_from_float(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = value;
    return v.u;
}

static int pack_bvh_aux_buffer(
    const float * points,
    int64_t point_count,
    const vkmesh_bvh_node * nodes,
    uint32_t node_count,
    const uint32_t * tri_indices,
    int64_t tri_count,
    uint32_t ** words_out,
    size_t * word_count_out) {
    *words_out = NULL;
    *word_count_out = 0;
    size_t point_words = (size_t) point_count * 3u;
    size_t node_words = (size_t) node_count * 8u;
    size_t tri_words = (size_t) tri_count;
    size_t total_words = point_words + node_words + tri_words;
    uint32_t * words = (uint32_t *) malloc(total_words * sizeof(uint32_t));
    if (words == NULL) return 0;
    for (int64_t i = 0; i < point_count * 3; ++i) words[i] = u32_from_float(points[i]);
    size_t node_base = point_words;
    for (uint32_t i = 0; i < node_count; ++i) {
        size_t base = node_base + (size_t) i * 8u;
        words[base + 0u] = u32_from_float(nodes[i].bmin[0]);
        words[base + 1u] = u32_from_float(nodes[i].bmin[1]);
        words[base + 2u] = u32_from_float(nodes[i].bmin[2]);
        words[base + 3u] = nodes[i].left;
        words[base + 4u] = u32_from_float(nodes[i].bmax[0]);
        words[base + 5u] = u32_from_float(nodes[i].bmax[1]);
        words[base + 6u] = u32_from_float(nodes[i].bmax[2]);
        words[base + 7u] = nodes[i].meta;
    }
    memcpy(words + point_words + node_words, tri_indices, tri_words * sizeof(uint32_t));
    *words_out = words;
    *word_count_out = total_words;
    return 1;
}

static int load_points(const char * path, float ** points_out, int64_t * count_out) {
    *points_out = NULL;
    *count_out = 0;
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open points file %s\n", path);
        return 0;
    }
    float * points = NULL;
    int64_t count = 0;
    int64_t capacity = 0;
    char line[4096];
    int64_t line_no = 0;
    int ok = 1;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        char * p = line;
        while (*p != '\0' && isspace((unsigned char) *p)) ++p;
        if (*p == '\0' || *p == '#') continue;
        float x, y, z;
        if (sscanf(p, "%f %f %f", &x, &y, &z) != 3) {
            fprintf(stderr, "vkmesh: bad point at %s:%" PRId64 "\n", path, line_no);
            ok = 0;
            break;
        }
        if (count == capacity) {
            int64_t next_cap = capacity > 0 ? capacity * 2 : 1024;
            float * next = (float *) realloc(points, (size_t) next_cap * 3u * sizeof(float));
            if (next == NULL) {
                ok = 0;
                break;
            }
            points = next;
            capacity = next_cap;
        }
        points[(size_t) count * 3u + 0u] = x;
        points[(size_t) count * 3u + 1u] = y;
        points[(size_t) count * 3u + 2u] = z;
        ++count;
    }
    fclose(f);
    if (!ok || count == 0) {
        free(points);
        if (ok) fprintf(stderr, "vkmesh: points file %s is empty\n", path);
        return 0;
    }
    *points_out = points;
    *count_out = count;
    return 1;
}

static float float_from_u32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

static int vkmesh_unsigned_distance_vulkan(
    const vkmesh_mesh * mesh,
    const float * points,
    int64_t point_count,
    float ** distances_out,
    uint32_t ** face_ids_out) {
    *distances_out = NULL;
    *face_ids_out = NULL;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX || point_count <= 0 || point_count > UINT32_MAX) return 0;

    vkmesh_bvh_node * nodes = NULL;
    uint32_t node_count = 0;
    uint32_t * tri_indices = NULL;
    uint32_t * aux_words = NULL;
    size_t aux_word_count = 0;
    if (!vkmesh_build_bvh(mesh, &nodes, &node_count, &tri_indices) ||
        !pack_bvh_aux_buffer(points, point_count, nodes, node_count, tri_indices, mesh->n_faces, &aux_words, &aux_word_count)) {
        free(nodes);
        free(tri_indices);
        free(aux_words);
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        free(nodes);
        free(tri_indices);
        free(aux_words);
        return 0;
    }
    vkmesh_vk_buffer buffers[4];
    memset(buffers, 0, sizeof(buffers));
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t out_bytes = (size_t) point_count * 2u * sizeof(uint32_t);
    const size_t aux_bytes = aux_word_count * sizeof(uint32_t);
    int ok = 0;
    if (!vk_buffer_create(vk, faces_bytes, &buffers[0]) ||
        !vk_buffer_create(vk, vertices_bytes, &buffers[1]) ||
        !vk_buffer_create(vk, out_bytes, &buffers[2]) ||
        !vk_buffer_create(vk, aux_bytes, &buffers[3])) {
        goto cleanup;
    }
    memcpy(buffers[0].mapped, mesh->faces, faces_bytes);
    memcpy(buffers[1].mapped, mesh->vertices, vertices_bytes);
    memcpy(buffers[3].mapped, aux_words, aux_bytes);

    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) point_count;
    push.aux0 = (uint32_t) mesh->n_faces;
    push.aux1 = node_count;
    uint32_t groups = (uint32_t) ((point_count + 127) / 128);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNSIGNED_DISTANCE, buffers, &push, groups)) goto cleanup;

    float * distances = (float *) malloc((size_t) point_count * sizeof(float));
    uint32_t * face_ids = (uint32_t *) malloc((size_t) point_count * sizeof(uint32_t));
    if (distances == NULL || face_ids == NULL) {
        free(distances); free(face_ids);
        goto cleanup;
    }
    const uint32_t * raw = (const uint32_t *) buffers[2].mapped;
    for (int64_t i = 0; i < point_count; ++i) {
        distances[i] = float_from_u32(raw[(size_t) i * 2u + 0u]);
        face_ids[i] = raw[(size_t) i * 2u + 1u];
    }
    *distances_out = distances;
    *face_ids_out = face_ids;
    ok = 1;

cleanup:
    for (uint32_t i = 0; i < 4; ++i) vk_buffer_destroy(vk, &buffers[i]);
    if (owns_vk) vkmesh_vk_destroy(vk);
    free(nodes);
    free(tri_indices);
    free(aux_words);
    return ok;
}

static int write_distances(const char * path, const float * points, int64_t point_count, const float * distances, const uint32_t * face_ids) {
    FILE * f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "vkmesh: failed to open %s for writing\n", path);
        return 0;
    }
    fprintf(f, "# x y z unsigned_distance face_id\n");
    for (int64_t i = 0; i < point_count; ++i) {
        const float * p = points + (size_t) i * 3u;
        fprintf(f, "%.9g %.9g %.9g %.9g %u\n", p[0], p[1], p[2], distances[i], face_ids[i]);
    }
    fclose(f);
    return 1;
}

static uint32_t vkmesh_find_parent_u32(const uint32_t * parent, uint32_t count, uint32_t x) {
    if (parent == NULL || x >= count) return UINT32_MAX;
    uint32_t r = x;
    for (uint32_t i = 0; i < 256u && parent[r] < count && parent[r] != r; ++i) r = parent[r];
    return r < count ? r : UINT32_MAX;
}

static int compute_hole_components_vulkan(
    const vkmesh_mesh * mesh,
    vkmesh_boundary_edge ** boundary_out,
    uint32_t ** parent_out,
    uint32_t ** comp_edges_out,
    uint32_t ** comp_invalid_out,
    float ** comp_perim_out,
    float ** comp_mid_out,
    int64_t * boundary_count_out) {
    *boundary_out = NULL;
    *parent_out = NULL;
    *comp_edges_out = NULL;
    *comp_invalid_out = NULL;
    *comp_perim_out = NULL;
    *comp_mid_out = NULL;
    *boundary_count_out = 0;
    if (mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) {
        return 0;
    }

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const size_t faces_bytes = (size_t) mesh->n_faces * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) mesh->n_vertices * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) mesh->n_faces * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(
            vk,
            mesh,
            &faces_buffer,
            &vertices_buffer,
            &edges_buffer,
            &dummy_buffer,
            &edge_count,
            &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count > max_boundary_count) goto cleanup;
    if (boundary_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    const uint32_t comp_edges_off = owner_off + vertex_count;
    const uint32_t comp_invalid_off = comp_edges_off + boundary_count;
    const uint32_t comp_perim_off = comp_invalid_off + boundary_count;
    const uint32_t comp_mid_off = comp_perim_off + boundary_count;
    const size_t hole_words = hole_words_size;
    if (!vk_buffer_create(vk, hole_words * sizeof(uint32_t), &hole_data)) goto cleanup;
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = counter_buffer;
    init_buffers[2] = counter_buffer;
    init_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = vertices_buffer;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_boundary_edge * boundary = (vkmesh_boundary_edge *) malloc((size_t) boundary_count * sizeof(*boundary));
    uint32_t * parents = (uint32_t *) malloc((size_t) boundary_count * sizeof(*parents));
    uint32_t * comp_edges = (uint32_t *) malloc((size_t) boundary_count * sizeof(*comp_edges));
    uint32_t * comp_invalid = (uint32_t *) malloc((size_t) boundary_count * sizeof(*comp_invalid));
    float * comp_perim = (float *) malloc((size_t) boundary_count * sizeof(*comp_perim));
    float * comp_mid = (float *) malloc((size_t) boundary_count * 3u * sizeof(*comp_mid));
    if (boundary == NULL || parents == NULL || comp_edges == NULL || comp_invalid == NULL ||
        comp_perim == NULL || comp_mid == NULL) {
        free(boundary);
        free(parents);
        free(comp_edges);
        free(comp_invalid);
        free(comp_perim);
        free(comp_mid);
        goto cleanup;
    }
    uint32_t * hole_words_u32 = (uint32_t *) hole_data.mapped;
    memcpy(boundary, boundary_buffer.mapped, (size_t) boundary_count * sizeof(*boundary));
    memcpy(parents, hole_words_u32, (size_t) boundary_count * sizeof(*parents));
    memcpy(comp_edges, hole_words_u32 + comp_edges_off, (size_t) boundary_count * sizeof(*comp_edges));
    memcpy(comp_invalid, hole_words_u32 + comp_invalid_off, (size_t) boundary_count * sizeof(*comp_invalid));
    memcpy(comp_perim, hole_words_u32 + comp_perim_off, (size_t) boundary_count * sizeof(*comp_perim));
    memcpy(comp_mid, hole_words_u32 + comp_mid_off, (size_t) boundary_count * 3u * sizeof(*comp_mid));
    *boundary_out = boundary;
    *parent_out = parents;
    *comp_edges_out = comp_edges;
    *comp_invalid_out = comp_invalid;
    *comp_perim_out = comp_perim;
    *comp_mid_out = comp_mid;
    *boundary_count_out = (int64_t) boundary_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_fill_holes_cpu(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    vkmesh_boundary_edge * boundary = NULL;
    int64_t boundary_count = 0;
    if (!get_boundary_edges(mesh, &boundary, &boundary_count)) return 0;
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

static int vkmesh_scan_flags_to_offsets(
    vkmesh_vk * vk,
    vkmesh_vk_buffer * flags,
    uint32_t count,
    vkmesh_vk_buffer * offsets_a,
    vkmesh_vk_buffer * offsets_b,
    vkmesh_vk_buffer * dummy,
    vkmesh_vk_buffer ** offsets_out) {
    if (vk == NULL || flags == NULL || offsets_a == NULL || offsets_b == NULL || dummy == NULL || offsets_out == NULL) return 0;
    *offsets_out = NULL;
    vkmesh_vk_buffer seed_buffers[4];
    seed_buffers[0] = *flags;
    seed_buffers[1] = *offsets_a;
    seed_buffers[2] = *dummy;
    seed_buffers[3] = *dummy;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = count + 1u;
    uint32_t groups = (push.n + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_SEED_VERTEX_OFFSETS, seed_buffers, &push, groups)) return 0;

    vkmesh_vk_buffer * scan_in = offsets_a;
    vkmesh_vk_buffer * scan_out = offsets_b;
    for (uint64_t stride = 1u; stride < (uint64_t) push.n; stride <<= 1u) {
        vkmesh_vk_buffer scan_buffers[4];
        scan_buffers[0] = *scan_in;
        scan_buffers[1] = *scan_out;
        scan_buffers[2] = *dummy;
        scan_buffers[3] = *dummy;
        memset(&push, 0, sizeof(push));
        push.n = count + 1u;
        push.aux0 = (uint32_t) stride;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_SCAN_U32_STRIDE, scan_buffers, &push, groups)) return 0;
        vkmesh_vk_buffer * tmp = scan_in;
        scan_in = scan_out;
        scan_out = tmp;
    }
    *offsets_out = scan_in;
    return 1;
}

static int vkmesh_fill_holes_vulkan(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX / 3u ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_vk_buffer faces_buffer;
    vkmesh_vk_buffer vertices_buffer;
    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    vkmesh_vk_buffer root_flags;
    vkmesh_vk_buffer root_offsets_a;
    vkmesh_vk_buffer root_offsets_b;
    vkmesh_vk_buffer face_flags;
    vkmesh_vk_buffer face_offsets_a;
    vkmesh_vk_buffer face_offsets_b;
    vkmesh_vk_buffer aux;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&faces_buffer, 0, sizeof(faces_buffer));
    memset(&vertices_buffer, 0, sizeof(vertices_buffer));
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));
    memset(&root_flags, 0, sizeof(root_flags));
    memset(&root_offsets_a, 0, sizeof(root_offsets_a));
    memset(&root_offsets_b, 0, sizeof(root_offsets_b));
    memset(&face_flags, 0, sizeof(face_flags));
    memset(&face_offsets_a, 0, sizeof(face_offsets_a));
    memset(&face_offsets_b, 0, sizeof(face_offsets_b));
    memset(&aux, 0, sizeof(aux));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t vertex_count = (uint32_t) mesh->n_vertices;
    const uint32_t face_count = (uint32_t) mesh->n_faces;
    const size_t faces_bytes = (size_t) face_count * 3u * sizeof(int32_t);
    const size_t vertices_bytes = (size_t) vertex_count * 3u * sizeof(float);
    const size_t max_boundary_count = (size_t) face_count * 3u;
    int ok = 0;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;

    if (!vk_buffer_create(vk, faces_bytes, &faces_buffer) ||
        !vk_buffer_create(vk, vertices_bytes, &vertices_buffer) ||
        !vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }
    memcpy(faces_buffer.mapped, mesh->faces, faces_bytes);
    memcpy(vertices_buffer.mapped, mesh->vertices, vertices_bytes);

    if (!expand_edges_device(vk, mesh, &faces_buffer, &vertices_buffer, &edges_buffer, &dummy_buffer, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count > max_boundary_count) goto cleanup;
    if (boundary_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    (void) owner_off;
    if (!vk_buffer_create(vk, hole_words_size * sizeof(uint32_t), &hole_data) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &root_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_b) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &face_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_b)) {
        goto cleanup;
    }
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = dummy_buffer;
    init_buffers[2] = dummy_buffer;
    init_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = vertices_buffer;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer mark_root_buffers[4];
    mark_root_buffers[0] = hole_data;
    mark_root_buffers[1] = root_flags;
    mark_root_buffers[2] = dummy_buffer;
    mark_root_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_ROOTS, mark_root_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * root_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &root_flags, boundary_count, &root_offsets_a, &root_offsets_b, &dummy_buffer, &root_offsets)) goto cleanup;
    uint32_t new_vertex_count = ((const uint32_t *) root_offsets->mapped)[boundary_count];
    if (new_vertex_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    vkmesh_vk_buffer mark_face_buffers[4];
    mark_face_buffers[0] = hole_data;
    mark_face_buffers[1] = root_flags;
    mark_face_buffers[2] = face_flags;
    mark_face_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_FACES, mark_face_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * face_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &face_flags, boundary_count, &face_offsets_a, &face_offsets_b, &dummy_buffer, &face_offsets)) goto cleanup;
    uint32_t new_face_count = ((const uint32_t *) face_offsets->mapped)[boundary_count];
    if (new_face_count == 0u ||
        vertex_count > UINT32_MAX - new_vertex_count ||
        face_count > UINT32_MAX - new_face_count) {
        goto cleanup;
    }

    const uint32_t out_vertex_count = vertex_count + new_vertex_count;
    const uint32_t out_face_count = face_count + new_face_count;
    const size_t aux_words = (size_t) boundary_count * 2u + 1u;
    if (!vk_buffer_create(vk, (size_t) out_vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) out_face_count * 3u * sizeof(int32_t), &out_faces) ||
        !vk_buffer_create(vk, aux_words * sizeof(uint32_t), &aux)) {
        goto cleanup;
    }
    memset(aux.mapped, 0xff, (size_t) boundary_count * sizeof(uint32_t));

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = vertices_buffer;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = dummy_buffer;
    copy_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = faces_buffer;
    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_vertex_buffers[4];
    write_vertex_buffers[0] = hole_data;
    write_vertex_buffers[1] = *root_offsets;
    write_vertex_buffers[2] = aux;
    write_vertex_buffers[3] = out_vertices;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.aux1 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_VERTICES, write_vertex_buffers, &push, boundary_groups)) goto cleanup;

    copy_buffers[0] = *face_offsets;
    copy_buffers[1] = aux;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count + 1u;
    push.aux0 = 0u;
    push.aux1 = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_face_buffers[4];
    write_face_buffers[0] = boundary_buffer;
    write_face_buffers[1] = hole_data;
    write_face_buffers[2] = aux;
    write_face_buffers[3] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux1 = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_FACES, write_face_buffers, &push, boundary_groups)) goto cleanup;

    float * new_vertices = (float *) malloc((size_t) out_vertex_count * 3u * sizeof(float));
    int32_t * new_faces = (int32_t *) malloc((size_t) out_face_count * 3u * sizeof(int32_t));
    if (new_vertices == NULL || new_faces == NULL) {
        free(new_vertices);
        free(new_faces);
        goto cleanup;
    }
    memcpy(new_vertices, out_vertices.mapped, (size_t) out_vertex_count * 3u * sizeof(float));
    memcpy(new_faces, out_faces.mapped, (size_t) out_face_count * 3u * sizeof(int32_t));

    free(mesh->vertices);
    free(mesh->uvs);
    free(mesh->faces);
    mesh->vertices = new_vertices;
    mesh->uvs = NULL;
    mesh->faces = new_faces;
    mesh->n_vertices = out_vertex_count;
    mesh->n_faces = out_face_count;
    mesh->vertex_capacity = out_vertex_count;
    mesh->face_capacity = out_face_count;
    mesh->has_uvs = 0;
    *filled_loops = (int) new_vertex_count;
    *added_faces = (int) new_face_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &faces_buffer);
    vk_buffer_destroy(vk, &vertices_buffer);
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    vk_buffer_destroy(vk, &root_flags);
    vk_buffer_destroy(vk, &root_offsets_a);
    vk_buffer_destroy(vk, &root_offsets_b);
    vk_buffer_destroy(vk, &face_flags);
    vk_buffer_destroy(vk, &face_offsets_a);
    vk_buffer_destroy(vk, &face_offsets_b);
    vk_buffer_destroy(vk, &aux);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_device_fill_holes(
    vkmesh_device_mesh * dm,
    float max_hole_perimeter,
    uint32_t * filled_loops,
    uint32_t * added_faces) {
    if (filled_loops != NULL) *filled_loops = 0u;
    if (added_faces != NULL) *added_faces = 0u;
    if (dm == NULL || dm->vk == NULL || dm->n_faces == 0u || dm->n_vertices == 0u ||
        dm->n_faces > UINT32_MAX / 3u) {
        return 0;
    }

    vkmesh_vk * vk = dm->vk;
    vkmesh_mesh view;
    memset(&view, 0, sizeof(view));
    view.n_vertices = dm->n_vertices;
    view.n_faces = dm->n_faces;

    vkmesh_vk_buffer edges_buffer;
    vkmesh_vk_buffer dummy_buffer;
    vkmesh_vk_buffer boundary_buffer;
    vkmesh_vk_buffer counter_buffer;
    vkmesh_vk_buffer hole_data;
    vkmesh_vk_buffer root_flags;
    vkmesh_vk_buffer root_offsets_a;
    vkmesh_vk_buffer root_offsets_b;
    vkmesh_vk_buffer face_flags;
    vkmesh_vk_buffer face_offsets_a;
    vkmesh_vk_buffer face_offsets_b;
    vkmesh_vk_buffer aux;
    vkmesh_vk_buffer out_vertices;
    vkmesh_vk_buffer out_faces;
    memset(&edges_buffer, 0, sizeof(edges_buffer));
    memset(&dummy_buffer, 0, sizeof(dummy_buffer));
    memset(&boundary_buffer, 0, sizeof(boundary_buffer));
    memset(&counter_buffer, 0, sizeof(counter_buffer));
    memset(&hole_data, 0, sizeof(hole_data));
    memset(&root_flags, 0, sizeof(root_flags));
    memset(&root_offsets_a, 0, sizeof(root_offsets_a));
    memset(&root_offsets_b, 0, sizeof(root_offsets_b));
    memset(&face_flags, 0, sizeof(face_flags));
    memset(&face_offsets_a, 0, sizeof(face_offsets_a));
    memset(&face_offsets_b, 0, sizeof(face_offsets_b));
    memset(&aux, 0, sizeof(aux));
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));

    const uint32_t vertex_count = dm->n_vertices;
    const uint32_t face_count = dm->n_faces;
    const size_t max_boundary_count = (size_t) face_count * 3u;
    size_t edge_count = 0;
    size_t edge_sort_count = 0;
    int ok = 0;

    if (!vk_buffer_create(vk, max_boundary_count * sizeof(vkmesh_boundary_edge), &boundary_buffer) ||
        !vk_buffer_create(vk, sizeof(uint32_t), &counter_buffer)) {
        goto cleanup;
    }

    if (!expand_edges_device(vk, &view, &dm->faces, &dm->vertices, &edges_buffer, &dummy_buffer, &edge_count, &edge_sort_count)) {
        goto cleanup;
    }

    vkmesh_vk_buffer compact_buffers[4];
    compact_buffers[0] = edges_buffer;
    compact_buffers[1] = boundary_buffer;
    compact_buffers[2] = counter_buffer;
    compact_buffers[3] = dummy_buffer;
    vkmesh_push push;
    memset(&push, 0, sizeof(push));
    push.n = (uint32_t) edge_count;
    uint32_t edge_groups = (uint32_t) ((edge_count + 127u) / 128u);
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPACT_BOUNDARY_EDGES, compact_buffers, &push, edge_groups)) goto cleanup;

    uint32_t boundary_count = ((const uint32_t *) counter_buffer.mapped)[0];
    if ((size_t) boundary_count > max_boundary_count) goto cleanup;
    if (boundary_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    size_t hole_words_size = (size_t) boundary_count * 7u + (size_t) vertex_count * 2u;
    if (hole_words_size > UINT32_MAX) goto cleanup;
    const uint32_t degree_off = boundary_count;
    const uint32_t owner_off = degree_off + vertex_count;
    if (!vk_buffer_create(vk, hole_words_size * sizeof(uint32_t), &hole_data) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &root_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &root_offsets_b) ||
        !vk_buffer_create(vk, (size_t) boundary_count * sizeof(uint32_t), &face_flags) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_a) ||
        !vk_buffer_create(vk, ((size_t) boundary_count + 1u) * sizeof(uint32_t), &face_offsets_b)) {
        goto cleanup;
    }
    memset((uint32_t *) hole_data.mapped + owner_off, 0xff, (size_t) vertex_count * sizeof(uint32_t));

    vkmesh_vk_buffer init_buffers[4];
    init_buffers[0] = hole_data;
    init_buffers[1] = dummy_buffer;
    init_buffers[2] = dummy_buffer;
    init_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    uint32_t boundary_groups = (boundary_count + 127u) / 128u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_INIT_U32_SEQUENCE, init_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer degree_buffers[4];
    degree_buffers[0] = boundary_buffer;
    degree_buffers[1] = dummy_buffer;
    degree_buffers[2] = hole_data;
    degree_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_BOUNDARY_DEGREE_OWNER, degree_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer union_buffers[4];
    union_buffers[0] = boundary_buffer;
    union_buffers[1] = dummy_buffer;
    union_buffers[2] = hole_data;
    union_buffers[3] = counter_buffer;
    vkmesh_vk_buffer compress_buffers[4];
    compress_buffers[0] = hole_data;
    compress_buffers[1] = counter_buffer;
    compress_buffers[2] = counter_buffer;
    compress_buffers[3] = counter_buffer;
    int converged = 0;
    for (int iter = 0; iter < 64; ++iter) {
        ((uint32_t *) counter_buffer.mapped)[0] = 0u;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        push.aux0 = vertex_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_UNION_BOUNDARY_EDGES, union_buffers, &push, boundary_groups)) goto cleanup;
        memset(&push, 0, sizeof(push));
        push.n = boundary_count;
        if (!vkmesh_dispatch(vk, VKMESH_PIPE_COMPRESS_PARENTS, compress_buffers, &push, boundary_groups)) goto cleanup;
        if (((const uint32_t *) counter_buffer.mapped)[0] == 0u) {
            converged = 1;
            break;
        }
    }
    if (!converged) goto cleanup;

    vkmesh_vk_buffer accum_buffers[4];
    accum_buffers[0] = boundary_buffer;
    accum_buffers[1] = dm->vertices;
    accum_buffers[2] = hole_data;
    accum_buffers[3] = counter_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_ACCUMULATE_HOLE_COMPONENTS, accum_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer mark_root_buffers[4];
    mark_root_buffers[0] = hole_data;
    mark_root_buffers[1] = root_flags;
    mark_root_buffers[2] = dummy_buffer;
    mark_root_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_ROOTS, mark_root_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * root_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &root_flags, boundary_count, &root_offsets_a, &root_offsets_b, &dummy_buffer, &root_offsets)) goto cleanup;
    uint32_t new_vertex_count = ((const uint32_t *) root_offsets->mapped)[boundary_count];
    if (new_vertex_count == 0u) {
        ok = 1;
        goto cleanup;
    }

    vkmesh_vk_buffer mark_face_buffers[4];
    mark_face_buffers[0] = hole_data;
    mark_face_buffers[1] = root_flags;
    mark_face_buffers[2] = face_flags;
    mark_face_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_MARK_HOLE_FACES, mark_face_buffers, &push, boundary_groups)) goto cleanup;

    vkmesh_vk_buffer * face_offsets = NULL;
    if (!vkmesh_scan_flags_to_offsets(vk, &face_flags, boundary_count, &face_offsets_a, &face_offsets_b, &dummy_buffer, &face_offsets)) goto cleanup;
    uint32_t new_face_count = ((const uint32_t *) face_offsets->mapped)[boundary_count];
    if (new_face_count == 0u ||
        vertex_count > UINT32_MAX - new_vertex_count ||
        face_count > UINT32_MAX - new_face_count) {
        goto cleanup;
    }

    const uint32_t out_vertex_count = vertex_count + new_vertex_count;
    const uint32_t out_face_count = face_count + new_face_count;
    const size_t aux_words = (size_t) boundary_count * 2u + 1u;
    if (!vk_buffer_create(vk, (size_t) out_vertex_count * 3u * sizeof(float), &out_vertices) ||
        !vk_buffer_create(vk, (size_t) out_face_count * 3u * sizeof(int32_t), &out_faces) ||
        !vk_buffer_create(vk, aux_words * sizeof(uint32_t), &aux)) {
        goto cleanup;
    }
    memset(aux.mapped, 0xff, (size_t) boundary_count * sizeof(uint32_t));

    vkmesh_vk_buffer copy_buffers[4];
    copy_buffers[0] = dm->vertices;
    copy_buffers[1] = out_vertices;
    copy_buffers[2] = dummy_buffer;
    copy_buffers[3] = dummy_buffer;
    memset(&push, 0, sizeof(push));
    push.n = vertex_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    copy_buffers[0] = dm->faces;
    copy_buffers[1] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = face_count * 3u;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_vertex_buffers[4];
    write_vertex_buffers[0] = hole_data;
    write_vertex_buffers[1] = *root_offsets;
    write_vertex_buffers[2] = aux;
    write_vertex_buffers[3] = out_vertices;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux0 = vertex_count;
    push.aux1 = vertex_count;
    push.eps = max_hole_perimeter;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_VERTICES, write_vertex_buffers, &push, boundary_groups)) goto cleanup;

    copy_buffers[0] = *face_offsets;
    copy_buffers[1] = aux;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count + 1u;
    push.aux0 = 0u;
    push.aux1 = boundary_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_COPY_U32, copy_buffers, &push, (push.n + 127u) / 128u)) goto cleanup;

    vkmesh_vk_buffer write_face_buffers[4];
    write_face_buffers[0] = boundary_buffer;
    write_face_buffers[1] = hole_data;
    write_face_buffers[2] = aux;
    write_face_buffers[3] = out_faces;
    memset(&push, 0, sizeof(push));
    push.n = boundary_count;
    push.aux1 = face_count;
    if (!vkmesh_dispatch(vk, VKMESH_PIPE_WRITE_HOLE_FACES, write_face_buffers, &push, boundary_groups)) goto cleanup;

    vk_buffer_destroy(vk, &dm->vertices);
    vk_buffer_destroy(vk, &dm->faces);
    dm->vertices = out_vertices;
    dm->faces = out_faces;
    memset(&out_vertices, 0, sizeof(out_vertices));
    memset(&out_faces, 0, sizeof(out_faces));
    dm->n_vertices = out_vertex_count;
    dm->n_faces = out_face_count;
    if (filled_loops != NULL) *filled_loops = new_vertex_count;
    if (added_faces != NULL) *added_faces = new_face_count;
    ok = 1;

cleanup:
    vk_buffer_destroy(vk, &edges_buffer);
    vk_buffer_destroy(vk, &dummy_buffer);
    vk_buffer_destroy(vk, &boundary_buffer);
    vk_buffer_destroy(vk, &counter_buffer);
    vk_buffer_destroy(vk, &hole_data);
    vk_buffer_destroy(vk, &root_flags);
    vk_buffer_destroy(vk, &root_offsets_a);
    vk_buffer_destroy(vk, &root_offsets_b);
    vk_buffer_destroy(vk, &face_flags);
    vk_buffer_destroy(vk, &face_offsets_a);
    vk_buffer_destroy(vk, &face_offsets_b);
    vk_buffer_destroy(vk, &aux);
    vk_buffer_destroy(vk, &out_vertices);
    vk_buffer_destroy(vk, &out_faces);
    return ok;
}

static int vkmesh_fill_holes_device_vulkan(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (mesh == NULL || mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t filled_u32 = 0u;
    uint32_t added_u32 = 0u;
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled_u32, &added_u32)) goto cleanup;
    if (filled_u32 > (uint32_t) INT_MAX || added_u32 > (uint32_t) INT_MAX) goto cleanup;
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    *filled_loops = (int) filled_u32;
    *added_faces = (int) added_u32;
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_fill_holes(vkmesh_mesh * mesh, float max_hole_perimeter, int * filled_loops, int * added_faces) {
    *filled_loops = 0;
    *added_faces = 0;
    if (vkmesh_fill_holes_device_vulkan(mesh, max_hole_perimeter, filled_loops, added_faces)) return 1;
    if (vkmesh_fill_holes_vulkan(mesh, max_hole_perimeter, filled_loops, added_faces)) return 1;
    vkmesh_boundary_edge * boundary = NULL;
    uint32_t * parents = NULL;
    uint32_t * comp_edges = NULL;
    uint32_t * comp_invalid = NULL;
    float * comp_perim = NULL;
    float * comp_mid = NULL;
    int64_t boundary_count_i64 = 0;
    if (!compute_hole_components_vulkan(
            mesh,
            &boundary,
            &parents,
            &comp_edges,
            &comp_invalid,
            &comp_perim,
            &comp_mid,
            &boundary_count_i64)) {
        return vkmesh_fill_holes_cpu(mesh, max_hole_perimeter, filled_loops, added_faces);
    }
    if (boundary_count_i64 <= 0) {
        free(boundary);
        free(parents);
        free(comp_edges);
        free(comp_invalid);
        free(comp_perim);
        free(comp_mid);
        return 1;
    }

    uint32_t boundary_count = (uint32_t) boundary_count_i64;
    int32_t * comp_new_vertex = (int32_t *) malloc((size_t) boundary_count * sizeof(*comp_new_vertex));
    if (comp_new_vertex == NULL) {
        free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
        return 0;
    }
    memset(comp_new_vertex, 0xff, (size_t) boundary_count * sizeof(*comp_new_vertex));

    for (uint32_t root = 0; root < boundary_count; ++root) {
        if (comp_edges[root] == 0u || comp_invalid[root] != 0u ||
            comp_perim[root] >= max_hole_perimeter) {
            continue;
        }
        float inv = 1.0f / (float) comp_edges[root];
        if (!mesh_add_vertex(
                mesh,
                comp_mid[(size_t) root * 3u + 0u] * inv,
                comp_mid[(size_t) root * 3u + 1u] * inv,
                comp_mid[(size_t) root * 3u + 2u] * inv)) {
            free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
            free(comp_new_vertex);
            return 0;
        }
        comp_new_vertex[root] = (int32_t) (mesh->n_vertices - 1);
        ++(*filled_loops);
    }

    for (uint32_t i = 0; i < boundary_count; ++i) {
        uint32_t root = vkmesh_find_parent_u32(parents, boundary_count, i);
        if (root == UINT32_MAX) continue;
        int32_t center = comp_new_vertex[root];
        if (center < 0) continue;
        if (!mesh_add_face(mesh, boundary[i].b, boundary[i].a, center)) {
            free(boundary); free(parents); free(comp_edges); free(comp_invalid); free(comp_perim); free(comp_mid);
            free(comp_new_vertex);
            return 0;
        }
        ++(*added_faces);
    }

    free(boundary);
    free(parents);
    free(comp_edges);
    free(comp_invalid);
    free(comp_perim);
    free(comp_mid);
    free(comp_new_vertex);
    return 1;
}

static int vkmesh_log_fill_holes(vkmesh_mesh * mesh, float max_hole_perimeter, const char * label) {
    int filled = 0;
    int added_faces = 0;
    int64_t before_v = mesh->n_vertices;
    int64_t before_f = mesh->n_faces;
    if (!vkmesh_fill_holes(mesh, max_hole_perimeter, &filled, &added_faces)) return 0;
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%d added_vertices=%" PRId64 " added_faces=%d faces=%" PRId64 "->%" PRId64 "\n",
        label,
        filled,
        mesh->n_vertices - before_v,
        added_faces,
        before_f,
        mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_duplicate_faces(vkmesh_mesh * mesh, const char * label) {
    int removed = 0;
    if (!vkmesh_remove_duplicate_faces(mesh, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_degenerate_faces(
    vkmesh_mesh * mesh,
    float degenerate_abs,
    float degenerate_rel,
    const char * label) {
    int removed = 0;
    if (!vkmesh_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_repair_non_manifold_edges(vkmesh_mesh * mesh, const char * label) {
    int old_v = 0;
    int new_v = 0;
    if (!vkmesh_repair_non_manifold_edges(mesh, &old_v, &new_v)) return 0;
    fprintf(stderr, "vkmesh: %s repair_non_manifold_edges vertices=%d->%d faces=%" PRId64 "\n", label, old_v, new_v, mesh->n_faces);
    return 1;
}

static int vkmesh_log_remove_small_connected_components(vkmesh_mesh * mesh, float min_component_area, const char * label) {
    int removed = 0;
    if (!vkmesh_remove_small_connected_components(mesh, min_component_area, &removed)) return 0;
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%d faces=%" PRId64 "\n", label, removed, mesh->n_faces);
    return 1;
}

static int vkmesh_log_small_components_fill_holes_device_cluster(
    vkmesh_mesh * mesh,
    float min_component_area,
    float max_hole_perimeter,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t removed = 0u;
    uint32_t filled = 0u;
    uint32_t added_faces = 0u;
    uint32_t after_small_faces = 0u;
    uint32_t before_fill_vertices = 0u;
    uint32_t before_fill_faces = 0u;
    uint32_t after_fill_vertices = 0u;
    uint32_t after_fill_faces = 0u;
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;
    if (!vkmesh_device_remove_small_connected_components(&dm, min_component_area, &removed)) goto cleanup;
    after_small_faces = dm.n_faces;
    before_fill_vertices = dm.n_vertices;
    before_fill_faces = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &added_faces)) goto cleanup;
    after_fill_vertices = dm.n_vertices;
    after_fill_faces = dm.n_faces;
    if (filled > (uint32_t) INT_MAX || added_faces > (uint32_t) INT_MAX) goto cleanup;
    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;

    fprintf(stderr,
        "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n",
        label,
        removed,
        after_small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        filled,
        after_fill_vertices - before_fill_vertices,
        added_faces,
        before_fill_faces,
        after_fill_faces);
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

typedef struct vkmesh_trellis_cleanup_stats {
    uint32_t duplicate_removed;
    uint32_t duplicate_faces;
    uint32_t degenerate_removed;
    uint32_t degenerate_faces;
    uint32_t repair_old_vertices;
    uint32_t repair_new_vertices;
    uint32_t repair_faces;
    uint32_t small_removed;
    uint32_t small_faces;
    uint32_t filled;
    uint32_t fill_added_faces;
    uint32_t before_fill_vertices;
    uint32_t before_fill_faces;
    uint32_t after_fill_vertices;
    uint32_t after_fill_faces;
    uint32_t flipped_faces;
} vkmesh_trellis_cleanup_stats;

static int vkmesh_device_trellis_cleanup(
    vkmesh_device_mesh * dm,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float max_hole_perimeter,
    int run_final_orientation,
    vkmesh_trellis_cleanup_stats * stats) {
    if (dm == NULL || stats == NULL) return 0;
    memset(stats, 0, sizeof(*stats));

    if (!vkmesh_device_remove_duplicate_faces(dm, &stats->duplicate_removed)) return 0;
    stats->duplicate_faces = dm->n_faces;

    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(dm, degenerate_abs, degenerate_rel, &stats->degenerate_removed)) return 0;
        stats->degenerate_faces = dm->n_faces;
    }

    if (!vkmesh_device_repair_non_manifold_edges(dm, &stats->repair_old_vertices, &stats->repair_new_vertices)) return 0;
    stats->repair_faces = dm->n_faces;

    if (!vkmesh_device_remove_small_connected_components(dm, min_component_area, &stats->small_removed)) return 0;
    stats->small_faces = dm->n_faces;

    stats->before_fill_vertices = dm->n_vertices;
    stats->before_fill_faces = dm->n_faces;
    if (!vkmesh_device_fill_holes(dm, max_hole_perimeter, &stats->filled, &stats->fill_added_faces)) return 0;
    stats->after_fill_vertices = dm->n_vertices;
    stats->after_fill_faces = dm->n_faces;

    if (run_final_orientation) {
        if (!vkmesh_device_unify_face_orientations(dm, &stats->flipped_faces)) return 0;
    }
    return 1;
}

static void vkmesh_log_trellis_cleanup_stats(
    const char * label,
    int run_degenerate_cleanup,
    int run_final_orientation,
    const char * orientation_label,
    const vkmesh_trellis_cleanup_stats * stats) {
    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, stats->duplicate_removed, stats->duplicate_faces);
    if (run_degenerate_cleanup) {
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, stats->degenerate_removed, stats->degenerate_faces);
    }
    fprintf(stderr,
        "vkmesh: %s repair_non_manifold_edges vertices=%u->%u faces=%u\n",
        label,
        stats->repair_old_vertices,
        stats->repair_new_vertices,
        stats->repair_faces);
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n", label, stats->small_removed, stats->small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        stats->filled,
        stats->after_fill_vertices - stats->before_fill_vertices,
        stats->fill_added_faces,
        stats->before_fill_faces,
        stats->after_fill_faces);
    if (run_final_orientation) {
        fprintf(stderr,
            "vkmesh: %s unify_face_orientations flipped=%u\n",
            orientation_label != NULL ? orientation_label : label,
            stats->flipped_faces);
    }
}

static int vkmesh_log_trellis_cleanup_device_cluster(
    vkmesh_mesh * mesh,
    int run_degenerate_cleanup,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float max_hole_perimeter,
    int run_final_orientation,
    const char * orientation_label,
    const char * label) {
    if (mesh == NULL || label == NULL ||
        mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    uint32_t duplicate_removed = 0u;
    uint32_t duplicate_faces = 0u;
    uint32_t degenerate_removed = 0u;
    uint32_t degenerate_faces = 0u;
    uint32_t repair_old_vertices = 0u;
    uint32_t repair_new_vertices = 0u;
    uint32_t repair_faces = 0u;
    uint32_t small_removed = 0u;
    uint32_t small_faces = 0u;
    uint32_t filled = 0u;
    uint32_t fill_added_faces = 0u;
    uint32_t before_fill_vertices = 0u;
    uint32_t before_fill_faces = 0u;
    uint32_t after_fill_vertices = 0u;
    uint32_t after_fill_faces = 0u;
    uint32_t flipped_faces = 0u;
    int ok = 0;

    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;

    if (!vkmesh_device_remove_duplicate_faces(&dm, &duplicate_removed)) goto cleanup;
    duplicate_faces = dm.n_faces;

    if (run_degenerate_cleanup) {
        if (!vkmesh_device_remove_degenerate_faces(&dm, degenerate_abs, degenerate_rel, &degenerate_removed)) goto cleanup;
        degenerate_faces = dm.n_faces;
    }

    if (!vkmesh_device_repair_non_manifold_edges(&dm, &repair_old_vertices, &repair_new_vertices)) goto cleanup;
    repair_faces = dm.n_faces;

    if (!vkmesh_device_remove_small_connected_components(&dm, min_component_area, &small_removed)) goto cleanup;
    small_faces = dm.n_faces;

    before_fill_vertices = dm.n_vertices;
    before_fill_faces = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &fill_added_faces)) goto cleanup;
    after_fill_vertices = dm.n_vertices;
    after_fill_faces = dm.n_faces;
    if (filled > (uint32_t) INT_MAX || fill_added_faces > (uint32_t) INT_MAX) goto cleanup;

    if (run_final_orientation) {
        if (!vkmesh_device_unify_face_orientations(&dm, &flipped_faces)) goto cleanup;
        if (flipped_faces > (uint32_t) INT_MAX) goto cleanup;
    }

    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;

    fprintf(stderr, "vkmesh: %s remove_duplicate_faces removed=%u faces=%u\n", label, duplicate_removed, duplicate_faces);
    if (run_degenerate_cleanup) {
        fprintf(stderr, "vkmesh: %s remove_degenerate_faces removed=%u faces=%u\n", label, degenerate_removed, degenerate_faces);
    }
    fprintf(stderr,
        "vkmesh: %s repair_non_manifold_edges vertices=%u->%u faces=%u\n",
        label,
        repair_old_vertices,
        repair_new_vertices,
        repair_faces);
    fprintf(stderr, "vkmesh: %s remove_small_connected_components removed=%u faces=%u\n", label, small_removed, small_faces);
    fprintf(stderr,
        "vkmesh: %s fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        label,
        filled,
        after_fill_vertices - before_fill_vertices,
        fill_added_faces,
        before_fill_faces,
        after_fill_faces);
    if (run_final_orientation) {
        fprintf(stderr,
            "vkmesh: %s unify_face_orientations flipped=%u\n",
            orientation_label != NULL ? orientation_label : label,
            flipped_faces);
    }
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_trellis_postprocess_device_inner(
    vkmesh_mesh * mesh,
    const char * projection_output,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup) {
    if (mesh == NULL || mesh->n_vertices <= 0 || mesh->n_vertices > UINT32_MAX ||
        mesh->n_faces <= 0 || mesh->n_faces > UINT32_MAX) {
        return 0;
    }
    if (decimation_target <= 0) decimation_target = 1000000;

    vkmesh_vk local_vk;
    vkmesh_vk * vk = NULL;
    int owns_vk = 0;
    if (!vkmesh_acquire_vk(&vk, &local_vk, &owns_vk)) return 0;

    vkmesh_device_mesh dm;
    memset(&dm, 0, sizeof(dm));
    int ok = 0;
    if (!vkmesh_device_mesh_upload(vk, mesh, &dm)) goto cleanup;

    fprintf(stderr,
        "vkmesh: trellis_postprocess begin target=%d max_hole_perimeter=%.9g min_component_area=%.9g\n",
        decimation_target,
        max_hole_perimeter,
        min_component_area);

    uint32_t filled = 0u;
    uint32_t added_faces = 0u;
    uint32_t before_v = dm.n_vertices;
    uint32_t before_f = dm.n_faces;
    if (!vkmesh_device_fill_holes(&dm, max_hole_perimeter, &filled, &added_faces)) goto cleanup;
    fprintf(stderr,
        "vkmesh: trellis.pre fill_holes loops=%u added_vertices=%u added_faces=%u faces=%u->%u\n",
        filled,
        dm.n_vertices - before_v,
        added_faces,
        before_f,
        dm.n_faces);
    if (projection_output != NULL && projection_output[0] != '\0') {
        vkmesh_mesh projection_mesh;
        memset(&projection_mesh, 0, sizeof(projection_mesh));
        if (!vkmesh_device_mesh_download(&dm, &projection_mesh)) {
            mesh_free(&projection_mesh);
            goto cleanup;
        }
        int wrote_projection = write_meshbin(projection_output, &projection_mesh);
        fprintf(stderr,
            "vkmesh: trellis.pre projection_mesh_output %s vertices=%" PRId64 " faces=%" PRId64 "\n",
            projection_output,
            projection_mesh.n_vertices,
            projection_mesh.n_faces);
        mesh_free(&projection_mesh);
        if (!wrote_projection) goto cleanup;
    }

    int64_t first_target = (int64_t) decimation_target * 3;
    if (first_target > INT_MAX) first_target = INT_MAX;
    if (dm.n_faces > (uint32_t) first_target) {
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr, "vkmesh: trellis.pass1 simplify begin target=%" PRId64 " faces=%u\n", first_target, dm.n_faces);
        fflush(stderr);
        if (!vkmesh_device_simplify(
                &dm,
                (int) first_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                &collapsed,
                &removed)) {
            goto cleanup;
        }
        fprintf(stderr,
            "vkmesh: trellis.pass1 simplify target=%" PRId64 " collapsed=%d removed_faces=%d faces=%u\n",
            first_target,
            collapsed,
            removed,
            dm.n_faces);
    } else {
        fprintf(stderr, "vkmesh: trellis.pass1 simplify skipped faces=%u target=%" PRId64 "\n", dm.n_faces, first_target);
    }

    vkmesh_trellis_cleanup_stats pass1_stats;
    if (!vkmesh_device_trellis_cleanup(
            &dm,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            0,
            &pass1_stats)) {
        goto cleanup;
    }
    vkmesh_log_trellis_cleanup_stats("trellis.pass1", run_degenerate_cleanup, 0, NULL, &pass1_stats);

    if (dm.n_faces > (uint32_t) decimation_target) {
        int collapsed = 0;
        int removed = 0;
        fprintf(stderr, "vkmesh: trellis.pass2 simplify begin target=%d faces=%u\n", decimation_target, dm.n_faces);
        fflush(stderr);
        if (!vkmesh_device_simplify(
                &dm,
                decimation_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                &collapsed,
                &removed)) {
            goto cleanup;
        }
        fprintf(stderr,
            "vkmesh: trellis.pass2 simplify target=%d collapsed=%d removed_faces=%d faces=%u\n",
            decimation_target,
            collapsed,
            removed,
            dm.n_faces);
    } else {
        fprintf(stderr, "vkmesh: trellis.pass2 simplify skipped faces=%u target=%d\n", dm.n_faces, decimation_target);
    }

    vkmesh_trellis_cleanup_stats pass2_stats;
    if (!vkmesh_device_trellis_cleanup(
            &dm,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            1,
            &pass2_stats)) {
        goto cleanup;
    }
    vkmesh_log_trellis_cleanup_stats("trellis.pass2", run_degenerate_cleanup, 1, "trellis.final", &pass2_stats);

    if (!vkmesh_device_mesh_download(&dm, mesh)) goto cleanup;
    fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
    ok = 1;

cleanup:
    vkmesh_device_mesh_destroy(&dm);
    if (owns_vk) vkmesh_vk_destroy(vk);
    return ok;
}

static int vkmesh_log_unify_face_orientations(vkmesh_mesh * mesh, const char * label) {
    int flipped = 0;
    if (!vkmesh_unify_face_orientations(mesh, &flipped)) return 0;
    fprintf(stderr, "vkmesh: %s unify_face_orientations flipped=%d\n", label, flipped);
    return 1;
}

static int vkmesh_log_simplify(
    vkmesh_mesh * mesh,
    int target_faces,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    const char * label) {
    int collapsed = 0;
    int removed = 0;
    if (target_faces <= 0) target_faces = 1;
    fprintf(stderr,
        "vkmesh: %s simplify begin target=%d faces=%" PRId64 "\n",
        label,
        target_faces,
        mesh->n_faces);
    fflush(stderr);
    if (!vkmesh_simplify(
            mesh,
            target_faces,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            &collapsed,
            &removed)) {
        return 0;
    }
    fprintf(stderr,
        "vkmesh: %s simplify target=%d collapsed=%d removed_faces=%d faces=%" PRId64 "\n",
        label,
        target_faces,
        collapsed,
        removed,
        mesh->n_faces);
    return 1;
}

static int vkmesh_trellis_postprocess_inner(
    vkmesh_mesh * mesh,
    const char * projection_output,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup) {
    if (vkmesh_trellis_postprocess_device_inner(
            mesh,
            projection_output,
            decimation_target,
            max_hole_perimeter,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            run_degenerate_cleanup)) {
        return 1;
    }

    if (decimation_target <= 0) decimation_target = 1000000;
    fprintf(stderr,
        "vkmesh: trellis_postprocess begin target=%d max_hole_perimeter=%.9g min_component_area=%.9g\n",
        decimation_target,
        max_hole_perimeter,
        min_component_area);

    if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pre")) return 0;
    if (projection_output != NULL && projection_output[0] != '\0') {
        if (!write_meshbin(projection_output, mesh)) return 0;
        fprintf(stderr,
            "vkmesh: trellis.pre projection_mesh_output %s vertices=%" PRId64 " faces=%" PRId64 "\n",
            projection_output,
            mesh->n_vertices,
            mesh->n_faces);
    }

    int64_t first_target = (int64_t) decimation_target * 3;
    if (first_target > INT_MAX) first_target = INT_MAX;
    if (mesh->n_faces > first_target) {
        if (!vkmesh_log_simplify(
                mesh,
                (int) first_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                "trellis.pass1")) {
            return 0;
        }
    } else {
        fprintf(stderr, "vkmesh: trellis.pass1 simplify skipped faces=%" PRId64 " target=%" PRId64 "\n", mesh->n_faces, first_target);
    }

    if (!vkmesh_log_trellis_cleanup_device_cluster(
            mesh,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            0,
            NULL,
            "trellis.pass1")) {
        if (!vkmesh_log_duplicate_degenerate_device_cluster(
                mesh,
                run_degenerate_cleanup,
                degenerate_abs,
                degenerate_rel,
                "trellis.pass1")) {
            if (!vkmesh_log_remove_duplicate_faces(mesh, "trellis.pass1")) return 0;
            if (run_degenerate_cleanup && !vkmesh_log_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, "trellis.pass1")) return 0;
        }
        if (!vkmesh_log_repair_non_manifold_edges(mesh, "trellis.pass1")) return 0;
        if (!vkmesh_log_remove_small_connected_components(mesh, min_component_area, "trellis.pass1")) return 0;
        if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pass1")) return 0;
    }

    if (mesh->n_faces > decimation_target) {
        if (!vkmesh_log_simplify(
                mesh,
                decimation_target,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                "trellis.pass2")) {
            return 0;
        }
    } else {
        fprintf(stderr, "vkmesh: trellis.pass2 simplify skipped faces=%" PRId64 " target=%d\n", mesh->n_faces, decimation_target);
    }

    int final_orientation_done = 0;
    if (vkmesh_log_trellis_cleanup_device_cluster(
            mesh,
            run_degenerate_cleanup,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            max_hole_perimeter,
            1,
            "trellis.final",
            "trellis.pass2")) {
        final_orientation_done = 1;
    } else {
        if (!vkmesh_log_duplicate_degenerate_device_cluster(
                mesh,
                run_degenerate_cleanup,
                degenerate_abs,
                degenerate_rel,
                "trellis.pass2")) {
            if (!vkmesh_log_remove_duplicate_faces(mesh, "trellis.pass2")) return 0;
            if (run_degenerate_cleanup && !vkmesh_log_remove_degenerate_faces(mesh, degenerate_abs, degenerate_rel, "trellis.pass2")) return 0;
        }
        if (!vkmesh_log_repair_non_manifold_edges(mesh, "trellis.pass2")) return 0;
        if (!vkmesh_log_remove_small_connected_components(mesh, min_component_area, "trellis.pass2")) return 0;
        if (!vkmesh_log_fill_holes(mesh, max_hole_perimeter, "trellis.pass2")) return 0;
    }
    if (!final_orientation_done && !vkmesh_log_unify_face_orientations(mesh, "trellis.final")) return 0;

    fprintf(stderr, "vkmesh: trellis_postprocess done vertices=%" PRId64 " faces=%" PRId64 "\n", mesh->n_vertices, mesh->n_faces);
    return 1;
}

static int vkmesh_trellis_postprocess(
    vkmesh_mesh * mesh,
    const char * projection_output,
    int decimation_target,
    float max_hole_perimeter,
    float degenerate_abs,
    float degenerate_rel,
    float min_component_area,
    float lambda_edge_length,
    float lambda_skinny,
    float simplify_threshold,
    int simplify_steps,
    int run_degenerate_cleanup) {
    if (g_active_vkmesh_vk != NULL) {
        return vkmesh_trellis_postprocess_inner(
            mesh,
            projection_output,
            decimation_target,
            max_hole_perimeter,
            degenerate_abs,
            degenerate_rel,
            min_component_area,
            lambda_edge_length,
            lambda_skinny,
            simplify_threshold,
            simplify_steps,
            run_degenerate_cleanup);
    }

    vkmesh_vk vk;
    if (!vkmesh_vk_init(&vk)) {
        vkmesh_vk_destroy(&vk);
        return 0;
    }
    g_active_vkmesh_vk = &vk;
    fprintf(stderr, "vkmesh: trellis_postprocess using one persistent Vulkan context\n");
    int ok = vkmesh_trellis_postprocess_inner(
        mesh,
        projection_output,
        decimation_target,
        max_hole_perimeter,
        degenerate_abs,
        degenerate_rel,
        min_component_area,
        lambda_edge_length,
        lambda_skinny,
        simplify_threshold,
        simplify_steps,
        run_degenerate_cleanup);
    g_active_vkmesh_vk = NULL;
    vkmesh_vk_destroy(&vk);
    return ok;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --input in.meshbin --output out.meshbin [options]\n"
        "\n"
        "Postprocess stages:\n"
        "  --postprocess                 Run PyTorch/o-voxel TRELLIS postprocess preset\n"
        "  --trellis-postprocess         Alias of --postprocess\n"
        "  --cleanup                     Run one cleanup pass using selected primitive stages\n"
        "  --fill-holes                  Fill small manifold boundary loops (default)\n"
        "  --no-fill-holes               Disable default fill_holes\n"
        "  --remove-duplicate-faces      Remove duplicate triangle index sets\n"
        "  --remove-degenerate-faces     Remove duplicate-vertex / tiny-area faces\n"
        "  --repair-non-manifold-edges   Split vertices across non-manifold sheets\n"
        "  --remove-small-components     Remove components below --min-component-area\n"
        "  --unify-face-orientations     Make winding consistent per manifold component\n"
        "  --simplify                    Run CuMesh-style simplify loop\n"
        "  --no-simplify                 Disable simplify even if --target-faces was set\n"
        "  --uv-unwrap                   Run xatlas unwrap and store UVs in meshbin\n"
        "  --no-uv-unwrap                Disable UV unwrap after a preset enabled it\n"
        "\n"
        "Parameters:\n"
        "  --max-hole-perimeter X        Default 0.03, matching CuMesh/TRELLIS.2\n"
        "  --degenerate-abs X            Default 1e-24\n"
        "  --degenerate-rel X            Default 1e-12\n"
        "  --min-component-area X        Default 1e-5\n"
        "  --target-faces N              Enables simplify to N faces; TRELLIS default is 1000000\n"
        "  --decimation-target N         Alias of --target-faces\n"
        "  --simplify-steps N            0 means no explicit step limit\n"
        "  --simplify-threshold X        Default 1e-8\n"
        "  --lambda-edge-length X        Default 1e-2\n"
        "  --lambda-skinny X             Default 1e-3\n"
        "  --texture-size N              xatlas pack resolution, default 1024\n"
        "  --unsigned-distance pts.txt   Compute UDF for text points: x y z per line\n"
        "  --distance-output out.txt     Required with --unsigned-distance unless --output is enough for mesh only\n"
        "  --projection-mesh-output FILE With --postprocess, write post-fill source meshbin for texture projection\n"
        "\n"
        "Note: unsigned_distance is currently a Vulkan compute brute-force kernel;\n"
        "      remesh_narrow_band_dc is the remaining optional stage.\n",
        argv0);
}

int main(int argc, char ** argv) {
    const char * input = NULL;
    const char * output = NULL;
    const char * projection_output = NULL;
    const char * points_path = NULL;
    const char * distance_output = NULL;
    float max_hole_perimeter = 3e-2f;
    float degenerate_abs = 1e-24f;
    float degenerate_rel = 1e-12f;
    float min_component_area = 1e-5f;
    float simplify_threshold = 1e-8f;
    float lambda_edge_length = 1e-2f;
    float lambda_skinny = 1e-3f;
    int target_faces = 0;
    int simplify_steps = 0;
    int texture_size = 1024;
    int trellis_postprocess = 0;
    int fill_holes = 1;
    int remove_duplicate_faces = 0;
    int remove_degenerate_faces = 0;
    int repair_non_manifold_edges = 0;
    int remove_small_components = 0;
    int unify_face_orientations = 0;
    int simplify = 0;
    int disable_simplify = 0;
    int uv_unwrap = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--projection-mesh-output") == 0 && i + 1 < argc) {
            projection_output = argv[++i];
        } else if (strcmp(argv[i], "--postprocess") == 0 || strcmp(argv[i], "--trellis-postprocess") == 0) {
            trellis_postprocess = 1;
            uv_unwrap = 1;
        } else if (strcmp(argv[i], "--cleanup") == 0) {
            fill_holes = 1;
            remove_duplicate_faces = 1;
            remove_degenerate_faces = 1;
            repair_non_manifold_edges = 1;
            remove_small_components = 1;
            unify_face_orientations = 1;
        } else if (strcmp(argv[i], "--max-hole-perimeter") == 0 && i + 1 < argc) {
            max_hole_perimeter = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--degenerate-abs") == 0 && i + 1 < argc) {
            degenerate_abs = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--degenerate-rel") == 0 && i + 1 < argc) {
            degenerate_rel = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-component-area") == 0 && i + 1 < argc) {
            min_component_area = (float) atof(argv[++i]);
        } else if ((strcmp(argv[i], "--target-faces") == 0 || strcmp(argv[i], "--decimation-target") == 0) && i + 1 < argc) {
            target_faces = atoi(argv[++i]);
            simplify = 1;
            disable_simplify = 0;
        } else if (strcmp(argv[i], "--simplify-steps") == 0 && i + 1 < argc) {
            simplify_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--simplify-threshold") == 0 && i + 1 < argc) {
            simplify_threshold = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--lambda-edge-length") == 0 && i + 1 < argc) {
            lambda_edge_length = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--lambda-skinny") == 0 && i + 1 < argc) {
            lambda_skinny = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--texture-size") == 0 && i + 1 < argc) {
            texture_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--unsigned-distance") == 0 && i + 1 < argc) {
            points_path = argv[++i];
        } else if (strcmp(argv[i], "--distance-output") == 0 && i + 1 < argc) {
            distance_output = argv[++i];
        } else if (strcmp(argv[i], "--fill-holes") == 0) {
            fill_holes = 1;
        } else if (strcmp(argv[i], "--no-fill-holes") == 0) {
            fill_holes = 0;
        } else if (strcmp(argv[i], "--remove-duplicate-faces") == 0) {
            remove_duplicate_faces = 1;
        } else if (strcmp(argv[i], "--remove-degenerate-faces") == 0) {
            remove_degenerate_faces = 1;
        } else if (strcmp(argv[i], "--repair-non-manifold-edges") == 0) {
            repair_non_manifold_edges = 1;
        } else if (strcmp(argv[i], "--remove-small-components") == 0) {
            remove_small_components = 1;
        } else if (strcmp(argv[i], "--unify-face-orientations") == 0) {
            unify_face_orientations = 1;
        } else if (strcmp(argv[i], "--simplify") == 0) {
            simplify = 1;
            disable_simplify = 0;
        } else if (strcmp(argv[i], "--no-simplify") == 0) {
            simplify = 0;
            disable_simplify = 1;
        } else if (strcmp(argv[i], "--uv-unwrap") == 0) {
            uv_unwrap = 1;
        } else if (strcmp(argv[i], "--no-uv-unwrap") == 0) {
            uv_unwrap = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "vkmesh: unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (input == NULL || (output == NULL && distance_output == NULL)) {
        print_usage(argv[0]);
        return 2;
    }
    if (points_path != NULL && distance_output == NULL) {
        fprintf(stderr, "vkmesh: --distance-output is required with --unsigned-distance\n");
        return 2;
    }

    vkmesh_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (!load_meshbin(input, &mesh)) {
        mesh_free(&mesh);
        return 1;
    }
    fprintf(stderr, "vkmesh: loaded %" PRId64 " vertices, %" PRId64 " faces\n", mesh.n_vertices, mesh.n_faces);
    if (trellis_postprocess) {
        int decimation_target = disable_simplify ? INT_MAX : target_faces;
        if (!vkmesh_trellis_postprocess(
                &mesh,
                projection_output,
                decimation_target,
                max_hole_perimeter,
                degenerate_abs,
                degenerate_rel,
                min_component_area,
                lambda_edge_length,
                lambda_skinny,
                simplify_threshold,
                simplify_steps,
                remove_degenerate_faces)) {
            mesh_free(&mesh);
            return 1;
        }
    } else {
        if (fill_holes && !vkmesh_log_fill_holes(&mesh, max_hole_perimeter, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remove_duplicate_faces && !vkmesh_log_remove_duplicate_faces(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remove_degenerate_faces && !vkmesh_log_remove_degenerate_faces(&mesh, degenerate_abs, degenerate_rel, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (repair_non_manifold_edges && !vkmesh_log_repair_non_manifold_edges(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (remove_small_components && !vkmesh_log_remove_small_connected_components(&mesh, min_component_area, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (unify_face_orientations && !vkmesh_log_unify_face_orientations(&mesh, "stage")) {
            mesh_free(&mesh);
            return 1;
        }
        if (simplify && !disable_simplify) {
            if (target_faces <= 0) target_faces = (int) (mesh.n_faces / 2);
            if (target_faces <= 0) target_faces = 1;
            if (!vkmesh_log_simplify(
                    &mesh,
                    target_faces,
                    lambda_edge_length,
                    lambda_skinny,
                    simplify_threshold,
                    simplify_steps,
                    "stage")) {
                mesh_free(&mesh);
                return 1;
            }
        }
    }
    if (uv_unwrap) {
        int old_v = 0;
        int new_v = 0;
        int charts = 0;
        if (!vkmesh_uv_unwrap(&mesh, texture_size, &old_v, &new_v, &charts)) {
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: uv_unwrap vertices=%d->%d charts=%d\n", old_v, new_v, charts);
    }
    if (points_path != NULL) {
        float * points = NULL;
        float * distances = NULL;
        uint32_t * face_ids = NULL;
        int64_t point_count = 0;
        if (!load_points(points_path, &points, &point_count) ||
            !vkmesh_unsigned_distance_vulkan(&mesh, points, point_count, &distances, &face_ids) ||
            !write_distances(distance_output, points, point_count, distances, face_ids)) {
            free(points);
            free(distances);
            free(face_ids);
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: unsigned_distance points=%" PRId64 " wrote=%s\n", point_count, distance_output);
        free(points);
        free(distances);
        free(face_ids);
    }
    if (output != NULL) {
        if (!write_meshbin(output, &mesh)) {
            mesh_free(&mesh);
            return 1;
        }
        fprintf(stderr, "vkmesh: wrote %s (%" PRId64 " vertices, %" PRId64 " faces)\n", output, mesh.n_vertices, mesh.n_faces);
    }
    mesh_free(&mesh);
    return 0;
}
