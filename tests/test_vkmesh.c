#include "trellis.h"
#include "trellis_platform.h"
#include "trellis_pipeline_internal.h"

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct test_mesh {
    float * vertices;
    int32_t * faces;
    int64_t n_vertices;
    int64_t n_faces;
} test_mesh;

static int g_failures = 0;

#define CHECK_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++g_failures; \
        return 0; \
    } \
} while (0)

static void test_mesh_free(test_mesh * mesh) {
    if (mesh == NULL) return;
    free(mesh->vertices);
    free(mesh->faces);
    memset(mesh, 0, sizeof(*mesh));
}

static int write_meshbin(const char * path, const float * vertices, uint64_t n_vertices,
                         const int32_t * faces, uint64_t n_faces) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) return 0;
    const char magic[8] = { 'T', 'R', 'L', 'M', 'E', 'S', 'H', '1' };
    const uint32_t flags = 0;
    const uint32_t reserved = 0;
    int ok =
        fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        fwrite(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
        fwrite(&n_faces, sizeof(n_faces), 1, f) == 1 &&
        fwrite(&flags, sizeof(flags), 1, f) == 1 &&
        fwrite(&reserved, sizeof(reserved), 1, f) == 1 &&
        fwrite(vertices, sizeof(float), (size_t) n_vertices * 3u, f) == (size_t) n_vertices * 3u &&
        fwrite(faces, sizeof(int32_t), (size_t) n_faces * 3u, f) == (size_t) n_faces * 3u;
    if (fclose(f) != 0) ok = 0;
    return ok;
}

static int read_meshbin(const char * path, test_mesh * mesh_out) {
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) return 0;
    char magic[8];
    uint64_t n_vertices = 0;
    uint64_t n_faces = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
    int ok =
        fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        fread(&n_vertices, sizeof(n_vertices), 1, f) == 1 &&
        fread(&n_faces, sizeof(n_faces), 1, f) == 1 &&
        fread(&flags, sizeof(flags), 1, f) == 1 &&
        fread(&reserved, sizeof(reserved), 1, f) == 1 &&
        memcmp(magic, "TRLMESH1", 8) == 0 && n_vertices > 0 && n_faces > 0 &&
        n_vertices <= SIZE_MAX / (3u * sizeof(float)) &&
        n_faces <= SIZE_MAX / (3u * sizeof(int32_t));
    (void) reserved;
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    if (ok) {
        mesh.vertices = (float *) malloc((size_t) n_vertices * 3u * sizeof(float));
        mesh.faces = (int32_t *) malloc((size_t) n_faces * 3u * sizeof(int32_t));
        mesh.n_vertices = (int64_t) n_vertices;
        mesh.n_faces = (int64_t) n_faces;
        ok = mesh.vertices != NULL && mesh.faces != NULL &&
            fread(mesh.vertices, sizeof(float), (size_t) n_vertices * 3u, f) == (size_t) n_vertices * 3u &&
            fread(mesh.faces, sizeof(int32_t), (size_t) n_faces * 3u, f) == (size_t) n_faces * 3u;
        if (ok && (flags & 1u) != 0) {
            float uv[2];
            for (uint64_t i = 0; i < n_vertices && ok; ++i) {
                ok = fread(uv, sizeof(float), 2u, f) == 2u;
            }
        }
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        test_mesh_free(&mesh);
        return 0;
    }
    *mesh_out = mesh;
    return 1;
}

static void sort3(int32_t v[3]) {
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { int32_t t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { int32_t t = v[0]; v[0] = v[1]; v[1] = t; }
}

static int mesh_is_closed_and_clean(const float * vertices, const int32_t * faces,
                                    int64_t n_vertices, int64_t n_faces) {
    (void) vertices;
    if (faces == NULL || n_vertices <= 0 || n_faces <= 0) return 0;
    for (int64_t i = 0; i < n_faces; ++i) {
        const int32_t * f = faces + (size_t) i * 3u;
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= n_vertices || f[1] >= n_vertices || f[2] >= n_vertices ||
            f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) return 0;
        int32_t key_i[3] = { f[0], f[1], f[2] };
        sort3(key_i);
        for (int64_t j = i + 1; j < n_faces; ++j) {
            const int32_t * g = faces + (size_t) j * 3u;
            int32_t key_j[3] = { g[0], g[1], g[2] };
            sort3(key_j);
            if (memcmp(key_i, key_j, sizeof(key_i)) == 0) return 0;
        }
    }
    for (int64_t i = 0; i < n_faces; ++i) {
        const int32_t * f = faces + (size_t) i * 3u;
        for (int e = 0; e < 3; ++e) {
            int32_t a = f[e];
            int32_t b = f[(e + 1) % 3];
            if (a > b) { int32_t t = a; a = b; b = t; }
            int count = 0;
            for (int64_t j = 0; j < n_faces; ++j) {
                const int32_t * g = faces + (size_t) j * 3u;
                for (int ge = 0; ge < 3; ++ge) {
                    int32_t ga = g[ge];
                    int32_t gb = g[(ge + 1) % 3];
                    if (ga > gb) { int32_t t = ga; ga = gb; gb = t; }
                    if (ga == a && gb == b) ++count;
                }
            }
            if (count != 2) return 0;
        }
    }
    return 1;
}

static int has_compute_device(int device_index) {
    VkInstance instance = VK_NULL_HANDLE;
    VkInstanceCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (vkCreateInstance(&info, NULL, &instance) != VK_SUCCESS) return 0;
    uint32_t count = 0;
    int found = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS &&
        device_index >= 0 && (uint32_t) device_index < count) {
        VkPhysicalDevice * devices = (VkPhysicalDevice *) malloc((size_t) count * sizeof(*devices));
        if (devices != NULL && vkEnumeratePhysicalDevices(instance, &count, devices) == VK_SUCCESS) {
            uint32_t queue_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, NULL);
            VkQueueFamilyProperties * queues =
                (VkQueueFamilyProperties *) malloc((size_t) queue_count * sizeof(*queues));
            if (queues != NULL) {
                vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, queues);
                for (uint32_t i = 0; i < queue_count; ++i) {
                    if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) found = 1;
                }
                free(queues);
            }
        }
        free(devices);
    }
    vkDestroyInstance(instance, NULL);
    return found;
}

static int test_cli_cleanup(const char * vkmesh_path) {
    static const float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    static const int32_t dirty_faces[15] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 2, 1, 0, 0, 1,
    };
    char input[PATH_MAX];
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(input, sizeof(input), "vkmesh_cleanup_in", ".meshbin"));
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_cleanup_out", ".meshbin"));
    CHECK_TRUE(write_meshbin(input, vertices, 4, dirty_faces, 5));
    char * cleanup_argv[] = {
        (char *) vkmesh_path, (char *) "--input", input, (char *) "--output", output,
        (char *) "--cleanup", (char *) "--max-hole-perimeter", (char *) "10",
        (char *) "--device", (char *) "0", NULL,
    };
    int ran = trellis_run_process_exact(cleanup_argv);
    test_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    int read_ok = ran && read_meshbin(output, &mesh);
    trellis_unlink(input);
    trellis_unlink(output);
    CHECK_TRUE(read_ok);
    CHECK_TRUE(mesh_is_closed_and_clean(mesh.vertices, mesh.faces, mesh.n_vertices, mesh.n_faces));
    test_mesh_free(&mesh);
    return 1;
}

static int test_cli_rejects_nan(const char * vkmesh_path) {
    char * args[] = {
        (char *) vkmesh_path, (char *) "--input", (char *) "unused.meshbin",
        (char *) "--output", (char *) "unused-output.meshbin",
        (char *) "--remesh-band", (char *) "nan", NULL,
    };
    CHECK_TRUE(!trellis_run_process_exact(args));
    return 1;
}

static int test_gltf_bake(void) {
    float vertices[12] = {
        -0.25f, -0.25f, -0.25f, 0.25f, -0.25f, -0.25f,
        -0.25f, 0.25f, -0.25f, -0.25f, -0.25f, 0.25f,
    };
    int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    int32_t coords[32];
    float attrs[48];
    int index = 0;
    for (int z = 1; z <= 2; ++z) {
        for (int y = 1; y <= 2; ++y) {
            for (int x = 1; x <= 2; ++x) {
                coords[index * 4 + 0] = 0;
                coords[index * 4 + 1] = x;
                coords[index * 4 + 2] = y;
                coords[index * 4 + 3] = z;
                attrs[index * 6 + 0] = (float) x * 0.25f;
                attrs[index * 6 + 1] = (float) y * 0.25f;
                attrs[index * 6 + 2] = (float) z * 0.25f;
                attrs[index * 6 + 3] = 0.1f;
                attrs[index * 6 + 4] = 0.7f;
                attrs[index * 6 + 5] = 1.0f;
                ++index;
            }
        }
    }
    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    mesh.vertices = vertices;
    mesh.faces = faces;
    mesh.n_vertices = 4;
    mesh.n_faces = 4;
    trellis_pbr_voxels voxels;
    memset(&voxels, 0, sizeof(voxels));
    voxels.coords_bxyz = coords;
    voxels.attrs = attrs;
    voxels.n_coords = index;
    voxels.channels = 6;
    voxels.resolution = 4;
    char output[PATH_MAX];
    CHECK_TRUE(trellis_make_temp_path(output, sizeof(output), "vkmesh_bake", ".glb"));
    trellis_status status = trellis_pipeline_write_gltf(output, &mesh, NULL, &voxels, 64, 0);
    FILE * f = status == TRELLIS_STATUS_OK ? fopen(output, "rb") : NULL;
    char magic[4] = {0, 0, 0, 0};
    int valid = f != NULL && fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        memcmp(magic, "glTF", 4) == 0;
    if (f != NULL) fclose(f);
    trellis_unlink(output);
    CHECK_TRUE(status == TRELLIS_STATUS_OK);
    CHECK_TRUE(valid);
    return 1;
}

typedef struct api_thread_args {
    const trellis_mesh_host * input;
    trellis_status status;
    int valid;
} api_thread_args;

static void run_api_thread(api_thread_args * args) {
    trellis_vkmesh_postprocess_options options;
    memset(&options, 0, sizeof(options));
    options.no_simplify = 1;
    options.max_hole_perimeter = 10.0f;
    options.device = 0;
    trellis_mesh_host output;
    memset(&output, 0, sizeof(output));
    args->status = trellis_vkmesh_postprocess(args->input, &output, NULL, &options);
    args->valid = args->status == TRELLIS_STATUS_OK &&
        mesh_is_closed_and_clean(output.vertices, output.faces, output.n_vertices, output.n_faces);
    trellis_mesh_free(&output);
}

#ifdef _WIN32
static DWORD WINAPI api_thread_entry(LPVOID opaque) {
    run_api_thread((api_thread_args *) opaque);
    return 0;
}
#else
static void * api_thread_entry(void * opaque) {
    run_api_thread((api_thread_args *) opaque);
    return NULL;
}
#endif

static int test_api_concurrent(void) {
    static float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    static int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    trellis_mesh_host input;
    memset(&input, 0, sizeof(input));
    input.vertices = vertices;
    input.faces = faces;
    input.n_vertices = 4;
    input.n_faces = 4;
    api_thread_args args[2] = { { &input, TRELLIS_STATUS_ERROR, 0 }, { &input, TRELLIS_STATUS_ERROR, 0 } };
#ifdef _WIN32
    HANDLE threads[2] = {
        CreateThread(NULL, 0, api_thread_entry, &args[0], 0, NULL),
        CreateThread(NULL, 0, api_thread_entry, &args[1], 0, NULL),
    };
    CHECK_TRUE(threads[0] != NULL && threads[1] != NULL);
    CHECK_TRUE(WaitForMultipleObjects(2, threads, TRUE, INFINITE) == WAIT_OBJECT_0);
    CloseHandle(threads[0]);
    CloseHandle(threads[1]);
#else
    pthread_t threads[2];
    CHECK_TRUE(pthread_create(&threads[0], NULL, api_thread_entry, &args[0]) == 0);
    CHECK_TRUE(pthread_create(&threads[1], NULL, api_thread_entry, &args[1]) == 0);
    CHECK_TRUE(pthread_join(threads[0], NULL) == 0);
    CHECK_TRUE(pthread_join(threads[1], NULL) == 0);
#endif
    CHECK_TRUE(args[0].status == TRELLIS_STATUS_OK && args[0].valid);
    CHECK_TRUE(args[1].status == TRELLIS_STATUS_OK && args[1].valid);
    return 1;
}

static int test_api_validation(void) {
    float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    int32_t faces[12] = {
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3,
    };
    trellis_mesh_host input;
    trellis_mesh_host output;
    trellis_vkmesh_postprocess_options options;
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    memset(&options, 0, sizeof(options));
    input.vertices = vertices;
    input.faces = faces;
    input.n_vertices = 4;
    input.n_faces = 4;
    options.device = INT_MAX;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    options.device = 0;
    options.remesh_band = NAN;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    options.remesh_band = 0.0f;
    vertices[0] = NAN;
    CHECK_TRUE(trellis_vkmesh_postprocess(&input, &output, NULL, &options) == TRELLIS_STATUS_INVALID_ARGUMENT);
    CHECK_TRUE(output.vertices == NULL && output.faces == NULL);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/vkmesh\n", argv[0]);
        return 2;
    }
    if (!has_compute_device(0)) {
        fprintf(stderr, "vkmesh tests skipped: Vulkan device 0 has no compute queue\n");
        return 77;
    }
    (void) test_cli_cleanup(argv[1]);
    (void) test_cli_rejects_nan(argv[1]);
    (void) test_api_concurrent();
    (void) test_api_validation();
    (void) test_gltf_bake();
    if (g_failures != 0) {
        fprintf(stderr, "vkmesh tests failed: %d failure(s)\n", g_failures);
        return 1;
    }
    fprintf(stderr, "vkmesh tests passed\n");
    return 0;
}
