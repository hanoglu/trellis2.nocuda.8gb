#include "trellis.h"
#include "../../src/pipeline/trellis_pipeline_internal.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE * out, const char * argv0) {
    fprintf(out,
        "Usage:\n"
        "  %s --mesh processed.obj --voxels pbr_voxels.bin --gltf out.glb [options]\n"
        "\n"
        "Re-bakes a GLB from material inputs dumped by TRELLIS_MATERIAL_DUMP_DIR.\n"
        "\n"
        "Options:\n"
        "  --sample projection_source.obj  Optional source mesh for CuMesh-style texture projection\n"
        "  --texture-size N                Texture size, default 1024\n",
        argv0);
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static int parse_int_arg(const char * text, int * out) {
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    char * end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return 0;
    }
    *out = (int) v;
    return 1;
}

static int mesh_reserve_vertices(trellis_mesh_host * mesh, int64_t need, int64_t * capacity) {
    if (need <= *capacity) {
        return 1;
    }
    int64_t cap = *capacity > 0 ? *capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) {
            return 0;
        }
        cap *= 2;
    }
    float * vertices = (float *) realloc(mesh->vertices, (size_t) cap * 3u * sizeof(float));
    if (vertices == NULL) {
        return 0;
    }
    mesh->vertices = vertices;
    *capacity = cap;
    return 1;
}

static int mesh_reserve_faces(trellis_mesh_host * mesh, int64_t need, int64_t * capacity) {
    if (need <= *capacity) {
        return 1;
    }
    int64_t cap = *capacity > 0 ? *capacity : 1024;
    while (cap < need) {
        if (cap > INT64_MAX / 2) {
            return 0;
        }
        cap *= 2;
    }
    int32_t * faces = (int32_t *) realloc(mesh->faces, (size_t) cap * 3u * sizeof(int32_t));
    if (faces == NULL) {
        return 0;
    }
    mesh->faces = faces;
    *capacity = cap;
    return 1;
}

static int mesh_add_obj_vertex(
    trellis_mesh_host * mesh,
    int64_t * vertex_capacity,
    float obj_x,
    float obj_y,
    float obj_z) {
    if (!mesh_reserve_vertices(mesh, mesh->n_vertices + 1, vertex_capacity)) {
        return 0;
    }
    float * v = mesh->vertices + (size_t) mesh->n_vertices * 3u;
    v[0] = obj_x;
    v[1] = -obj_z;
    v[2] = obj_y;
    ++mesh->n_vertices;
    return 1;
}

static int mesh_add_face(trellis_mesh_host * mesh, int64_t * face_capacity, int32_t a, int32_t b, int32_t c) {
    if (a < 0 || b < 0 || c < 0 ||
        a >= mesh->n_vertices || b >= mesh->n_vertices || c >= mesh->n_vertices) {
        return 0;
    }
    if (!mesh_reserve_faces(mesh, mesh->n_faces + 1, face_capacity)) {
        return 0;
    }
    int32_t * f = mesh->faces + (size_t) mesh->n_faces * 3u;
    f[0] = a;
    f[1] = b;
    f[2] = c;
    ++mesh->n_faces;
    return 1;
}

static int parse_obj_index(const char * token, int64_t vertex_count, int32_t * out) {
    char * end = NULL;
    long idx = strtol(token, &end, 10);
    if (end == token || idx == 0) {
        return 0;
    }
    int64_t resolved = idx > 0 ? (int64_t) idx - 1 : vertex_count + (int64_t) idx;
    if (resolved < 0 || resolved >= vertex_count || resolved > INT32_MAX) {
        return 0;
    }
    *out = (int32_t) resolved;
    return 1;
}

static trellis_status load_obj_mesh(const char * path, trellis_mesh_host * mesh_out) {
    if (path == NULL || mesh_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(mesh_out, 0, sizeof(*mesh_out));
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "trellis-rebake-gltf: failed to open OBJ %s: %s\n", path, strerror(errno));
        return TRELLIS_STATUS_IO_ERROR;
    }

    trellis_mesh_host mesh;
    memset(&mesh, 0, sizeof(mesh));
    int64_t vertex_capacity = 0;
    int64_t face_capacity = 0;
    char line[8192];
    int64_t line_no = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    while (fgets(line, sizeof(line), f) != NULL) {
        ++line_no;
        char * p = line;
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (p[0] == 'v' && isspace((unsigned char) p[1])) {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            if (sscanf(p + 1, "%f %f %f", &x, &y, &z) != 3 ||
                !mesh_add_obj_vertex(&mesh, &vertex_capacity, x, y, z)) {
                fprintf(stderr, "trellis-rebake-gltf: bad vertex at %s:%lld\n", path, (long long) line_no);
                status = TRELLIS_STATUS_IO_ERROR;
                break;
            }
        } else if (p[0] == 'f' && isspace((unsigned char) p[1])) {
            int32_t ids[256];
            int n = 0;
            char * q = p + 1;
            while (*q != '\0') {
                while (*q != '\0' && isspace((unsigned char) *q)) {
                    ++q;
                }
                if (*q == '\0' || *q == '#') {
                    break;
                }
                if (n >= (int) (sizeof(ids) / sizeof(ids[0])) ||
                    !parse_obj_index(q, mesh.n_vertices, &ids[n])) {
                    fprintf(stderr, "trellis-rebake-gltf: bad face index at %s:%lld\n", path, (long long) line_no);
                    status = TRELLIS_STATUS_IO_ERROR;
                    break;
                }
                ++n;
                while (*q != '\0' && !isspace((unsigned char) *q)) {
                    ++q;
                }
            }
            if (status != TRELLIS_STATUS_OK) {
                break;
            }
            if (n < 3) {
                fprintf(stderr, "trellis-rebake-gltf: face has fewer than 3 vertices at %s:%lld\n", path, (long long) line_no);
                status = TRELLIS_STATUS_IO_ERROR;
                break;
            }
            for (int i = 1; i + 1 < n; ++i) {
                if (!mesh_add_face(&mesh, &face_capacity, ids[0], ids[i], ids[i + 1])) {
                    fprintf(stderr, "trellis-rebake-gltf: failed to append face at %s:%lld\n", path, (long long) line_no);
                    status = TRELLIS_STATUS_IO_ERROR;
                    break;
                }
            }
            if (status != TRELLIS_STATUS_OK) {
                break;
            }
        }
    }
    fclose(f);
    if (status == TRELLIS_STATUS_OK && (mesh.n_vertices <= 0 || mesh.n_faces <= 0)) {
        fprintf(stderr, "trellis-rebake-gltf: OBJ has no usable triangle mesh: %s\n", path);
        status = TRELLIS_STATUS_IO_ERROR;
    }
    if (status != TRELLIS_STATUS_OK) {
        trellis_mesh_free(&mesh);
        return status;
    }
    *mesh_out = mesh;
    return TRELLIS_STATUS_OK;
}

static int checked_mul_size(size_t a, size_t b, size_t * out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static trellis_status read_pbr_voxels(const char * path, trellis_pbr_voxels * voxels_out) {
    if (path == NULL || voxels_out == NULL) {
        return TRELLIS_STATUS_INVALID_ARGUMENT;
    }
    memset(voxels_out, 0, sizeof(*voxels_out));
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "trellis-rebake-gltf: failed to open PBR voxels %s: %s\n", path, strerror(errno));
        return TRELLIS_STATUS_IO_ERROR;
    }

    char magic[8];
    int64_t n_coords = 0;
    int channels = 0;
    int resolution = 0;
    trellis_status status = TRELLIS_STATUS_OK;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(&n_coords, sizeof(n_coords), 1, f) != 1 ||
        fread(&channels, sizeof(channels), 1, f) != 1 ||
        fread(&resolution, sizeof(resolution), 1, f) != 1 ||
        memcmp(magic, "TRLPBR1", 7) != 0 ||
        n_coords <= 0 || channels <= 0 || channels > 64) {
        status = TRELLIS_STATUS_PARSE_ERROR;
    }

    trellis_pbr_voxels voxels;
    memset(&voxels, 0, sizeof(voxels));
    if (status == TRELLIS_STATUS_OK) {
        size_t coords_count = 0;
        size_t attrs_count = 0;
        if (!checked_mul_size((size_t) n_coords, 4u, &coords_count) ||
            !checked_mul_size((size_t) n_coords, (size_t) channels, &attrs_count)) {
            status = TRELLIS_STATUS_OUT_OF_MEMORY;
        } else {
            voxels.coords_bxyz = (int32_t *) malloc(coords_count * sizeof(int32_t));
            voxels.attrs = (float *) malloc(attrs_count * sizeof(float));
            voxels.n_coords = n_coords;
            voxels.channels = channels;
            voxels.resolution = resolution;
            if (voxels.coords_bxyz == NULL || voxels.attrs == NULL) {
                status = TRELLIS_STATUS_OUT_OF_MEMORY;
            } else if (fread(voxels.coords_bxyz, sizeof(int32_t), coords_count, f) != coords_count ||
                       fread(voxels.attrs, sizeof(float), attrs_count, f) != attrs_count) {
                status = TRELLIS_STATUS_IO_ERROR;
            }
        }
    }
    fclose(f);
    if (status != TRELLIS_STATUS_OK) {
        trellis_pbr_voxels_free(&voxels);
        fprintf(stderr, "trellis-rebake-gltf: failed to read PBR voxels %s: %s\n", path, trellis_status_string(status));
        return status;
    }
    *voxels_out = voxels;
    return TRELLIS_STATUS_OK;
}

int main(int argc, char ** argv) {
    const char * mesh_path = NULL;
    const char * sample_path = NULL;
    const char * voxels_path = NULL;
    const char * gltf_path = NULL;
    int texture_size = 1024;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mesh") == 0) {
            mesh_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--sample") == 0) {
            sample_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--voxels") == 0) {
            voxels_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--gltf") == 0 || strcmp(argv[i], "--glb") == 0) {
            gltf_path = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "--texture-size") == 0) {
            const char * value = arg_value(argc, argv, &i);
            if (!parse_int_arg(value, &texture_size)) {
                fprintf(stderr, "trellis-rebake-gltf: invalid --texture-size\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "trellis-rebake-gltf: unknown or incomplete argument: %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (mesh_path == NULL || voxels_path == NULL || gltf_path == NULL) {
        usage(stderr, argv[0]);
        return 2;
    }

    trellis_mesh_host mesh;
    trellis_mesh_host sample_mesh;
    trellis_pbr_voxels voxels;
    memset(&mesh, 0, sizeof(mesh));
    memset(&sample_mesh, 0, sizeof(sample_mesh));
    memset(&voxels, 0, sizeof(voxels));

    trellis_status status = load_obj_mesh(mesh_path, &mesh);
    if (status == TRELLIS_STATUS_OK && sample_path != NULL) {
        status = load_obj_mesh(sample_path, &sample_mesh);
    }
    if (status == TRELLIS_STATUS_OK) {
        status = read_pbr_voxels(voxels_path, &voxels);
    }
    if (status == TRELLIS_STATUS_OK) {
        const trellis_mesh_host * sample =
            sample_mesh.vertices != NULL && sample_mesh.faces != NULL ? &sample_mesh : NULL;
        status = trellis_pipeline_write_gltf(gltf_path, &mesh, sample, &voxels, texture_size);
    }

    trellis_mesh_free(&sample_mesh);
    trellis_mesh_free(&mesh);
    trellis_pbr_voxels_free(&voxels);

    if (status != TRELLIS_STATUS_OK) {
        fprintf(stderr, "trellis-rebake-gltf: failed: %s\n", trellis_status_string(status));
        return 1;
    }
    fprintf(stderr, "trellis-rebake-gltf: wrote %s\n", gltf_path);
    return 0;
}
