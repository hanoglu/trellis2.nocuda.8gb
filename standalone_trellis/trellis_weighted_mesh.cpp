#include "trellis_ops.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "third_party/safetensors-cpp/safetensors.hh"

namespace fs = std::filesystem;

struct LoadedSafetensors {
    std::vector<fs::path> paths;
    std::vector<safetensors::safetensors_t> files;
    size_t tensor_count = 0;
    size_t tensor_bytes = 0;
    std::map<std::string, size_t> dtype_counts;
    uint64_t signature = 1469598103934665603ull;
};

static void fnv1a_update(uint64_t &h, const void *data, size_t n) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
}

static void fnv1a_update_string(uint64_t &h, const std::string &s) {
    fnv1a_update(h, s.data(), s.size());
}

static std::vector<fs::path> collect_safetensors_paths(const fs::path &input) {
    std::vector<fs::path> paths;
    if (fs::is_regular_file(input) && input.extension() == ".safetensors") {
        paths.push_back(input);
    } else if (fs::is_directory(input)) {
        for (const auto &entry : fs::recursive_directory_iterator(input)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                paths.push_back(entry.path());
            }
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

static const uint8_t *storage_base(const safetensors::safetensors_t &st) {
    if (st.mmaped) {
        return st.databuffer_addr;
    }
    return st.storage.data();
}

static int load_weights(const fs::path &ckpt_path, LoadedSafetensors &out) {
    out.paths = collect_safetensors_paths(ckpt_path);
    if (out.paths.empty()) {
        std::cerr << "no .safetensors files found under " << ckpt_path << "\n";
        return 1;
    }

    out.files.reserve(out.paths.size());
    for (const auto &path : out.paths) {
        out.files.emplace_back();
        safetensors::safetensors_t &st = out.files.back();
        std::string warn;
        std::string err;
        if (!safetensors::mmap_from_file(path.string(), &st, &warn, &err)) {
            std::cerr << "failed to mmap " << path << "\n" << err << "\n";
            return 1;
        }
        if (!warn.empty()) {
            std::cerr << "warn: " << path << ": " << warn << "\n";
        }
        if (!safetensors::validate_data_offsets(st, err)) {
            std::cerr << "invalid data offsets in " << path << "\n" << err << "\n";
            return 1;
        }

        const uint8_t *base = storage_base(st);
        fnv1a_update_string(out.signature, path.filename().string());
        for (size_t i = 0; i < st.tensors.size(); ++i) {
            safetensors::tensor_t t;
            st.tensors.at(i, &t);
            const std::string key = st.tensors.keys()[i];
            const std::string dtype = safetensors::get_dtype_str(t.dtype);
            const size_t bytes = safetensors::get_shape_size(t) * safetensors::get_dtype_bytes(t.dtype);

            ++out.tensor_count;
            out.tensor_bytes += bytes;
            out.dtype_counts[dtype] += 1;

            fnv1a_update_string(out.signature, key);
            fnv1a_update_string(out.signature, dtype);
            for (size_t dim : t.shape) {
                fnv1a_update(out.signature, &dim, sizeof(dim));
            }
            fnv1a_update(out.signature, t.data_offsets.data(), t.data_offsets.size() * sizeof(t.data_offsets[0]));

            const size_t begin = t.data_offsets[0];
            const size_t end = t.data_offsets[1];
            if (base != nullptr && end > begin) {
                const size_t n = end - begin;
                const size_t head = std::min<size_t>(n, 256);
                fnv1a_update(out.signature, base + begin, head);
                if (n > head) {
                    const size_t mid = begin + n / 2;
                    const size_t mid_n = std::min<size_t>(n - (mid - begin), 64);
                    fnv1a_update(out.signature, base + mid, mid_n);
                    const size_t tail = std::min<size_t>(n, 256);
                    fnv1a_update(out.signature, base + end - tail, tail);
                }
            }
        }

    }
    return 0;
}

static bool write_obj(const fs::path &path, const strellis_mesh &mesh) {
    FILE *f = fopen(path.string().c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    for (int64_t i = 0; i < mesh.n_vertices; ++i) {
        const float *v = mesh.vertices + static_cast<size_t>(i) * 3u;
        fprintf(f, "v %.9g %.9g %.9g\n", v[0], v[1], v[2]);
    }
    for (int64_t i = 0; i < mesh.n_faces; ++i) {
        const uint32_t *tri = mesh.faces + static_cast<size_t>(i) * 3u;
        fprintf(f, "f %u %u %u\n", tri[0] + 1u, tri[1] + 1u, tri[2] + 1u);
    }
    return fclose(f) == 0;
}

static void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0 << " [--ckpt-dir PATH] [--out mesh.obj]\n"
        << "       [--latent-size N] [--stage1-steps N] [--stage2-steps N]\n"
        << "       [--seed N] [--threshold X] [--no-weight-signature-seed]\n";
}

static bool parse_int(const char *s, int &out) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_u32(const char *s, uint32_t &out) {
    char *end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v > 0xfffffffful) {
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

static bool parse_float(const char *s, float &out) {
    char *end = nullptr;
    float v = std::strtof(s, &end);
    if (end == s || *end != '\0') {
        return false;
    }
    out = v;
    return true;
}

int main(int argc, char **argv) {
    fs::path ckpt_dir = "/home/wimaxs/Documents/TRELLIS.2/TRELLIS.2-4B/ckpts";
    fs::path out_path = "/tmp/trellis_c_weighted_mesh.obj";
    bool use_weight_signature_seed = true;

    strellis_infer_options opt;
    strellis_infer_options_default(&opt);
    opt.latent_size = 10;
    opt.stage1_steps = 4;
    opt.stage2_steps = 4;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--ckpt-dir" && i + 1 < argc) {
            ckpt_dir = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--latent-size" && i + 1 < argc) {
            if (!parse_int(argv[++i], opt.latent_size)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--stage1-steps" && i + 1 < argc) {
            if (!parse_int(argv[++i], opt.stage1_steps)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--stage2-steps" && i + 1 < argc) {
            if (!parse_int(argv[++i], opt.stage2_steps)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            if (!parse_u32(argv[++i], opt.seed)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--threshold" && i + 1 < argc) {
            if (!parse_float(argv[++i], opt.voxel_threshold)) {
                usage(argv[0]);
                return 2;
            }
        } else if (arg == "--no-weight-signature-seed") {
            use_weight_signature_seed = false;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    LoadedSafetensors weights;
    if (load_weights(ckpt_dir, weights) != 0) {
        return 1;
    }
    if (use_weight_signature_seed) {
        opt.seed ^= static_cast<uint32_t>(weights.signature);
        opt.seed ^= static_cast<uint32_t>(weights.signature >> 32);
    }

    std::cout << "Loaded TRELLIS2 safetensors\n";
    std::cout << "  files: " << weights.paths.size() << "\n";
    std::cout << "  tensors: " << weights.tensor_count << "\n";
    std::cout << "  tensor_bytes: " << weights.tensor_bytes << "\n";
    std::cout << "  signature: 0x" << std::hex << weights.signature << std::dec << "\n";
    std::cout << "  dtypes:";
    for (const auto &kv : weights.dtype_counts) {
        std::cout << " " << kv.first << "=" << kv.second;
    }
    std::cout << "\n";
    std::cout << "Running standalone C inference and OBJ mesh export\n";
    std::cout << "  note: real weights are loaded and validated; full 4B module topology is not wired yet.\n";

    strellis_infer_result result;
    std::memset(&result, 0, sizeof(result));
    const strellis_status status = strellis_run_inference_compute(&opt, &result);
    if (status != STRELLIS_STATUS_OK) {
        std::cerr << "strellis_run_inference_compute failed: " << strellis_status_name(status) << "\n";
        return 1;
    }

    const bool wrote = write_obj(out_path, result.mesh);
    std::cout << "  latent_size: " << opt.latent_size << "\n";
    std::cout << "  active_coords: " << static_cast<long long>(result.n_coords) << "\n";
    std::cout << "  slat_shape: [" << static_cast<long long>(result.n_coords) << "," << result.slat_channels << "]\n";
    std::cout << "  mesh_vertices: " << static_cast<long long>(result.mesh.n_vertices) << "\n";
    std::cout << "  mesh_faces: " << static_cast<long long>(result.mesh.n_faces) << "\n";
    if (wrote) {
        std::cout << "  obj: " << out_path << "\n";
    } else {
        std::cerr << "failed to write OBJ: " << out_path << "\n";
        strellis_infer_result_free(&result);
        return 1;
    }

    strellis_infer_result_free(&result);
    return 0;
}
