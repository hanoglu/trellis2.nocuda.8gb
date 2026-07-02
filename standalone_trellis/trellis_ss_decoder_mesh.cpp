#include "trellis_ops.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "third_party/safetensors-cpp/safetensors.hh"

namespace fs = std::filesystem;

struct TensorF32 {
    std::vector<float> data;
    std::vector<size_t> shape;
};

static size_t ncdhw_index(int c, int d, int h, int w, int channels, int depth, int height, int width) {
    return (((static_cast<size_t>(c) * static_cast<size_t>(depth) + static_cast<size_t>(d)) * static_cast<size_t>(height) + static_cast<size_t>(h)) * static_cast<size_t>(width) + static_cast<size_t>(w));
}

static bool load_safetensors(const fs::path &path, safetensors::safetensors_t &st) {
    std::string warn;
    std::string err;
    if (!safetensors::mmap_from_file(path.string(), &st, &warn, &err)) {
        std::cerr << "failed to mmap " << path << "\n" << err << "\n";
        return false;
    }
    if (!warn.empty()) {
        std::cerr << "warn: " << warn << "\n";
    }
    if (!safetensors::validate_data_offsets(st, err)) {
        std::cerr << "invalid data offsets: " << err << "\n";
        return false;
    }
    return true;
}

static const uint8_t *storage_base(const safetensors::safetensors_t &st) {
    return st.mmaped ? st.databuffer_addr : st.storage.data();
}

static bool tensor_to_f32(const safetensors::safetensors_t &st, const std::string &key, TensorF32 &out) {
    safetensors::tensor_t t;
    if (!st.tensors.at(key, &t)) {
        std::cerr << "missing tensor: " << key << "\n";
        return false;
    }
    const uint8_t *base = storage_base(st);
    if (base == nullptr) {
        std::cerr << "empty safetensors storage\n";
        return false;
    }
    const size_t n = safetensors::get_shape_size(t);
    out.shape = t.shape;
    out.data.resize(n);
    const uint8_t *src = base + t.data_offsets[0];
    if (t.dtype == safetensors::dtype::kFLOAT32) {
        const float *p = reinterpret_cast<const float *>(src);
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = p[i];
        }
    } else if (t.dtype == safetensors::dtype::kFLOAT16) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = safetensors::fp16_to_float(p[i]);
        }
    } else if (t.dtype == safetensors::dtype::kBFLOAT16) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = safetensors::bfloat16_to_float(p[i]);
        }
    } else {
        std::cerr << "unsupported tensor dtype for " << key << ": " << safetensors::get_dtype_str(t.dtype) << "\n";
        return false;
    }
    return true;
}

static bool get_tensor(const safetensors::safetensors_t &st, const std::string &key, TensorF32 &out) {
    if (!tensor_to_f32(st, key, out)) {
        return false;
    }
    return true;
}

static bool conv3d_same(
    const std::vector<float> &x,
    int channels,
    int depth,
    int height,
    int width,
    const TensorF32 &weight,
    const TensorF32 &bias,
    int out_channels,
    std::vector<float> &y) {
    if (weight.shape.size() != 5 || weight.shape[0] != static_cast<size_t>(out_channels) ||
        weight.shape[1] != static_cast<size_t>(channels) || weight.shape[2] != 3 ||
        weight.shape[3] != 3 || weight.shape[4] != 3) {
        std::cerr << "conv weight shape mismatch\n";
        return false;
    }
    if (bias.shape.size() != 1 || bias.shape[0] != static_cast<size_t>(out_channels)) {
        std::cerr << "conv bias shape mismatch\n";
        return false;
    }
    y.assign(static_cast<size_t>(out_channels) * static_cast<size_t>(depth) * static_cast<size_t>(height) * static_cast<size_t>(width), 0.0f);
    const strellis_status st = strellis_conv3d_ncdhw_f32(
        x.data(),
        weight.data.data(),
        bias.data.data(),
        y.data(),
        1,
        channels,
        depth,
        height,
        width,
        out_channels,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1);
    if (st != STRELLIS_STATUS_OK) {
        std::cerr << "conv3d failed: " << strellis_status_name(st) << "\n";
        return false;
    }
    return true;
}

static bool channel_norm_silu(
    const std::vector<float> &x,
    int channels,
    int depth,
    int height,
    int width,
    const TensorF32 &gamma,
    const TensorF32 &beta,
    std::vector<float> &y) {
    y.assign(x.size(), 0.0f);
    strellis_status st = strellis_channel_layer_norm_3d_ncdhw_f32(
        x.data(),
        gamma.data.data(),
        beta.data.data(),
        y.data(),
        1,
        channels,
        depth,
        height,
        width,
        1e-6f);
    if (st != STRELLIS_STATUS_OK) {
        std::cerr << "channel layer norm failed: " << strellis_status_name(st) << "\n";
        return false;
    }
    st = strellis_silu_f32(y.data(), y.data(), y.size());
    if (st != STRELLIS_STATUS_OK) {
        std::cerr << "silu failed: " << strellis_status_name(st) << "\n";
        return false;
    }
    return true;
}

static bool resblock(
    const safetensors::safetensors_t &st,
    const std::string &prefix,
    std::vector<float> &x,
    int channels,
    int depth,
    int height,
    int width) {
    TensorF32 norm1_w, norm1_b, norm2_w, norm2_b, conv1_w, conv1_b, conv2_w, conv2_b;
    if (!get_tensor(st, prefix + ".norm1.weight", norm1_w) ||
        !get_tensor(st, prefix + ".norm1.bias", norm1_b) ||
        !get_tensor(st, prefix + ".norm2.weight", norm2_w) ||
        !get_tensor(st, prefix + ".norm2.bias", norm2_b) ||
        !get_tensor(st, prefix + ".conv1.weight", conv1_w) ||
        !get_tensor(st, prefix + ".conv1.bias", conv1_b) ||
        !get_tensor(st, prefix + ".conv2.weight", conv2_w) ||
        !get_tensor(st, prefix + ".conv2.bias", conv2_b)) {
        return false;
    }

    std::vector<float> h, c1, h2, c2;
    if (!channel_norm_silu(x, channels, depth, height, width, norm1_w, norm1_b, h)) {
        return false;
    }
    if (!conv3d_same(h, channels, depth, height, width, conv1_w, conv1_b, channels, c1)) {
        return false;
    }
    if (!channel_norm_silu(c1, channels, depth, height, width, norm2_w, norm2_b, h2)) {
        return false;
    }
    if (!conv3d_same(h2, channels, depth, height, width, conv2_w, conv2_b, channels, c2)) {
        return false;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += c2[i];
    }
    return true;
}

static bool upsample_block(
    const safetensors::safetensors_t &st,
    const std::string &prefix,
    std::vector<float> &x,
    int in_channels,
    int out_channels,
    int &depth,
    int &height,
    int &width) {
    TensorF32 conv_w, conv_b;
    if (!get_tensor(st, prefix + ".conv.weight", conv_w) ||
        !get_tensor(st, prefix + ".conv.bias", conv_b)) {
        return false;
    }
    std::vector<float> conv;
    if (!conv3d_same(x, in_channels, depth, height, width, conv_w, conv_b, out_channels * 8, conv)) {
        return false;
    }
    std::vector<float> y(static_cast<size_t>(out_channels) * static_cast<size_t>(depth * 2) * static_cast<size_t>(height * 2) * static_cast<size_t>(width * 2));
    const strellis_status st_status = strellis_pixel_shuffle_3d_ncdhw_f32(
        conv.data(),
        y.data(),
        1,
        out_channels * 8,
        depth,
        height,
        width,
        2);
    if (st_status != STRELLIS_STATUS_OK) {
        std::cerr << "pixel shuffle failed: " << strellis_status_name(st_status) << "\n";
        return false;
    }
    depth *= 2;
    height *= 2;
    width *= 2;
    x.swap(y);
    return true;
}

static std::vector<float> make_latent(int latent_size) {
    const int channels = 8;
    std::vector<float> x(static_cast<size_t>(channels) * static_cast<size_t>(latent_size) * static_cast<size_t>(latent_size) * static_cast<size_t>(latent_size));
    for (int c = 0; c < channels; ++c) {
        for (int d = 0; d < latent_size; ++d) {
            for (int h = 0; h < latent_size; ++h) {
                for (int w = 0; w < latent_size; ++w) {
                    const float fd = ((float)d + 0.5f) / (float)latent_size - 0.5f;
                    const float fh = ((float)h + 0.5f) / (float)latent_size - 0.5f;
                    const float fw = ((float)w + 0.5f) / (float)latent_size - 0.5f;
                    const float dist = std::sqrt(fd * fd + fh * fh + fw * fw);
                    float v = 0.25f * std::sin((float)(c + 1) * (fd * 2.1f + fh * 1.3f - fw * 0.7f));
                    v += 0.11f * std::cos((float)(c + 3) * (fd - fh + fw));
                    if (c == 0) {
                        v += 0.45f - dist;
                    }
                    x[ncdhw_index(c, d, h, w, channels, latent_size, latent_size, latent_size)] = v;
                }
            }
        }
    }
    return x;
}

static bool decode_sparse_structure(
    const safetensors::safetensors_t &st,
    int latent_size,
    std::vector<float> &logits,
    int &resolution) {
    int depth = latent_size;
    int height = latent_size;
    int width = latent_size;
    std::vector<float> x = make_latent(latent_size);

    TensorF32 w, b;
    if (!get_tensor(st, "input_layer.weight", w) ||
        !get_tensor(st, "input_layer.bias", b)) {
        return false;
    }
    if (!conv3d_same(x, 8, depth, height, width, w, b, 512, x)) {
        return false;
    }

    if (!resblock(st, "middle_block.0", x, 512, depth, height, width) ||
        !resblock(st, "middle_block.1", x, 512, depth, height, width) ||
        !resblock(st, "blocks.0", x, 512, depth, height, width) ||
        !resblock(st, "blocks.1", x, 512, depth, height, width) ||
        !upsample_block(st, "blocks.2", x, 512, 128, depth, height, width) ||
        !resblock(st, "blocks.3", x, 128, depth, height, width) ||
        !resblock(st, "blocks.4", x, 128, depth, height, width) ||
        !upsample_block(st, "blocks.5", x, 128, 32, depth, height, width) ||
        !resblock(st, "blocks.6", x, 32, depth, height, width) ||
        !resblock(st, "blocks.7", x, 32, depth, height, width)) {
        return false;
    }

    TensorF32 norm_w, norm_b, out_w, out_b;
    if (!get_tensor(st, "out_layer.0.weight", norm_w) ||
        !get_tensor(st, "out_layer.0.bias", norm_b) ||
        !get_tensor(st, "out_layer.2.weight", out_w) ||
        !get_tensor(st, "out_layer.2.bias", out_b)) {
        return false;
    }
    std::vector<float> h;
    if (!channel_norm_silu(x, 32, depth, height, width, norm_w, norm_b, h)) {
        return false;
    }
    if (!conv3d_same(h, 32, depth, height, width, out_w, out_b, 1, logits)) {
        return false;
    }
    resolution = depth;
    return true;
}

static bool logits_to_mesh(const std::vector<float> &logits, int resolution, float threshold, strellis_mesh &mesh) {
    float actual_threshold = threshold;
    int64_t n_active = 0;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (v > max_logit) {
            max_logit = v;
        }
        if (v >= actual_threshold) {
            ++n_active;
        }
    }
    if (n_active == 0) {
        actual_threshold = max_logit - std::max(1e-3f, std::fabs(max_logit) * 1e-5f);
        for (float v : logits) {
            if (v >= actual_threshold) {
                ++n_active;
            }
        }
        std::cerr << "no logits above requested threshold; using max-logit fallback threshold " << actual_threshold << "\n";
    }

    std::vector<int32_t> coords(static_cast<size_t>(n_active) * 4u);
    std::vector<float> feats(static_cast<size_t>(n_active));
    int64_t row = 0;
    for (int d = 0; d < resolution; ++d) {
        for (int h = 0; h < resolution; ++h) {
            for (int w = 0; w < resolution; ++w) {
                const size_t idx = ncdhw_index(0, d, h, w, 1, resolution, resolution, resolution);
                const float v = logits[idx];
                if (v >= actual_threshold) {
                    coords[static_cast<size_t>(row) * 4u + 0u] = 0;
                    coords[static_cast<size_t>(row) * 4u + 1u] = d;
                    coords[static_cast<size_t>(row) * 4u + 2u] = h;
                    coords[static_cast<size_t>(row) * 4u + 3u] = w;
                    feats[static_cast<size_t>(row)] = v - actual_threshold;
                    ++row;
                }
            }
        }
    }
    const strellis_status st = strellis_flexible_dual_grid_mesh_from_decoder_logits_f32(
        coords.data(),
        feats.data(),
        n_active,
        1,
        resolution,
        &mesh);
    if (st != STRELLIS_STATUS_OK) {
        std::cerr << "mesh extraction failed: " << strellis_status_name(st) << "\n";
        return false;
    }
    return true;
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

static bool write_f32(const fs::path &path, const std::vector<float> &data) {
    FILE *f = fopen(path.string().c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const size_t wrote = fwrite(data.data(), sizeof(float), data.size(), f);
    const bool ok = wrote == data.size() && fclose(f) == 0;
    return ok;
}

int main(int argc, char **argv) {
    fs::path ckpt = "/home/wimaxs/Documents/TRELLIS.2/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
    fs::path out = "/tmp/trellis_c_real_ss_decoder_mesh.obj";
    fs::path dump_logits;
    int latent_size = 2;
    float threshold = 0.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ckpt" && i + 1 < argc) {
            ckpt = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else if (arg == "--dump-logits" && i + 1 < argc) {
            dump_logits = argv[++i];
        } else if (arg == "--latent-size" && i + 1 < argc) {
            latent_size = std::atoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::strtof(argv[++i], nullptr);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [--ckpt ss_dec.safetensors] [--latent-size N] [--threshold X] [--out mesh.obj] [--dump-logits logits.f32]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }
    if (latent_size <= 0 || latent_size > 4) {
        std::cerr << "latent-size must be in [1,4] for this CPU reference decoder\n";
        return 2;
    }

    safetensors::safetensors_t st;
    if (!load_safetensors(ckpt, st)) {
        return 1;
    }
    std::vector<float> logits;
    int resolution = 0;
    if (!decode_sparse_structure(st, latent_size, logits, resolution)) {
        return 1;
    }
    if (!dump_logits.empty() && !write_f32(dump_logits, logits)) {
        std::cerr << "failed to write logits: " << dump_logits << "\n";
        return 1;
    }
    strellis_mesh mesh;
    std::memset(&mesh, 0, sizeof(mesh));
    if (!logits_to_mesh(logits, resolution, threshold, mesh)) {
        return 1;
    }
    if (!write_obj(out, mesh)) {
        std::cerr << "failed to write OBJ: " << out << "\n";
        strellis_mesh_free(&mesh);
        return 1;
    }
    std::cout << "C real SparseStructureDecoder mesh written\n";
    std::cout << "  ckpt: " << ckpt << "\n";
    std::cout << "  latent_shape: [1,8," << latent_size << "," << latent_size << "," << latent_size << "]\n";
    std::cout << "  logits_shape: [1,1," << resolution << "," << resolution << "," << resolution << "]\n";
    std::cout << "  mesh_vertices: " << static_cast<long long>(mesh.n_vertices) << "\n";
    std::cout << "  mesh_faces: " << static_cast<long long>(mesh.n_faces) << "\n";
    std::cout << "  obj: " << out << "\n";
    if (!dump_logits.empty()) {
        std::cout << "  logits: " << dump_logits << "\n";
    }
    strellis_mesh_free(&mesh);
    return 0;
}
