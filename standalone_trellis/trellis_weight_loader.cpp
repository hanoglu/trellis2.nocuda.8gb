#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "third_party/safetensors-cpp/safetensors.hh"

namespace fs = std::filesystem;

static std::string shape_string(const std::vector<size_t> &shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
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

static int load_one(const fs::path &path, size_t preview_count, size_t &total_tensors, size_t &total_bytes, std::map<std::string, size_t> &dtype_counts) {
    safetensors::safetensors_t st;
    std::string warn;
    std::string err;
    const bool ok = safetensors::mmap_from_file(path.string(), &st, &warn, &err);
    if (!warn.empty()) {
        std::cerr << "warn: " << path << ": " << warn << "\n";
    }
    if (!ok) {
        std::cerr << "failed to mmap safetensors: " << path << "\n";
        std::cerr << err << "\n";
        return 1;
    }
    if (!safetensors::validate_data_offsets(st, err)) {
        std::cerr << "invalid safetensors offsets: " << path << "\n";
        std::cerr << err << "\n";
        return 1;
    }

    size_t file_bytes = 0;
    for (size_t i = 0; i < st.tensors.size(); ++i) {
        safetensors::tensor_t t;
        st.tensors.at(i, &t);
        const size_t bytes = safetensors::get_shape_size(t) * safetensors::get_dtype_bytes(t.dtype);
        file_bytes += bytes;
        dtype_counts[safetensors::get_dtype_str(t.dtype)] += 1;
    }
    total_tensors += st.tensors.size();
    total_bytes += file_bytes;

    std::cout << path << "\n";
    std::cout << "  tensors=" << st.tensors.size()
              << " mapped_bytes=" << st.databuffer_size
              << " tensor_bytes=" << file_bytes << "\n";

    const size_t n_preview = std::min(preview_count, st.tensors.size());
    for (size_t i = 0; i < n_preview; ++i) {
        safetensors::tensor_t t;
        st.tensors.at(i, &t);
        const std::string key = st.tensors.keys()[i];
        std::cout << "    " << std::left << std::setw(72) << key
                  << " " << std::setw(8) << safetensors::get_dtype_str(t.dtype)
                  << " " << shape_string(t.shape) << "\n";
    }
    if (st.tensors.size() > n_preview) {
        std::cout << "    ...\n";
    }
    return 0;
}

int main(int argc, char **argv) {
    fs::path input = "/home/wimaxs/Documents/TRELLIS.2/TRELLIS.2-4B/ckpts";
    size_t preview_count = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--preview") && i + 1 < argc) {
            preview_count = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [ckpt-dir-or-file] [--preview N]\n";
            return 0;
        } else {
            input = arg;
        }
    }

    const auto paths = collect_safetensors_paths(input);
    if (paths.empty()) {
        std::cerr << "no .safetensors files found under " << input << "\n";
        return 1;
    }

    size_t total_tensors = 0;
    size_t total_bytes = 0;
    std::map<std::string, size_t> dtype_counts;
    for (const auto &path : paths) {
        const int rc = load_one(path, preview_count, total_tensors, total_bytes, dtype_counts);
        if (rc != 0) {
            return rc;
        }
    }

    std::cout << "\nsummary\n";
    std::cout << "  files=" << paths.size() << "\n";
    std::cout << "  tensors=" << total_tensors << "\n";
    std::cout << "  tensor_bytes=" << total_bytes << "\n";
    std::cout << "  dtypes";
    for (const auto &kv : dtype_counts) {
        std::cout << " " << kv.first << "=" << kv.second;
    }
    std::cout << "\n";
    return 0;
}
