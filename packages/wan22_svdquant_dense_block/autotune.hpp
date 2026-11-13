// Reusable SYCL / C++17 runtime autotuning engine with persistent JSON selection caching.
//
// Designed for SYCL kernels (including KernelFoundry PyTorch `.sycl` extensions built with `icpx`)
// where tile sizes, subgroup layouts, and GRF modes (`grf_size<128|256>`) are C++ compile-time
// template parameters. Instantiate a curated table of template specializations and let
// `SyclRuntimeAutotuner` filter, benchmark, validate, and cache the winner per bucketed shape.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__INTEL_LLVM_COMPILER) || defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__)
#include <sycl/sycl.hpp>
#define KF_HAS_SYCL_HEADER 1
#else
#define KF_HAS_SYCL_HEADER 0
#endif

namespace kf_sycl_autotune {

struct SyclDeviceInfo {
    std::string name = "sycl_device";
    std::string vendor = "Intel";
    std::string driver_version = "0";
    int max_compute_units = 160;
    int local_mem_kb = 128;
    int max_work_group_size = 1024;
    int xe_cores = 20;
    int gpu_eu_count = -1;                   // vector engines (ext_intel_gpu_eu_count); -1 = unknown
    std::vector<size_t> sub_group_sizes;     // empty = unknown
};

#if KF_HAS_SYCL_HEADER
inline SyclDeviceInfo query_sycl_device_info(const sycl::queue& q) {
    auto dev = q.get_device();
    if (!dev.is_gpu()) throw std::invalid_argument("SYCL autotuning requires a GPU queue");
    SyclDeviceInfo info;
    info.name = dev.get_info<sycl::info::device::name>();
    info.vendor = dev.get_info<sycl::info::device::vendor>();
    info.driver_version = dev.get_info<sycl::info::device::driver_version>();
    info.max_compute_units = static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
    info.local_mem_kb = static_cast<int>(dev.get_info<sycl::info::device::local_mem_size>() / 1024);
    info.max_work_group_size = static_cast<int>(dev.get_info<sycl::info::device::max_work_group_size>());
    int eus_per_xe = (info.local_mem_kb >= 128) ? 8 : 16;
    info.xe_cores = std::max(1, info.max_compute_units / eus_per_xe);
    try { info.sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>(); } catch (const sycl::exception&) {}
    if (dev.has(sycl::aspect::ext_intel_gpu_eu_count))
        info.gpu_eu_count = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
    return info;
}

// Limits of ONE compiled kernel on this device (static-dispatch legality: the device maximum is not enough, a
// register-heavy kernel can launch with fewer work-items). Unknown values stay -1. Kernels must be named.
struct CompiledKernelLimits {
    int64_t max_work_group_size = -1;
    int64_t private_mem_size = -1;   // spill / private bytes per work-item
};

template <typename KernelName>
CompiledKernelLimits compiled_kernel_limits(const sycl::queue& q) {
    CompiledKernelLimits limits;
    try {
        const auto id = sycl::get_kernel_id<KernelName>();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {id});
        const auto kernel = bundle.get_kernel(id);
        limits.max_work_group_size =
            int64_t(kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device()));
        limits.private_mem_size =
            int64_t(kernel.template get_info<sycl::info::kernel_device_specific::private_mem_size>(q.get_device()));
    } catch (const sycl::exception&) {}
    return limits;
}
#endif

inline int64_t bucket_dim(int64_t m) {
    if (m <= 16) return std::max<int64_t>(1, m);
    if (m <= 512) {
        int64_t p = 16;
        while (p < m) p <<= 1;
        return p;
    }
    return ((m + 255) / 256) * 256;
}

inline std::string fnv1a64(const std::string& value) {
    uint64_t result = 0xCBF29CE484222325ULL;
    for (unsigned char byte : value) {
        result ^= static_cast<uint64_t>(byte);
        result *= 0x100000001B3ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << result;
    return out.str();
}

inline std::filesystem::path default_cache_dir() {
    if (const char* override_dir = std::getenv("SYCL_AUTOTUNE_CACHE_DIR")) {
        if (*override_dir) {
            return std::filesystem::path(override_dir) / "kernelfoundry-sycl-autotune";
        }
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg) {
            return std::filesystem::path(xdg) / "sycl" / "kernelfoundry-autotune";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home) {
            return std::filesystem::path(home) / ".cache" / "sycl" / "kernelfoundry-autotune";
        }
    }
    return std::filesystem::temp_directory_path() / "kernelfoundry-sycl-autotune";
}

struct SyclVariantSpec {
    std::string name;
    std::map<std::string, int> parameters;
    std::function<bool(const SyclDeviceInfo&)> is_feasible;
    std::function<void()> launch;
};

struct TuneResult {
    int winner_index = -1;
    std::string winner_name;
    std::map<std::string, int> parameters;
    double elapsed_seconds = 0.0;
    bool from_cache = false;
    // Cold searches only: per-candidate screening seconds, and median seconds for finalists. Set
    // SelectionSettings::finalists to the pool size for a full sweep.
    std::map<std::string, double> screen_seconds;
    std::map<std::string, double> median_seconds;
};

// Work actually performed, counted at the real search/validate/benchmark boundaries. Candidate IDs are names.
struct SelectionStats {
    int searches = 0, benchmarks = 0, validations = 0;
    std::set<std::string> benchmarked, validated;
};

struct SelectionSettings {
    int finalists = 3;
    int batches = 5;
    double relative_noise = 0.02;
    double slow_factor = 3.0;
};

using StaticPolicy = std::function<std::pair<std::string, std::vector<std::string>>(const std::string&)>;

class SyclRuntimeAutotuner {
public:
    static SyclRuntimeAutotuner& instance() {
        static SyclRuntimeAutotuner inst;
        return inst;
    }

    TuneResult select_and_launch(
        const std::string& op_id,
        const std::string& workload_key,
        const SyclDeviceInfo& dev_info,
        const std::vector<SyclVariantSpec>& variants,
        const std::function<void()>& sync_fn,
        const std::function<bool()>& validate_fn = nullptr,
        const std::filesystem::path& cache_dir = default_cache_dir(),
        int warmup_iters = 2,
        int bench_iters = 5,
        const StaticPolicy& policy = nullptr,
        const std::string& policy_version = "",
        const std::string& compiler_flags = "",
        SelectionSettings selection = {})
    {
        if (variants.size() < 2) {
            throw std::invalid_argument("SyclRuntimeAutotuner requires at least 2 candidate variants");
        }
        if (dev_info.name == "sycl_device" || dev_info.driver_version.empty() || dev_info.driver_version == "0")
            throw std::invalid_argument("Actual SYCL GPU device and driver identity are required");

        if (warmup_iters < 1 || bench_iters < 1 || selection.finalists < 2 || selection.batches < 3 ||
            !std::isfinite(selection.relative_noise) || selection.relative_noise < 0 ||
            selection.relative_noise >= 1 || !std::isfinite(selection.slow_factor) || selection.slow_factor <= 1) {
            throw std::invalid_argument("Invalid SYCL selection settings");
        }
        std::vector<int> eligible;
        std::string default_name;
        std::vector<std::string> names;
        if (policy) {
            auto resolved = policy(workload_key);
            default_name = resolved.first;
            names = resolved.second;
        }
        for (int index = 0; index < static_cast<int>(variants.size()); ++index) {
            const auto& variant = variants[index];
            if (policy && std::find(names.begin(), names.end(), variant.name) == names.end()) continue;
            // Static dispatch owns eligibility: an infeasible pool member is a dispatch bug, never silently dropped.
            if (variant.is_feasible && !variant.is_feasible(dev_info))
                throw std::invalid_argument("Pool member '" + variant.name + "' is infeasible; fix static dispatch");
            eligible.push_back(index);
        }
        if (eligible.empty()) throw std::runtime_error("No eligible SYCL GPU specialization");
        if (!policy) default_name = variants[eligible.front()].name;
        auto default_it = std::find_if(eligible.begin(), eligible.end(), [&](int index) {
            return variants[index].name == default_name;
        });
        if (default_it == eligible.end()) throw std::invalid_argument("Policy default is not eligible");
        std::rotate(eligible.begin(), default_it, default_it + 1);
        std::ostringstream identity;
        identity << std::quoted(op_id) << std::quoted(dev_info.name) << std::quoted(dev_info.vendor)
                 << std::quoted(dev_info.driver_version) << std::quoted(workload_key)
                 << std::quoted(policy_version) << std::quoted(compiler_flags) << std::quoted(default_name)
                 << "|protocol3|" << selection.finalists << '|' << selection.batches << '|'
                 << std::setprecision(17) << selection.relative_noise << '|' << selection.slow_factor;
        for (int index : eligible) {
            identity << std::quoted(variants[index].name);
            for (const auto& parameter : variants[index].parameters)
                identity << std::quoted(parameter.first) << ':' << parameter.second << ';';
        }
        const std::string composite_key = identity.str();

        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = winner_cache_.find(composite_key);
            if (it != winner_cache_.end() &&
                std::find(eligible.begin(), eligible.end(), it->second) != eligible.end()) {
                int idx = it->second;
                variants[idx].launch();
                return make_result(idx, variants[idx], 0.0, true);
            }
        }

        std::filesystem::path cache_file = cache_dir / (fnv1a64(composite_key) + ".json");
        int cached_idx = load_cached_winner(cache_file, composite_key, variants, dev_info);
        if (std::find(eligible.begin(), eligible.end(), cached_idx) == eligible.end()) cached_idx = -1;
        if (cached_idx >= 0) {
            try {
                sync_fn();
                variants[cached_idx].launch();
                sync_fn();
                if (validate_fn && !validate_fn()) throw std::runtime_error("Loaded selection validation failed");
                std::lock_guard<std::mutex> lock(mu_);
                ++stats_.validations;
                stats_.validated.insert(variants[cached_idx].name);
                winner_cache_[composite_key] = cached_idx;
            } catch (...) {
                cached_idx = -1;
            }
        }
        if (cached_idx >= 0) {
            variants[cached_idx].launch();
            return make_result(cached_idx, variants[cached_idx], 0.0, true);
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            ++stats_.searches;
        }
        double incumbent_sec = std::numeric_limits<double>::infinity();
        auto measure = [&](int index) {
            const auto& variant = variants[index];
            sync_fn();
            variant.launch();
            sync_fn();
            if (validate_fn && !validate_fn()) throw std::runtime_error("Candidate validation failed");
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++stats_.validations;
                stats_.validated.insert(variant.name);
            }
            for (int warmup = 1; warmup < warmup_iters; ++warmup) variant.launch();
            sync_fn();
            auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < bench_iters; ++iteration) variant.launch();
            sync_fn();
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / bench_iters;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++stats_.benchmarks;
                stats_.benchmarked.insert(variant.name);
            }
            if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("Nonfinite GPU timing");
            if (elapsed > selection.slow_factor * incumbent_sec)
                throw std::runtime_error("Slow candidate rejected after completed synchronized batch");
            return elapsed;
        };
        std::vector<std::pair<double, int>> screened;
        std::map<std::string, double> screen_seconds;
        for (int index : eligible) {
            try {
                double elapsed = measure(index);
                if (screened.empty()) incumbent_sec = elapsed;
                screened.push_back({elapsed, index});
                screen_seconds[variants[index].name] = elapsed;
            } catch (...) {}
        }
        if (screened.empty()) throw std::runtime_error("Every eligible SYCL candidate failed");
        int incumbent = screened.front().second;
        std::stable_sort(screened.begin(), screened.end());
        if (static_cast<int>(screened.size()) > selection.finalists) screened.resize(selection.finalists);
        if (std::none_of(screened.begin(), screened.end(), [&](const auto& item) { return item.second == incumbent; }))
            screened.back() = {incumbent_sec, incumbent};
        std::map<int, std::vector<double>> samples;
        std::vector<int> failed;
        for (int batch = 0; batch < selection.batches; ++batch) {
            for (int offset = 0; offset < static_cast<int>(screened.size()); ++offset) {
                int position = batch % 2 == 0 ? offset : static_cast<int>(screened.size()) - offset - 1;
                int index = screened[position].second;
                if (std::find(failed.begin(), failed.end(), index) != failed.end()) continue;
                try { samples[index].push_back(measure(index)); }
                catch (...) { failed.push_back(index); }
            }
        }
        int best_idx = -1;
        double best_sec = std::numeric_limits<double>::infinity();
        std::map<int, double> medians;
        for (auto& sample : samples) {
            if (std::find(failed.begin(), failed.end(), sample.first) != failed.end()) continue;
            auto& durations = sample.second;
            std::sort(durations.begin(), durations.end());
            const auto middle = durations.size() / 2;
            double elapsed = durations.size() % 2 ? durations[middle] : (durations[middle - 1] + durations[middle]) / 2;
            medians[sample.first] = elapsed;
            if (elapsed < best_sec) { best_sec = elapsed; best_idx = sample.first; }
        }
        if (best_idx < 0) throw std::runtime_error("Every SYCL finalist failed");
        auto incumbent_median = medians.find(incumbent);
        if (incumbent_median != medians.end() && incumbent_median->second <= best_sec * (1 + selection.relative_noise)) {
            best_idx = incumbent;
            best_sec = incumbent_median->second;
        }

        save_cached_winner(cache_file, composite_key, dev_info, best_idx, variants[best_idx], best_sec);
        {
            std::lock_guard<std::mutex> lock(mu_);
            winner_cache_[composite_key] = best_idx;
        }
        variants[best_idx].launch();
        TuneResult result = make_result(best_idx, variants[best_idx], best_sec, false);
        result.screen_seconds = screen_seconds;
        for (const auto& item : medians) result.median_seconds[variants[item.first].name] = item.second;
        return result;
    }

#if KF_HAS_SYCL_HEADER
    TuneResult select_and_launch(
        sycl::queue& q,
        const std::string& op_id,
        const std::string& workload_key,
        const std::vector<SyclVariantSpec>& variants,
        const std::function<bool()>& validate_fn = nullptr,
        const std::filesystem::path& cache_dir = default_cache_dir(),
        int warmup_iters = 2,
        int bench_iters = 5,
        const StaticPolicy& policy = nullptr,
        const std::string& policy_version = "",
        const std::string& compiler_flags = "",
        SelectionSettings selection = {})
    {
        SyclDeviceInfo info = query_sycl_device_info(q);
        return select_and_launch(
            op_id,
            workload_key,
            info,
            variants,
            [&q]() { q.wait_and_throw(); },
            validate_fn,
            cache_dir,
            warmup_iters,
            bench_iters,
            policy,
            policy_version,
            compiler_flags,
            selection);
    }
#endif

    void clear_memory_cache() {
        std::lock_guard<std::mutex> lock(mu_);
        winner_cache_.clear();
    }

    SelectionStats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }
    void reset_stats() {
        std::lock_guard<std::mutex> lock(mu_);
        stats_ = SelectionStats{};
    }

private:
    static TuneResult make_result(int index, const SyclVariantSpec& variant, double seconds, bool from_cache) {
        TuneResult result;
        result.winner_index = index;
        result.winner_name = variant.name;
        result.parameters = variant.parameters;
        result.elapsed_seconds = seconds;
        result.from_cache = from_cache;
        return result;
    }

    static int load_cached_winner(
        const std::filesystem::path& path,
        const std::string& composite_key,
        const std::vector<SyclVariantSpec>& variants,
        const SyclDeviceInfo& dev_info)
    {
        std::ifstream in(path);
        if (!in) return -1;
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();

        if (text.find("\"key_hash\": \"" + fnv1a64(composite_key) + "\"") == std::string::npos) {
            return -1;
        }
        if (text.find("\"autotuned\": true") == std::string::npos) {
            return -1;
        }
        std::regex name_re("\"winner_name\"\\s*:\\s*\"([^\"]+)\"");
        std::smatch m;
        if (!std::regex_search(text, m, name_re)) {
            return -1;
        }
        std::string winner_name = m[1].str();
        for (int i = 0; i < static_cast<int>(variants.size()); ++i) {
            if (variants[i].name == winner_name) {
                return i;
            }
        }
        return -1;
    }

    static void save_cached_winner(
        const std::filesystem::path& path,
        const std::string& composite_key,
        const SyclDeviceInfo& dev_info,
        int winner_idx,
        const SyclVariantSpec& winner,
        double elapsed_seconds)
    {
        std::filesystem::create_directories(path.parent_path());
        std::filesystem::path tmp = path;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out << "{\n"
                << "  \"key_hash\": \"" << fnv1a64(composite_key) << "\",\n"
                << "  \"device\": \"" << dev_info.name << "\",\n"
                << "  \"driver_version\": \"" << dev_info.driver_version << "\",\n"
                << "  \"autotuned\": true,\n"
                << "  \"winner_index\": " << winner_idx << ",\n"
                << "  \"winner_name\": \"" << winner.name << "\",\n"
                << "  \"elapsed_seconds\": " << std::setprecision(9) << elapsed_seconds << ",\n"
                << "  \"parameters\": {";
            bool first = true;
            for (const auto& kv : winner.parameters) {
                out << (first ? "\n" : ",\n") << "    \"" << kv.first << "\": " << kv.second;
                first = false;
            }
            out << "\n  }\n}\n";
        }
        std::filesystem::rename(tmp, path);
    }

    mutable std::mutex mu_;
    SelectionStats stats_;
    std::unordered_map<std::string, int> winner_cache_;
};

} // namespace kf_sycl_autotune
