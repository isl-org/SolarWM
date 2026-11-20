#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CL_CHECK(expr) do { \
    const cl_int _error = (expr); \
    if (_error != CL_SUCCESS) throw std::runtime_error("OpenCL error " + std::to_string(_error) + " at line " + std::to_string(__LINE__)); \
} while (0)

namespace {

struct Options {
    int batch = 1;
    int q_len = 256;
    int kv_len = 256;
    int q_heads = 32;
    int kv_heads = 4;
    int head_dim = 128;
    bool causal = true;
    bool bf16 = true;
    int warmup = 5;
    int iters = 20;
    bool check = false;
    bool large_grf = false;
    int debug_scores = 0;
    std::string dump_prefix;
};

Options parse_args(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto get_int = [&]() {
            if (++i >= argc) throw std::runtime_error("missing integer after " + std::string(argv[i - 1]));
            return std::atoi(argv[i]);
        };
        const std::string arg(argv[i]);
        if (arg == "--b") o.batch = get_int();
        else if (arg == "--sq") o.q_len = get_int();
        else if (arg == "--sk") o.kv_len = get_int();
        else if (arg == "--hq") o.q_heads = get_int();
        else if (arg == "--hkv") o.kv_heads = get_int();
        else if (arg == "--d") o.head_dim = get_int();
        else if (arg == "--noncausal") o.causal = false;
        else if (arg == "--fp16") o.bf16 = false;
        else if (arg == "--warmup") o.warmup = get_int();
        else if (arg == "--iters") o.iters = get_int();
        else if (arg == "--check") o.check = true;
        else if (arg == "--large-grf") o.large_grf = true;
        else if (arg == "--debug-scores") o.debug_scores = get_int();
        else if (arg == "--dump") { if (++i >= argc) throw std::runtime_error("missing dump prefix"); o.dump_prefix = argv[i]; }
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if ((o.head_dim != 64 && o.head_dim != 128) || o.batch <= 0 || o.q_len <= 0 ||
        o.kv_len <= 0 || o.q_heads <= 0 || o.kv_heads <= 0 || o.q_heads % o.kv_heads ||
        (o.causal && o.kv_len < o.q_len))
        throw std::runtime_error("invalid or unsupported shape");
    return o;
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

uint16_t f32_to_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

uint16_t f32_to_fp16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const int exp = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    const uint32_t frac = bits & 0x7fffffU;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    if (exp <= 0) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(exp << 10) | (frac >> 13));
}

float bits_to_f32(uint16_t x, bool bf16) {
    if (bf16) {
        uint32_t bits = static_cast<uint32_t>(x) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    const uint32_t sign = (x & 0x8000U) ? 0x80000000U : 0;
    const uint32_t exponent = (x >> 10) & 0x1fU;
    const uint32_t mantissa = x & 0x3ffU;
    uint32_t bits;
    if (exponent == 0) bits = sign;
    else if (exponent == 31) bits = sign | 0x7f800000U | (mantissa << 13);
    else bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void set_kernel_arg(cl_kernel k, cl_mem mem, cl_uint index) {
    CL_CHECK(clSetKernelArg(k, index, sizeof(mem), &mem));
}

template <typename T>
void set_kernel_arg(cl_kernel k, T value, cl_uint index) {
    CL_CHECK(clSetKernelArg(k, index, sizeof(value), &value));
}

double event_ms(cl_event e) {
    cl_ulong start = 0, end = 0;
    CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return static_cast<double>(end - start) * 1e-6;
}

struct Runner {
    const Options &o;
    cl_context context{};
    cl_command_queue queue{};
    cl_program program{};
    cl_device_id device{};
    cl_kernel centre{}, quant_q{}, quant_k{}, attention{}, debug_kernel{}, debug_v_kernel{};
    cl_mem q{}, k{}, v{}, km{}, q8{}, qs{}, k8{}, ks{}, out{};
    cl_mem debug{};
    std::vector<uint16_t> q_host, k_host, v_host;
    size_t q_pad, kv_pad;

    explicit Runner(const Options &opts) : o(opts), q_pad((o.q_len + (o.head_dim == 128 ? 255 : 127)) /
                (o.head_dim == 128 ? 256 : 128) * (o.head_dim == 128 ? 256 : 128)),
                                         kv_pad((o.kv_len + 63) / 64 * 64) {
        cl_uint np = 0;
        CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
        std::vector<cl_platform_id> platforms(np);
        CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));
        cl_int err = CL_DEVICE_NOT_FOUND;
        for (cl_platform_id p : platforms) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || !nd) continue;
            std::vector<cl_device_id> devices(nd);
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devices.data(), nullptr) != CL_SUCCESS) continue;
            device = devices[0];
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err == CL_SUCCESS) break;
        }
        if (!context) throw std::runtime_error("no OpenCL GPU device");
        const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue = clCreateCommandQueueWithProperties(context, device, props, &err);
        CL_CHECK(err);

        const std::string source = read_file("sage_causal.cl");
        const char *src = source.c_str();
        const size_t len = source.size();
        program = clCreateProgramWithSource(context, 1, &src, &len, &err);
        CL_CHECK(err);
        std::ostringstream opts_text;
        opts_text << "-cl-std=CL3.0 -cl-mad-enable -DSAGE_HEAD_DIM=" << o.head_dim
                  << " -DSAGE_INPUT_BF16=" << (o.bf16 ? 1 : 0);
        if (o.large_grf) opts_text << " -cl-intel-256-GRF-per-thread";
        if (const char *extra = std::getenv("SAGE_OCL_OPTIONS")) opts_text << ' ' << extra;
        err = clBuildProgram(program, 1, &device, opts_text.str().c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL build failed: " + log);
        }
        centre = clCreateKernel(program, "sage_k_centre", &err); CL_CHECK(err);
        quant_q = clCreateKernel(program, "sage_quantize_q", &err); CL_CHECK(err);
        quant_k = clCreateKernel(program, "sage_quantize_k", &err); CL_CHECK(err);
        attention = clCreateKernel(program, "sage_attention", &err); CL_CHECK(err);
        if (o.debug_scores > 0) {
            debug_kernel = clCreateKernel(program, "sage_debug_scores", &err);
            if (err != CL_SUCCESS) debug_kernel = nullptr;
            debug_v_kernel = clCreateKernel(program, "sage_debug_v_transform", &err); CL_CHECK(err);
        }
    }

    ~Runner() {
        for (cl_kernel x : {centre, quant_q, quant_k, attention, debug_kernel, debug_v_kernel}) if (x) clReleaseKernel(x);
        for (cl_mem x : {q, k, v, km, q8, qs, k8, ks, out, debug}) if (x) clReleaseMemObject(x);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    cl_mem buffer(size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        cl_int err;
        cl_mem m = clCreateBuffer(context, flags, bytes, nullptr, &err);
        CL_CHECK(err);
        return m;
    }

    void allocate() {
        const size_t qn = static_cast<size_t>(o.batch) * o.q_len * o.q_heads * o.head_dim;
        const size_t kn = static_cast<size_t>(o.batch) * o.kv_len * o.kv_heads * o.head_dim;
        q = buffer(qn * sizeof(uint16_t));
        k = buffer(kn * sizeof(uint16_t));
        v = buffer(kn * sizeof(uint16_t));
        km = buffer(static_cast<size_t>(o.batch) * o.kv_heads * o.head_dim * sizeof(float));
        q8 = buffer(static_cast<size_t>(o.batch) * o.q_heads * q_pad * o.head_dim);
        qs = buffer(static_cast<size_t>(o.batch) * o.q_heads * q_pad * sizeof(float));
        k8 = buffer(static_cast<size_t>(o.batch) * o.kv_heads * (kv_pad / 16) * (o.head_dim / 32) * 8 * 16 * sizeof(uint32_t));
        ks = buffer(static_cast<size_t>(o.batch) * o.kv_heads * kv_pad * sizeof(float));
        out = buffer(qn * sizeof(uint16_t));
        const size_t debug_count = o.debug_scores > 0 ? o.debug_scores : 1;
        debug = buffer(debug_count * 2 * 16 * 16 * sizeof(int));
        std::vector<int> debug_zero(debug_count * 2 * 16 * 16, 0x7fffffff);
        CL_CHECK(clEnqueueWriteBuffer(queue, debug, CL_TRUE, 0,
                                      debug_zero.size() * sizeof(int), debug_zero.data(),
                                      0, nullptr, nullptr));
    }

    std::vector<uint16_t> random_input(size_t n) const {
        std::mt19937 rng(12345);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<uint16_t> x(n);
        for (auto &v : x) v = o.bf16 ? f32_to_bf16(normal(rng) * 0.35f)
                                     : f32_to_fp16(normal(rng) * 0.35f);
        return x;
    }

    void set_inputs() {
        const size_t qn = static_cast<size_t>(o.batch) * o.q_len * o.q_heads * o.head_dim;
        const size_t kn = static_cast<size_t>(o.batch) * o.kv_len * o.kv_heads * o.head_dim;
        q_host = random_input(qn); k_host = random_input(kn); v_host = random_input(kn);
        CL_CHECK(clEnqueueWriteBuffer(queue, q, CL_FALSE, 0, qn * 2, q_host.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(queue, k, CL_FALSE, 0, kn * 2, k_host.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(queue, v, CL_FALSE, 0, kn * 2, v_host.data(), 0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));
        if (!o.check) return;
        check_reference(q_host, k_host, v_host);
    }

    void bind() {
        set_kernel_arg(centre, k, 0); set_kernel_arg(centre, km, 1); set_kernel_arg(centre, o.batch, 2);
        set_kernel_arg(centre, o.kv_len, 3); set_kernel_arg(centre, o.kv_heads, 4);

        set_kernel_arg(quant_q, q, 0); set_kernel_arg(quant_q, q8, 1); set_kernel_arg(quant_q, qs, 2);
        set_kernel_arg(quant_q, o.batch, 3); set_kernel_arg(quant_q, o.q_len, 4);
        set_kernel_arg(quant_q, o.q_heads, 5); set_kernel_arg(quant_q, static_cast<int>(q_pad), 6);

        set_kernel_arg(quant_k, k, 0); set_kernel_arg(quant_k, km, 1); set_kernel_arg(quant_k, k8, 2);
        set_kernel_arg(quant_k, ks, 3); set_kernel_arg(quant_k, o.batch, 4); set_kernel_arg(quant_k, o.kv_len, 5);
        set_kernel_arg(quant_k, o.kv_heads, 6); set_kernel_arg(quant_k, static_cast<int>(kv_pad), 7);

        set_kernel_arg(attention, q8, 0); set_kernel_arg(attention, qs, 1); set_kernel_arg(attention, k8, 2);
        set_kernel_arg(attention, ks, 3); set_kernel_arg(attention, v, 4); set_kernel_arg(attention, out, 5);
        set_kernel_arg(attention, o.batch, 6); set_kernel_arg(attention, o.q_len, 7);
        set_kernel_arg(attention, o.kv_len, 8); set_kernel_arg(attention, o.q_heads, 9);
        set_kernel_arg(attention, o.kv_heads, 10); set_kernel_arg(attention, static_cast<int>(q_pad), 11);
        set_kernel_arg(attention, static_cast<int>(kv_pad), 12); set_kernel_arg(attention, o.causal ? 1 : 0, 13);
    }

    std::vector<double> run_once() {
        cl_event events[4]{};
        size_t cglobal = static_cast<size_t>(o.batch) * o.kv_heads * o.head_dim;
        size_t clocal = static_cast<size_t>(o.head_dim);
        CL_CHECK(clEnqueueNDRangeKernel(queue, centre, 1, nullptr, &cglobal, &clocal, 0, nullptr, &events[0]));
        const size_t qtile = o.head_dim == 128 ? 256 : 128;
        const size_t qglobal[2] = {static_cast<size_t>(o.batch) * o.q_heads * qtile, q_pad / qtile};
        const size_t qlocal[2] = {qtile, 1};
        CL_CHECK(clEnqueueNDRangeKernel(queue, quant_q, 2, nullptr, qglobal, qlocal, 0, nullptr, &events[1]));
        const size_t kblocks = ((kv_pad / 16) + 7) / 8;
        const size_t kglobal = static_cast<size_t>(o.batch) * o.kv_heads * kblocks * 128;
        const size_t klocal = 128;
        CL_CHECK(clEnqueueNDRangeKernel(queue, quant_k, 1, nullptr, &kglobal, &klocal, 0, nullptr, &events[2]));
        const size_t subgroups = qtile / 16;
        const size_t aglobal[2] = {((static_cast<size_t>(o.q_len) + qtile - 1) / qtile) * 16 * subgroups,
                                  static_cast<size_t>(o.batch) * o.q_heads};
        const size_t alocal[2] = {16 * subgroups, 1};
        CL_CHECK(clEnqueueNDRangeKernel(queue, attention, 2, nullptr, aglobal, alocal, 0, nullptr, &events[3]));
        if (o.debug_scores > 0) {
            if (debug_kernel) {
                set_kernel_arg(debug_kernel, q8, 0); set_kernel_arg(debug_kernel, k8, 1);
                set_kernel_arg(debug_kernel, debug, 2); set_kernel_arg(debug_kernel, static_cast<int>(q_pad), 3);
                set_kernel_arg(debug_kernel, static_cast<int>(kv_pad), 4);
                set_kernel_arg(debug_kernel, 0, 5); set_kernel_arg(debug_kernel, 0, 6);
                const size_t dglobal = 16, dlocal = 16;
                CL_CHECK(clEnqueueNDRangeKernel(queue, debug_kernel, 1, nullptr, &dglobal, &dlocal, 0, nullptr, nullptr));
                for (int nb = 1; nb < 2; ++nb) {
                    set_kernel_arg(debug_kernel, nb, 6);
                    CL_CHECK(clEnqueueNDRangeKernel(queue, debug_kernel, 1, nullptr, &dglobal, &dlocal, 0, nullptr, nullptr));
                }
            }
            set_kernel_arg(debug_v_kernel, v, 0); set_kernel_arg(debug_v_kernel, debug, 1);
            set_kernel_arg(debug_v_kernel, static_cast<int>(kv_pad), 2); set_kernel_arg(debug_v_kernel, o.kv_heads, 3);
            set_kernel_arg(debug_v_kernel, 0, 4); set_kernel_arg(debug_v_kernel, 0, 5); set_kernel_arg(debug_v_kernel, 0, 6);
            const size_t vglobal = 16, vlocal = 16;
            CL_CHECK(clEnqueueNDRangeKernel(queue, debug_v_kernel, 1, nullptr, &vglobal, &vlocal, 0, nullptr, nullptr));
        }
        CL_CHECK(clFinish(queue));
        std::vector<double> times(4);
        for (int i = 0; i < 4; ++i) { times[i] = event_ms(events[i]); clReleaseEvent(events[i]); }
        return times;
    }

    void check_reference(const std::vector<uint16_t> &qh, const std::vector<uint16_t> &kh,
                        const std::vector<uint16_t> &vh) {
        bind();
        run_once();
        const size_t qn = static_cast<size_t>(o.batch) * o.q_len * o.q_heads * o.head_dim;
        std::vector<uint16_t> actual(qn);
        CL_CHECK(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, actual.size() * 2, actual.data(), 0, nullptr, nullptr));
        std::vector<float> actual_qs(o.batch * o.q_heads * q_pad);
        std::vector<char> actual_q8(actual_qs.size() * o.head_dim);
        CL_CHECK(clEnqueueReadBuffer(queue, qs, CL_TRUE, 0, actual_qs.size() * sizeof(float), actual_qs.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(queue, q8, CL_TRUE, 0, actual_q8.size(), actual_q8.data(), 0, nullptr, nullptr));
        int q8_diff = 0;
        double qs_diff = 0.0;
        int row_diffs[8] = {};
        for (size_t bh = 0, t = 0; bh < static_cast<size_t>(o.batch) * o.q_heads; ++bh)
            for (size_t row = 0; row < q_pad; ++row, ++t) {
                const size_t qi = row < static_cast<size_t>(o.q_len)
                                ? (bh / o.q_heads * o.q_len + row) * o.q_heads * o.head_dim
+                                  + (bh % o.q_heads) * o.head_dim
                                : 0;
                float amax = 0.0f;
                if (row < static_cast<size_t>(o.q_len)) for (int d = 0; d < o.head_dim; ++d)
                    amax = std::max(amax, std::abs(bits_to_f32(qh[qi + d], o.bf16)));
                const float expected_scale = amax / 127.0f * (1.4426950408889634f / std::sqrt(float(o.head_dim)));
                qs_diff = std::max(qs_diff, double(std::abs(actual_qs[t] - expected_scale)));
                for (int d = 0; d < o.head_dim; ++d) {
                    const float value = row < static_cast<size_t>(o.q_len) ? bits_to_f32(qh[qi + d], o.bf16) : 0.0f;
                    const int expected = amax > 0.0f ? std::max(-128, std::min(127, int(std::nearbyint(value * 127.0f / amax)))) : 0;
                    const int g = d / 32;
                    const int lane = (d % 32) / 2;
                    const size_t word = (bh * (q_pad / 8) + row / 8) * (o.head_dim / 32) * 128
                                      + g * 128 + (row % 8) * 16 + lane;
                    const int packed = static_cast<unsigned char>(actual_q8[word * 2 + (d % 2)]);
                    const int got = packed >= 128 ? packed - 256 : packed;
                    q8_diff += got != expected;
                }
            }
        std::printf("q-quant: scale_max_diff=%.6g int8_diffs=%d/%zu\n", qs_diff, q8_diff, actual_q8.size());
        std::printf("q-quant rows0-7:"); for (int r = 0; r < 8; ++r) std::printf(" %d", row_diffs[r]); std::printf("\n");
        double max_abs = 0.0, sq = 0.0, first_ref = 0.0;
        int one_token_bad = 0;
        for (int b = 0; b < o.batch; ++b) for (int i = 0; i < o.q_len; ++i) for (int hq = 0; hq < o.q_heads; ++hq) {
            const int hkv = hq / (o.q_heads / o.kv_heads);
            const int visible = o.causal ? std::min(o.kv_len, i + o.kv_len - o.q_len + 1) : o.kv_len;
            std::vector<double> scores(visible), sums(o.head_dim, 0.0);
            double maximum = -std::numeric_limits<double>::infinity(), total = 0.0;
            for (int j = 0; j < visible; ++j) {
                double dot = 0.0;
                for (int d = 0; d < o.head_dim; ++d) {
                    const size_t qi = ((static_cast<size_t>(b) * o.q_len + i) * o.q_heads + hq) * o.head_dim + d;
                    const size_t ki = ((static_cast<size_t>(b) * o.kv_len + j) * o.kv_heads + hkv) * o.head_dim + d;
                    dot += bits_to_f32(qh[qi], o.bf16) * bits_to_f32(kh[ki], o.bf16);
                }
                scores[j] = dot / std::sqrt(static_cast<double>(o.head_dim));
                maximum = std::max(maximum, scores[j]);
            }
            for (int j = 0; j < visible; ++j) { scores[j] = std::exp(scores[j] - maximum); total += scores[j]; }
            for (int j = 0; j < visible; ++j) for (int d = 0; d < o.head_dim; ++d) {
                const size_t vi = ((static_cast<size_t>(b) * o.kv_len + j) * o.kv_heads + hkv) * o.head_dim + d;
                sums[d] += scores[j] / total * bits_to_f32(vh[vi], o.bf16);
            }
            for (int d = 0; d < o.head_dim; ++d) {
                const size_t oi = ((static_cast<size_t>(b) * o.q_len + i) * o.q_heads + hq) * o.head_dim + d;
                const double got = bits_to_f32(actual[oi], o.bf16);
                max_abs = std::max(max_abs, std::abs(got - sums[d]));
                sq += (got - sums[d]) * (got - sums[d]);
                if (b == 0 && i == 0 && hq == 0 && d == 0) first_ref = sums[d];
                if (o.q_len == 1 && o.kv_len == 1 && std::abs(got - sums[d]) > 1e-3)
                    ++one_token_bad;
            }
        }
        const double rms = std::sqrt(sq / static_cast<double>(qn));
        std::printf("reference: max_abs=%.6g rms=%.6g\n", max_abs, rms);
        if (o.q_len == 1 && o.kv_len == 1)
            std::printf("one-token out0=%.8g v0=%.8g sumref=%.8g bad=%d/%d\n",
                        bits_to_f32(actual[0], o.bf16), bits_to_f32(vh[0], o.bf16), first_ref,
                        one_token_bad, o.head_dim);
    }

    void dump() const {
        if (o.dump_prefix.empty()) return;
        auto write = [&](const std::string &suffix, const void *data, size_t bytes) {
            std::ofstream f(o.dump_prefix + suffix, std::ios::binary);
            if (!f) throw std::runtime_error("cannot write " + o.dump_prefix + suffix);
            f.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        };
        write(".q.bits", q_host.data(), q_host.size() * sizeof(q_host[0]));
        write(".k.bits", k_host.data(), k_host.size() * sizeof(k_host[0]));
        write(".v.bits", v_host.data(), v_host.size() * sizeof(v_host[0]));
        auto dump_buffer = [&](cl_mem mem, const std::string &suffix, size_t bytes) {
            std::vector<char> data(bytes);
            CL_CHECK(clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes, data.data(), 0, nullptr, nullptr));
            write(suffix, data.data(), data.size());
        };
        dump_buffer(km, ".kmean.f32", static_cast<size_t>(o.batch) * o.kv_heads * o.head_dim * sizeof(float));
        dump_buffer(qs, ".qscale.f32", static_cast<size_t>(o.batch) * o.q_heads * q_pad * sizeof(float));
        dump_buffer(q8, ".q8.packed", static_cast<size_t>(o.batch) * o.q_heads * q_pad * o.head_dim);
        dump_buffer(ks, ".kscale.f32", static_cast<size_t>(o.batch) * o.kv_heads * kv_pad * sizeof(float));
        dump_buffer(k8, ".k8.packed", static_cast<size_t>(o.batch) * o.kv_heads * (kv_pad / 16)
                                     * (o.head_dim / 32) * 128 * sizeof(uint32_t));
        if (o.debug_scores > 0)
             dump_buffer(debug, ".scores.i32", static_cast<size_t>(o.debug_scores) * 2 * 16 * 16 * sizeof(int));
        if (o.debug_scores > 0)
            dump_buffer(debug, ".vprobe.u32", 16 * 8 * sizeof(int));
        dump_buffer(debug, ".debug.i32", 16 * sizeof(int));
        std::vector<uint16_t> output(static_cast<size_t>(o.batch) * o.q_len * o.q_heads * o.head_dim);
        CL_CHECK(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, output.size() * sizeof(output[0]),
                                     output.data(), 0, nullptr, nullptr));
        write(".opencl.bits", output.data(), output.size() * sizeof(output[0]));
    }
};
}  // namespace

int main(int argc, char **argv) {
    try {
        const Options o = parse_args(argc, argv);
        Runner r(o);
        r.allocate();
        r.set_inputs();
        r.bind();
        for (int i = 0; i < o.warmup; ++i) r.run_once();
        std::vector<std::vector<double>> samples(o.iters);
        for (auto &s : samples) s = r.run_once();
        const char *names[] = {"k_centre", "quant_q", "quant_k", "attention"};
        std::vector<double> totals(o.iters, 0.0);
        std::printf("device OpenCL | %s D=%d B=%d Sq=%d Skv=%d Hq=%d Hkv=%d causal=%d grf=%s\n",
                    o.bf16 ? "BF16" : "FP16", o.head_dim, o.batch, o.q_len, o.kv_len,
                    o.q_heads, o.kv_heads, o.causal, o.large_grf ? "256" : "default");
        for (int stage = 0; stage < 4; ++stage) {
            std::vector<double> values;
            for (int i = 0; i < o.iters; ++i) { values.push_back(samples[i][stage]); totals[i] += samples[i][stage]; }
            std::sort(values.begin(), values.end());
            std::printf("  %-10s median %.3f ms\n", names[stage], values[values.size() / 2]);
        }
        std::sort(totals.begin(), totals.end());
        std::printf("  %-10s median %.3f ms\n", "total", totals[totals.size() / 2]);
        r.dump();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
