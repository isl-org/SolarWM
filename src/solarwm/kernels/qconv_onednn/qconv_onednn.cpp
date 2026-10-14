#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>

namespace solarwm {
namespace qconv {

struct ConvKey {
    void* engine_handle;
    int64_t n, c_in, c_out, d, h, w;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd_l, ph_l, pw_l;
    int64_t pd_r, ph_r, pw_r;
    int64_t dd, dh, dw;
    int64_t groups;
    bool has_bias;
    int64_t src_stride_1;
    int64_t weights_stride_1;
    int64_t dst_stride_1;
    int32_t out_dtype_enum;
    std::string post_op;

    bool operator==(const ConvKey& o) const {
        return engine_handle == o.engine_handle &&
               n == o.n && c_in == o.c_in && c_out == o.c_out &&
               d == o.d && h == o.h && w == o.w &&
               kd == o.kd && kh == o.kh && kw == o.kw &&
               sd == o.sd && sh == o.sh && sw == o.sw &&
               pd_l == o.pd_l && ph_l == o.ph_l && pw_l == o.pw_l &&
               pd_r == o.pd_r && ph_r == o.ph_r && pw_r == o.pw_r &&
               dd == o.dd && dh == o.dh && dw == o.dw &&
               groups == o.groups && has_bias == o.has_bias &&
               src_stride_1 == o.src_stride_1 &&
               weights_stride_1 == o.weights_stride_1 &&
               dst_stride_1 == o.dst_stride_1 &&
               out_dtype_enum == o.out_dtype_enum &&
               post_op == o.post_op;
    }
};

struct ConvKeyHash {
    std::size_t operator()(const ConvKey& k) const {
        std::size_t h = std::hash<void*>()(k.engine_handle);
        auto hash_combine = [&h](int64_t val) {
            h ^= std::hash<int64_t>()(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(k.n); hash_combine(k.c_in); hash_combine(k.c_out);
        hash_combine(k.d); hash_combine(k.h); hash_combine(k.w);
        hash_combine(k.kd); hash_combine(k.kh); hash_combine(k.kw);
        hash_combine(k.sd); hash_combine(k.sh); hash_combine(k.sw);
        hash_combine(k.pd_l); hash_combine(k.ph_l); hash_combine(k.pw_l);
        hash_combine(k.pd_r); hash_combine(k.ph_r); hash_combine(k.pw_r);
        hash_combine(k.dd); hash_combine(k.dh); hash_combine(k.dw);
        hash_combine(k.groups); hash_combine(k.has_bias ? 1 : 0);
        hash_combine(k.src_stride_1); hash_combine(k.weights_stride_1);
        hash_combine(k.dst_stride_1);
        hash_combine(k.out_dtype_enum);
        h ^= std::hash<std::string>()(k.post_op) + 0x9e3779b9;
        return h;
    }
};

struct CachedConv {
    dnnl::convolution_forward primitive;
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc scratchpad_md;
    size_t scratchpad_size;
    int64_t out_d, out_h, out_w;
};

static std::unordered_map<ConvKey, CachedConv, ConvKeyHash> g_conv_cache;
static std::mutex g_conv_cache_mutex;

/**
 * Fast QuaRot INT8 3D convolution with native oneDNN primitives on Intel XPU.
 *
 * Avoids any call to aten::item() on scale / zero-point tensors:
 * - x_scale and w_scale device tensors are passed directly to oneDNN via USM pointers
 * - Avoids host CPU blocking synchronization (zeEventHostSynchronize)
 * - Thread-safe primitive caching reuses compiled convolution descriptors across steps
 */
at::Tensor fast_qconv3d(
    const at::Tensor& qx,
    const at::Tensor& x_scale,
    const at::Tensor& qw,
    const at::Tensor& w_scale,
    const std::optional<at::Tensor>& bias_opt,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    std::optional<c10::ScalarType> output_dtype_opt,
    const std::string& post_op,
    const std::optional<at::Tensor>& accumulate_into_opt = std::nullopt
) {
    c10::ScalarType output_dtype = output_dtype_opt.value_or(at::kBFloat16);
    TORCH_CHECK(qx.is_xpu(), "fast_qconv3d: qx must be an XPU tensor");
    TORCH_CHECK(qw.is_xpu(), "fast_qconv3d: qw must be an XPU tensor");
    TORCH_CHECK(x_scale.is_xpu(), "fast_qconv3d: x_scale must be an XPU tensor");
    TORCH_CHECK(w_scale.is_xpu(), "fast_qconv3d: w_scale must be an XPU tensor");
    TORCH_CHECK(qx.dim() == 5, "fast_qconv3d: expected 5D input tensor [N, C, D, H, W]");
    TORCH_CHECK(qw.dim() == 5, "fast_qconv3d: expected 5D weight tensor [OC, IC, kD, kH, kW]");

    bool do_accumulate = accumulate_into_opt.has_value() && accumulate_into_opt->defined();
    at::Tensor accumulate_into;
    if (do_accumulate) {
        accumulate_into = accumulate_into_opt.value();
        TORCH_CHECK(accumulate_into.is_xpu(), "fast_qconv3d: accumulate_into must be an XPU tensor");
        TORCH_CHECK(accumulate_into.scalar_type() == output_dtype, "fast_qconv3d: accumulate_into dtype must match output_dtype");
        TORCH_CHECK(
            accumulate_into.is_contiguous(at::MemoryFormat::ChannelsLast3d) || accumulate_into.is_contiguous(),
            "fast_qconv3d: accumulate_into must be contiguous or channels_last_3d"
        );
    }

    at::Tensor x_scale_f32 = (x_scale.scalar_type() == at::kFloat) ? x_scale : x_scale.to(at::kFloat);
    at::Tensor w_scale_f32 = (w_scale.scalar_type() == at::kFloat) ? w_scale : w_scale.to(at::kFloat);

    auto q = c10::xpu::getCurrentXPUStream().queue();
    auto engine = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, q);

    bool has_bias = bias_opt.has_value() && bias_opt->defined();
    at::Tensor bias;
    if (has_bias) {
        bias = bias_opt.value().contiguous();
    }

    int64_t pd_l = padding[0], ph_l = padding[1], pw_l = padding[2];
    int64_t pd_r = padding[0], ph_r = padding[1], pw_r = padding[2];
    if (padding.size() == 6) {
        pd_r = padding[3];
        ph_r = padding[4];
        pw_r = padding[5];
    }

    std::string eff_post_op = post_op;
    if (do_accumulate) {
        if (eff_post_op.empty() || eff_post_op == "none") {
            eff_post_op = "sum";
        } else if (eff_post_op.find("sum") == std::string::npos) {
            eff_post_op = eff_post_op + "+sum";
        }
    }

    int64_t dst_stride_1 = do_accumulate ? accumulate_into.stride(1)
        : (qx.is_contiguous(at::MemoryFormat::ChannelsLast3d) ? 1 : 0);

    ConvKey key{
        engine.get(),
        qx.size(0), qx.size(1), qw.size(0), qx.size(2), qx.size(3), qx.size(4),
        qw.size(2), qw.size(3), qw.size(4),
        stride[0], stride[1], stride[2],
        pd_l, ph_l, pw_l,
        pd_r, ph_r, pw_r,
        dilation[0], dilation[1], dilation[2],
        groups, has_bias,
        qx.stride(1), qw.stride(1),
        dst_stride_1,
        static_cast<int32_t>(output_dtype),
        eff_post_op
    };

    CachedConv cached;
    {
        std::lock_guard<std::mutex> lock(g_conv_cache_mutex);
        auto it = g_conv_cache.find(key);
        if (it != g_conv_cache.end()) {
            cached = it->second;
        } else {
            // Source memory desc
            dnnl::memory::dims src_dims = {qx.size(0), qx.size(1), qx.size(2), qx.size(3), qx.size(4)};
            dnnl::memory::dims src_strides = {qx.stride(0), qx.stride(1), qx.stride(2), qx.stride(3), qx.stride(4)};
            cached.src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::s8, src_strides);

            // Weights memory desc
            dnnl::memory::dims weights_dims;
            dnnl::memory::dims weights_strides;
            if (groups > 1) {
                int64_t oc_per_g = qw.size(0) / groups;
                weights_dims = {groups, oc_per_g, qw.size(1), qw.size(2), qw.size(3), qw.size(4)};
                weights_strides = {qw.stride(0) * oc_per_g, qw.stride(0), qw.stride(1), qw.stride(2), qw.stride(3), qw.stride(4)};
            } else {
                weights_dims = {qw.size(0), qw.size(1), qw.size(2), qw.size(3), qw.size(4)};
                weights_strides = {qw.stride(0), qw.stride(1), qw.stride(2), qw.stride(3), qw.stride(4)};
            }
            cached.weights_md = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::s8, weights_strides);

            // Compute spatial output dimensions
            cached.out_d = (qx.size(2) + pd_l + pd_r - (dilation[0] * (qw.size(2) - 1) + 1)) / stride[0] + 1;
            cached.out_h = (qx.size(3) + ph_l + ph_r - (dilation[1] * (qw.size(3) - 1) + 1)) / stride[1] + 1;
            cached.out_w = (qx.size(4) + pw_l + pw_r - (dilation[2] * (qw.size(4) - 1) + 1)) / stride[2] + 1;

            // Destination memory desc
            dnnl::memory::data_type dst_dt = dnnl::memory::data_type::bf16;
            if (output_dtype == at::kFloat) {
                dst_dt = dnnl::memory::data_type::f32;
            } else if (output_dtype == at::kChar || output_dtype == at::kByte) {
                dst_dt = dnnl::memory::data_type::s8;
            }

            int64_t d_sz = cached.out_d, h_sz = cached.out_h, w_sz = cached.out_w, c_sz = qw.size(0);
            dnnl::memory::dims dst_dims = {qx.size(0), c_sz, d_sz, h_sz, w_sz};
            dnnl::memory::dims dst_strides;
            if (do_accumulate) {
                dst_strides = {accumulate_into.stride(0), accumulate_into.stride(1),
                               accumulate_into.stride(2), accumulate_into.stride(3),
                               accumulate_into.stride(4)};
            } else if (qx.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
                // PyTorch channels_last_3d: (N, C, D, H, W) -> strides: (C*D*H*W, 1, H*W*C, W*C, C)
                dst_strides = {d_sz * h_sz * w_sz * c_sz, 1, h_sz * w_sz * c_sz, w_sz * c_sz, c_sz};
            } else {
                // Contiguous (channels-first): (N, C, D, H, W) -> strides: (C*D*H*W, D*H*W, H*W, W, 1)
                dst_strides = {c_sz * d_sz * h_sz * w_sz, d_sz * h_sz * w_sz, h_sz * w_sz, w_sz, 1};
            }
            cached.dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);

            dnnl::memory::dims strides = {stride[0], stride[1], stride[2]};
            dnnl::memory::dims dilates = {dilation[0] - 1, dilation[1] - 1, dilation[2] - 1};
            dnnl::memory::dims padding_l = {pd_l, ph_l, pw_l};
            dnnl::memory::dims padding_r = {pd_r, ph_r, pw_r};

            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            attr.set_scales_mask(DNNL_ARG_SRC, 0); // scalar scale
            int w_scale_mask = 0;
            if (w_scale.numel() > 1) {
                w_scale_mask = (groups > 1) ? ((1 << 0) | (1 << 1)) : (1 << 0);
            }
            attr.set_scales_mask(DNNL_ARG_WEIGHTS, w_scale_mask);

            dnnl::post_ops po;
            if (eff_post_op.find("sum") != std::string::npos) {
                po.append_sum(1.0f);
            }
            if (eff_post_op.find("relu") != std::string::npos) {
                po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
            }
            if (po.len() > 0) {
                attr.set_post_ops(po);
            }

            if (has_bias) {
                dnnl::memory::dims bias_dims = {bias.size(0)};
                dnnl::memory::dims bias_strides = {bias.stride(0)};
                auto bias_dt = (bias.scalar_type() == at::kFloat) ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
                cached.bias_md = dnnl::memory::desc(bias_dims, bias_dt, bias_strides);
            }

            auto conv_pd = has_bias
                ? dnnl::convolution_forward::primitive_desc(
                      engine, dnnl::prop_kind::forward_inference,
                      dnnl::algorithm::convolution_auto,
                      cached.src_md, cached.weights_md, cached.bias_md, cached.dst_md,
                      strides, dilates, padding_l, padding_r, attr)
                : dnnl::convolution_forward::primitive_desc(
                      engine, dnnl::prop_kind::forward_inference,
                      dnnl::algorithm::convolution_auto,
                      cached.src_md, cached.weights_md, cached.dst_md,
                      strides, dilates, padding_l, padding_r, attr);

            cached.primitive = dnnl::convolution_forward(conv_pd);
            cached.scratchpad_md = conv_pd.scratchpad_desc();
            cached.scratchpad_size = cached.scratchpad_md.get_size();

            g_conv_cache[key] = cached;
        }
    }

    at::Tensor out;
    if (do_accumulate) {
        out = accumulate_into;
    } else {
        auto mem_fmt = qx.is_contiguous(at::MemoryFormat::ChannelsLast3d)
            ? at::MemoryFormat::ChannelsLast3d
            : at::MemoryFormat::Contiguous;

        out = at::empty(
            {qx.size(0), qw.size(0), cached.out_d, cached.out_h, cached.out_w},
            qx.options().dtype(output_dtype).memory_format(mem_fmt)
        );
    }

    at::Tensor scratchpad;
    if (cached.scratchpad_size > 0) {
        scratchpad = at::empty({static_cast<int64_t>(cached.scratchpad_size)}, qx.options().dtype(at::kByte));
    }

    dnnl::memory src_mem(cached.src_md, engine, qx.data_ptr());
    dnnl::memory weights_mem(cached.weights_md, engine, qw.data_ptr());
    dnnl::memory dst_mem(cached.dst_md, engine, out.data_ptr());

    // Scale memories wrapping device USM pointers directly without host sync:
    dnnl::memory::desc x_scale_md({1}, dnnl::memory::data_type::f32, {1});
    dnnl::memory x_scale_mem(x_scale_md, engine, x_scale_f32.data_ptr());

    dnnl::memory::desc w_scale_md({w_scale_f32.numel()}, dnnl::memory::data_type::f32, {1});
    dnnl::memory w_scale_mem(w_scale_md, engine, w_scale_f32.data_ptr());

    std::unordered_map<int, dnnl::memory> exec_args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, x_scale_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, w_scale_mem},
    };

    if (has_bias) {
        dnnl::memory bias_mem(cached.bias_md, engine, bias.data_ptr());
        exec_args.insert({DNNL_ARG_BIAS, bias_mem});
    }

    if (cached.scratchpad_size > 0) {
        dnnl::memory scratchpad_mem(cached.scratchpad_md, engine, scratchpad.data_ptr());
        exec_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
    }

    cached.primitive.execute(stream, exec_args);
    return out;
}

// Drop-in wrapper matching onednn.qconv_pointwise.tensor argument conventions
at::Tensor qconv_pointwise_tensor_fast(
    const at::Tensor& qx,
    const at::Tensor& x_scale,
    const at::Tensor& /*x_zero_point*/,
    const at::Tensor& qw,
    const at::Tensor& w_scale,
    const std::optional<at::Tensor>& /*w_zero_point*/,
    const std::optional<at::Tensor>& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    double /*output_scale*/,
    int64_t /*output_zero_point*/,
    std::optional<c10::ScalarType> output_dtype,
    const std::string& attr,
    const std::vector<std::optional<c10::Scalar>>& /*scalars*/,
    const std::optional<std::string>& /*algorithm*/,
    const std::optional<at::Tensor>& accumulate_into = std::nullopt
) {
    c10::ScalarType dt = output_dtype.value_or(at::kBFloat16);
    return fast_qconv3d(qx, x_scale, qw, w_scale, bias, stride, padding, dilation, groups, dt, attr, accumulate_into);
}

void clear_cache() {
    std::lock_guard<std::mutex> lock(g_conv_cache_mutex);
    g_conv_cache.clear();
}

} // namespace qconv
} // namespace solarwm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fast QuaRot INT8 oneDNN Conv3D on Intel XPU without aten::item() host synchronizations";
    m.def("fast_qconv3d", &solarwm::qconv::fast_qconv3d,
          py::arg("qx"), py::arg("x_scale"), py::arg("qw"), py::arg("w_scale"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1, 1},
          py::arg("groups") = 1,
          py::arg("output_dtype") = py::none(),
          py::arg("post_op") = "none",
          py::arg("accumulate_into") = py::none(),
          "Executes 3D quantized convolution without item() host synchronizations.");
    m.def("qconv_pointwise_tensor_fast", &solarwm::qconv::qconv_pointwise_tensor_fast,
          py::arg("qx"), py::arg("x_scale"), py::arg("x_zero_point"),
          py::arg("qw"), py::arg("w_scale"), py::arg("w_zero_point") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1, 1},
          py::arg("groups") = 1,
          py::arg("output_scale") = 1.0,
          py::arg("output_zero_point") = 0,
          py::arg("output_dtype") = py::none(),
          py::arg("attr") = "none",
          py::arg("scalars") = std::vector<std::optional<c10::Scalar>>{},
          py::arg("algorithm") = py::none(),
          py::arg("accumulate_into") = py::none(),
          "Drop-in fast replacement for torch.ops.onednn.qconv_pointwise.tensor");
    m.def("clear_cache", &solarwm::qconv::clear_cache, "Clear primitive descriptor cache");
}
