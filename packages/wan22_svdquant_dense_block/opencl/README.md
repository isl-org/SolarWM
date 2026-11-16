# OpenCL port of the SVDQuant projection (PyOpenCL)

Results, final variants and timings against SYCL: [../opencl_porting.md](../opencl_porting.md).

| File | Role |
|---|---|
| `svdquant_w4a4_v_layer.cl` | Three kernels (quantize, LoRA-A, INT4 GEMM + LoRA-B + bias) and the `AUTOTUNE_METADATA` pool block |
| `ocl_svdquant.py` | Static dispatch (`StaticDispatch`), autotuned op (`SvdquantLinear`), device buffers (`Weights`, `Layer`) |
| `static_priority_ocl.json` | Measured ranking per hardware class and shape class |
| `sweep_ocl.py`, `sweep_ocl_results.json` | Measures every legal candidate with the runtime tuner and writes the ranking |
| `bench_end_to_end.py` | Wall-clock timing of any implementation (SYCL original/current, OpenCL fixed/tuned) on the same operation |
| `prepare_v_inputs.py` | Saves inputs and SYCL reference outputs (the only script that needs torch) |
| `autotune.py`, `runtime_selection.py`, `dispatch_metadata.py` | Copies of the skills' shared code |

```
python opencl/prepare_v_inputs.py /tmp/ocl_probe/v_inputs2.npz
python opencl/sweep_ocl.py /tmp/ocl_probe/v_inputs2.npz 2
python opencl/bench_end_to_end.py ocl-tuned          # also: ocl-fixed, sycl-current, sycl-static, sycl-original <build dir>
```
