# Bug report draft: `torch.softmax` incorrect for specific XPU reduction sizes

> Draft only. This has not been submitted to `intel/torch-xpu-ops`.
>
> Prepared against the bug-report form in `intel/torch-xpu-ops`:
> `.github/ISSUE_TEMPLATE/bug-report.yml`.

## Proposed title

`torch.softmax` on Linux XPU is incorrect for specific reduction sizes in torch 2.13+

## 🐛 Describe the bug

On Intel XPU under Linux, `torch.softmax` produces numerically incorrect results
for specific last-dimension sizes with recent PyTorch XPU wheels.

The issue has the following behavior:

- `float32` returns finite but incorrect values for particular sizes.
- `bfloat16` returns non-finite values for some sizes.
- The result is deterministic across repeated runs.
- Nearby tensor sizes are correct.
- The same test passes on Linux with `torch 2.12.1+xpu`, `2.11.0+xpu`,
  and `2.10.0+xpu`.
- The issue was **not reproduced on Windows with torch-xpu 2.13**.

The Linux issue was observed on an Intel discrete GPU with approximately 32 GB
of device memory. The affected sizes overlap with production attention shapes
in a Wan 2.2 VAE and cause visibly corrupted decoded output.

### Minimal reproducer

This example has no random inputs or external data. The CPU float64 result is
used as the reference.

```python
import torch

device = torch.device("xpu")


def check(seq, dtype):
    # Fixed input; no RNG or external data.
    x = (
        torch.sin(torch.arange(seq * seq, dtype=torch.float32) * 7e-4) * 4.0
    ).view(1, 1, seq, seq)

    # Higher-precision CPU reference.
    reference = torch.softmax(x.double(), dim=-1).float()

    result = torch.softmax(x.to(device=device, dtype=dtype), dim=-1)
    result = result.float().cpu()

    finite = bool(torch.isfinite(result).all())
    if finite:
        diff = (reference - result).abs()
        print(
            f"seq={seq:4d} dtype={dtype} "
            f"finite={finite} "
            f"max_abs_diff={diff.max().item():.6g} "
            f"mean_abs_diff={diff.mean().item():.6g}"
        )
    else:
        print(f"seq={seq:4d} dtype={dtype} finite=False")


for seq in (900, 1215, 1620):
    for dtype in (torch.float32, torch.bfloat16):
        check(seq, dtype)
```

Run it with:

```bash
python repro.py
```

### Observed output on affected Linux XPU builds

Representative output from `torch 2.14.0+xpu`:

```text
seq= 900 dtype=torch.float32 finite=True max_abs_diff=6.98e-10 mean_abs_diff=5.53e-11
seq= 900 dtype=torch.bfloat16 finite=True max_abs_diff=2.79e-05 mean_abs_diff=3.71e-06

seq=1215 dtype=torch.float32 finite=True max_abs_diff=0.002857 mean_abs_diff=0.000555
seq=1215 dtype=torch.bfloat16 finite=False

seq=1620 dtype=torch.float32 finite=True max_abs_diff=0.002813 mean_abs_diff=0.000538
seq=1620 dtype=torch.bfloat16 finite=True max_abs_diff=2.76e-05 mean_abs_diff=2.06e-06
```

The complete size sweep is available in
`scripts/debug/xpu_softmax_bug_mre.py`. It tests fixed inputs in both fp32 and
bf16 and exits with status 1 if any tested size fails.

The complete sweep on `torch 2.14.0+xpu` reported:

```text
fp32 failures:
576, 640, 768, 1215, 1620, 2430, 3240

bf16 non-finite results:
1215, 2430, 3240
```

### Related `scaled_dot_product_attention` behavior

The affected softmax behavior also affects the following VAE-like attention
shape:

```text
[batch=1, heads=1, sequence_length, head_dim=640]
```

On `torch 2.14.0+xpu`, compared with a float64 CPU reference:

```text
sequence length 576:
  fp32 max_abs_diff ~= 0.0899
  bf16 max_abs_diff ~= 0.0920

sequence length 1620:
  fp32 max_abs_diff ~= 0.143
  bf16 max_abs_diff ~= 5.19
```

These sizes correspond to production VAE attention shapes. The incorrect result
causes corrupted XPU VAE decode output.

## Version bisect

The following versions were tested on the same Linux Intel XPU system using
clean Python 3.12 environments:

| PyTorch XPU wheel | XPU build version | Result |
|---|---:|---|
| `2.14.0+xpu` | `20260100` | Fails |
| `2.13.0+xpu` | `20260000` | Fails |
| `2.12.1+xpu` | `20250302` | Passes all tested sizes |
| `2.11.0+xpu` | `20250302` | Passes all tested sizes |
| `2.10.0+xpu` | `20250301` | Passes all tested sizes |

The regression is therefore between `2.12.1+xpu` and `2.13.0+xpu`.

The same issue was not reproduced on Windows with torch-xpu 2.13. This may
indicate a Linux-specific runtime, driver, or bundled oneAPI regression.

## Versions

The collected environment below is from the affected Linux system. The
collection file was named `collect_env_torch_2.14_xpu.txt`; it reports the
installed affected wheel as `torch 2.14.1+xpu`.

```text
Collecting environment information...
PyTorch version: 2.14.1+xpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: Could not collect
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.18.0-rc3-lgci-xe-kernel-xe-internal-1065-469253aa484f+-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
Is XPU available: True
XPU used to build PyTorch: 20260100
Intel GPU driver version:
* libze1: 1.27.0-1~24.04~ppa2
* intel-opencl-icd: 26.05.37020.3-1~24.04~ppa3
Intel GPU models onboard:
N/A
Intel GPU models detected:
* [0] _XpuDeviceProperties(name='Intel(R) Graphics [0xe223]', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero V2', type='gpu', device_id=0xE223, uuid=868023e2-0000-0000-0400-000000000000, driver_version='1.14.37020+3', total_memory=31023MB, local_mem_size=128KB, last_level_cache_size=24576KB, max_compute_units=256, memory_clock_rate=2800MHz, memory_bus_width=64-bit, gpu_eu_count=256, gpu_subslice_count=32, max_work_group_size=1024, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1, is_integrated_gpu=0)
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  14
On-line CPU(s) list:                     0-13
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Core(TM) Ultra 5 245K
CPU family:                              6
Model:                                   198
Thread(s) per core:                      1
Core(s) per socket:                      14
Socket(s):                               1
Stepping:                                2

Versions of relevant libraries:
[pip3] Could not collect
[conda] Could not collect
```

## Requested investigation

Could you identify the XPU softmax implementation or kernel change between the
`20250302` and `20260000` XPU builds that causes incorrect results for these
reduction sizes? In particular:

1. Why are some fp32 reductions finite but numerically incorrect?
2. Why do bf16 reductions produce NaNs for some sizes?
3. Why does the behavior differ between Linux and Windows for torch-xpu 2.13?

The expected behavior is that `torch.softmax` returns finite results matching
the CPU reference within normal fp32/bf16 numerical error for all reduction
sizes.
