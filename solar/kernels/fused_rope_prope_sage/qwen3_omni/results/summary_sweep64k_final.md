Peaks: bf16 GEMM 154 TFLOPS, bandwidth 812 GB/s (Intel(R) Arc(TM) Pro B70 Graphics)

| L | arm | n | TTFT s | ×SDPA | attn s | attn share | ×SDPA | attn TFLOPS (util) | attn GB/s (util) | prefill TFLOPS (util) | prefill GB/s (util) | peak prefill GiB | act GiB | E2E s | TPOT ms | RULER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 63488 | sage_int4 | 180 | 20.66 | 1.45 | 9.95 | 48% | 1.96 | 156 (101%) | 17 (2%) | 91.5 (60%) | 9 (1%) | 27.71 | 9.80 | 23.96 | 235 | 61.4 |
| 63488 | sage_int8 | 180 | 23.50 | 1.27 | 12.88 | 55% | 1.51 | 120 (78%) | 13 (2%) | 80.4 (52%) | 8 (1%) | 27.85 | 9.94 | 26.59 | 236 | 68.9 |
| 63488 | sdpa | 180 | 29.91 | 1.00 | 19.50 | 65% | 1.00 | 79 (52%) | 3 (0%) | 63.2 (41%) | 3 (0%) | 26.91 | 9.00 | 32.99 | 235 | 68.8 |
