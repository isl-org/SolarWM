#!/usr/bin/env bash
# KernelFoundry non-performance gate for a candidate built as module `sage_kernel`.
#   kernel/kf_gate.sh path/to/sage_kernel.so LOGNAME
# Uses /home/gta/kf_task_run (task.py + v3 reference built as sage_reference).
set -euo pipefail
SO=$(realpath "$1"); LOG=/home/gta/omni_bench/logs/$2.log
T=/home/gta/kf_task_run
cp "$SO" "$T/sage_kernel.so"
cd "$T"
PYTHONPATH=/home/gta/kernelfoundry_private /home/gta/omni_bench/scripts/rss_guard.sh 10 "$LOG" -- \
  /home/gta/omni_bench/env.sh python -u -m pytest --color=no -p no:cacheprovider -m "not performance" task.py >/dev/null 2>&1 || true
grep -E "^=+ .*(passed|failed)" "$LOG" | tail -1
