#!/usr/bin/env bash
# Run a command in omni_env: auto-round's XPU kernel needs the pip-installed MKL on the loader path.
export LD_LIBRARY_PATH=/home/gta/omni_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# transformers 5 async (threaded) weight loading spikes host RAM by ~17 GB on this 30 GB box
export HF_DEACTIVATE_ASYNC_LOAD=1
# expandable segments tested and rejected: 32K TTFT 17 -> 41 s, decode 5x slower (remaps every step)
export PATH=/home/gta/omni_env/bin:$PATH
exec "$@"
