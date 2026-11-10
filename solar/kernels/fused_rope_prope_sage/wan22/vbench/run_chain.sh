#!/usr/bin/env bash
# Wan2.2-T2V-A14B VBench-2.0 subset: smoke test, SVDQuant calibration + conversion, then the 4 arms in
# 3 balanced rounds (--limit 11 / 22 / 33). Resumable: rerun to continue (finished videos are skipped).
#   nohup wan/vbench/run_chain.sh > logs/vb_chain.log 2>&1 &
set -uo pipefail
cd /home/gta/omni_bench
export PYTHONPATH=/home/gta/pylibs/diffusers
G="scripts/rss_guard.sh 24"
B="./env.sh python wan/run_wan_t2v_batch.py"
CAL="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
until grep -q "\[t2v\] done" logs/wan_t2v_convert.log 2>/dev/null; do
  grep -q "\[t2v\].*failed" logs/wan_t2v_convert.log 2>/dev/null && { echo "[chain] int8 conversion failed"; exit 1; }
  sleep 60
done
echo "[chain] int8 experts ready $(date)"
if [[ ! -f wan/vbench/smoke/smoke/custom/"${CAL:0:180}"-0.mp4 ]]; then
  $G logs/vb_smoke.log -- $B --arm sage --arm-name smoke --prompt "$CAL" --frames 17 --steps 4 --out wan/vbench/smoke \
    || { echo "[chain] smoke failed"; exit 1; }
fi
if [[ ! -f /home/gta/models/wan22-t2v-a14b-svdq/transformer_2/svdq_index.json ]]; then
  [[ -f wan/outputs/t2v_calib_amax.pt ]] || $G logs/vb_calib.log -- $B --arm sdpa --arm-name calib --prompt "$CAL" --seed 7 \
    --frames 33 --out wan/vbench/calib --calib-out wan/outputs/t2v_calib_amax.pt || { echo "[chain] calib failed"; exit 1; }
  $G logs/vb_svdq_convert.log -- ./env.sh python wan/wan_svdq.py convert /home/gta/models/wan22-t2v-a14b-int8 \
    wan/outputs/t2v_calib_amax.pt /home/gta/models/wan22-t2v-a14b-svdq || { echo "[chain] svdq convert failed"; exit 1; }
fi
for limit in 11 22 33; do
  $G logs/vb_sdpa.log -- $B --arm sdpa --limit $limit || { echo "[chain] sdpa failed"; exit 1; }
  $G logs/vb_sage.log -- $B --arm sage --limit $limit || { echo "[chain] sage failed"; exit 1; }
  SAGE_BUILD_DIR=$PWD/kernel/build_int4 $G logs/vb_sage_int4.log -- $B --arm sage --arm-name sage_int4 --limit $limit \
    || { echo "[chain] sage_int4 failed"; exit 1; }
  WAN_T2V_SVDQ_DIR=/home/gta/models/wan22-t2v-a14b-svdq $G logs/vb_sdpa_svdq.log -- $B --arm sdpa --linear svdq --limit $limit \
    || { echo "[chain] sdpa_svdq failed"; exit 1; }
  echo "[chain] round limit=$limit done $(date)"
done
echo "[chain] all done $(date)"
