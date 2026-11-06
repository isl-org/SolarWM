#!/usr/bin/env bash
# rss_guard.sh LIMIT_GB LOG -- cmd...
# Run cmd, sample memory every 0.25 s into LOG.mem.csv, and kill cmd if its RSS > LIMIT_GB or
# system MemAvailable < 4 GB. Protects the host (30 GB RAM, shared with the Claude session)
# from the global OOM killer, which also counts GPU-driver spill that is not in the process RSS.
LIMIT_KB=$(( $1 * 1024 * 1024 )); LOG=$2; shift 3
CSV=$LOG.mem.csv; echo "t,avail_mb,anon_pages_mb,shmem_mb,swap_used_mb,p_anon_mb,p_file_mb,p_shmem_mb" > $CSV
"$@" > "$LOG" 2>&1 & PID=$!
PEAK=0; MINAV=99999999; T0=$SECONDS
while kill -0 $PID 2>/dev/null; do
  read AV AP SH ST SF < <(awk '/MemAvailable/{a=$2}/AnonPages/{b=$2}/^Shmem:/{c=$2}/SwapTotal/{e=$2}/SwapFree/{f=$2}END{print a,b,c,e,f}' /proc/meminfo)
  read PA PF PS < <(awk '/RssAnon/{a=$2}/RssFile/{b=$2}/RssShmem/{c=$2}END{print a+0,b+0,c+0}' /proc/$PID/status 2>/dev/null)
  RSS=$(( PA + PF + PS )); (( RSS > PEAK )) && PEAK=$RSS; (( AV < MINAV )) && MINAV=$AV
  echo "$((SECONDS-T0)),$((AV/1024)),$((AP/1024)),$((SH/1024)),$(((ST-SF)/1024)),$((PA/1024)),$((PF/1024)),$((PS/1024))" >> $CSV
  if (( AV < 4 * 1024 * 1024 )); then
    echo "[rss_guard] system MemAvailable ${AV} kB < 4 GB, killing $PID" | tee -a "$LOG"; kill -9 $PID
  elif (( RSS > LIMIT_KB )); then
    echo "[rss_guard] RSS ${RSS} kB > limit, killing $PID" | tee -a "$LOG"; kill -9 $PID
  fi
  sleep 0.25
done
wait $PID; RC=$?
echo "[rss_guard] exit=$RC peak_rss=$(( PEAK / 1048576 )).$(( PEAK % 1048576 * 10 / 1048576 )) GB min_sys_avail=$(( MINAV / 1048576 )).$(( MINAV % 1048576 * 10 / 1048576 )) GB" | tee -a "$LOG"
exit $RC
