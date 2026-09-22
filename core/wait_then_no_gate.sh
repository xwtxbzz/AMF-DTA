#!/bin/sh
# Hold the no-gate sweep until the seed-52 FingerDTA pilot releases the GPU.
#
# Both schedulers run MAX_PARALLEL=3 trainings; starting the second while the
# first is live would put six trainings on one 8 GB RTX 4060 and roughly
# halve everyone's throughput, so we serialise them.
#
# The completion signal is a line the pilot itself prints once its pool has
# drained: either "WARNING: incomplete runs: [...]" (there is no --seeds
# argument, so this is what a seed-52-only run prints) or "All FingerDTA
# ensemble members ... complete."  Do NOT grep for an "exit=" line: the
# launcher appends that outside the redirect, so it never reaches the log.
#
# A log that has stopped growing for 30 minutes is also treated as terminal,
# so a hard crash cannot leave this waiting forever. The pilot emits an epoch
# line every ~30 s while alive, so a 30-minute gap is unambiguous.
cd /d/try/rail_dta || exit 1

PILOT_LOG=outputs/pilot_fens_seed52.log
IDLE_LIMIT=1800

while :; do
    if grep -qE '^(WARNING: incomplete runs:|All FingerDTA ensemble members)' \
            "$PILOT_LOG" 2>/dev/null; then
        echo "[waiter] pilot reported completion at $(date '+%F %T')"
        break
    fi
    if [ -f "$PILOT_LOG" ]; then
        now=$(date +%s)
        mod=$(stat -c %Y "$PILOT_LOG" 2>/dev/null || echo 0)
        if [ "$mod" -gt 0 ] && [ $((now - mod)) -gt "$IDLE_LIMIT" ]; then
            echo "[waiter] pilot log idle >$((IDLE_LIMIT / 60))min at $(date '+%F %T'); assuming finished"
            break
        fi
    fi
    sleep 120
done

echo "[waiter] starting no-gate sweep at $(date '+%F %T')"
python run_no_gate_5seeds.py > outputs/no_gate_5seeds.log 2>&1
status=$?
echo "[waiter] no-gate sweep exit=$status at $(date '+%F %T')"
tail -20 outputs/no_gate_5seeds.log
