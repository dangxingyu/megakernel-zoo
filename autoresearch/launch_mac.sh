#!/usr/bin/env bash
# bash autoresearch/launch_mac.sh RUN_DIR [HOURS=6]: an Inkling-2-Mini track-C autoresearch run on this Mac.
#   1. waits until the float64 reference is in MK_REF_DIR (default ~/mk-ref/thinkingmachines-inkling-2-mini),
#   2. prepares the evaluator (scripts/eval_setup_mac.sh) and the run directory (autoresearch/start.py --track C),
#   3. runs the agent (MODEL, default claude-fable-5-2) in RUN_DIR/workspace with RUN_DIR/PROMPT.md, and resumes it with
#      `claude -p --continue` whenever it stops while the board still has time left (at most MAX_RESUMES times),
#   4. keeps the Mac awake (caffeinate) for the whole run.
# Logs: RUN_DIR/agent-N.jsonl (stream-json of each agent session) and RUN_DIR/launch.log.
set -uo pipefail
zoo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="${1:?run dir}"; hours="${2:-6}"
model="${MODEL:-claude-fable-5-2}"; resumes="${MAX_RESUMES:-20}"
export MK_MODEL_DIR="${MK_MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
export MK_REF_DIR="${MK_REF_DIR:-$HOME/mk-ref/thinkingmachines-inkling-2-mini}"
need=(ids.npy meta.json logits64.npy long_positions.npy long_logits64.npy)

log() { echo "$(date '+%F %T') $*" | tee -a "$run.launch.log"; }
missing() { for f in "${need[@]}"; do [ -s "$MK_REF_DIR/$f" ] || { echo "$f"; return 0; }; done; return 1; }
# REF_RELEASE=OWNER/REPO@TAG: fetch the reference from that GitHub release (gh) once it exists
release="${REF_RELEASE:-dangxingyu/megakernel-refs@inkling-2-mini-v3}"
fetch() {
  local repo="${release%@*}" tag="${release#*@}"
  gh release view "$tag" --repo "$repo" > /dev/null 2>&1 || return 1
  mkdir -p "$MK_REF_DIR"
  gh release download "$tag" --repo "$repo" --dir "$MK_REF_DIR" --skip-existing >> "$run.launch.log" 2>&1
  # ids.npy is in the repository; the release may leave it out
  [ -s "$MK_REF_DIR/ids.npy" ] || cp "$zoo/eval/refs/thinkingmachines-inkling-2-mini/ids.npy" "$MK_REF_DIR/" 2>/dev/null
  log "fetched $release: $(ls "$MK_REF_DIR" | tr '\n' ' ')"
}
while m="$(missing)"; do
  [ -n "$release" ] && fetch && ! missing > /dev/null && break
  log "waiting for the reference: $MK_REF_DIR/$m (or GitHub release $release)"; sleep 120
done
log "reference present in $MK_REF_DIR"

bash "$zoo/scripts/eval_setup_mac.sh" 2>&1 | tee -a "$run.launch.log" || { log "eval_setup_mac.sh failed"; exit 3; }
[ -d "$run" ] || "$zoo/.venv-mac/bin/python" "$zoo/autoresearch/start.py" "$run" --model thinkingmachines/Inkling-2-Mini \
  --card m5max --track C --hours "$hours" --model-dir "$MK_MODEL_DIR" --ref-dir "$MK_REF_DIR" 2>&1 | tee -a "$run.launch.log"
mv "$run.launch.log" "$run/launch.log" 2>/dev/null; exec >> "$run/launch.log" 2>&1
log "run $run, $hours h, agent $model"

time_left() { PATH="$run/bin:$PATH" mk board 2>/dev/null | sed -n 's/.* \([0-9.]*\) left;.*/\1/p' | head -1; }
cd "$run/workspace" || exit 4
caffeinate -dims -w $$ &
for i in $(seq 0 "$resumes"); do
  if [ "$i" = 0 ]; then
    log "agent session 0"
    claude -p "$(cat ../PROMPT.md)" --model "$model" --dangerously-skip-permissions --output-format stream-json --verbose \
      > "$run/agent-0.jsonl" 2>&1
  else
    left="$(time_left)"; log "agent stopped; board says ${left:-?} h left"
    awk -v l="${left:-0}" 'BEGIN { exit !(l > 0.1) }' || break
    claude -p --continue "You stopped, but the run still has ${left} hours on the board. Continue the run as PROMPT.md says: read your notes and \`mk board\`, decide the next experiment, and keep going until the time is used up." \
      --model "$model" --dangerously-skip-permissions --output-format stream-json --verbose > "$run/agent-$i.jsonl" 2>&1
  fi
done
log "run finished: $(PATH="$run/bin:$PATH" mk board 2>/dev/null | head -1)"
