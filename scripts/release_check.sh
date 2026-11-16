#!/usr/bin/env bash
# bash scripts/release_check.sh: eval/release_check.py on every engine source named in $RELCHECK_SOURCES (files
# rescore_inbox/NAME.tgz), each in two fresh processes whose outputs are then compared (restart determinism).
# MK_MODEL / MK_HW as for scripts/eval_setup.sh; RELCHECK_CONTEXT = max_context given to the engine (default 2048).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
env_report
source scripts/cuda_toolkit.sh > "$MK_ROOT/logs/cuda-toolkit.log" 2>&1; grep MK_ENV "$MK_ROOT/logs/cuda-toolkit.log"
set +e +u; set -o pipefail
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
bash scripts/eval_setup.sh 2>&1 | grep -E "MK_ENV|MK_STAGE|rror" | cut -c1-300
model="$(mk_model "${MK_MODEL:-Qwen/Qwen3-8B}")" || exit 11
py="$MK_ROOT/venvs/eval/bin/python"; export PATH="$MK_ROOT/venvs/eval/bin:$PATH"
ref="$("$py" -c 'import json,os; print(json.load(open(os.environ["MK_ROOT"] + "/eval_env.json"))["private"]["ref"])')"
cpu_args=(); [ "${MK_TARGET:-card}" = node ] && cpu_args=(--server $(( $(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l) - 12 )) --client 4)
# correctness only: when the node has no quiet cores, run unpinned instead of failing
picked="$(python3 scripts/pick_cpus.py "${cpu_args[@]}" 2>&1)" && eval "$picked" || SERVER_CPUS="0-$(( $(nproc) - 1 ))"
echo "MK_ENV cpus $picked -> SERVER_CPUS=$SERVER_CPUS"
for name in $RELCHECK_SOURCES; do
  ws="$MK_ROOT/relcheck/$name"; rm -rf "$ws"; mkdir -p "$ws"; tar xzf "rescore_inbox/$name.tgz" -C "$ws" || { echo "MK_RELCHECK_FAIL $name unpack"; continue; }
  echo "MK_ENV relcheck source=$name sha256=$(sha256sum "rescore_inbox/$name.tgz" | cut -c1-16) track=$(cat "$ws/engine/TRACK" 2>/dev/null | tr -d '\n')"
  (cd "$ws" && { [ ! -f engine/build.sh ] || bash engine/build.sh; }) > "$MK_ROOT/logs/build-$name.log" 2>&1 || { tail -n 15 "$MK_ROOT/logs/build-$name.log" | cut -c1-300; echo "MK_RELCHECK_FAIL $name build"; continue; }
  for k in 1 2; do
    t=$(date +%s)
    (cd "$ws" && timeout "${RELCHECK_TIMEOUT:-3600}" taskset -c "$SERVER_CPUS" "$py" "$MK_REPO/eval/release_check.py" --workspace "$ws" --model-dir "$model" --ref "$ref" \
       --max-context "${RELCHECK_CONTEXT:-2048}" --out "$MK_ROOT/run/relcheck-$name-$k.json") > "$MK_ROOT/logs/relcheck-$name-$k.log" 2>&1; rc=$?
    grep MK_RELCHECK "$MK_ROOT/logs/relcheck-$name-$k.log" | sed "s/^MK_RELCHECK /MK_RELCHECK $name p$k /" | cut -c1-2500
    [ $rc = 0 ] || { echo "MK_RELCHECK_FAIL $name process $k rc=$rc"; tail -n 12 "$MK_ROOT/logs/relcheck-$name-$k.log" | cut -c1-300; }
    stage "relcheck $name $k" "$([ $rc = 0 ] && echo ok || echo failed)" $t
    pkill -9 -f "release_check.py --workspace" 2>/dev/null; sleep 3
  done
  [ -f "$MK_ROOT/run/relcheck-$name-1.json" ] && [ -f "$MK_ROOT/run/relcheck-$name-2.json" ] && \
    "$py" eval/release_check.py --compare "$MK_ROOT/run/relcheck-$name-1.json" "$MK_ROOT/run/relcheck-$name-2.json" | sed "s/^MK_RELCHECK /MK_RELCHECK $name /"
done
mkdir -p "$MK_ROOT/run/logs" && cp "$MK_ROOT"/logs/*.log "$MK_ROOT/run/logs/" 2>/dev/null
store put "$MK_ROOT/run" "$MK_S3/runs/${MK_JOB}" "$MK_S3" 2>&1 | tail -n 1 | cut -c1-200
stage "release check" ok
