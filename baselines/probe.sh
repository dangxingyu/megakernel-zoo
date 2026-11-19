#!/usr/bin/env bash
# What a pod gives us (H100 x86 or GB200 aarch64): GPU, driver, CPU and memory limits, toolchain, network, object store.
# Prints names of credential variables, never their values.
set -uo pipefail
section() { echo; echo "=== $1"; }

section gpu
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode,clocks.max.sm --format=csv 2>&1
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1
nvidia-smi -q 2>/dev/null | grep -iE "CUDA Version|Persistence|ECC|Product Architecture" | head -6
ls /usr/local | grep -i cuda; command -v nvcc && nvcc --version | tail -2

section cpu-and-memory
echo "nproc $(nproc)  online $(getconf _NPROCESSORS_ONLN)"
lscpu | grep -E "Architecture|Model name|Socket|Core|Thread|NUMA node|MHz|L2|L3" | head -16
echo "threads per core (cpu0 siblings): $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"
echo "distinct (package, core_id) pairs: $(for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(cat $c/topology/physical_package_id):$(cat $c/topology/core_id)"; done | sort -u | wc -l)"
for g in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do b=$(echo "$g" | tr A-Z a-z); echo "gpu $g numa_node $(cat /sys/bus/pci/devices/${b:4}/numa_node 2>&1)"; done
python3 "$(dirname "$0")/../scripts/pick_cpus.py" 2>&1 | cut -c1-400
grep -E "^(Cpus|Mems)_allowed_list" /proc/self/status
cat /sys/fs/cgroup/cpu.max 2>/dev/null | sed 's/^/cgroup cpu.max /'
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null | sed 's/^/cpuset /'
cat /sys/fs/cgroup/memory.max 2>/dev/null | awk '{printf "cgroup memory.max %.0f GB\n", $1/1e9}'
free -g | head -2
uptime
ps -eo pcpu,pmem,comm --sort=-pcpu 2>/dev/null | head -8

section os-and-tools
. /etc/os-release && echo "$PRETTY_NAME"; uname -rm; id -u
for tool in python3 python3.12 uv pip gcc g++ cmake ninja git curl nsys stress-ng perf taskset numactl; do
  printf "%-10s %s\n" "$tool" "$(command -v $tool || echo -)"
done
python3 --version 2>&1; gcc --version 2>/dev/null | head -1; cmake --version 2>/dev/null | head -1

section disk
df -h / /tmp /dev/shm 2>/dev/null
echo "MK vars: $(env | grep -E '^MK_' | cut -d= -f1 | tr '\n' ' ')"
echo "store vars present: $(env | cut -d= -f1 | grep -iE 'AWS|S3|R2|ENDPOINT|BUCKET|MARIN' | tr '\n' ' ')"

section network
for url in https://huggingface.co https://pypi.org/simple/ https://github.com https://developer.download.nvidia.com https://download.pytorch.org/whl/cu130/ https://download.pytorch.org/whl/cu132/torch/; do
  printf "%-50s %s\n" "$url" "$(curl -s -o /dev/null -m 15 -w '%{http_code} %{time_total}s' "$url")"
done

section apt
(apt-get update -qq 2>&1 | tail -1; apt-cache policy stress-ng linux-tools-common 2>/dev/null | grep -E "^[a-z]|Candidate") 2>&1 | head -8

section neighbours
# Other tenants on the node show up as CPU load and as GPUs we cannot see.
nvidia-smi -L 2>&1 | head -10
grep -c ^processor /proc/cpuinfo; awk '{print "loadavg", $1, $2, $3}' /proc/loadavg

section wheels
# Which torch would the engines get on this architecture? (aarch64 has resolved to a CPU-only wheel before.)
tag="$(uname -m)"; [ "$tag" = x86_64 ] && tag="x86_64" || tag="aarch64"
for idx in cu128 cu130 cu132; do
  printf "torch %s %s wheels: %s\n" "$idx" "$tag" "$(curl -s -m 20 https://download.pytorch.org/whl/$idx/torch/ | grep -oE "torch-[0-9.]+%2B$idx-cp312-cp312-[a-z0-9_]*$tag[a-z0-9_.]*whl" | sort -uV | tail -3 | tr '\n' ' ')"
done
for pkg in vllm sglang flashinfer-python sglang-kernel sgl-kernel; do
  printf "%-18s %s\n" "$pkg" "$(curl -s -m 20 https://pypi.org/simple/$pkg/ | grep -oE ">$(echo $pkg | tr - '[-_]')[^<]*(aarch64|none-any)[^<]*whl<" | tr -d '<>' | sort -uV | tail -2 | tr '\n' ' ')"
done
echo PROBE_DONE
