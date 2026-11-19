#!/usr/bin/env python3
"""Choose idle CPU cores on a shared node: prints `SERVER_CPUS=... CLIENT_CPUS=...` for eval-ing in bash.

Machines here see every core of the node and have no CPU quota, and other tenants (and our own other jobs) run
on the same cores. Per-CPU busy time is sampled for two seconds; physical cores all of whose hardware threads
(two on the x86 H100 hosts, one on the Grace GB200 hosts, which have no SMT) were under 3% busy are taken first,
on the GPU's NUMA node when it has enough of them; then the next quietest (cores over 20% busy are refused). The server
gets `--server` physical cores and the client `--client` others, so the client never takes time from the
server. This avoids load that is already there; it cannot stop a neighbour from starting later, so runs
also record the load average and repeat every point.
"""

from __future__ import annotations

import os
import argparse
import subprocess
import time
from pathlib import Path


def busy_ticks() -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        parts = line.split()
        if parts[0].startswith("cpu") and parts[0] != "cpu":
            values = list(map(int, parts[1:9]))
            idle = values[3] + values[4]
            out[int(parts[0][3:])] = (sum(values) - idle, sum(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=int, default=8, help="physical cores for the server")
    parser.add_argument("--client", type=int, default=4, help="physical cores for the client")
    parser.add_argument("--extra", type=int, default=0, help="further physical cores, printed as EXTRA_CPUS")
    parser.add_argument("--gpu", type=int, default=0, help="index of the card whose NUMA node is preferred (machines with several cards)")
    parser.add_argument("--exclude", default="", help="comma-separated CPUs that are already given to other jobs")
    args = parser.parse_args()
    excluded = {int(x) for x in args.exclude.split(",") if x.strip().isdigit()}
    a = busy_ticks(); time.sleep(2.0); b = busy_ticks()
    busy = {cpu: (b[cpu][0] - a[cpu][0]) / max(1, b[cpu][1] - a[cpu][1]) for cpu in b}
    cores: dict[tuple[int, int], list[int]] = {}
    node_of: dict[int, int] = {}
    for cpu in busy:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        cores.setdefault(key, []).append(cpu)
        nodes = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node*"))
        node_of[cpu] = int(nodes[0].name[4:]) if nodes else 0
    gpu_node = 0
    try:
        buses = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                               text=True).stdout.split()
        bus = buses[args.gpu if 0 <= args.gpu < len(buses) else 0].lower()
        gpu_node = max(0, int(Path(f"/sys/bus/pci/devices/{bus[4:] if len(bus) > 12 else bus}/numa_node").read_text()))
    except Exception:
        pass
    # Quietest physical cores first, the GPU's NUMA node preferred among equally quiet ones. Every core under 3%
    # busy counts as quiet (one class); busier ones are ordered by their busier thread, rounded to 2%. Without
    # the one quiet class, background noise of 1-2% on the GPU's node sends everything to the other socket
    # (seen on the two-socket Grace hosts, where neighbours' threads roam over all 144 cores).
    # A node where even the chosen cores are busy is reported, not silently used.
    def level(threads: list[int]) -> int:
        worst_thread = max(busy[t] for t in threads)
        return 0 if worst_thread < 0.03 else 1 + round(worst_thread / 0.02)

    idle = sorted((sorted(threads) for threads in cores.values() if not excluded & set(threads)),
                  key=lambda threads: (level(threads), node_of[threads[0]] != gpu_node, threads[0]))
    need = args.server + args.client + args.extra
    if len(idle) < need:
        raise SystemExit(f"only {len(idle)} physical cores are left for {need}")
    worst = max(busy[t] for threads in idle[:need] for t in threads)
    # MK_CPU_BUSY_MAX: how busy the busiest chosen core may be before the node is refused (default 0.20). A comparison that
    # serves every system in turn on the same cores can accept more: all systems see the same neighbours, and the
    # chosen level is printed (CHOSEN_MAX_BUSY) with the results.
    limit = float(os.environ.get("MK_CPU_BUSY_MAX", "0.20"))
    if worst > limit:
        raise SystemExit(f"busiest chosen core is {worst:.0%} busy: no {need} quiet physical cores on this node")
    quiet = sum(1 for threads in idle if all(busy[t] < 0.03 for t in threads))

    def fmt(group: list[list[int]]) -> str:
        return ",".join(str(t) for threads in group for t in threads)

    server, client, extra = idle[:args.server], idle[args.server:args.server + args.client], idle[args.server + args.client:need]
    by_node: dict[int, int] = {}
    for threads in idle:
        if level(threads) == 0:
            by_node[node_of[threads[0]]] = by_node.get(node_of[threads[0]], 0) + 1
    print(f"SERVER_CPUS={fmt(server)} CLIENT_CPUS={fmt(client)} EXTRA_CPUS={fmt(extra)} GPU_NUMA={gpu_node} "
          f"IDLE_CORES={quiet} CHOSEN_MAX_BUSY={worst:.3f} NODE_BUSY={sum(busy.values()) / len(busy):.3f} "
          f"THREADS_PER_CORE={max(len(t) for t in idle)} CORES={len(idle)} "
          f"IDLE_BY_NUMA={','.join(f'{n}:{c}' for n, c in sorted(by_node.items()))} "
          f"CHOSEN_NUMA={','.join(str(n) for n in sorted({node_of[t[0]] for t in idle[:need]}))}")


if __name__ == "__main__":
    main()
