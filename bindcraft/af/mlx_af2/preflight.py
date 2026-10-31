"""Checks every real-model MLX run starts with (the design run, captures and benchmarks; adapted from the RFD4 port).

The disk must be below ``BINDCRAFT_MLX_MAX_DISK_PCT`` percent full (default 90; ESMFold2Score's README
records a kernel panic from a nearly full disk plus load pressure). ``BINDCRAFT_MLX_MIN_FREE_GB=<gb>``
is an explicit opt-in for running over that limit with at least that much free (the 90% limit stays the default), and says so on stderr.
No other process may hold more than 8 GB.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

MIN_FREE_ENV = "BINDCRAFT_MLX_MIN_FREE_GB"
MAX_DISK_ENV = "BINDCRAFT_MLX_MAX_DISK_PCT"


def check_disk(max_disk_pct: float | None = None) -> None:
    limit = float(os.environ.get(MAX_DISK_ENV, 90.0)) if max_disk_pct is None else max_disk_pct
    usage = shutil.disk_usage("/")
    pct = 100.0 * usage.used / usage.total
    if pct < limit:
        return
    free_gb = usage.free / 1e9
    floor = os.environ.get(MIN_FREE_ENV)
    if floor is None:
        raise SystemExit(f"preflight: disk is {pct:.0f}% full (limit {limit:.0f}%)")
    if free_gb < float(floor):
        raise SystemExit(f"preflight: disk is {pct:.0f}% full with {free_gb:.0f} GB free, below {MIN_FREE_ENV}={floor}")
    print(f"preflight WARNING: disk is {pct:.0f}% full (limit {limit:.0f}%); running on the "
          f"opt-in {MIN_FREE_ENV}={floor} with {free_gb:.0f} GB free", file=sys.stderr, flush=True)


def heavy_processes(max_other_gb: float) -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,rss=,comm="], capture_output=True, text=True, timeout=10).stdout
    mine = {os.getpid(), os.getppid()}
    heavy = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) not in mine:
            gb = int(parts[1]) / 1024 / 1024  # ps reports KiB
            if gb > max_other_gb:
                heavy.append(f"pid {parts[0]} {parts[2]} {gb:.1f} GB")
    return heavy


def checks(max_disk_pct: float | None = None, max_other_gb: float = 8.0) -> None:
    """Raise SystemExit (message starting ``preflight:``) if this machine is not ready for a model run."""
    check_disk(max_disk_pct)
    heavy = heavy_processes(max_other_gb)
    if heavy:
        raise SystemExit(f"preflight: other processes hold more than {max_other_gb} GB: {heavy}")
