"""Checks every real-model MLX run starts with (the design run, captures and benchmarks; adapted from the RFD4 port).

The disk must be below ``BINDCRAFT_MLX_MAX_DISK_PCT`` percent full (default 90; ESMFold2Score's README
records a kernel panic from a nearly full disk plus load pressure). ``BINDCRAFT_MLX_MIN_FREE_GB=<gb>``
is an explicit opt-in for running over that limit with at least that much free (the 90% limit stays the default), and says so on stderr.
Only the disk is checked: how much memory other processes hold is not (a rule that refused to start while another process held more than 8 GB was dropped).
"""

from __future__ import annotations

import os
import shutil
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


def checks(max_disk_pct: float | None = None) -> None:
    """Raise SystemExit (message starting ``preflight:``) if this machine is not ready for a model run."""
    check_disk(max_disk_pct)
