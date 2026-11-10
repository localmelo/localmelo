"""Check that a running app is at rest: its process uses little CPU while nothing happens.

usage: python scripts/check_app_idle.py PID [--seconds 20] [--max 10]

Reads the CPU time of PID with ps once a second for --seconds and prints the mean share of one core it used and the
busiest second. Exit 0 when the mean is at most --max percent, 1 when it is above, 2 when PID is gone. The menu-bar item
once redrew without pause and held a core; that can only be seen on a real menu bar, so this is the check to run on a
built app, started and left alone (window open, nothing running) for the minute before. Standard library only; it
changes nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def cpu_seconds(pid: int) -> float | None:
    """The CPU time PID has used, in seconds; None when the process is gone."""
    done = subprocess.run(
        ["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True
    )
    text = done.stdout.strip()
    if done.returncode != 0 or not text:
        return None
    seconds = 0.0
    for part in text.split(":"):  # [[h:]m:]s.cc
        seconds = seconds * 60 + float(part)
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pid", type=int)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--max", type=float, default=10.0, help="percent of one core")
    args = parser.parse_args()
    cpu = cpu_seconds(args.pid)
    if cpu is None:
        print(f"process {args.pid} is not running")
        return 2
    samples = [(time.monotonic(), cpu)]  # (when, CPU seconds used so far)
    for _ in range(args.seconds):
        time.sleep(1)
        cpu = cpu_seconds(args.pid)
        if cpu is None:
            print(f"process {args.pid} went away")
            return 2
        samples.append((time.monotonic(), cpu))
    shares = [
        (b[1] - a[1]) * 100 / (b[0] - a[0])
        for a, b in zip(samples, samples[1:], strict=False)
    ]
    mean = (samples[-1][1] - samples[0][1]) * 100 / (samples[-1][0] - samples[0][0])
    print(
        f"process {args.pid}: mean {mean:.1f}% of one core over {args.seconds} s, "
        f"busiest second {max(shares):.0f}% (limit {args.max:g}%)"
    )
    return 0 if mean <= args.max else 1


if __name__ == "__main__":
    sys.exit(main())
