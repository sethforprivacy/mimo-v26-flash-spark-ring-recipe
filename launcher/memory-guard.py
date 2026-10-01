#!/usr/bin/env python3
"""Local host-memory guard for one rank container; survives SSH disconnects.

Started by vllm-rank.sh up (nohup, on the host, stdlib only). GB10 memory is unified: driver allocations have no
OOM killer, and an exhausted Spark can hang and reboot. Every 5 s the guard reads MemAvailable; it stops this rank's
container (logs preserved) below 4 GiB at once, or after 3 consecutive samples below 6 GiB (a single deep prefill
chunk's transient must not count as exhaustion). It stops only its own rank: after a trip, stop the peers yourself
(launcher/ring-up.sh down). This reduces gradual-exhaustion risk; it cannot guarantee protection against an
instantaneous allocation or a driver hang.

State: <state>/memory-guard.json (last sample), <state>/MEMORY_GUARD_TRIPPED (written on a trip).
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def pressure(available_gib, previous_strikes):
    # Trip below the 4 GiB hard floor at once, or on the third consecutive sample below 6 GiB.
    strikes = previous_strikes + 1 if available_gib < 6 else 0
    return strikes, available_gib < 4 or strikes >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--container', default='vllm_mimo26')
    args = ap.parse_args()
    lock = (args.state / 'memory-guard.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    strikes, seen, consecutive_idle = 0, False, 0
    started = time.monotonic()
    while True:
        try:
            result = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', args.container],
                                    capture_output=True, text=True, timeout=5)
            running = result.returncode == 0 and result.stdout.strip() == 'true'
        except subprocess.TimeoutExpired:
            running = seen  # Do not skip memory checks when Docker becomes unresponsive.
        if not running:
            consecutive_idle += 1
            if (seen and consecutive_idle >= 3) or time.monotonic() - started > 3000:
                return
            time.sleep(5)
            continue
        seen, consecutive_idle = True, 0
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(memory['MemAvailable'].split()[0]) / 1048576
        strikes, trip = pressure(available, strikes)
        state = {'utc': datetime.now(timezone.utc).isoformat(), 'available_gib': round(available, 3),
                 'strikes': strikes, 'tripped': trip, 'container': args.container}
        tmp = args.state / 'memory-guard.tmp'
        tmp.write_text(json.dumps(state) + '\n')
        tmp.replace(args.state / 'memory-guard.json')
        if trip:
            (args.state / 'MEMORY_GUARD_TRIPPED').write_text(json.dumps(state) + '\n')
            print(json.dumps(state), flush=True)
            try:
                subprocess.run(['docker', 'stop', '-t', '5', args.container], timeout=15, check=False)
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'kill', args.container], timeout=10, check=False)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
