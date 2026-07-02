"""run_awake.py — run a script while preventing Windows from sleeping.

Usage: python run_awake.py <script.py> [args...]

Sets ES_SYSTEM_REQUIRED only for the lifetime of the child process; no
permanent power-plan changes. The display may still turn off.
"""
import ctypes
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_awake.py <script.py> [args...]")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        rc = subprocess.call([sys.executable] + sys.argv[1:])
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(rc)
