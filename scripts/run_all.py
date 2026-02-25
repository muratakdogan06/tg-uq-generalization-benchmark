import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cmd = [sys.executable, "scripts/run_benchmark.py", "--config", args.config]
    print(">>> Running benchmark:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
