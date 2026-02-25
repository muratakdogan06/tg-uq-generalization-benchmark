import argparse
from cms_tg.config import load_config
from cms_tg.eval import run_benchmark

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_benchmark(cfg)
    print(f"Done. See: {cfg.run_dir}")

if __name__ == "__main__":
    main()
