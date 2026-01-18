import argparse
import os

from ti.config.defaults import load_config
from ti.online.train import run_online_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--env", default="periodicity")
    parser.add_argument("--method", default="CRTR")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg["runtime"]
    output_dir = os.path.join(runtime.get("table_dir", os.path.join(runtime["output_dir"], "tables")), "online")
    run_online_training(cfg, args.env, args.method, args.seed, args.alpha, output_dir)


if __name__ == "__main__":
    main()
