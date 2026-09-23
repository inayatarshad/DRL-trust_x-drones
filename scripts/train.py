"""Train a DRL agent on the UAV environment.

    python -m scripts.train --config configs/td3.yaml
    python -m scripts.train --config configs/ppo.yaml --out checkpoints/ppo train.episodes=200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from trustx.training import train
from trustx.utils.config import load_config, parse_overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config (see configs/)")
    parser.add_argument("--out", default=None, help="output directory (default: checkpoints/<algo>)")
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. train.episodes=50")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
    out = Path(args.out or f"checkpoints/{cfg['algo']}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"training {cfg['algo']} -> {out}")
    train(cfg, out)
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
