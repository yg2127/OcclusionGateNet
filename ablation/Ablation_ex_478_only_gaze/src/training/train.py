from __future__ import annotations

import argparse

from src.training.runner import train_and_evaluate


def main(config_path: str) -> None:
    train_and_evaluate(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/region_transformer.yaml")
    args = parser.parse_args()
    main(args.config)
