#!/usr/bin/env python3
"""Convert configured truth or reconstructed ROOT samples to Parquet."""

from __future__ import annotations

import argparse

from hww.config import load_config
from hww.preprocessing import preprocess_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.toml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="append", help="Configured sample name; repeat as needed")
    group.add_argument(
        "--domain", choices=("truth", "reco"), help="Process every sample in a domain"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sample:
        missing = sorted(set(args.sample) - set(config.samples))
        if missing:
            raise SystemExit("Unknown sample(s): " + ", ".join(missing))
        samples = [config.samples[name] for name in args.sample]
    else:
        samples = [sample for sample in config.samples.values() if sample.domain == args.domain]
    if not samples:
        raise SystemExit("No matching samples are configured")

    for sample in samples:
        output = preprocess_sample(config, sample, overwrite=args.overwrite)
        print(f"[{sample.domain}] {sample.name} -> {output}")


if __name__ == "__main__":
    main()
