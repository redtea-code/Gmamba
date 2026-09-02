"""Run one NACC ResNet baseline with binary focal loss."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_baseline_comparison import build_parser as _build_baseline_parser, train_one


def build_parser() -> argparse.ArgumentParser:
    parser = _build_baseline_parser()
    parser.set_defaults(dataset="nacc", model="resnet", device="cuda")
    # Keep the dedicated entry point constrained to the approved experiment.
    for action in parser._actions:
        if action.dest in {"dataset", "model"}:
            action.required = False
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.focal_loss = True
    if args.dataset != "nacc" or args.model != "resnet":
        raise ValueError("NACC focal runner requires --dataset nacc and --model resnet")
    if args.output_dir is None:
        args.output_dir = str(Path("runs/nacc_resnet_focal_seed2026") / f"alpha{args.alpha:.2f}_gamma{args.gamma:.1f}".replace(".", ""))
    train_one(args)


if __name__ == "__main__":
    main()
