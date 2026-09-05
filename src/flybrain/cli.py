from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flybrain")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="compile published connectivity files")
    pack.add_argument("--completeness", required=True)
    pack.add_argument("--connectivity", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--materialization", required=True)
    verify = subparsers.add_parser("verify-pack", help="audit a compiled data pack")
    verify.add_argument("--pack", required=True)
    verify.add_argument("--completeness")
    verify.add_argument("--connectivity")
    engine = subparsers.add_parser(
        "verify-engine", help="compare the accelerated engine with the NumPy reference"
    )
    engine.add_argument("--pack", required=True)
    engine.add_argument("--steps", type=int, default=1000)
    engine.add_argument("--rate-hz", type=float, default=150.0)
    engine.add_argument("--seed", type=int, default=20260816)
    engine.add_argument("--propagation", choices=("scatter", "metal"), default="metal")
    simulate = subparsers.add_parser("simulate", help="run a seeded sugar-input experiment")
    simulate.add_argument("--pack", required=True)
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--duration-ms", type=float, default=1000.0)
    simulate.add_argument("--rate-hz", type=float, default=150.0)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--propagation", choices=("scatter", "metal"), default="metal")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pack":
        from flybrain.pack import pack_connectome

        manifest = pack_connectome(
            completeness_path=args.completeness,
            connectivity_path=args.connectivity,
            output_path=args.output,
            materialization=args.materialization,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "verify-pack":
        from flybrain.verify import audit_pack

        audit = audit_pack(
            args.pack,
            completeness_path=args.completeness,
            connectivity_path=args.connectivity,
        )
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    elif args.command == "verify-engine":
        from flybrain.connectome import PackedConnectome
        from flybrain.validation import validate_engine

        connectome = PackedConnectome.load(args.pack)
        validation = validate_engine(
            connectome,
            steps=args.steps,
            rate_hz=args.rate_hz,
            seed=args.seed,
            propagation=args.propagation,
        )
        print(json.dumps(validation.to_dict(), indent=2, sort_keys=True))
    elif args.command == "simulate":
        from flybrain.connectome import PackedConnectome
        from flybrain.runner import run_sugar_experiment

        connectome = PackedConnectome.load(args.pack)
        manifest = run_sugar_experiment(
            connectome,
            args.output,
            duration_ms=args.duration_ms,
            rate_hz=args.rate_hz,
            seed=args.seed,
            propagation=args.propagation,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
