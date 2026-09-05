from __future__ import annotations

import argparse
import json
from pathlib import Path

from flybrain.male_cns import import_male_cns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile the published MaleCNS v1.0 tables into CSR")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--materialization",
        default="male-cns-v1.0-superclass-non-null-known-nt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = import_male_cns(
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectivity_path=args.connectivity,
        output_path=args.output,
        materialization=args.materialization,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
