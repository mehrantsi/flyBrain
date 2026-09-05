from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HEIGHT = 512
WIDTH = 450
OMMATIDIA = 721
DISTORTION_COEFFICIENT = 3.8
ZOOM = 2.72
COVERED_PIXELS = 170_288
PALE_OMMATIDIA = 216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flygym-repo", type=Path, default=Path("work/upstream/flygym")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("assets/neuromechfly/vision")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_retina_assets(repo: Path, output: Path, force: bool = False) -> dict:
    repo = repo.resolve()
    output = output.resolve()
    if output == Path(output.anchor) or output == Path.home():
        raise ValueError(f"refusing unsafe output directory: {output}")

    source = repo / "src/flygym/assets/model/neuromechfly/vision"
    id_map = np.load(source / "ommatidia_id_map.npy", allow_pickle=False)
    pale_mask = np.load(source / "pale_mask.npy", allow_pickle=False)
    if id_map.dtype != np.uint16 or id_map.shape != (HEIGHT, WIDTH):
        raise ValueError(f"unexpected ommatidia map: {id_map.dtype} {id_map.shape}")
    if pale_mask.dtype != np.bool_ or pale_mask.shape != (OMMATIDIA,):
        raise ValueError(f"unexpected pale mask: {pale_mask.dtype} {pale_mask.shape}")
    if not np.array_equal(np.unique(id_map), np.arange(OMMATIDIA + 1)):
        raise ValueError("ommatidia IDs are not contiguous 0..721")
    if int(np.count_nonzero(id_map)) != COVERED_PIXELS:
        raise ValueError("ommatidia map covered-pixel count changed")
    if int(np.count_nonzero(pale_mask)) != PALE_OMMATIDIA:
        raise ValueError("pale ommatidia count changed")

    map_bytes = id_map.astype("<u2", copy=False).tobytes(order="C")
    mask_bytes = pale_mask.astype(np.uint8, copy=False).tobytes(order="C")
    files = {
        "ommatidia_id_map_u16le.bin": map_bytes,
        "pale_mask_u8.bin": mask_bytes,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = output / name
        if path.exists() and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force")
        path.write_bytes(data)

    metadata = {
        "schema": "flygym-retina-v1",
        "height": HEIGHT,
        "width": WIDTH,
        "ommatidia_per_eye": OMMATIDIA,
        "fovy_degrees": 157.0,
        "fisheye_distortion_coefficient": DISTORTION_COEFFICIENT,
        "fisheye_zoom": ZOOM,
        "covered_pixels": COVERED_PIXELS,
        "pale_ommatidia": PALE_OMMATIDIA,
        "yellow_ommatidia": OMMATIDIA - PALE_OMMATIDIA,
        "files": {name: sha256_bytes(data) for name, data in files.items()},
        "source": {
            "repository": "https://github.com/NeLy-EPFL/flygym",
            "tag": subprocess.check_output(
                ["git", "-C", str(repo), "describe", "--tags", "--exact-match"],
                text=True,
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "license": "Apache-2.0",
            "fisheye_lineage": "Gil-Mor/iFish (MIT), via FlyGym Retina",
        },
    }
    manifest = output / "manifest.json"
    if manifest.exists() and not force:
        raise FileExistsError(f"output already exists: {manifest}; pass --force")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    args = parse_args()
    metadata = export_retina_assets(args.flygym_repo, args.output, args.force)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
