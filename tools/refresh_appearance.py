from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from fly_appearance import improve_fly_appearance
from habitat_assets import improve_habitat_appearance


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate cosmetic meshes/materials without re-exporting the physical fly")
    parser.add_argument("--assets", type=Path, default=Path("assets/neuromechfly"))
    args = parser.parse_args()
    assets = args.assets.resolve()
    source = Path(__file__).resolve().parents[1] / "assets/materials"
    shutil.copytree(source, assets / "textures", dirs_exist_ok=True)
    path = assets / "fly.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            name = child.get("name", "")
            if name.startswith(("detail/", "detail_", "fly/detail_")):
                parent.remove(child)
    improve_habitat_appearance(root)
    improve_fly_appearance(root, assets)
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in [path, *sorted((assets / "textures").glob("*"))]:
        if artifact.is_file():
            manifest["files"][str(artifact.relative_to(assets))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Refreshed appearance: {path}")


if __name__ == "__main__":
    main()
