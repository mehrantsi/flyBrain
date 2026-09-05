from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    assets = project / "assets/neuromechfly"
    native = json.loads((assets / "manifest.json").read_text())
    model = ET.parse(assets / "fly.xml").getroot()
    files = {
        "manifest.json", "fly.xml", "aerodynamics.json", "tripod_gait.json", "habitat.json",
        "male_cns_v1_neural_io.json", "flywire_v783_neural_io.json",
        "vision/ommatidia_id_map_u16le.bin", "vision/pale_mask_u8.bin",
    }
    files.update(element.get("file") for element in model.findall("asset/*") if element.get("file"))
    hashes = {}
    for name in sorted(files):
        digest = hashlib.sha256((assets / name).read_bytes()).hexdigest()
        expected = native["files"].get(name)
        if expected is not None and digest != expected:
            raise ValueError(f"Native runtime asset hash mismatch: {name}")
        hashes[name] = digest
    destination = project / "web/dist/runtime-assets.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema": "flybrain.browser-runtime-assets.v1", "files": hashes}, indent=2) + "\n")
    print(f"Packaged {len(files)} hashed runtime assets")


if __name__ == "__main__":
    main()
