import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.habitat_assets import HABITAT_CONTACT_ATTRIBUTES, add_habitat


ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = ROOT / "assets" / "neuromechfly" / "fly.xml"


def _minimal_exported_root() -> ET.Element:
    root = ET.Element("mujoco")
    ET.SubElement(root, "option")
    ET.SubElement(root, "asset")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "map")
    ET.SubElement(root, "statistic")
    ET.SubElement(root, "worldbody")
    ET.SubElement(root, "contact")
    return root


def test_habitat_ccd_budget_matches_checked_in_model():
    root = _minimal_exported_root()
    add_habitat(root)
    checked_in = ET.parse(ASSET_PATH).getroot()
    for model in (root, checked_in):
        option = model.find("option")
        assert option.get("ccd_iterations") == "100"
        assert option.get("ccd_tolerance") is None


def test_habitat_manifest_hashes_match_current_assets():
    manifest = json.loads((ASSET_PATH.parent / "manifest.json").read_text())
    for name in ("fly.xml", "habitat.json"):
        assert manifest["files"][name] == hashlib.sha256(
            (ASSET_PATH.parent / name).read_bytes()
        ).hexdigest()


def test_generator_applies_floor_contact_calibration_to_habitat_supports():
    root = _minimal_exported_root()
    add_habitat(root)

    geoms = [
        geom
        for geom in root.findall("worldbody/geom")
        if geom.get("conaffinity") == "1"
    ]
    assert len(geoms) == 34
    for geom in geoms:
        for attribute, value in HABITAT_CONTACT_ATTRIBUTES.items():
            assert geom.get(attribute) == value


def test_checked_in_habitat_contact_calibration_matches_generator():
    root = ET.parse(ASSET_PATH).getroot()
    geoms = [
        geom
        for geom in root.findall("worldbody/geom")
        if geom.get("conaffinity") == "1"
    ]
    assert len(geoms) == 34
    for geom in geoms:
        for attribute, value in HABITAT_CONTACT_ATTRIBUTES.items():
            assert geom.get(attribute) == value
