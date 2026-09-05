# ruff: noqa: I001

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.flight_assets import _quaternion_multiply, _quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "neuromechfly"


def _vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _quat_equivalent(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-6) -> bool:
    return min(np.max(np.abs(left - right)), np.max(np.abs(left + right))) < tolerance


def test_flybody_reframed_wing_meshes_preserve_world_pose():
    root = ET.parse(ASSETS / "fly.xml").getroot()
    expected_old = {
        "left": np.array([0.649074, 0.0360041, -0.0650074, 0.757087]),
        "right": np.array([0.649074, -0.0360041, -0.0650074, -0.757087]),
    }
    expected_positions = {
        "left": np.array([-0.674, 0.395, 0.081]),
        "right": np.array([-0.674, -0.395, 0.081]),
    }
    expected_frames = {
        "left": np.array([0.0, -0.40274668985873724, 0.0, -0.9153114791194471]),
        "right": np.array([0.0, -0.9153114791194471, 0.0, 0.40274668985873724]),
    }
    for side, old_quat in expected_old.items():
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        mesh = root.find(f".//geom[@name='{body_name}']")
        assert body is not None and mesh is not None
        body_quat = _vector(body.get("quat", "1 0 0 0"))
        mesh_quat = _vector(mesh.get("quat", "1 0 0 0"))
        assert _quat_equivalent(body_quat, expected_frames[side])
        assert _quat_equivalent(_quaternion_multiply(body_quat, mesh_quat), old_quat)
        assert np.allclose(_vector(body.get("pos", "0 0 0")), expected_positions[side])
        assert np.allclose(_vector(mesh.get("pos", "0 0 0")), np.zeros(3), atol=1e-12)


def test_official_fluid_geometry_is_thin_and_world_mirrored():
    root = ET.parse(ASSETS / "fly.xml").getroot()
    expected_size = np.array([0.005, 0.551, 1.14])
    world_positions = {}
    for side in ("left", "right"):
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        fluid = root.find(f".//geom[@name='{body_name}-fluid']")
        assert body is not None and fluid is not None
        assert np.allclose(_vector(fluid.get("size", "")), expected_size)
        assert fluid.get("fluidshape") == "ellipsoid"
        assert fluid.get("fluidcoef") == "1.0 0.5 1.5 1.7 1.0"
        mesh = root.find(f".//geom[@name='{body_name}']")
        inertial = root.find(f".//geom[@name='{body_name}-inertial']")
        assert mesh is not None and inertial is not None
        assert float(mesh.get("mass")) == 0.0
        assert inertial.get("type") == "box"
        assert np.allclose(_vector(inertial.get("size", "")), expected_size)
        assert math.isclose(float(inertial.get("mass")), 8e-6, abs_tol=1e-15)
        body_pos = _vector(body.get("pos", "0 0 0"))
        body_quat = _vector(body.get("quat", "1 0 0 0"))
        local_pos = _vector(fluid.get("pos", "0 0 0"))
        world_positions[side] = body_pos + _quaternion_rotate(body_quat, local_pos)
    assert np.isclose(world_positions["left"][0], world_positions["right"][0], atol=1e-6)
    assert np.isclose(world_positions["left"][2], world_positions["right"][2], atol=1e-6)
    assert np.isclose(world_positions["left"][1], -world_positions["right"][1], atol=1e-6)


def test_wing_axes_and_task_mechanics_are_pinned():
    root = ET.parse(ASSETS / "fly.xml").getroot()
    expected_axes = {
        "yaw": "0.0 0.0 1.0",
        "pitch": "0.0 1.0 0.0",
        "roll": "1.0 0.0 0.0",
    }
    actuators = root.find("actuator")
    assert actuators is not None
    wing_actuators = actuators.findall("general")[50:56]
    assert len(wing_actuators) == 6
    for index, actuator in enumerate(wing_actuators):
        side = "l" if index < 3 else "r"
        axis = ("yaw", "pitch", "roll")[index % 3]
        joint_name = f"fly/c_thorax-{side}_wing-{axis}"
        joint = root.find(f".//joint[@name='{joint_name}']")
        assert joint is not None
        assert joint.get("axis") == expected_axes[axis]
        assert float(joint.get("stiffness")) == 1.0
        assert math.isclose(float(joint.get("damping")), 0.776923, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(float(joint.get("armature")), 1e-4, rel_tol=0, abs_tol=1e-12)
        assert float(actuator.get("gainprm")) == 1800.0
        assert actuator.get("name") == f"{joint_name}-flight-position"
    for side in ("l", "r"):
        body = root.find(f".//body[@name='fly/{side}_wing']")
        assert body is not None
        joint_axes = [
            joint.get("name").rsplit("-", 1)[-1]
            for joint in body.findall("joint")
        ]
        assert joint_axes == ["yaw", "roll", "pitch"]


def test_strip_frames_are_right_handed():
    metadata = json.loads((ASSETS / "aerodynamics.json").read_text())
    for wing in metadata["wings"]:
        for strip in wing["strips"]:
            chord = np.asarray(strip["chord_hat_local"], dtype=np.float64)
            span = np.asarray(strip["span_hat_local"], dtype=np.float64)
            normal = np.asarray(strip["normal_hat_local"], dtype=np.float64)
            assert np.isclose(np.dot(np.cross(chord, span), normal), 1.0, atol=1e-6)


def test_fourier_and_policy_artifact_provenance_survives_export():
    metadata = json.loads((ASSETS / "aerodynamics.json").read_text())
    assert metadata["wingbeat"]["frequency_hz"] == 218.0
    assert metadata["wingbeat"]["waveform"]["type"] == "flybody_fourier"
    assert metadata["model"]["flight_pose"]["body_pitch_deg"] == 47.5

    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for filename in (
        "flywire_v783_neural_io.json",
        "flybody_flight_policy_v1.json",
        "flybody_flight_policy_v1.f32le",
        "flybody_flight_policy_fixture_v1.json",
    ):
        path = ASSETS / filename
        assert path.is_file()
        assert manifest["files"][filename] == hashlib.sha256(path.read_bytes()).hexdigest()
