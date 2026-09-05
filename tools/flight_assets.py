from __future__ import annotations

import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

WING_JOINTS = (
    {
        "side": "left",
        "body": "fly/l_wing",
        "body_frame": "fly/l_wing",
        "joint": "fly/c_thorax-l_wing-yaw",
        "axis": "yaw",
        "range_rad": [-1.5, 1.5],
        "springref_rad": 1.5,
        "flybody_axis": [0.0, 0.0, 1.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
    {
        "side": "left",
        "body": "fly/l_wing",
        "body_frame": "fly/l_wing",
        "joint": "fly/c_thorax-l_wing-pitch",
        "axis": "pitch",
        "range_rad": [-1.27, 2.92],
        "springref_rad": -1.0,
        "flybody_axis": [0.0, 1.0, 0.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
    {
        "side": "left",
        "body": "fly/l_wing",
        "body_frame": "fly/l_wing",
        "joint": "fly/c_thorax-l_wing-roll",
        "axis": "roll",
        "range_rad": [-1.0, 1.5],
        "springref_rad": 0.7,
        "flybody_axis": [1.0, 0.0, 0.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
    {
        "side": "right",
        "body": "fly/r_wing",
        "body_frame": "fly/r_wing",
        "joint": "fly/c_thorax-r_wing-yaw",
        "axis": "yaw",
        "range_rad": [-1.5, 1.5],
        "springref_rad": 1.5,
        "flybody_axis": [0.0, 0.0, 1.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
    {
        "side": "right",
        "body": "fly/r_wing",
        "body_frame": "fly/r_wing",
        "joint": "fly/c_thorax-r_wing-pitch",
        "axis": "pitch",
        "range_rad": [-1.27, 2.92],
        "springref_rad": -1.0,
        "flybody_axis": [0.0, 1.0, 0.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
    {
        "side": "right",
        "body": "fly/r_wing",
        "body_frame": "fly/r_wing",
        "joint": "fly/c_thorax-r_wing-roll",
        "axis": "roll",
        "range_rad": [-1.0, 1.5],
        "springref_rad": 0.7,
        "flybody_axis": [1.0, 0.0, 0.0],
        "flybody_gain": 18.0,
        "gain": 1800.0,
        "stiffness": 1.0,
        "damping": 0.776923,
        "armature": 0.0001,
    },
)

WING_ACTUATOR_NAMES = tuple(f"{item['joint']}-flight-position" for item in WING_JOINTS)
WING_MESH_SCALE = 1000.0
STRIPS_PER_WING = 12
FLYBODY_DENSITY_G_PER_CM3 = 0.00128
FLYBODY_VISCOSITY_G_PER_CM_S = 0.000185
AIR_DENSITY_G_PER_MM3 = FLYBODY_DENSITY_G_PER_CM3 / 1000.0
AIR_VISCOSITY_G_PER_MM_S = FLYBODY_VISCOSITY_G_PER_CM_S / 10.0
FLYBODY_FLUIDCOEF = [1.0, 0.5, 1.5, 1.7, 1.0]
FLYBODY_SOURCE_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
FLYBODY_WING_PATTERN_FILENAME = "datasets_flight-imitation/wing_pattern_fmech.npy"
FLYBODY_WING_PATTERN_SOURCE_SHA256 = "f97b975ef1b5adbe42c208ea9665f3c37fa432914f56f3943f1697cdb090d1ce"
FLYBODY_WING_PATTERN_DOI = "10.1038/s41586-025-09029-4"
FLYBODY_DATA_DOI = "10.25378/janelia.25309105.v4"
FLYBODY_DATA_LICENSE = "GPL-3.0-or-later"
FLYBODY_DATA_LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
FLYBODY_WING_PATTERN_HARMONICS = 12
FLYBODY_WING_PATTERN_FREQUENCY_HZ = 218.0
FLYBODY_WING_PATTERN_SOURCE_COLUMNS = ("yaw", "roll", "pitch")
FLYBODY_WING_PATTERN_PROJECT_AXES = ("yaw", "pitch", "roll")
FLYBODY_WING_JOINT_ORDER = ("yaw", "roll", "pitch")
FLYBODY_BODY_PITCH_DEG = 47.5
FLYBODY_STROKE_PLANE_DEG = 0.0
FLYBODY_FLIGHT_PHYSICS_TIMESTEP_S = 5e-5
PROJECT_PHYSICS_TIMESTEP_S = 1e-4
FLYBODY_FLUID_SIZE_MM = [0.005, 0.551, 1.14]
FLYBODY_WING_INERTIAL_MASS_G = 8e-6
FLYBODY_FLUID_GEOMETRY = {
    "left": {
        "body_quat_wxyz": [0.0, -0.403, 0.0, -0.915],
        "pos_cm": [0.0263, -0.148, -0.0289],
        "quat_wxyz": [-0.685, -0.634, 0.265, -0.243],
    },
    "right": {
        "body_quat_wxyz": [0.0, 0.915, 0.0, -0.403],
        "pos_cm": [-0.0263, 0.148, 0.0289],
        "quat_wxyz": [0.243, 0.265, 0.634, -0.685],
    },
}


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("quaternion must have a finite non-zero norm")
    return quaternion / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    )


def _quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, pure), _quaternion_conjugate(quaternion)
    )[1:]


def _rotation_y(angle_rad: float) -> np.ndarray:
    return np.array(
        [math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0],
        dtype=np.float64,
    )


def _flybody_wing_frame_quaternions(
    body_pitch_deg: float = FLYBODY_BODY_PITCH_DEG,
    stroke_plane_deg: float = FLYBODY_STROKE_PLANE_DEG,
) -> dict[str, np.ndarray]:
    body_pitch = math.radians(body_pitch_deg)
    stroke_plane = math.radians(stroke_plane_deg)
    up_dir = _rotation_y(body_pitch)
    stroke_plane_quat = _rotation_y(stroke_plane)
    left_seed = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    right_seed = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    left = _quaternion_multiply(
        _quaternion_multiply(_quaternion_conjugate(stroke_plane_quat), left_seed),
        _quaternion_conjugate(up_dir),
    )
    right = _quaternion_multiply(
        _quaternion_multiply(_quaternion_conjugate(stroke_plane_quat), right_seed),
        _quaternion_conjugate(up_dir),
    )
    return {
        "left": _normalize_quaternion(left),
        "right": _normalize_quaternion(right),
    }


def _format_vector(values: np.ndarray | list[float]) -> str:
    return " ".join(str(float(value)) for value in values)


def _set_child_frame(child: ET.Element, frame_delta: np.ndarray) -> None:
    if "pos" in child.attrib or child.tag in {"geom", "joint", "site", "camera", "body"}:
        position = np.asarray(_float_list(child.get("pos", "0 0 0")), dtype=np.float64)
        child.set("pos", _format_vector(_quaternion_rotate(frame_delta, position)))
    if "quat" in child.attrib or child.tag in {"geom", "site", "camera", "body"}:
        child_quat = np.asarray(_float_list(child.get("quat", "1 0 0 0")), dtype=np.float64)
        child.set(
            "quat",
            _format_vector(_normalize_quaternion(_quaternion_multiply(frame_delta, child_quat))),
        )


def apply_flybody_wing_frames(
    root: ET.Element,
    body_pitch_deg: float = FLYBODY_BODY_PITCH_DEG,
    stroke_plane_deg: float = FLYBODY_STROKE_PLANE_DEG,
) -> dict[str, dict[str, list[float]]]:
    target_frames = _flybody_wing_frame_quaternions(body_pitch_deg, stroke_plane_deg)
    metadata = {}
    for side in ("left", "right"):
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"exported model is missing wing body {body_name}")
        old_quat = _normalize_quaternion(
            np.asarray(_float_list(body.get("quat", "1 0 0 0")), dtype=np.float64)
        )
        new_quat = target_frames[side]
        frame_delta = _normalize_quaternion(
            _quaternion_multiply(_quaternion_conjugate(new_quat), old_quat)
        )
        for child in list(body):
            _set_child_frame(child, frame_delta)
        body.set("quat", _format_vector(new_quat))
        metadata[side] = {
            "body_name": body_name,
            "body_pitch_deg": body_pitch_deg,
            "stroke_plane_deg": stroke_plane_deg,
            "old_body_quat_wxyz": [float(value) for value in old_quat],
            "new_body_quat_wxyz": [float(value) for value in new_quat],
            "child_frame_delta_wxyz": [float(value) for value in frame_delta],
            "position_preserved": True,
        }
    return metadata


def _official_fluid_ellipsoid_geometry(side: str) -> dict:
    source = FLYBODY_FLUID_GEOMETRY[side]
    target = _flybody_wing_frame_quaternions()[side]
    source_body_quat = _normalize_quaternion(np.asarray(source["body_quat_wxyz"], dtype=np.float64))
    frame_delta = _normalize_quaternion(
        _quaternion_multiply(_quaternion_conjugate(target), source_body_quat)
    )
    position_mm = _quaternion_rotate(
        frame_delta, np.asarray(source["pos_cm"], dtype=np.float64) * 10.0
    )
    geom_quat = _normalize_quaternion(
        _quaternion_multiply(
            frame_delta, np.asarray(source["quat_wxyz"], dtype=np.float64)
        )
    )
    return {
        "geom_name": f"fly/{'l' if side == 'left' else 'r'}_wing-fluid",
        "pos_local_mm": [float(value) for value in position_mm],
        "quat_wxyz": [float(value) for value in geom_quat],
        "size_mm": list(FLYBODY_FLUID_SIZE_MM),
        "fluidshape": "ellipsoid",
        "fluidcoef": FLYBODY_FLUIDCOEF,
        "fit": "official FlyBody wing-fluid ellipsoid, cm-g-s geometry converted to NMF mm-g-s and reframed",
        "source_body_quat_wxyz": source["body_quat_wxyz"],
        "source_pos_cm": source["pos_cm"],
        "source_quat_wxyz": source["quat_wxyz"],
    }


def add_flight_actuators(root: ET.Element) -> dict[str, dict[str, list[float]]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("exported model has no worldbody")
    frame_metadata = apply_flybody_wing_frames(root)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = actuator.findall("general")
    if len(existing) != 50:
        raise RuntimeError(
            f"expected 50 pre-flight actuators after feeding extension, found {len(existing)}"
        )
    existing_names = {item.get("name") for item in existing}
    for index, item in enumerate(WING_JOINTS):
        joint = root.find(f".//joint[@name='{item['joint']}']")
        if joint is None:
            raise RuntimeError(f"exported model is missing wing joint {item['joint']}")
        joint.set("limited", "true")
        joint.set("range", " ".join(str(value) for value in item["range_rad"]))
        joint.set("springref", str(item["springref_rad"]))
        joint.set("stiffness", str(item["stiffness"]))
        joint.set("damping", str(item["damping"]))
        joint.set("armature", str(item["armature"]))
        joint.set("axis", _format_vector(item["flybody_axis"]))
        name = WING_ACTUATOR_NAMES[index]
        if name in existing_names:
            raise RuntimeError(f"duplicate wing actuator {name}")
        gain = item["gain"]
        ET.SubElement(
            actuator,
            "general",
            {
                "name": name,
                "joint": item["joint"],
                "ctrllimited": "true",
                "ctrlrange": " ".join(str(value) for value in item["range_rad"]),
                "biastype": "affine",
                "gainprm": str(gain),
                "biasprm": f"0 {-gain}",
            },
        )
    for side in ("left", "right"):
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"exported model is missing wing body {body_name}")
        joints_by_axis = {
            item["axis"]: root.find(f".//joint[@name='{item['joint']}']")
            for item in WING_JOINTS
            if item["side"] == side
        }
        if any(joint is None for joint in joints_by_axis.values()):
            raise RuntimeError(f"exported model has an incomplete {side} wing joint chain")
        for joint in joints_by_axis.values():
            body.remove(joint)
        for index, axis in enumerate(FLYBODY_WING_JOINT_ORDER):
            body.insert(index, joints_by_axis[axis])
    return frame_metadata


def _read_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"wing mesh is too short to be an STL: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    record_size = 50
    expected_size = 84 + record_size * triangle_count
    if expected_size != len(payload):
        raise ValueError(
            f"wing mesh is not the expected binary STL: {path} ({len(payload)} != {expected_size})"
        )
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (9,)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(payload, dtype=dtype, offset=84, count=triangle_count)
    triangles = records["vertices"].reshape(triangle_count, 3, 3).astype(np.float64)
    return triangles * WING_MESH_SCALE


def _body_frame(root: ET.Element, body_name: str) -> dict:
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise RuntimeError(f"exported model is missing wing body {body_name}")
    return {
        "body_name": body_name,
        "pos_mm": _float_list(body.get("pos", "0 0 0")),
        "quat_wxyz": _float_list(body.get("quat", "1 0 0 0")),
    }


def _strip_geometry(triangles: np.ndarray) -> dict:
    tri_centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangle_surface_area = 0.5 * np.linalg.norm(cross, axis=1)
    surface_area = float(triangle_surface_area.sum())
    if surface_area <= 0.0:
        raise ValueError("wing mesh has no surface area")
    planform_area = surface_area / 2.0
    span_min = min(0.0, float(triangles[:, :, 1].min()))
    span_max = float(triangles[:, :, 1].max())
    span = span_max - span_min
    edges = np.linspace(span_min, span_max, STRIPS_PER_WING + 1)
    strip_area = np.zeros(STRIPS_PER_WING, dtype=np.float64)
    strip_centers = np.zeros((STRIPS_PER_WING, 3), dtype=np.float64)
    strip_x_min = np.full(STRIPS_PER_WING, np.inf)
    strip_x_max = np.full(STRIPS_PER_WING, -np.inf)
    for triangle_index, center in enumerate(tri_centers):
        strip_index = int(np.searchsorted(edges, center[1], side="right") - 1)
        strip_index = min(max(strip_index, 0), STRIPS_PER_WING - 1)
        area = triangle_surface_area[triangle_index] / 2.0
        strip_area[strip_index] += area
        strip_centers[strip_index] += area * center
        strip_x_min[strip_index] = min(
            strip_x_min[strip_index], float(triangles[triangle_index, :, 0].min())
        )
        strip_x_max[strip_index] = max(
            strip_x_max[strip_index], float(triangles[triangle_index, :, 0].max())
        )
    if np.any(strip_area <= 0.0):
        raise ValueError(f"wing mesh produced an empty strip: {strip_area.tolist()}")
    strip_area *= planform_area / float(strip_area.sum())
    strip_centers /= strip_area[:, None]
    strips = []
    for index, area in enumerate(strip_area):
        r_inner = float(edges[index] - span_min)
        r_outer = float(edges[index + 1] - span_min)
        r_mid = 0.5 * (r_inner + r_outer)
        dr = r_outer - r_inner
        x_quarter = float(strip_x_min[index] + 0.25 * (strip_x_max[index] - strip_x_min[index]))
        strips.append(
            {
                "index": index,
                "r_inner_mm": r_inner,
                "r_outer_mm": r_outer,
                "r_mid_mm": r_mid,
                "dr_mm": dr,
                "chord_mm": float(area / dr),
                "area_mm2": float(area),
                "position_local_mm": [float(value) for value in strip_centers[index]],
                "quarter_chord_local_mm": [
                    x_quarter,
                    float(strip_centers[index, 1]),
                    float(strip_centers[index, 2]),
                ],
                "chord_bounds_local_x_mm": [
                    float(strip_x_min[index]),
                    float(strip_x_max[index]),
                ],
            }
        )
    projected_area = float(np.abs(cross[:, 2]).sum() / 2.0)
    return {
        "mesh_triangles": int(triangles.shape[0]),
        "span_min_local_y_mm": span_min,
        "span_max_local_y_mm": span_max,
        "span_mm": span,
        "planform_area_mm2": planform_area,
        "projected_area_xy_mm2": projected_area,
        "area_definition": "half of closed mesh surface area; projected area is reported separately",
        "span_axis_local": [0.0, 1.0, 0.0],
        "chord_axis_local": [1.0, 0.0, 0.0],
        "normal_axis_local": [0.0, 0.0, 1.0],
        "strips": strips,
    }


def _mesh_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_fourier_waveform(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"published FlyBody wing pattern is missing: {path}")
    source_sha256 = _mesh_sha256(path)
    if source_sha256 != FLYBODY_WING_PATTERN_SOURCE_SHA256:
        raise ValueError(
            f"unexpected FlyBody wing pattern SHA256: {source_sha256} != "
            f"{FLYBODY_WING_PATTERN_SOURCE_SHA256}"
        )
    samples = np.load(path, allow_pickle=False)
    if samples.shape != (500, 3) or samples.dtype.kind != "f":
        raise ValueError(f"FlyBody wing pattern must be a floating (500, 3) array: {samples.shape}")
    samples = samples.astype(np.float64, copy=False)
    sample_count = samples.shape[0]
    sample_phase = 2.0 * math.pi * np.arange(sample_count) / sample_count
    coefficients = {}
    fit = {}
    centers = {}
    amplitudes = {}
    dense_phase = np.linspace(0.0, 2.0 * math.pi, 100_000, endpoint=False)
    for project_axis_index, axis in enumerate(FLYBODY_WING_PATTERN_PROJECT_AXES):
        source_column = FLYBODY_WING_PATTERN_SOURCE_COLUMNS.index(axis)
        values = samples[:, source_column]
        offset = float(values.mean())
        cosine = [
            float(
                2.0
                / sample_count
                * np.sum(values * np.cos(harmonic * sample_phase))
            )
            for harmonic in range(1, FLYBODY_WING_PATTERN_HARMONICS + 1)
        ]
        sine = [
            float(
                2.0
                / sample_count
                * np.sum(values * np.sin(harmonic * sample_phase))
            )
            for harmonic in range(1, FLYBODY_WING_PATTERN_HARMONICS + 1)
        ]
        reconstructed = offset + sum(
            cosine[harmonic - 1] * np.cos(harmonic * sample_phase)
            + sine[harmonic - 1] * np.sin(harmonic * sample_phase)
            for harmonic in range(1, FLYBODY_WING_PATTERN_HARMONICS + 1)
        )
        dense_reconstructed = offset + sum(
            cosine[harmonic - 1] * np.cos(harmonic * dense_phase)
            + sine[harmonic - 1] * np.sin(harmonic * dense_phase)
            for harmonic in range(1, FLYBODY_WING_PATTERN_HARMONICS + 1)
        )
        error = reconstructed - values
        coefficients[axis] = {
            "offset_rad": offset,
            "cos_rad": cosine,
            "sin_rad": sine,
        }
        fit[axis] = {
            "max_abs_error_rad": float(np.max(np.abs(error))),
            "rmse_rad": float(np.sqrt(np.mean(error * error))),
            "source_min_rad": float(np.min(values)),
            "source_max_rad": float(np.max(values)),
            "fit_min_rad": float(np.min(reconstructed)),
            "fit_max_rad": float(np.max(reconstructed)),
        }
        centers[axis] = offset
        amplitudes[axis] = float(np.max(np.abs(dense_reconstructed - offset)))
    all_errors = np.concatenate(
        [
            np.asarray(
                [
                    coefficients[axis]["offset_rad"]
                    + sum(
                        coefficients[axis]["cos_rad"][harmonic - 1]
                        * np.cos(harmonic * sample_phase)
                        + coefficients[axis]["sin_rad"][harmonic - 1]
                        * np.sin(harmonic * sample_phase)
                        for harmonic in range(1, FLYBODY_WING_PATTERN_HARMONICS + 1)
                    )
                    - samples[:, FLYBODY_WING_PATTERN_SOURCE_COLUMNS.index(axis)]
                ]
            )
            for axis in FLYBODY_WING_PATTERN_PROJECT_AXES
        ]
    )
    fit_error = {
        "max_abs_error_rad": float(max(item["max_abs_error_rad"] for item in fit.values())),
        "rmse_rad": float(np.sqrt(np.mean(all_errors * all_errors))),
        "per_axis": fit,
    }
    return {
        "frequency_hz": FLYBODY_WING_PATTERN_FREQUENCY_HZ,
        "waveform": {
            "type": "flybody_fourier",
            "fourier": {
                "harmonics": FLYBODY_WING_PATTERN_HARMONICS,
                "sample_count": sample_count,
                "source_columns": list(FLYBODY_WING_PATTERN_SOURCE_COLUMNS),
                "project_axes": list(FLYBODY_WING_PATTERN_PROJECT_AXES),
                "coefficient_convention": (
                    "angle(axis, phase) = offset + sum(cos[n]*cos(n*phase) + "
                    "sin[n]*sin(n*phase)); phase is cycles at the published beat frequency"
                ),
                "offset_rad_by_axis": {
                    axis: coefficients[axis]["offset_rad"]
                    for axis in FLYBODY_WING_PATTERN_PROJECT_AXES
                },
                "cos_rad_by_axis": {
                    axis: coefficients[axis]["cos_rad"]
                    for axis in FLYBODY_WING_PATTERN_PROJECT_AXES
                },
                "sin_rad_by_axis": {
                    axis: coefficients[axis]["sin_rad"]
                    for axis in FLYBODY_WING_PATTERN_PROJECT_AXES
                },
                "provenance": {
                    "source_file": FLYBODY_WING_PATTERN_FILENAME,
                    "source_sha256": source_sha256,
                    "doi": FLYBODY_WING_PATTERN_DOI,
                    "figshare_doi": FLYBODY_DATA_DOI,
                    "data_license": FLYBODY_DATA_LICENSE,
                    "license_url": FLYBODY_DATA_LICENSE_URL,
                    "flybody_commit": FLYBODY_SOURCE_COMMIT,
                    "fit_error": fit_error,
                },
            },
        },
        "fit_error": fit_error,
        "center_rad_by_axis": centers,
        "amplitude_rad_by_axis": amplitudes,
    }


def _rotation_to_quaternion(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        largest = int(np.argmax(diagonal))
        if largest == 0:
            scale = math.sqrt(max(1e-15, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif largest == 1:
            scale = math.sqrt(max(1e-15, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1e-15, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return [float(value) for value in quaternion]


def _fluid_ellipsoid_geometry(triangles: np.ndarray, side: str) -> dict:
    del triangles
    return _official_fluid_ellipsoid_geometry(side)


def add_flight_fluid_geoms(root: ET.Element, mesh_path: Path) -> list[dict]:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("density", str(AIR_DENSITY_G_PER_MM3))
    option.set("viscosity", str(AIR_VISCOSITY_G_PER_MM_S))
    if not mesh_path.exists():
        raise FileNotFoundError(f"exported wing mesh is missing: {mesh_path}")
    fluid_geometries = []
    for side in ("left", "right"):
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"exported model is missing wing body {body_name}")
        geometry = _official_fluid_ellipsoid_geometry(side)
        if root.find(f".//geom[@name='{geometry['geom_name']}']") is not None:
            raise RuntimeError(f"duplicate fluid geom {geometry['geom_name']}")
        mesh_geom = root.find(f".//geom[@name='{body_name}']")
        if mesh_geom is None:
            raise RuntimeError(f"exported model is missing wing mesh geom {body_name}")
        mesh_geom.set("mass", "0")
        ET.SubElement(
            body,
            "geom",
            {
                "name": geometry["geom_name"],
                "type": "ellipsoid",
                "pos": " ".join(str(value) for value in geometry["pos_local_mm"]),
                "quat": " ".join(str(value) for value in geometry["quat_wxyz"]),
                "size": " ".join(str(value) for value in geometry["size_mm"]),
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
                "fluidshape": geometry["fluidshape"],
                "fluidcoef": " ".join(str(value) for value in geometry["fluidcoef"]),
                "rgba": "0 0 0 0",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{body_name}-inertial",
                "type": "box",
                "pos": " ".join(str(value) for value in geometry["pos_local_mm"]),
                "quat": " ".join(str(value) for value in geometry["quat_wxyz"]),
                "size": " ".join(str(value) for value in geometry["size_mm"]),
                "mass": str(FLYBODY_WING_INERTIAL_MASS_G),
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
                "rgba": "0 0 0 0",
            },
        )
        fluid_geometries.append(geometry)
    return fluid_geometries


def _side_geometry(geometry: dict, mirrored: bool) -> dict:
    if not mirrored:
        return geometry
    side_geometry = dict(geometry)
    side_geometry["span_min_local_y_mm"] = -geometry["span_max_local_y_mm"]
    side_geometry["span_max_local_y_mm"] = -geometry["span_min_local_y_mm"]
    side_geometry["span_axis_local"] = [0.0, -1.0, 0.0]
    side_geometry["normal_axis_local"] = [0.0, 0.0, -1.0]
    side_geometry["strips"] = []
    for strip in geometry["strips"]:
        side_strip = dict(strip)
        side_strip["position_local_mm"] = [
            strip["position_local_mm"][0],
            -strip["position_local_mm"][1],
            strip["position_local_mm"][2],
        ]
        side_strip["quarter_chord_local_mm"] = [
            strip["quarter_chord_local_mm"][0],
            -strip["quarter_chord_local_mm"][1],
            strip["quarter_chord_local_mm"][2],
        ]
        side_strip["chord_bounds_local_x_mm"] = list(strip["chord_bounds_local_x_mm"])
        side_geometry["strips"].append(side_strip)
    return side_geometry


def _transform_side_geometry(geometry: dict, frame_delta: np.ndarray) -> dict:
    transformed = dict(geometry)
    transformed["span_axis_local"] = [
        float(value) for value in _quaternion_rotate(frame_delta, np.asarray(geometry["span_axis_local"]))
    ]
    transformed["chord_axis_local"] = [
        float(value) for value in _quaternion_rotate(frame_delta, np.asarray(geometry["chord_axis_local"]))
    ]
    transformed["normal_axis_local"] = [
        float(value) for value in _quaternion_rotate(frame_delta, np.asarray(geometry["normal_axis_local"]))
    ]
    transformed["strips"] = []
    for strip in geometry["strips"]:
        side_strip = dict(strip)
        side_strip["position_local_mm"] = [
            float(value)
            for value in _quaternion_rotate(
                frame_delta, np.asarray(strip["position_local_mm"], dtype=np.float64)
            )
        ]
        side_strip["quarter_chord_local_mm"] = [
            float(value)
            for value in _quaternion_rotate(
                frame_delta, np.asarray(strip["quarter_chord_local_mm"], dtype=np.float64)
            )
        ]
        side_strip["chord_bounds_local_x_mm"] = list(strip["chord_bounds_local_x_mm"])
        transformed["strips"].append(side_strip)
    return transformed


def _mesh_frame_delta(root: ET.Element, body_name: str) -> np.ndarray:
    mesh_geom = root.find(f".//geom[@name='{body_name}']")
    if mesh_geom is None:
        raise RuntimeError(f"exported model is missing wing mesh geom {body_name}")
    return _normalize_quaternion(
        np.asarray(_float_list(mesh_geom.get("quat", "1 0 0 0")), dtype=np.float64)
    )


def _source_revision(repo: Path) -> dict:
    import subprocess

    return {
        "repository": "https://github.com/NeLy-EPFL/flygym",
        "tag": subprocess.check_output(
            ["git", "-C", str(repo), "describe", "--tags", "--exact-match"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def write_aerodynamics(
    path: Path,
    world_xml: Path,
    repo: Path,
    wing_pattern_path: Path | None = None,
) -> dict:
    world_root = ET.parse(world_xml).getroot()
    mesh_path = world_xml.parent / "l_wing.stl"
    if not mesh_path.exists():
        raise FileNotFoundError(f"exported wing mesh is missing: {mesh_path}")
    if wing_pattern_path is None:
        wing_pattern_path = Path("work/upstream/flybody-data") / FLYBODY_WING_PATTERN_FILENAME
    wing_pattern = _fit_fourier_waveform(wing_pattern_path)
    triangles = _read_binary_stl(mesh_path)
    geometry = _strip_geometry(triangles)
    wings = []
    for side in ("left", "right"):
        body_name = f"fly/{'l' if side == 'left' else 'r'}_wing"
        frame_delta = _mesh_frame_delta(world_root, body_name)
        side_geometry = _transform_side_geometry(
            _side_geometry(geometry, side == "right"), frame_delta
        )
        fluid_geometry = _official_fluid_ellipsoid_geometry(side)
        wing = {
            "side": side,
            "body": body_name,
            "body_frame": _body_frame(world_root, body_name),
            "mesh": {
                "file": "l_wing.stl",
                "sha256": _mesh_sha256(mesh_path),
                "source_mesh": "FlyGym neuromechfly l_wing.stl",
                "mirror_for_right": side == "right",
            },
            "fluid_ellipsoid": fluid_geometry,
            "mesh_body_frame_delta_wxyz": [float(value) for value in frame_delta],
            **side_geometry,
            "joint_names": [item["joint"] for item in WING_JOINTS if item["side"] == side],
            "actuator_names": [
                WING_ACTUATOR_NAMES[index]
                for index, item in enumerate(WING_JOINTS)
                if item["side"] == side
            ],
        }
        wing["strips"] = [
            {
                "index": strip["index"],
                "center_local_mm": strip["quarter_chord_local_mm"],
                "centroid_local_mm": strip["position_local_mm"],
                "span_hat_local": side_geometry["span_axis_local"],
                "chord_hat_local": side_geometry["chord_axis_local"],
                "normal_hat_local": side_geometry["normal_axis_local"],
                "chord_mm": strip["chord_mm"],
                "width_mm": strip["dr_mm"],
                "area_mm2": strip["area_mm2"],
                "r_inner_mm": strip["r_inner_mm"],
                "r_outer_mm": strip["r_outer_mm"],
                "r_mid_mm": strip["r_mid_mm"],
            }
            for strip in side_geometry["strips"]
        ]
        wings.append(wing)
    actuators = []
    xml_actuators = world_root.find("actuator")
    if xml_actuators is None:
        raise RuntimeError("exported model has no actuator section")
    xml_names = [item.get("name") for item in xml_actuators.findall("general")]
    for index, item in enumerate(WING_JOINTS):
        name = WING_ACTUATOR_NAMES[index]
        if name not in xml_names:
            raise RuntimeError(f"wing actuator was not emitted: {name}")
        actuator_index = xml_names.index(name)
        if actuator_index != 50 + index:
            raise RuntimeError(
                f"wing actuator {name} has index {actuator_index}, expected {50 + index}"
            )
        actuators.append(
            {
                "index": actuator_index,
                "name": name,
                "joint": item["joint"],
                "side": item["side"],
                "axis": item["axis"],
                "axis_local": item["flybody_axis"],
                "control_semantics": "desired_joint_angle_rad",
                "ctrlrange_rad": item["range_rad"],
                "joint_range_rad": item["range_rad"],
                "springref_rad": item["springref_rad"],
                "position_gain": item["gain"],
                "flybody_gain": item["flybody_gain"],
                "gain_conversion_factor": item["gain"] / item["flybody_gain"],
                "joint_stiffness": item["stiffness"],
                "joint_damping": item["damping"],
                "joint_armature": item["armature"],
                "source": "FlyBody flight-task wing defaults, converted for NMF mm-g-s dynamics",
            }
        )
    metadata = {
        "schema": "flybrain-aerodynamics-v1",
        "units": {
            "length": "mm",
            "time": "s",
            "mass": "g",
            "velocity": "mm/s",
            "density": "g/mm^3",
            "dynamic_viscosity": "g/(mm*s)",
            "force": "g*mm/s^2",
            "moment": "g*mm^2/s^2",
            "angle": "rad",
        },
        "model": {
            "name": "mujoco_ellipsoid",
            "backend": "mujoco_ellipsoid",
            "version": 1,
            "strips_per_wing": STRIPS_PER_WING,
            "wings": ["left", "right"],
            "force_frame": "world",
            "input_frame": "wing_body_local_then_world",
            "relative_velocity": "wing_strip_velocity_minus_air_velocity",
            "dynamic_pressure": "MuJoCo passive ellipsoid fluid model",
            "strip_force": "legacy translational_quasi_steady backend only",
            "strip_moment": "legacy translational_quasi_steady backend only",
            "air_velocity_mm_s": [0.0, 0.0, 0.0],
            "force_application": "MuJoCo qfrc_fluid from massless wing ellipsoid geoms; no xfrc_applied wing forces",
            "fluid_geom_names": [wing["fluid_ellipsoid"]["geom_name"] for wing in wings],
            "fluidcoef": FLYBODY_FLUIDCOEF,
            "flight_pose": {
                "body_pitch_deg": FLYBODY_BODY_PITCH_DEG,
                "stroke_plane_deg": FLYBODY_STROKE_PLANE_DEG,
                "physics_timestep_seconds": PROJECT_PHYSICS_TIMESTEP_S,
                "source_physics_timestep_seconds": FLYBODY_FLIGHT_PHYSICS_TIMESTEP_S,
                "timestep_note": "Project retains 0.1 ms physics steps for brain coupling; FlyBody flight used 0.05 ms.",
                "frame_construction": "inverse(stroke_plane_quat) * side_seed * inverse(hover_up_dir_quat)",
                "left_seed_quat_wxyz": [0.0, 0.0, 0.0, 1.0],
                "right_seed_quat_wxyz": [0.0, -1.0, 0.0, 0.0],
                "child_geometry_compensation": "child_pos_new=R(frame_delta)*child_pos_old; child_quat_new=frame_delta*child_quat_old",
                "child_world_pose_preserved_at_zero_qpos": True,
                "wing_inertial": {
                    "shape": "box",
                    "size_mm": list(FLYBODY_FLUID_SIZE_MM),
                    "mass_g": FLYBODY_WING_INERTIAL_MASS_G,
                    "visual_mesh_mass_g": 0.0,
                },
            },
            "wing_axis_convention": {
                "source_joint_order": ["yaw", "roll", "pitch"],
                "project_control_order": ["yaw", "pitch", "roll"],
                "physical_joint_order": list(FLYBODY_WING_JOINT_ORDER),
                "physical_local_axes": {
                    "yaw": [0.0, 0.0, 1.0],
                    "pitch": [0.0, 1.0, 0.0],
                    "roll": [1.0, 0.0, 0.0],
                },
            },
        },
        "air": {
            "rho_g_per_mm3": AIR_DENSITY_G_PER_MM3,
            "dynamic_viscosity_g_per_mm_s": AIR_VISCOSITY_G_PER_MM_S,
            "source_flybody_defaults": {
                "density_g_per_cm3": FLYBODY_DENSITY_G_PER_CM3,
                "viscosity_g_per_cm_s": FLYBODY_VISCOSITY_G_PER_CM_S,
                "gravity_cm_per_s2": -981.0,
            },
            "rho_kg_per_m3": FLYBODY_DENSITY_G_PER_CM3 * 1000.0,
            "viscosity_pa_s": FLYBODY_VISCOSITY_G_PER_CM_S * 0.1,
            "conversion_assumption": (
                "NMF uses millimetres, grams, and seconds; FlyBody's cm-g-s air defaults are "
                "converted by cm^3=1000 mm^3 and cm=10 mm."
            ),
        },
        "coefficients": {
            "source_fit": "Dickinson et al. 1999 translational force-coefficient fit",
            "lift": {
                "formula": "CL(alpha) = 0.225 + 1.58*sin(2.13*alpha - 0.125663706)",
                "offset": 0.225,
                "amplitude": 1.58,
                "angle_gain": 2.13,
                "phase_rad": -0.125663706,
            },
            "drag": {
                "formula": "CD(alpha) = 1.92 - 1.55*cos(2.04*alpha - 0.171391)",
                "offset": 1.92,
                "amplitude": -1.55,
                "angle_gain": 2.04,
                "phase_rad": -0.171391,
            },
            "moment": {
                "formula": "CM(alpha) = 0",
                "value": 0.0,
                "status": "explicit zero engineering assumption; not identified for this mesh",
            },
            "angle_domain_rad": [-math.pi / 2.0, math.pi / 2.0],
            "outside_domain": "clamp angle before evaluating source fit",
        },
        "wingbeat": {
            "frequency_hz": wing_pattern["frequency_hz"],
            "waveform": wing_pattern["waveform"],
            "joint_order": ["yaw", "pitch", "roll"],
            "phase_rad_by_side": {"left": 0.0, "right": 0.0},
            "center_rad_by_axis": wing_pattern["center_rad_by_axis"],
            "amplitude_rad_by_axis": wing_pattern["amplitude_rad_by_axis"],
            "mirror_roll_for_right": False,
            "joint_ranges_rad": {
                axis: item["range_rad"]
                for axis, item in ((item["axis"], item) for item in WING_JOINTS[:3])
            },
            "joint_dynamics": {
                axis: {
                    "stiffness": item["stiffness"],
                    "damping": item["damping"],
                    "armature": item["armature"],
                }
                for axis, item in ((item["axis"], item) for item in WING_JOINTS[:3])
            },
            "engineering_conversion": {
                "source": "FlyBody wing_pattern_fmech.npy compact Fourier fit",
                "length_scale_source_to_nmf": 10.0,
                "position_gain_scale": 100.0,
                "task_gainprm_source": [18.0, 18.0, 18.0],
                "task_stiffness_source": 0.01,
                "task_damping_source": 0.00776923,
                "task_armature_source": 1e-06,
                "task_gainprm_nmf": [1800.0, 1800.0, 1800.0],
                "task_stiffness_nmf": 1.0,
                "task_damping_nmf": 0.776923,
                "task_armature_nmf": 0.0001,
                "joint_axis_convention": "FlyBody wing yaw=z, pitch=y, roll=x; project actuator API remains yaw,pitch,roll.",
                "note": "Published source columns yaw,roll,pitch are projected to the project API order yaw,pitch,roll; physical hinge axes follow FlyBody while XML actuator names/order remain project order.",
            },
            "smoke_fixture": {
                "frequency_hz": 20.0,
                "amplitude_rad_by_axis": {"yaw": 0.15, "pitch": 0.1, "roll": 0.05},
                "purpose": "actuator and rendering smoke test only; not a flight command",
            },
            "status": "published FlyBody waveform compressed to twelve harmonics; not a free-flight validation",
        },
        "wings": wings,
        "actuators": actuators,
        "provenance": {
            "source_revision": _source_revision(repo),
            "flybody_wing_pattern": {
                "source_file": FLYBODY_WING_PATTERN_FILENAME,
                "source_sha256": FLYBODY_WING_PATTERN_SOURCE_SHA256,
                "doi": FLYBODY_WING_PATTERN_DOI,
                "figshare_doi": FLYBODY_DATA_DOI,
                "data_license": FLYBODY_DATA_LICENSE,
                "license_url": FLYBODY_DATA_LICENSE_URL,
                "flybody_commit": FLYBODY_SOURCE_COMMIT,
                "harmonics": FLYBODY_WING_PATTERN_HARMONICS,
                "frequency_hz": FLYBODY_WING_PATTERN_FREQUENCY_HZ,
                "source_columns": list(FLYBODY_WING_PATTERN_SOURCE_COLUMNS),
                "project_axes": list(FLYBODY_WING_PATTERN_PROJECT_AXES),
                "fit_error": wing_pattern["fit_error"],
            },
            "flybody_source": {
                "repository": "https://github.com/TuragaLab/flybody",
                "commit": FLYBODY_SOURCE_COMMIT,
                "ellipsoid_fluid_model": "flybody/ellipsoid_fluid_model.py:mj_ellipsoidFluidModel",
                "fluid_defaults": "flybody/tasks/constants.py:_WING_PARAMS.fluidcoef",
                "xml": "src/flygym/assets/model/flybody/fruitfly.xml",
                "wing_defaults": "src/flygym/assets/model/flybody/fruitfly.xml:57-88",
                "wing_bodies": "src/flygym/assets/model/flybody/fruitfly.xml:391-425",
                "wing_actuators": "src/flygym/assets/model/flybody/fruitfly.xml:957-962",
                "ranges": "src/flygym/assets/model/flybody/joints.yaml:ranges.c_thorax-*_wing-*",
            },
            "literature": [
                {
                    "citation": "Dickinson MH et al. 1999, Science 284(5422):1954-1960",
                    "doi": "10.1126/science.284.5422.1954",
                    "url": "https://doi.org/10.1126/science.284.5422.1954",
                    "role": "translational lift and drag coefficient fit",
                },
                {
                    "citation": "Sane SP, Dickinson MH. 2002, J Exp Biol 205(8):1087-1096",
                    "doi": "10.1242/jeb.205.8.1087",
                    "url": "https://doi.org/10.1242/jeb.205.8.1087",
                    "role": "quasi-steady insect-flight force-model context",
                },
            ],
        },
        "limitations": [
            "Wing controls are an engineered six-DOF position interface; they are not recovered motor-neuron outputs.",
            "FlyBody ranges and gains are transferred engineering parameters, not a measurement of the NMF animal.",
            "The active backend is MuJoCo's published ellipsoid fluid model using the official thin FlyBody wing-fluid ellipsoids after unit and frame conversion.",
            "The NMF visual wing mesh is retained massless around FlyBody's invisible 8 microgram box inertia, so appearance and flight collision geometry are still not the original FlyBody mesh.",
            "The project keeps a 0.1 ms physics step for brain coupling; FlyBody flight used 0.05 ms, so this is not timestep-identical.",
            "MuJoCo's ellipsoid model includes added mass, viscous drag, Magnus, Kutta, and viscous resistance terms; wake interaction is not modeled.",
            "The coefficient fit is literature-derived and has not been identified or validated against this exported NMF mesh.",
            "The twelve-strip discretization is a runtime approximation of the exported mesh, not a new biomechanical reconstruction.",
            "No flight performance, stability, hover, takeoff, or landing claim follows from this asset alone.",
            "The legacy translational_quasi_steady backend remains available for explicit comparison but is not selected by this artifact.",
            "The published wing pattern is a twelve-harmonic fit with documented residual error, not a recovered motor-neuron waveform.",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
