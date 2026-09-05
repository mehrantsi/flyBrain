from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from .flight_assets import _read_binary_stl
except ImportError:
    from flight_assets import _read_binary_stl


_THORAX = "fly/c_thorax"
_HEAD = "fly/c_head"
_ABDOMEN = tuple(f"fly/c_abdomen{name}" for name in ("12", "3", "4", "5", "6"))
_TIBIA = tuple(
    f"fly/{side}{position}_tibia"
    for side in ("l", "r")
    for position in ("f", "m", "h")
)
_BRISTLE_COUNTS = {
    _THORAX: 60,
    _HEAD: 30,
    **{name: 15 for name in _ABDOMEN},
    **{name: 8 for name in _TIBIA},
}
_DETAIL_GEOM_PREFIX = "fly/detail_"
_MESH_SCALE_UNIT = 1000.0
_BRISTLE_BASE_OFFSET_MM = 0.0006
_BAND_OFFSET_MM = 0.001
_EYE_HEX_RADIUS_MM = 0.012
_EYE_CENTER_LIFT_MM = 0.003


class _Surface(NamedTuple):
    triangles: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    cumulative_areas: np.ndarray
    frame: dict[str, str]


def _numbers(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    output = []
    for value in values:
        number = float(value)
        if abs(number) < 5e-12:
            number = 0.0
        output.append(f"{number:.9g}")
    return " ".join(output)


def _mesh_numbers(values: list[tuple[float, float, float]]) -> str:
    return _numbers([component for vertex in values for component in vertex])


def _face_numbers(faces: list[tuple[int, int, int]]) -> str:
    return " ".join(str(index) for face in faces for index in face)


def _parse_vector(value: str, expected: int = 3) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {expected} finite values, got {value!r}")
    return values


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    return root.find(f".//body[@name='{body_name}']")


def _find_source_geom(body: ET.Element, body_name: str) -> ET.Element:
    geom = body.find(f"geom[@name='{body_name}']")
    if geom is None:
        geom = next((item for item in body.findall("geom") if item.get("mesh")), None)
    if geom is None or not geom.get("mesh"):
        raise RuntimeError(f"anatomical body is missing its mesh geom: {body_name}")
    return geom


def _load_surface(root: ET.Element, assets_dir: Path, body_name: str) -> _Surface:
    body = _find_body(root, body_name)
    if body is None:
        raise RuntimeError(f"anatomical body is missing: {body_name}")
    source_geom = _find_source_geom(body, body_name)
    mesh_name = source_geom.get("mesh")
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("fly appearance requires an asset section")
    mesh = asset.find(f"mesh[@name='{mesh_name}']")
    if mesh is None:
        raise RuntimeError(f"anatomical mesh asset is missing: {mesh_name}")
    mesh_file = mesh.get("file")
    if not mesh_file:
        raise RuntimeError(f"anatomical mesh asset has no file: {mesh_name}")
    mesh_path = Path(mesh_file)
    if not mesh_path.is_absolute():
        mesh_path = assets_dir / mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"anatomical mesh file is missing: {mesh_path}")
    triangles = _read_binary_stl(mesh_path)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"anatomical STL has unexpected shape: {mesh_path} {triangles.shape}")
    scale = _parse_vector(mesh.get("scale", _numbers([_MESH_SCALE_UNIT] * 3)))
    triangles = triangles * (scale / _MESH_SCALE_UNIT)
    if np.prod(scale) < 0.0:
        triangles = triangles[:, [0, 2, 1]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    if not np.any(valid):
        raise ValueError(f"anatomical STL has no non-degenerate triangles: {mesh_path}")
    triangles = triangles[valid]
    cross = cross[valid]
    lengths = lengths[valid]
    normals = cross / lengths[:, None]
    areas = lengths * 0.5
    cumulative_areas = np.cumsum(areas)
    frame = {name: source_geom.get(name) for name in ("pos", "quat", "euler") if name in source_geom.attrib}
    return _Surface(triangles, normals, areas, cumulative_areas, frame)


def _smooth_surface_mesh(
    surface: _Surface,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], np.ndarray]:
    vertices, inverse = np.unique(surface.triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    weighted_normals = surface.normals * surface.areas[:, None]
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], weighted_normals)
    lengths = np.linalg.norm(normals, axis=1)
    zero = lengths <= 1e-12
    if np.any(zero):
        fallback = np.full(len(vertices), -1, dtype=np.int64)
        for corner in range(3):
            vertex_ids = faces[:, corner]
            unset = fallback[vertex_ids] < 0
            fallback[vertex_ids[unset]] = np.flatnonzero(unset)
        normals[zero] = surface.normals[fallback[zero]]
        lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    return [tuple(float(value) for value in vertex) for vertex in vertices], [
        tuple(int(value) for value in face) for face in faces
    ], normals


def _stable_float(seed: str, index: int) -> float:
    digest = hashlib.blake2s(f"{seed}:{index}".encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "little")
    return (integer + 0.5) / 18446744073709551616.0


def _sample_surface(surface: _Surface, count: int, seed: str) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    total_area = float(surface.cumulative_areas[-1])
    points = np.empty((count, 3), dtype=np.float64)
    normals = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        area_target = ((index + 0.5) / count) * total_area
        triangle_index = int(np.searchsorted(surface.cumulative_areas, area_target, side="left"))
        triangle_index = min(triangle_index, len(surface.triangles) - 1)
        triangle = surface.triangles[triangle_index]
        first = _stable_float(seed, 2 * index)
        second = _stable_float(seed, 2 * index + 1)
        root_first = math.sqrt(first)
        barycentric = np.asarray(
            [1.0 - root_first, root_first * (1.0 - second), root_first * second],
            dtype=np.float64,
        )
        points[index] = barycentric @ triangle
        normals[index] = surface.normals[triangle_index]
    return points, normals


def _append_cone(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    base: np.ndarray,
    normal: np.ndarray,
    length: float,
    radius: float,
) -> None:
    axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = np.cross(axis, reference)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    bitangent = np.cross(axis, tangent)
    ring_start = len(vertices)
    for side in range(4):
        angle = 0.5 * math.pi * side + math.pi * 0.25
        point = base + radius * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
        vertices.append(tuple(float(value) for value in point))
    base_center = len(vertices)
    vertices.append(tuple(float(value) for value in base))
    tip = base + length * axis
    tip_index = len(vertices)
    vertices.append(tuple(float(value) for value in tip))
    for side in range(4):
        following = (side + 1) % 4
        first = ring_start + side
        second = ring_start + following
        faces.append((first, second, tip_index))
        faces.append((base_center, second, first))


def _bristle_mesh(surface: _Surface, body_name: str, count: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    points, normals = _sample_surface(surface, count, f"bristle:{body_name}")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for index, (point, normal) in enumerate(zip(points, normals)):
        length = 0.04 + 0.06 * _stable_float(f"bristle-length:{body_name}", index)
        radius = 0.003 + 0.003 * _stable_float(f"bristle-radius:{body_name}", index)
        base = point + _BRISTLE_BASE_OFFSET_MM * normal
        _append_cone(vertices, faces, base, normal, length, radius)
    return vertices, faces


def _band_mesh(surface: _Surface) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    centers = surface.triangles.mean(axis=1)
    minimum = float(surface.triangles[:, :, 0].min())
    maximum = float(surface.triangles[:, :, 0].max())
    span = max(maximum - minimum, 1e-12)
    selected = centers[:, 0] <= minimum + 0.24 * span
    if not np.any(selected):
        selected[np.argmin(centers[:, 0])] = True
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for triangle, normal in zip(surface.triangles[selected], surface.normals[selected]):
        start = len(vertices)
        offset_triangle = triangle + _BAND_OFFSET_MM * normal
        vertices.extend(tuple(float(value) for value in point) for point in offset_triangle)
        faces.append((start, start + 1, start + 2))
    return vertices, faces


def _project_front(
    triangles: np.ndarray,
    normals: np.ndarray,
    x: float,
    z: float,
    side_sign: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    xz = triangles[:, :, (0, 2)]
    first = xz[:, 0]
    edge_one = xz[:, 1] - first
    edge_two = xz[:, 2] - first
    offset = np.asarray([x, z], dtype=np.float64) - first
    denominator = edge_one[:, 0] * edge_two[:, 1] - edge_one[:, 1] * edge_two[:, 0]
    valid = np.abs(denominator) > 1e-12
    if not np.any(valid):
        return None
    inverse = np.zeros_like(denominator)
    inverse[valid] = 1.0 / denominator[valid]
    weight_one = (offset[:, 0] * edge_two[:, 1] - offset[:, 1] * edge_two[:, 0]) * inverse
    weight_two = (edge_one[:, 0] * offset[:, 1] - edge_one[:, 1] * offset[:, 0]) * inverse
    weight_zero = 1.0 - weight_one - weight_two
    valid &= weight_zero >= -1e-7
    valid &= weight_one >= -1e-7
    valid &= weight_two >= -1e-7
    if not np.any(valid):
        return None
    indices = np.flatnonzero(valid)
    y_values = (
        weight_zero[indices] * triangles[indices, 0, 1]
        + weight_one[indices] * triangles[indices, 1, 1]
        + weight_two[indices] * triangles[indices, 2, 1]
    )
    choice = int(indices[np.argmax(side_sign * y_values)])
    barycentric = np.asarray(
        [weight_zero[choice], weight_one[choice], weight_two[choice]], dtype=np.float64
    )
    point = barycentric @ triangles[choice]
    normal = normals[choice].copy()
    if float(np.dot(normal, np.asarray([0.0, side_sign, 0.0]))) < 0.0:
        normal *= -1.0
    return point, normal


def _eye_mesh(surface: _Surface, body_name: str) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    radius = _EYE_HEX_RADIUS_MM
    side_sign = 1.0 if body_name == "fly/l_eye" else -1.0
    minimum = surface.triangles.reshape(-1, 3).min(axis=0)
    maximum = surface.triangles.reshape(-1, 3).max(axis=0)
    x_step = math.sqrt(3.0) * radius
    z_step = 1.5 * radius
    x_values = np.arange(float(minimum[0] + radius), float(maximum[0] - radius) + x_step * 0.25, x_step)
    z_values = np.arange(float(minimum[2] + radius), float(maximum[2] - radius) + z_step * 0.25, z_step)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    angles = [math.pi / 6.0 + side * math.pi / 3.0 for side in range(6)]
    for row, center_z in enumerate(z_values):
        row_offset = x_step * 0.5 if row % 2 else 0.0
        for center_x in x_values + row_offset:
            center_projection = _project_front(surface.triangles, surface.normals, float(center_x), float(center_z), side_sign)
            if center_projection is None:
                continue
            perimeter = []
            valid = True
            for angle in angles:
                projection = _project_front(
                    surface.triangles,
                    surface.normals,
                    float(center_x + radius * math.cos(angle)),
                    float(center_z + radius * math.sin(angle)),
                    side_sign,
                )
                if projection is None:
                    valid = False
                    break
                perimeter.append(projection)
            if not valid:
                continue
            center_point, center_normal = center_projection
            normal_sum = center_normal + sum((normal for _, normal in perimeter), np.zeros(3))
            normal_sum /= max(float(np.linalg.norm(normal_sum)), 1e-12)
            center = center_point + _EYE_CENTER_LIFT_MM * normal_sum
            base = len(vertices)
            vertices.extend(tuple(float(value) for value in point + 0.0015 * normal) for point, normal in perimeter)
            vertices.append(tuple(float(value) for value in center))
            center_index = base + 6
            for side in range(6):
                face = (center_index, base + side, base + (side + 1) % 6)
                if side_sign > 0:
                    face = (face[0], face[2], face[1])
                faces.append(face)
    return vertices, faces


def _add_mesh(
    asset: ET.Element,
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    smoothnormal: bool,
    normals: np.ndarray | None = None,
    inertia: str | None = None,
) -> None:
    if asset.find(f"mesh[@name='{name}']") is not None:
        return
    attributes = {
        "name": name,
        "vertex": _mesh_numbers(vertices),
        "face": _face_numbers(faces),
        "smoothnormal": "true" if smoothnormal else "false",
    }
    if normals is not None:
        attributes["normal"] = _mesh_numbers([tuple(row) for row in normals])
    if inertia is not None:
        attributes["inertia"] = inertia
    ET.SubElement(
        asset,
        "mesh",
        attributes,
    )


def _add_detail_geom(
    body: ET.Element,
    name: str,
    mesh: str,
    material: str,
    frame: dict[str, str],
) -> None:
    if body.find(f"geom[@name='{name}']") is not None:
        return
    attributes = {
        "name": name,
        "type": "mesh",
        "mesh": mesh,
        "material": material,
        "mass": "0",
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
        "fluidshape": "none",
    }
    attributes.update(frame)
    ET.SubElement(body, "geom", attributes)


def _add_smooth_visual_geom(
    body: ET.Element,
    name: str,
    mesh: str,
    source_geom: ET.Element,
    frame: dict[str, str],
) -> None:
    if body.find(f"geom[@name='{name}']") is not None:
        return
    attributes = {
        "name": name,
        "type": "mesh",
        "mesh": mesh,
        "mass": "0",
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
        "fluidshape": "none",
    }
    for key in ("material", "rgba"):
        value = source_geom.get(key)
        if value is not None:
            attributes[key] = value
    attributes.update(frame)
    ET.SubElement(body, "geom", attributes)


def _set_material(asset: ET.Element, name: str, attributes: dict[str, str]) -> None:
    material = asset.find(f"material[@name='{name}']")
    if material is None:
        material = ET.SubElement(asset, "material", {"name": name})
    for key, value in attributes.items():
        material.set(key, value)


def _tune_fly_materials(asset: ET.Element) -> None:
    textures = {
        "fly/headthorax": {
            "rgb1": "0.38 0.20 0.065",
            "rgb2": "0.58 0.34 0.12",
            "markrgb": "0.46 0.25 0.08",
            "random": "0.06",
        },
        "fly/antennaproboscis": {
            "rgb1": "0.25 0.11 0.035",
            "rgb2": "0.43 0.22 0.07",
            "random": "0.035",
        },
        "fly/abdomen12345": {
            "rgb1": "0.36 0.16 0.045",
            "rgb2": "0.70 0.42 0.14",
            "markrgb": "0.48 0.24 0.07",
            "random": "0.055",
        },
        "fly/abdomen6": {
            "rgb1": "0.24 0.085 0.022",
            "rgb2": "0.56 0.29 0.085",
            "markrgb": "0.34 0.15 0.04",
            "random": "0.05",
        },
        "fly/coxa": {
            "rgb1": "0.33 0.16 0.045",
            "rgb2": "0.53 0.29 0.09",
            "random": "0.035",
        },
        "fly/trochanterfemur": {
            "rgb1": "0.38 0.19 0.055",
            "rgb2": "0.57 0.31 0.10",
            "random": "0.035",
        },
        "fly/tibia": {
            "rgb1": "0.42 0.21 0.065",
            "rgb2": "0.62 0.36 0.12",
            "random": "0.035",
        },
        "fly/tarsus": {
            "rgb1": "0.34 0.15 0.04",
            "rgb2": "0.52 0.27 0.08",
            "random": "0.035",
        },
    }
    for name, values in textures.items():
        texture = asset.find(f"texture[@name='{name}']")
        if texture is not None:
            texture.set("markrgb", values.get("markrgb", values["rgb1"]))
            for key, value in values.items():
                texture.set(key, value)
    material_names = (
        "fly/headthorax",
        "fly/antennaproboscis",
        "fly/abdomen12345",
        "fly/abdomen6",
        "fly/arista",
        "fly/coxa",
        "fly/eye",
        "fly/haltere",
        "fly/trochanterfemur",
        "fly/tibia",
        "fly/tarsus",
    )
    for name in material_names:
        material = asset.find(f"material[@name='{name}']")
        if material is not None:
            material.set("specular", "0.15")
            material.set("shininess", "0.25")
    _set_material(
        asset,
        "fly/wing",
        {"rgba": "0.56 0.62 0.65 0.30", "specular": "0.15", "shininess": "0.25"},
    )
    _set_material(
        asset,
        "fly/eye",
        {"rgba": "0.47 0.075 0.045 1", "specular": "0.15", "shininess": "0.25"},
    )
    _set_material(
        asset,
        "fly/detail_bristle",
        {"rgba": "0.20 0.075 0.022 1", "specular": "0.15", "shininess": "0.25"},
    )
    _set_material(
        asset,
        "fly/detail_abdomen_band",
        {"rgba": "0.12 0.045 0.015 1", "specular": "0.15", "shininess": "0.25"},
    )
    _set_material(
        asset,
        "fly/detail_eye_facet",
        {"rgba": "0.34 0.035 0.022 1", "specular": "0.16", "shininess": "0.22"},
    )


def improve_fly_appearance(root: ET.Element, assets_dir: Path) -> dict[str, int]:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("fly appearance requires asset and worldbody")
    assets_dir = Path(assets_dir)
    _tune_fly_materials(asset)
    for mesh in asset.findall("mesh"):
        if mesh.get("file") and mesh.get("name", "").startswith("fly/"):
            mesh.set("smoothnormal", "true")

    file_backed_body_names = tuple(
        mesh.get("name")
        for mesh in asset.findall("mesh")
        if mesh.get("file")
        and mesh.get("name", "").startswith("fly/")
        and not mesh.get("name", "").startswith("fly/detail")
        and mesh.get("name") is not None
    )
    surfaces: dict[str, _Surface] = {}
    for body_name in file_backed_body_names:
        if _find_body(root, body_name) is not None:
            surfaces[body_name] = _load_surface(root, assets_dir, body_name)

    smooth_visual_meshes = 0
    smooth_visual_triangles = 0
    for body_name in file_backed_body_names:
        surface = surfaces.get(body_name)
        body = _find_body(root, body_name)
        if surface is None or body is None:
            continue
        source_geom = _find_source_geom(body, body_name)
        source_geom.set("group", "5")
        geom_name = f"{_DETAIL_GEOM_PREFIX}surface_{body_name.split('/')[-1]}"
        if body.find(f"geom[@name='{geom_name}']") is None:
            vertices, faces, normals = _smooth_surface_mesh(surface)
            _add_mesh(asset, geom_name, vertices, faces, smoothnormal=True, normals=normals)
            _add_smooth_visual_geom(body, geom_name, geom_name, source_geom, surface.frame)
            smooth_visual_meshes += 1
            smooth_visual_triangles += len(faces)

    bristle_meshes = 0
    bristle_count = 0
    for body_name, count in _BRISTLE_COUNTS.items():
        surface = surfaces.get(body_name)
        body = _find_body(root, body_name)
        if surface is None or body is None:
            continue
        geom_name = f"{_DETAIL_GEOM_PREFIX}bristles_{body_name.split('/')[-1]}"
        mesh_name = geom_name
        if body.find(f"geom[@name='{geom_name}']") is None:
            vertices, faces = _bristle_mesh(surface, body_name, count)
            _add_mesh(asset, mesh_name, vertices, faces, smoothnormal=False)
            _add_detail_geom(body, geom_name, mesh_name, "fly/detail_bristle", surface.frame)
            bristle_meshes += 1
        bristle_count += count

    band_meshes = 0
    band_triangles = 0
    for body_name in _ABDOMEN:
        surface = surfaces.get(body_name)
        body = _find_body(root, body_name)
        if surface is None or body is None:
            continue
        geom_name = f"{_DETAIL_GEOM_PREFIX}abdomen_band_{body_name.split('/')[-1]}"
        if body.find(f"geom[@name='{geom_name}']") is None:
            vertices, faces = _band_mesh(surface)
            _add_mesh(asset, geom_name, vertices, faces, smoothnormal=False, inertia="shell")
            _add_detail_geom(
                body,
                geom_name,
                geom_name,
                "fly/detail_abdomen_band",
                surface.frame,
            )
            band_meshes += 1
            band_triangles += len(faces)

    eye_meshes = 0
    eye_facets = 0
    for body_name in ("fly/l_eye", "fly/r_eye"):
        surface = surfaces.get(body_name)
        body = _find_body(root, body_name)
        if surface is None or body is None:
            continue
        geom_name = f"{_DETAIL_GEOM_PREFIX}eye_facets_{body_name.split('/')[-1][0]}"
        if body.find(f"geom[@name='{geom_name}']") is None:
            vertices, faces = _eye_mesh(surface, body_name)
            _add_mesh(asset, geom_name, vertices, faces, smoothnormal=False)
            _add_detail_geom(body, geom_name, geom_name, "fly/detail_eye_facet", surface.frame)
            eye_meshes += 1
            eye_facets += len(faces)

    return {
        "bristle_meshes": bristle_meshes,
        "bristles": bristle_count,
        "abdomen_band_meshes": band_meshes,
        "abdomen_band_triangles": band_triangles,
        "eye_meshes": eye_meshes,
        "eye_facets": eye_facets,
        "smooth_visual_meshes": smooth_visual_meshes,
        "smooth_visual_triangles": smooth_visual_triangles,
    }


__all__ = ["improve_fly_appearance"]
