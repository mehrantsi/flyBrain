from __future__ import annotations

import copy
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.flight_assets import _read_binary_stl
from tools.fly_appearance import _load_surface, improve_fly_appearance
from tools.habitat_assets import improve_habitat_appearance
from tools.habitat_detail import _validate_mesh, add_habitat_details


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "neuromechfly"
MODEL_PATH = ASSETS / "fly.xml"
DETAIL_PREFIXES = ("detail/", "detail_", "fly/detail_")
COSMETIC_GEOM_ATTRIBUTES = {"group", "material", "smoothnormal"}


def _checked_in_root() -> ET.Element:
    return ET.parse(MODEL_PATH).getroot()


def _remove_details(root: ET.Element) -> ET.Element:
    for parent in root.iter():
        for child in list(parent):
            if child.get("name", "").startswith(DETAIL_PREFIXES):
                parent.remove(child)
    return root


def _mesh_data(
    mesh: ET.Element,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertex_values = np.fromstring(mesh.get("vertex", ""), sep=" ", dtype=np.float64)
    face_values = np.fromstring(mesh.get("face", ""), sep=" ", dtype=np.int64)
    assert vertex_values.size and vertex_values.size % 3 == 0
    assert face_values.size and face_values.size % 3 == 0
    vertices = [tuple(row) for row in vertex_values.reshape(-1, 3)]
    faces = [tuple(row) for row in face_values.reshape(-1, 3)]
    return vertices, faces


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    assert np.all(np.isfinite(lengths))
    assert np.all(lengths > 0.0)
    return cross / lengths[:, None]


def _signed_volume(
    vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]
) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(faces, dtype=np.int64)
    return float(
        np.einsum(
            "ij,ij->i",
            points[indices[:, 0]],
            np.cross(points[indices[:, 1]], points[indices[:, 2]]),
        ).sum()
        / 6.0
    )


def test_appearance_generators_are_idempotent_in_memory():
    root = _remove_details(copy.deepcopy(_checked_in_root()))

    improve_habitat_appearance(root)
    improve_fly_appearance(root, ASSETS)
    first_xml = ET.tostring(root, encoding="unicode")
    first_meshes = {
        mesh.get("name")
        for mesh in root.findall("asset/mesh")
        if mesh.get("name", "").startswith(DETAIL_PREFIXES)
    }
    first_geoms = {
        geom.get("name")
        for geom in root.iter("geom")
        if geom.get("name", "").startswith(("detail_", "fly/detail_"))
    }

    improve_habitat_appearance(root)
    improve_fly_appearance(root, ASSETS)

    assert ET.tostring(root, encoding="unicode") == first_xml
    assert {
        mesh.get("name")
        for mesh in root.findall("asset/mesh")
        if mesh.get("name", "").startswith(DETAIL_PREFIXES)
    } == first_meshes
    assert {
        geom.get("name")
        for geom in root.iter("geom")
        if geom.get("name", "").startswith(("detail_", "fly/detail_"))
    } == first_geoms


def test_detail_geoms_are_massless_noncolliding_and_fly_details_have_no_fluid():
    root = _checked_in_root()
    habitat_details = [
        geom
        for geom in root.iter("geom")
        if geom.get("name", "").startswith("detail_")
    ]
    fly_details = [
        geom
        for geom in root.iter("geom")
        if geom.get("name", "").startswith("fly/detail_")
    ]
    assert habitat_details
    assert fly_details
    for geom in (*habitat_details, *fly_details):
        assert float(geom.get("mass", "nan")) == 0.0
        assert geom.get("contype") == "0"
        assert geom.get("conaffinity") == "0"
    assert all(geom.get("fluidshape") == "none" for geom in fly_details)


def test_detail_generator_preserves_original_geometry_physics():
    root = _remove_details(copy.deepcopy(_checked_in_root()))
    worldbody = root.find("worldbody")
    assert worldbody is not None
    before = {
        geom.get("name"): {
            key: value
            for key, value in geom.attrib.items()
            if key not in COSMETIC_GEOM_ATTRIBUTES
        }
        for geom in worldbody.findall("geom")
        if geom.get("name")
    }

    add_habitat_details(root)

    after = {
        geom.get("name"): {
            key: value
            for key, value in geom.attrib.items()
            if key not in COSMETIC_GEOM_ATTRIBUTES
        }
        for geom in worldbody.findall("geom")
        if geom.get("name")
    }
    assert set(before) <= set(after)
    for name, attributes in before.items():
        assert after[name] == attributes


def test_detail_meshes_are_nondegenerate_and_closed_meshes_have_positive_volume():
    root = _checked_in_root()
    asset = root.find("asset")
    assert asset is not None
    detail_meshes = [
        mesh
        for mesh in asset.findall("mesh")
        if mesh.get("name", "").startswith(DETAIL_PREFIXES)
    ]
    assert detail_meshes
    for mesh in detail_meshes:
        name = mesh.get("name", "")
        vertices, faces = _mesh_data(mesh)
        points = np.asarray(vertices, dtype=np.float64)
        indices = np.asarray(faces, dtype=np.int64)
        cross = np.cross(
            points[indices[:, 1]] - points[indices[:, 0]],
            points[indices[:, 2]] - points[indices[:, 0]],
        )
        assert np.all(np.isfinite(cross))
        assert np.all(np.sum(cross * cross, axis=1) > 0.0), name
        if name.startswith("detail/"):
            _validate_mesh(vertices, faces)
            assert _signed_volume(vertices, faces) > 1e-10
        elif not name.startswith(("fly/detail_abdomen_band_", "fly/detail_surface_")):
            _validate_mesh(vertices, faces, shell=True)


def test_smooth_visual_clones_preserve_source_triangles_and_use_unit_normals():
    root = _checked_in_root()
    meshes = root.findall("asset/mesh")
    clones = [mesh for mesh in meshes if mesh.get("name", "").startswith("fly/detail_surface_")]
    assert clones
    for mesh in clones:
        suffix = mesh.get("name").removeprefix("fly/detail_surface_")
        body_name = f"fly/{suffix}"
        source = _load_surface(root, ASSETS, body_name)
        vertices, faces = _mesh_data(mesh)
        triangles = np.asarray(vertices)[np.asarray(faces)]
        np.testing.assert_allclose(triangles, source.triangles, rtol=0.0, atol=1e-8)
        normals = np.fromstring(mesh.get("normal"), sep=" ").reshape(-1, 3)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-8)
        assert len(np.unique(normals, axis=0)) > 1
        original = root.find(f".//geom[@name='{body_name}']")
        assert original.get("group") == "5"


def test_right_mirrored_fly_surfaces_reverse_winding_with_the_reflection():
    root = _checked_in_root()
    asset = root.find("asset")
    assert asset is not None
    reflection = np.asarray((1.0, -1.0, 1.0))
    for left_name, right_name in (("fly/l_wing", "fly/r_wing"), ("fly/l_eye", "fly/r_eye")):
        left = asset.find(f"mesh[@name='{left_name}']")
        right = asset.find(f"mesh[@name='{right_name}']")
        assert left is not None and right is not None
        assert left.get("file") == right.get("file")
        source = _read_binary_stl(ASSETS / right.get("file"))
        left_scale = np.asarray([float(value) for value in left.get("scale").split()]) / 1000.0
        right_scale = np.asarray([float(value) for value in right.get("scale").split()]) / 1000.0
        assert np.prod(right_scale) < 0.0
        left_triangles = source * left_scale
        right_triangles = source * right_scale
        right_triangles = right_triangles[:, [0, 2, 1]]
        expected_triangles = left_triangles[:, [0, 2, 1]] * reflection
        np.testing.assert_allclose(right_triangles, expected_triangles, rtol=0.0, atol=1e-12)
        left_normals = _triangle_normals(left_triangles)
        right_normals = _triangle_normals(right_triangles)
        np.testing.assert_allclose(right_normals, left_normals * reflection, rtol=0.0, atol=1e-12)


def test_eye_hex_facets_face_bilateral_outward():
    root = _checked_in_root()
    asset = root.find("asset")
    assert asset is not None
    for side, outward_y in (("l", 1.0), ("r", -1.0)):
        mesh = asset.find(f"mesh[@name='fly/detail_eye_facets_{side}']")
        assert mesh is not None
        vertices, faces = _mesh_data(mesh)
        points = np.asarray(vertices, dtype=np.float64)
        indices = np.asarray(faces, dtype=np.int64)
        normals = _triangle_normals(points[indices])
        assert np.all(normals[:, 1] * outward_y > 0.0)


def _remove_cosmetic_assets(root: ET.Element) -> ET.Element:
    _remove_details(root)
    asset = root.find("asset")
    assert asset is not None
    referenced_materials = {
        element.get("material")
        for element in root.iter()
        if element.get("material")
    }
    for material in list(asset.findall("material")):
        if material.get("name") not in referenced_materials:
            asset.remove(material)
    referenced_textures = {
        material.get("texture")
        for material in asset.findall("material")
        if material.get("texture")
    }
    for texture in list(asset.findall("texture")):
        if texture.get("name") not in referenced_textures:
            asset.remove(texture)
    referenced_meshes = {
        element.get("mesh")
        for element in root.iter()
        if element.get("mesh")
    }
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in referenced_meshes:
            asset.remove(mesh)
    return root


def _set_absolute_compiler_paths(root: ET.Element) -> None:
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(ASSETS.resolve()))
    compiler.set("texturedir", str(ASSETS.resolve()))


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="MuJoCo Python bindings are optional",
)
def test_cosmetic_removal_preserves_compiled_body_mass_and_inertia():
    mujoco = pytest.importorskip("mujoco")
    full_root = _checked_in_root()
    stripped_root = _remove_cosmetic_assets(copy.deepcopy(full_root))
    _set_absolute_compiler_paths(full_root)
    _set_absolute_compiler_paths(stripped_root)
    full_model = mujoco.MjModel.from_xml_string(ET.tostring(full_root, encoding="unicode"))
    stripped_model = mujoco.MjModel.from_xml_string(
        ET.tostring(stripped_root, encoding="unicode")
    )

    assert stripped_model.nbody == full_model.nbody
    assert stripped_model.njnt == full_model.njnt
    assert stripped_model.nq == full_model.nq
    assert stripped_model.nv == full_model.nv
    assert stripped_model.nu == full_model.nu
    assert stripped_model.ngeom < full_model.ngeom
    for attribute in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "dof_armature",
        "dof_damping",
        "jnt_stiffness",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(full_model, attribute)),
            np.asarray(getattr(stripped_model, attribute)),
        )
