from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOM_HALF_EXTENTS = (300.0, 220.0, 110.0)
AIRFLOW_MM_S = (35.0, 8.0, 0.0)
BANANA_SEGMENTS = (
    ((25.0, 15.0, 1.4), (29.0, 17.0, 1.8)),
    ((29.0, 17.0, 1.8), (34.0, 18.5, 2.0)),
    ((34.0, 18.5, 2.0), (39.0, 18.0, 1.8)),
    ((39.0, 18.0, 1.8), (43.0, 15.5, 1.4)),
)
BANANA_RADIUS_MM = 0.9

# Match the explicit ground-plane pairs while letting MuJoCo select the
# stationary habitat geometry's parameters by priority.
HABITAT_CONTACT_ATTRIBUTES = {
    "priority": "1",
    "condim": "3",
    "friction": "1 0.02 0.0001",
    "solref": "0.0002",
    "solimp": "0.98 0.99 1e-05 0.5 3",
    "margin": "0.001",
}

RESOURCES = [
    {
        "id": "sugar_drop",
        "kind": "sugar",
        "geom": "food_patch",
        "position": [1.1, 0.0, 0.25],
        "movable": True,
        "taste_radius_mm": 0.75,
        "odor_source_ppm": 0.0,
        "odor_length_mm": 22.0,
        "taste_valence": 1.0,
        "nutrition": 1.0,
        "hydration": 0.15,
    },
    {
        "id": "banana",
        "kind": "ripe_fruit",
        "geom": "resource_banana",
        "position": [32.0, 18.0, 1.6],
        "movable": False,
        "taste_radius_mm": 2.6,
        "taste_margin_mm": 1.7,
        "taste_capsules": [
            {"from_mm": start, "to_mm": end, "radius_mm": BANANA_RADIUS_MM}
            for start, end in BANANA_SEGMENTS
        ],
        "odor_source_ppm": 12.0,
        "odor_length_mm": 95.0,
        "taste_valence": 0.9,
        "nutrition": 0.8,
        "hydration": 0.3,
    },
    {
        "id": "orange_half",
        "kind": "citrus",
        "geom": "resource_orange",
        "position": [-48.0, 38.0, 6.5],
        "movable": False,
        "taste_radius_mm": 7.5,
        "odor_source_ppm": 3.0,
        "odor_length_mm": 85.0,
        "taste_valence": 0.65,
        "nutrition": 0.55,
        "hydration": 0.7,
    },
    {
        "id": "apple",
        "kind": "fruit",
        "geom": "resource_apple",
        "position": [105.0, 70.0, 58.0],
        "movable": False,
        "taste_radius_mm": 9.5,
        "odor_source_ppm": 3.0,
        "odor_length_mm": 105.0,
        "taste_valence": 0.75,
        "nutrition": 0.7,
        "hydration": 0.5,
    },
    {
        "id": "fermenting_juice",
        "kind": "ferment",
        "geom": "resource_ferment",
        "position": [151.0, 62.0, 55.2],
        "movable": False,
        "taste_radius_mm": 8.0,
        "odor_source_ppm": 32.0,
        "odor_length_mm": 165.0,
        "taste_valence": 0.72,
        "nutrition": 0.45,
        "hydration": 0.55,
    },
    {
        "id": "water_dish",
        "kind": "water",
        "geom": "resource_water",
        "position": [-58.0, -34.0, 0.7],
        "movable": False,
        "taste_radius_mm": 7.0,
        "odor_source_ppm": 0.0,
        "odor_length_mm": 22.0,
        "taste_valence": 0.25,
        "nutrition": 0.0,
        "hydration": 1.0,
    },
    {
        "id": "yeast_culture",
        "kind": "yeast",
        "geom": "resource_yeast",
        "position": [-178.0, 145.0, 73.2],
        "movable": False,
        "taste_radius_mm": 6.5,
        "odor_source_ppm": 12.0,
        "odor_length_mm": 145.0,
        "taste_valence": 0.85,
        "nutrition": 0.9,
        "hydration": 0.35,
    },
]


MATERIALS = {
    "habitat/wall": {"rgba": "0.73 0.72 0.66 1", "specular": "0.05", "shininess": "0.05"},
    "habitat/wood": {"rgba": "0.42 0.27 0.16 1", "specular": "0.12", "shininess": "0.12"},
    "habitat/darkwood": {"rgba": "0.22 0.15 0.10 1", "specular": "0.08", "shininess": "0.1"},
    "habitat/rug": {"rgba": "0.22 0.34 0.36 1", "specular": "0", "shininess": "0"},
    "habitat/ceramic": {"rgba": "0.81 0.83 0.78 1", "specular": "0.22", "shininess": "0.35"},
    "habitat/banana": {"rgba": "0.82 0.63 0.13 1", "specular": "0.10", "shininess": "0.15"},
    "habitat/orange": {"rgba": "0.85 0.30 0.05 1", "specular": "0.08", "shininess": "0.10"},
    "habitat/apple": {"rgba": "0.63 0.09 0.06 1", "specular": "0.2", "shininess": "0.3"},
    "habitat/leaf": {"rgba": "0.12 0.30 0.14 1", "specular": "0.08", "shininess": "0.1"},
    "habitat/water": {"rgba": "0.20 0.40 0.48 1", "specular": "0.4", "shininess": "0.5"},
    "habitat/juice": {"rgba": "0.35 0.08 0.09 1", "specular": "0.25", "shininess": "0.35"},
    "habitat/glass": {"rgba": "0.75 0.86 0.89 0.28", "specular": "0.3", "shininess": "0.6"},
    "habitat/soil": {"rgba": "0.19 0.12 0.07 1", "specular": "0", "shininess": "0"},
    "habitat/metal": {"rgba": "0.25 0.29 0.29 1", "specular": "0.35", "shininess": "0.4"},
    "habitat/window": {"rgba": "0.53 0.70 0.75 1", "specular": "0.08", "emission": "0.25"},
}


def _add_geom(
    worldbody: ET.Element,
    name: str,
    geom_type: str,
    position: tuple[float, float, float],
    size: tuple[float, ...],
    material: str,
    *,
    collidable: bool = False,
    **attributes: str,
) -> ET.Element:
    payload = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(str(value) for value in position),
        "size": " ".join(str(value) for value in size),
        "material": material,
        "mass": "0",
        "contype": "0",
        "conaffinity": "1" if collidable else "0",
    }
    if collidable:
        payload.update(HABITAT_CONTACT_ATTRIBUTES)
    payload.update(attributes)
    if "fromto" in payload:
        payload.pop("pos")
    return ET.SubElement(worldbody, "geom", payload)


def add_habitat(root: ET.Element) -> None:
    option = root.find("option")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    contact = root.find("contact")
    if option is None or asset is None or worldbody is None or contact is None:
        raise RuntimeError("exported model is missing option, asset, worldbody, or contact")
    option.set("ccd_iterations", "100")

    for name, attributes in MATERIALS.items():
        ET.SubElement(asset, "material", {"name": name, **attributes})

    map_element = root.find("visual/map")
    statistic = root.find("statistic")
    if map_element is None or statistic is None:
        raise RuntimeError("exported model is missing visual map or statistic")
    map_element.set("znear", "0.0001")
    map_element.set("zfar", "4")
    statistic.set("extent", "350")

    for pair in contact.findall("pair"):
        if pair.get("geom1") != "ground_plane":
            continue
        fly_geom = root.find(f".//geom[@name='{pair.get('geom2')}']")
        if fly_geom is not None:
            fly_geom.set("contype", "1")
            fly_geom.set("conaffinity", "0")

    _add_geom(worldbody, "room_wall_left", "box", (-300, 0, 110), (2, 220, 110), "habitat/wall", collidable=True)
    _add_geom(worldbody, "room_wall_right", "box", (300, 0, 110), (2, 220, 110), "habitat/wall", collidable=True)
    _add_geom(worldbody, "room_wall_back", "box", (0, 220, 110), (300, 2, 110), "habitat/wall", collidable=True)
    _add_geom(worldbody, "room_wall_front_left", "box", (-205, -220, 110), (95, 2, 110), "habitat/wall", collidable=True)
    _add_geom(worldbody, "room_wall_front_right", "box", (205, -220, 110), (95, 2, 110), "habitat/wall", collidable=True)
    _add_geom(worldbody, "room_wall_front_window", "box", (0, -220, 110), (110, 2, 110), "habitat/glass", collidable=True)
    _add_geom(worldbody, "room_wall_ceiling", "box", (0, 0, 220), (300, 220, 2), "habitat/glass", collidable=True)
    _add_geom(worldbody, "room_rug", "box", (-75, -45, 0.025), (68, 48, 0.025), "habitat/rug")

    _add_geom(worldbody, "table_top", "box", (120, 70, 48), (72, 52, 3), "habitat/wood", collidable=True)
    for index, (x, y) in enumerate(((55, 25), (185, 25), (55, 115), (185, 115))):
        _add_geom(worldbody, f"table_leg_{index}", "box", (x, y, 23), (4, 4, 23), "habitat/darkwood", collidable=True)
    _add_geom(worldbody, "wall_shelf", "box", (-180, 150, 68), (64, 22, 3), "habitat/wood", collidable=True)
    for index, x in enumerate((-235, -125)):
        _add_geom(worldbody, f"shelf_bracket_{index}", "box", (x, 167, 48), (3, 3, 18), "habitat/metal", collidable=True)

    _add_geom(worldbody, "banana_plate", "cylinder", (32, 18, 0.35), (8.5, 0.35), "habitat/ceramic", collidable=True)
    for index, (start, end) in enumerate(BANANA_SEGMENTS):
        attributes = {"fromto": " ".join(str(value) for value in (*start, *end))}
        _add_geom(worldbody, "resource_banana" if index == 1 else f"banana_segment_{index}", "capsule", (0, 0, 0), (BANANA_RADIUS_MM,), "habitat/banana", collidable=True, **attributes)

    _add_geom(worldbody, "orange_plate", "cylinder", (-48, 38, 0.4), (10, 0.4), "habitat/ceramic", collidable=True)
    _add_geom(worldbody, "resource_orange", "sphere", (-48, 38, 6.5), (6.2,), "habitat/orange", collidable=True)
    _add_geom(worldbody, "orange_leaf", "ellipsoid", (-44, 39, 12.1), (3.2, 1.4, 0.25), "habitat/leaf", euler="0.2 0.5 0.4")

    _add_geom(worldbody, "apple_plate", "cylinder", (105, 70, 51.5), (12, 0.5), "habitat/ceramic", collidable=True)
    _add_geom(worldbody, "resource_apple", "sphere", (105, 70, 58), (7.2,), "habitat/apple", collidable=True)
    _add_geom(worldbody, "apple_stem", "capsule", (105, 70, 66), (0.65, 2.2), "habitat/darkwood", euler="0.15 0.1 0")
    _add_geom(worldbody, "apple_leaf", "ellipsoid", (108, 70.5, 66.5), (3.0, 1.2, 0.22), "habitat/leaf", euler="0.1 0.35 0.2")

    _add_geom(worldbody, "wine_base", "cylinder", (151, 62, 51.5), (7.0, 0.55), "habitat/glass", collidable=True)
    _add_geom(worldbody, "wine_stem", "capsule", (151, 62, 54.0), (0.65, 2.0), "habitat/glass")
    _add_geom(worldbody, "wine_bowl", "cylinder", (151, 62, 58.0), (7.2, 3.0), "habitat/glass", collidable=True)
    _add_geom(worldbody, "resource_ferment", "cylinder", (151, 62, 55.2), (6.5, 0.8), "habitat/juice")

    _add_geom(worldbody, "water_dish", "cylinder", (-58, -34, 0.35), (8.0, 0.35), "habitat/ceramic", collidable=True)
    _add_geom(worldbody, "resource_water", "cylinder", (-58, -34, 0.72), (7.2, 0.08), "habitat/water")
    _add_geom(worldbody, "yeast_dish", "cylinder", (-178, 145, 71.4), (8.0, 0.5), "habitat/ceramic", collidable=True)
    _add_geom(worldbody, "resource_yeast", "cylinder", (-178, 145, 72.1), (7.0, 0.65), "habitat/soil")

    _add_geom(worldbody, "plant_pot", "cylinder", (220, -130, 11), (10, 11), "habitat/ceramic", collidable=True)
    _add_geom(worldbody, "plant_soil", "cylinder", (220, -130, 22.1), (9.2, 0.8), "habitat/soil")
    for index, (x, y, z, pitch, yaw) in enumerate(((215, -130, 31, 0.4, -0.5), (225, -132, 34, -0.35, 0.7), (220, -124, 38, 0.2, 1.1), (218, -136, 42, -0.2, -1.0))):
        _add_geom(worldbody, f"plant_leaf_{index}", "ellipsoid", (x, y, z), (2.0, 8.0, 0.45), "habitat/leaf", euler=f"{pitch} 0 {yaw}")

    for index, (x, y, radius) in enumerate(((12, -11, 0.7), (17, -9, 0.45), (-8, 14, 0.55), (67, -38, 0.6), (73, -41, 0.4))):
        _add_geom(worldbody, f"crumb_{index}", "sphere", (x, y, radius), (radius,), "habitat/banana", collidable=True)

    _add_geom(worldbody, "window_glass", "box", (299.3, 52, 132), (0.35, 55, 48), "habitat/window")
    for index, (y, z, sy, sz) in enumerate(((52, 82, 58, 2), (52, 182, 58, 2), (-5, 132, 2, 52), (109, 132, 2, 52))):
        _add_geom(worldbody, f"window_frame_{index}", "box", (298.5, y, z), (1.0, sy, sz), "habitat/darkwood")

    ET.SubElement(worldbody, "light", {"name": "room_sun", "pos": "-180 -120 210", "dir": "0.55 0.35 -1", "directional": "true", "diffuse": "0.82 0.80 0.72", "specular": "0.25 0.25 0.22", "castshadow": "true"})
    ET.SubElement(worldbody, "light", {"name": "table_light", "pos": "130 35 150", "dir": "0 0 -1", "directional": "false", "diffuse": "0.52 0.38 0.24", "specular": "0.2 0.16 0.1", "cutoff": "65", "castshadow": "true"})
    ET.SubElement(worldbody, "camera", {"name": "room_camera", "pos": "0 -360 185", "xyaxes": "1 0 0 0 0.456 0.89", "fovy": "58"})
    improve_habitat_appearance(root)


def improve_habitat_appearance(root: ET.Element) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    visual = root.find("visual")
    if asset is None or worldbody is None or visual is None:
        raise RuntimeError("room appearance requires asset, worldbody and visual")
    for name, attributes in MATERIALS.items():
        material = asset.find(f"material[@name='{name}']")
        if material is not None:
            material.attrib.clear()
            material.attrib.update(name=name, **attributes)
    if asset.find("texture[@name='habitat/oak']") is None:
        ET.SubElement(asset, "texture", name="habitat/oak", type="2d", file="textures/oak-v1.png")
    for name, tint in (("habitat/wood", "0.82 0.76 0.66 1"), ("habitat/darkwood", "0.43 0.36 0.28 1")):
        material = asset.find(f"material[@name='{name}']")
        material.attrib.update(texture="habitat/oak", texrepeat="1 1", rgba=tint)
    floor = worldbody.find("geom[@name='ground_plane']")
    if floor is not None:
        floor.set("size", "300 220 1")
    sugar = worldbody.find("geom[@name='food_patch']")
    if sugar is not None:
        sugar.set("rgba", "0.92 0.88 0.76 1")
    floor_material = asset.find("material[@name='grid']")
    if floor_material is not None:
        floor_material.set("reflectance", "0")
        floor_material.set("texture", "habitat/oak")
        floor_material.set("texrepeat", "4 3")
        floor_material.set("rgba", "0.72 0.71 0.68 1")
        floor_material.set("specular", "0.05")
        floor_material.set("shininess", "0.05")
    floor_texture = asset.find("texture[@name='checker']")
    if floor_texture is not None:
        floor_texture.set("rgb1", "0.44 0.43 0.39")
        floor_texture.set("rgb2", "0.40 0.39 0.35")
    sky = asset.find("texture[@name='skybox']")
    if sky is not None:
        sky.set("rgb1", "0.18 0.24 0.29")
        sky.set("rgb2", "0.37 0.43 0.45")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.attrib.update(shadowsize="4096", offsamples="4", numslices="48", numstacks="24")
    visual.find("map").set("znear", "0.0001")
    headlight = visual.find("headlight")
    if headlight is not None:
        headlight.attrib.update(ambient="0.32 0.32 0.32", diffuse="0.35 0.35 0.35", specular="0.1 0.1 0.1")
    for name in ("room_wall_front_window", "room_wall_ceiling"):
        worldbody.find(f"geom[@name='{name}']").set("material", "habitat/wall")
    overview = worldbody.find("camera[@name='room_camera']")
    overview.attrib.update(pos="360 -500 330", xyaxes="0.811534 0.584305 0 -0.234474 0.325657 0.916")
    overview.set("fovy", "52")
    room_light = worldbody.find("light[@name='room_sun']")
    room_light.attrib.update(pos="-140 -100 205", dir="0.45 0.35 -1", directional="false", cutoff="85", exponent="1", diffuse="0.65 0.62 0.55", specular="0.12 0.12 0.10")
    table_light = worldbody.find("light[@name='table_light']")
    table_light.attrib.update(pos="0 0 110", cutoff="180", exponent="0", ambient="0.25 0.25 0.25", diffuse="0.42 0.42 0.42", specular="0.08 0.08 0.06", castshadow="false")
    fill = worldbody.find("light[@name='room_fill']")
    if fill is not None:
        worldbody.remove(fill)
    worldbody.find("geom[@name='window_glass']").set("pos", "297.90 52 132")
    worldbody.find("geom[@name='window_glass']").set("size", "0.02 55 48")
    for index in range(4):
        frame = worldbody.find(f"geom[@name='window_frame_{index}']")
        position = frame.get("pos").split()
        position[0] = "297.0"
        frame.set("pos", " ".join(position))
    for index, tip in enumerate(((215, -130, 31), (225, -132, 34), (220, -124, 38), (218, -136, 42))):
        name = f"plant_stem_{index}"
        if worldbody.find(f"geom[@name='{name}']") is None:
            _add_geom(worldbody, name, "capsule", (0, 0, 0), (0.35,), "habitat/leaf", fromto=" ".join(map(str, (220, -130, 22, *tip))))
    try:
        from .habitat_detail import add_habitat_details
    except ImportError:
        from habitat_detail import add_habitat_details
    add_habitat_details(root)


def habitat_metadata() -> dict:
    return {
        "schema": "flybrain-habitat-v2",
        "units": {
            "length": "millimeter",
            "time": "second",
            "mass": "gram",
            "odor_concentration": "isobutylene-equivalent ppm",
        },
        "room": {
            "half_extents_mm": list(ROOM_HALF_EXTENTS),
            "open_ceiling": False,
            "front_doorway_width_mm": 0.0,
            "flight_altitude_bounds_mm": [5.0, 208.0],
        },
        "airflow_mm_s": list(AIRFLOW_MM_S),
        "resources": RESOURCES,
        "sensory_model": {
            "odor": "bilateral antenna food-volatile concentration from deterministic finite-core advection-diffusion fields",
            "taste": "haustellum distance to resource taste geometry with configured margin",
            "vision": "FlyGym v2.1.0 721-ommatidium retina",
        },
    }


def write_habitat(path: Path) -> dict:
    metadata = habitat_metadata()
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
