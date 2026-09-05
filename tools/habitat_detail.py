from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence


Point = tuple[float, float, float]

_BANANA_KNOTS: tuple[Point, ...] = (
    (25.0, 15.0, 1.4),
    (29.0, 17.0, 1.8),
    (34.0, 18.5, 2.0),
    (39.0, 18.0, 1.8),
    (43.0, 15.5, 1.4),
)


def _number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if abs(value) < 5e-12:
        value = 0.0
    return f"{value:.6g}"


def _numbers(values: Iterable[float | int]) -> str:
    return " ".join(_number(value) for value in values)


def _add_mesh(
    asset: ET.Element,
    name: str,
    vertices: Sequence[Point],
    faces: Sequence[tuple[int, int, int]],
    *,
    smoothnormal: bool = True,
    shell: bool = False,
) -> None:
    if asset.find(f"mesh[@name='{name}']") is not None:
        return
    _validate_mesh(vertices, faces, shell=shell)
    attributes = {
        "name": name,
        "vertex": _numbers(value for vertex in vertices for value in vertex),
        "face": _numbers(value for face in faces for value in face),
        "smoothnormal": "true" if smoothnormal else "false",
    }
    ET.SubElement(asset, "mesh", attributes)


def _validate_mesh(
    vertices: Sequence[Point],
    faces: Sequence[tuple[int, int, int]],
    *,
    shell: bool = False,
) -> None:
    edge_counts: dict[tuple[int, int], int] = {}
    signed_volume = 0.0
    for first, second, third in faces:
        if min(first, second, third) < 0 or max(first, second, third) >= len(vertices):
            raise ValueError("detail mesh face references a missing vertex")
        if len({first, second, third}) != 3:
            raise ValueError("detail mesh contains a zero-area face")
        a, b, c = vertices[first], vertices[second], vertices[third]
        cross = _cross(_sub(b, a), _sub(c, a))
        if sum(component * component for component in cross) <= 1e-14:
            raise ValueError("detail mesh contains a zero-area face")
        signed_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
        for edge in ((first, second), (second, third), (third, first)):
            key = tuple(sorted(edge))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("detail mesh has a non-manifold edge")
    if not shell and any(count != 2 for count in edge_counts.values()):
        raise ValueError("closed detail mesh has a boundary edge")
    if not shell and signed_volume <= 1e-10:
        raise ValueError("closed detail mesh is inward-wound")


def _add_mesh_geom(
    worldbody: ET.Element,
    name: str,
    mesh: str,
    material: str,
    position: Point = (0.0, 0.0, 0.0),
    *,
    euler: str | None = None,
) -> None:
    if worldbody.find(f"geom[@name='{name}']") is not None:
        return
    attributes = {
        "name": name,
        "type": "mesh",
        "mesh": mesh,
        "pos": _numbers(position),
        "material": material,
        "mass": "0",
        "contype": "0",
        "conaffinity": "0",
        "group": "0",
    }
    if euler is not None:
        attributes["euler"] = euler
    ET.SubElement(worldbody, "geom", attributes)


def _add_visual_primitive(
    worldbody: ET.Element,
    name: str,
    geom_type: str,
    position: Point,
    size: Sequence[float],
    material: str,
    *,
    euler: str | None = None,
    fromto: Sequence[float] | None = None,
) -> None:
    if worldbody.find(f"geom[@name='{name}']") is not None:
        return
    attributes = {
        "name": name,
        "type": geom_type,
        "pos": _numbers(position),
        "size": _numbers(size),
        "material": material,
        "mass": "0",
        "contype": "0",
        "conaffinity": "0",
        "group": "0",
    }
    if euler is not None:
        attributes["euler"] = euler
    if fromto is not None:
        attributes["fromto"] = _numbers(fromto)
        attributes.pop("pos")
    ET.SubElement(worldbody, "geom", attributes)


def _hide_original(worldbody: ET.Element, names: Iterable[str]) -> None:
    for name in names:
        geom = worldbody.find(f"geom[@name='{name}']")
        if geom is not None:
            geom.set("group", "5")


def _unit(vector: Point) -> Point:
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-12:
        return (1.0, 0.0, 0.0)
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _sub(left: Point, right: Point) -> Point:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _add(left: Point, right: Point) -> Point:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(vector: Point, factor: float) -> Point:
    return tuple(component * factor for component in vector)  # type: ignore[return-value]


def _cross(left: Point, right: Point) -> Point:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _tube_mesh(points: Sequence[Point], radii: Sequence[float], sides: int = 12) -> tuple[list[Point], list[tuple[int, int, int]]]:
    vertices: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    for index, point in enumerate(points):
        if index == 0:
            tangent = _sub(points[1], point)
        elif index == len(points) - 1:
            tangent = _sub(point, points[index - 1])
        else:
            tangent = _sub(points[index + 1], points[index - 1])
        tangent = _unit(tangent)
        normal = _cross(tangent, (0.0, 0.0, 1.0))
        if sum(component * component for component in normal) < 1e-12:
            normal = _cross(tangent, (0.0, 1.0, 0.0))
        normal = _unit(normal)
        binormal = _unit(_cross(tangent, normal))
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            offset = _add(
                _scale(normal, math.cos(angle) * radii[index]),
                _scale(binormal, math.sin(angle) * radii[index]),
            )
            vertices.append(_add(point, offset))
    for index in range(len(points) - 1):
        current = index * sides
        following = (index + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            faces.append((current + side, following + next_side, following + side))
            faces.append((current + side, current + next_side, following + next_side))
    start_center = len(vertices)
    vertices.append(points[0])
    end_center = len(vertices)
    vertices.append(points[-1])
    for side in range(sides):
        next_side = (side + 1) % sides
        faces.append((start_center, next_side, side))
        end = (len(points) - 1) * sides
        faces.append((end_center, end + side, end + next_side))
    return vertices, faces


def _catmull_rom(points: Sequence[Point], subdivisions: int = 4) -> list[Point]:
    result: list[Point] = []
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        for step in range(subdivisions):
            t = step / subdivisions
            t2 = t * t
            t3 = t2 * t
            result.append(
                tuple(
                    0.5
                    * (
                        2.0 * p1[axis]
                        + (-p0[axis] + p2[axis]) * t
                        + (2.0 * p0[axis] - 5.0 * p1[axis] + 4.0 * p2[axis] - p3[axis]) * t2
                        + (-p0[axis] + 3.0 * p1[axis] - 3.0 * p2[axis] + p3[axis]) * t3
                    )
                    for axis in range(3)
                )
            )  # type: ignore[arg-type]
    result.append(points[-1])
    return result


def _rounded_loop(half_x: float, half_y: float, radius: float, z: float, segments: int = 4) -> list[Point]:
    corners = (
        (half_x - radius, half_y - radius, 0.0),
        (-half_x + radius, half_y - radius, math.pi / 2.0),
        (-half_x + radius, -half_y + radius, math.pi),
        (half_x - radius, -half_y + radius, 3.0 * math.pi / 2.0),
    )
    points: list[Point] = []
    for center_x, center_y, start_angle in corners:
        for step in range(segments):
            angle = start_angle + math.pi * step / (2.0 * segments)
            points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle), z))
    return points


def _rounded_box_mesh(
    half_x: float,
    half_y: float,
    half_z: float,
    bevel: float,
    *,
    segments: int = 4,
) -> tuple[list[Point], list[tuple[int, int, int]]]:
    bevel = min(bevel, half_x * 0.5, half_y * 0.5, half_z * 0.8)
    loops = (
        _rounded_loop(half_x - bevel, half_y - bevel, bevel * 0.7, -half_z, segments),
        _rounded_loop(half_x, half_y, bevel, -half_z + bevel, segments),
        _rounded_loop(half_x, half_y, bevel, half_z - bevel, segments),
        _rounded_loop(half_x - bevel, half_y - bevel, bevel * 0.7, half_z, segments),
    )
    vertices = [vertex for loop in loops for vertex in loop]
    faces: list[tuple[int, int, int]] = []
    loop_size = len(loops[0])
    for loop_index in range(len(loops) - 1):
        current = loop_index * loop_size
        following = (loop_index + 1) * loop_size
        for side in range(loop_size):
            next_side = (side + 1) % loop_size
            faces.append((current + side, following + next_side, following + side))
            faces.append((current + side, current + next_side, following + next_side))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_z))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, half_z))
    bottom = 0
    top = (len(loops) - 1) * loop_size
    for side in range(loop_size):
        next_side = (side + 1) % loop_size
        faces.append((bottom_center, bottom + next_side, bottom + side))
        faces.append((top_center, top + side, top + next_side))
    return vertices, faces


def _lathe_mesh(
    profile: Sequence[tuple[float, float]],
    *,
    sides: int = 24,
    lobes: int = 0,
    lobe_amount: float = 0.0,
) -> tuple[list[Point], list[tuple[int, int, int]]]:
    vertices: list[Point] = []
    rings: list[int | list[int]] = []
    for radius, z in profile:
        if radius <= 1e-10:
            rings.append(len(vertices))
            vertices.append((0.0, 0.0, z))
            continue
        ring: list[int] = []
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            scale = 1.0 + lobe_amount * math.cos(lobes * angle + 0.25) if lobes else 1.0
            ring.append(len(vertices))
            vertices.append((radius * scale * math.cos(angle), radius * scale * math.sin(angle), z))
        rings.append(ring)
    faces: list[tuple[int, int, int]] = []
    for previous, current in zip(rings, rings[1:]):
        if isinstance(previous, int) and isinstance(current, list):
            for side in range(sides):
                faces.append((previous, current[(side + 1) % sides], current[side]))
        elif isinstance(previous, list) and isinstance(current, int):
            for side in range(sides):
                faces.append((current, previous[side], previous[(side + 1) % sides]))
        elif isinstance(previous, list) and isinstance(current, list):
            for side in range(sides):
                next_side = (side + 1) % sides
                faces.append((previous[side], current[next_side], current[side]))
                faces.append((previous[side], previous[next_side], current[next_side]))
    return vertices, faces


def _leaf_mesh(half_length: float, half_width: float, thickness: float) -> tuple[list[Point], list[tuple[int, int, int]]]:
    stations = 12
    sides = 8
    vertices: list[Point] = []
    rings: list[int | list[int]] = []
    for station in range(stations):
        fraction = station / (stations - 1)
        y = -half_length + 2.0 * half_length * fraction
        width = half_width * math.sin(math.pi * fraction) ** 0.7
        depth = thickness * math.sin(math.pi * fraction) ** 0.75
        if width < 1e-10:
            rings.append(len(vertices))
            vertices.append((0.0, y, 0.0))
            continue
        ring: list[int] = []
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            ring.append(len(vertices))
            vertices.append((width * math.cos(angle), y, depth * math.sin(angle)))
        rings.append(ring)
    faces: list[tuple[int, int, int]] = []
    for previous, current in zip(rings, rings[1:]):
        if isinstance(previous, int) and isinstance(current, list):
            for side in range(sides):
                faces.append((previous, current[side], current[(side + 1) % sides]))
        elif isinstance(previous, list) and isinstance(current, int):
            for side in range(sides):
                faces.append((current, previous[(side + 1) % sides], previous[side]))
        elif isinstance(previous, list) and isinstance(current, list):
            for side in range(sides):
                next_side = (side + 1) % sides
                faces.append((previous[side], current[side], current[next_side]))
                faces.append((previous[side], current[next_side], previous[next_side]))
    return vertices, faces


def _leaf_vein_mesh(half_length: float, half_width: float, thickness: float) -> tuple[list[Point], list[tuple[int, int, int]]]:
    components: list[tuple[list[Point], list[tuple[int, int, int]]]] = []
    components.append(_tube_mesh(((0.0, -half_length * 0.84, thickness * 0.8), (0.0, half_length * 0.84, thickness * 0.8)), (0.09, 0.07), 6))
    for fraction in (0.24, 0.42, 0.60, 0.77):
        y = -half_length + 2.0 * half_length * fraction
        width = half_width * math.sin(math.pi * fraction) ** 0.7
        ridge = thickness * math.sin(math.pi * fraction) ** 0.75
        reach = min(half_width * 0.82, width * 0.82)
        for side in (-1.0, 1.0):
            components.append(
                _tube_mesh(
                    ((0.0, y, ridge * 0.8), (side * reach, y + side * half_length * 0.07, 0.02)),
                    (0.055, 0.035),
                    6,
                )
            )
    vertices: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    for component_vertices, component_faces in components:
        offset = len(vertices)
        vertices.extend(component_vertices)
        faces.extend(tuple(index + offset for index in face) for face in component_faces)
    return vertices, faces


def _cup_mesh(sides: int = 28) -> tuple[list[Point], list[tuple[int, int, int]]]:
    outer_profile = ((5.8, -3.0), (6.7, -2.8), (7.2, -2.2), (7.35, 2.2), (7.2, 3.0))
    inner_profile = ((5.35, -2.35), (6.45, -1.9), (6.58, 2.3), (6.55, 2.72))
    vertices: list[Point] = []
    outer_rings: list[list[int]] = []
    inner_rings: list[list[int]] = []
    for radius, z in outer_profile:
        ring = []
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            ring.append(len(vertices))
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        outer_rings.append(ring)
    for radius, z in inner_profile:
        ring = []
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            ring.append(len(vertices))
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        inner_rings.append(ring)
    faces: list[tuple[int, int, int]] = []

    def connect(first: list[int], second: list[int], reverse: bool = False) -> None:
        for side in range(sides):
            next_side = (side + 1) % sides
            if reverse:
                faces.extend(((first[side], second[next_side], second[side]), (first[side], first[next_side], second[next_side])))
            else:
                faces.extend(((first[side], second[side], second[next_side]), (first[side], second[next_side], first[next_side])))

    for first, second in zip(outer_rings, outer_rings[1:]):
        connect(first, second, reverse=True)
    connect(outer_rings[-1], inner_rings[-1], reverse=True)
    for first, second in zip(reversed(inner_rings), reversed(inner_rings[:-1])):
        connect(first, second, reverse=True)
    inner_bottom_center = len(vertices)
    vertices.append((0.0, 0.0, inner_profile[0][1]))
    for side in range(sides):
        faces.append((inner_bottom_center, inner_rings[0][side], inner_rings[0][(side + 1) % sides]))
    outer_bottom_center = len(vertices)
    vertices.append((0.0, 0.0, outer_profile[0][1]))
    for side in range(sides):
        faces.append((outer_bottom_center, outer_rings[0][(side + 1) % sides], outer_rings[0][side]))
    return vertices, faces


def _shelf_bracket_mesh() -> tuple[list[Point], list[tuple[int, int, int]]]:
    thickness = 2.4
    triangle = ((-6.0, -9.0), (18.0, -9.0), (18.0, 9.0))
    vertices: list[Point] = []
    for x in (-thickness, thickness):
        vertices.extend((x, y, z) for y, z in triangle)
    faces = [
        (0, 2, 1),
        (3, 4, 5),
        (0, 4, 3),
        (0, 1, 4),
        (1, 5, 4),
        (1, 2, 5),
        (2, 3, 5),
        (2, 0, 3),
    ]
    return vertices, faces


def _ensure_orange_material(asset: ET.Element) -> None:
    texture_name = "detail/orange_pore"
    if asset.find(f"texture[@name='{texture_name}']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": texture_name,
                "type": "2d",
                "builtin": "flat",
                "width": "128",
                "height": "128",
                "rgb1": "0.78 0.20 0.025",
                "rgb2": "0.78 0.20 0.025",
                "mark": "random",
                "markrgb": "0.50 0.08 0.01",
                "random": "0.035",
            },
        )
    material_name = "habitat/orange_detail"
    if asset.find(f"material[@name='{material_name}']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": material_name,
                "texture": texture_name,
                "texrepeat": "4 4",
                "specular": "0.11",
                "shininess": "0.18",
            },
        )


def add_habitat_details(root: ET.Element) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("habitat details require asset and worldbody")
    if asset.find("mesh[@name='detail/banana_skin']") is not None:
        return

    _ensure_orange_material(asset)
    _hide_original(
        worldbody,
        (
            "banana_plate",
            "banana_segment_0",
            "resource_banana",
            "banana_segment_2",
            "banana_segment_3",
            "orange_plate",
            "resource_orange",
            "orange_leaf",
            "apple_plate",
            "resource_apple",
            "apple_stem",
            "apple_leaf",
            "wine_bowl",
            "resource_ferment",
            "water_dish",
            "yeast_dish",
            "plant_leaf_0",
            "plant_leaf_1",
            "plant_leaf_2",
            "plant_leaf_3",
            "table_top",
            "wall_shelf",
            "shelf_bracket_0",
            "shelf_bracket_1",
        ),
    )

    banana_points = _catmull_rom(_BANANA_KNOTS, subdivisions=5)
    banana_radii = [0.58 + 0.5 * math.sin(math.pi * index / (len(banana_points) - 1)) ** 0.8 for index in range(len(banana_points))]
    banana_vertices, banana_faces = _tube_mesh(banana_points, banana_radii, sides=16)
    _add_mesh(asset, "detail/banana_skin", banana_vertices, banana_faces)
    _add_mesh_geom(worldbody, "detail_banana_skin", "detail/banana_skin", "habitat/banana")
    banana_stalk_vertices, banana_stalk_faces = _tube_mesh(((24.3, 14.7, 1.25), (25.4, 15.2, 2.35)), (0.34, 0.23), sides=10)
    _add_mesh(asset, "detail/banana_stalk", banana_stalk_vertices, banana_stalk_faces)
    _add_mesh_geom(worldbody, "detail_banana_stalk", "detail/banana_stalk", "habitat/darkwood")

    for name, position, radius in (
        ("banana", (32.0, 18.0, 0.35), 8.5),
        ("orange", (-48.0, 38.0, 0.4), 10.0),
        ("apple", (105.0, 70.0, 51.5), 12.0),
        ("water", (-58.0, -34.0, 0.35), 8.0),
        ("yeast", (-178.0, 145.0, 71.4), 8.0),
    ):
        vertices, faces = _lathe_mesh(
            (
                (0.0, -0.35),
                (radius * 0.88, -0.35),
                (radius * 0.98, -0.14),
                (radius, 0.10),
                (radius * 0.91, 0.28),
                (radius * 0.78, 0.23),
                (radius * 0.55, -0.02),
                (0.0, -0.04),
            ),
            sides=32,
        )
        _add_mesh(asset, f"detail/{name}_plate", vertices, faces, smoothnormal=False)
        _add_mesh_geom(worldbody, f"detail_{name}_plate", f"detail/{name}_plate", "habitat/ceramic", position)

    orange_vertices, orange_faces = _lathe_mesh(
        (
            (0.0, -6.0),
            (3.2, -5.8),
            (5.4, -4.8),
            (6.2, -2.8),
            (6.35, 0.0),
            (6.15, 3.0),
            (5.2, 4.9),
            (3.1, 5.9),
            (0.0, 6.1),
        ),
        sides=32,
    )
    _add_mesh(asset, "detail/orange_skin", orange_vertices, orange_faces)
    _add_mesh_geom(worldbody, "detail_orange_skin", "detail/orange_skin", "habitat/orange_detail", (-48.0, 38.0, 6.5))

    apple_vertices, apple_faces = _lathe_mesh(
        (
            (0.0, -7.0),
            (0.48, -6.55),
            (0.78, -5.25),
            (0.96, -3.0),
            (1.0, 0.0),
            (0.93, 3.0),
            (0.76, 5.0),
            (0.50, 6.2),
            (0.24, 6.55),
            (0.0, 5.95),
        ),
        sides=40,
        lobes=5,
        lobe_amount=0.065,
    )
    apple_vertices = [(x * 7.2, y * 7.2, z) for x, y, z in apple_vertices]
    _add_mesh(asset, "detail/apple_skin", apple_vertices, apple_faces)
    _add_mesh_geom(worldbody, "detail_apple_skin", "detail/apple_skin", "habitat/apple", (105.0, 70.0, 58.0))
    apple_stalk_vertices, apple_stalk_faces = _tube_mesh(((0.0, 0.0, 5.55), (0.25, 0.05, 8.55)), (0.46, 0.24), sides=10)
    _add_mesh(asset, "detail/apple_stalk", apple_stalk_vertices, apple_stalk_faces)
    _add_mesh_geom(worldbody, "detail_apple_stalk", "detail/apple_stalk", "habitat/darkwood", (105.0, 70.0, 58.0))

    cup_vertices, cup_faces = _cup_mesh()
    _add_mesh(asset, "detail/glass_cup", cup_vertices, cup_faces, smoothnormal=False)
    _add_mesh_geom(worldbody, "detail_glass_cup", "detail/glass_cup", "habitat/glass", (151.0, 62.0, 58.0))
    _add_visual_primitive(worldbody, "detail_wine_liquid", "cylinder", (151.0, 62.0, 55.35), (6.35, 0.12), "habitat/juice")

    for mesh_name, half_length, half_width, thickness in (
        ("detail/leaf_long", 8.0, 2.0, 0.42),
        ("detail/leaf_small", 3.0, 1.2, 0.22),
    ):
        leaf_vertices, leaf_faces = _leaf_mesh(half_length, half_width, thickness)
        _add_mesh(asset, mesh_name, leaf_vertices, leaf_faces)
        vein_vertices, vein_faces = _leaf_vein_mesh(half_length, half_width, thickness)
        _add_mesh(asset, mesh_name.replace("leaf_", "leaf_veins_"), vein_vertices, vein_faces, smoothnormal=False)

    for index, (position, euler) in enumerate(
        (
            ((215.0, -130.0, 31.0), "0.4 0 -0.5"),
            ((225.0, -132.0, 34.0), "-0.35 0 0.7"),
            ((220.0, -124.0, 38.0), "0.2 0 1.1"),
            ((218.0, -136.0, 42.0), "-0.2 0 -1.0"),
        )
    ):
        _add_mesh_geom(worldbody, f"detail_plant_leaf_{index}", "detail/leaf_long", "habitat/leaf", position, euler=euler)
        _add_mesh_geom(worldbody, f"detail_plant_leaf_veins_{index}", "detail/leaf_veins_long", "habitat/darkwood", position, euler=euler)

    _add_mesh_geom(worldbody, "detail_orange_leaf", "detail/leaf_small", "habitat/leaf", (-44.0, 39.0, 12.1), euler="0.2 0.5 0.4")
    _add_mesh_geom(worldbody, "detail_orange_leaf_veins", "detail/leaf_veins_small", "habitat/darkwood", (-44.0, 39.0, 12.1), euler="0.2 0.5 0.4")
    _add_mesh_geom(worldbody, "detail_apple_leaf", "detail/leaf_small", "habitat/leaf", (108.0, 70.5, 66.5), euler="0.1 0.35 0.2")
    _add_mesh_geom(worldbody, "detail_apple_leaf_veins", "detail/leaf_veins_small", "habitat/darkwood", (108.0, 70.5, 66.5), euler="0.1 0.35 0.2")

    table_vertices, table_faces = _rounded_box_mesh(72.0, 52.0, 3.0, 2.0)
    _add_mesh(asset, "detail/table_top", table_vertices, table_faces, smoothnormal=False)
    _add_mesh_geom(worldbody, "detail_table_top", "detail/table_top", "habitat/wood", (120.0, 70.0, 48.0))
    for name, position, size in (
        ("detail_table_apron_front", (120.0, 20.5, 44.0), (67.0, 1.5, 4.0)),
        ("detail_table_apron_back", (120.0, 119.5, 44.0), (67.0, 1.5, 4.0)),
        ("detail_table_apron_left", (51.5, 70.0, 44.0), (1.5, 47.0, 4.0)),
        ("detail_table_apron_right", (188.5, 70.0, 44.0), (1.5, 47.0, 4.0)),
    ):
        _add_visual_primitive(worldbody, name, "box", position, size, "habitat/darkwood")
    for index, position in enumerate(((55.0, 25.0, 44.3), (185.0, 25.0, 44.3), (55.0, 115.0, 44.3), (185.0, 115.0, 44.3))):
        _add_visual_primitive(worldbody, f"detail_table_joint_{index}", "cylinder", position, (2.0, 0.35), "habitat/metal")

    shelf_vertices, shelf_faces = _rounded_box_mesh(64.0, 22.0, 3.0, 1.5)
    _add_mesh(asset, "detail/shelf", shelf_vertices, shelf_faces, smoothnormal=False)
    _add_mesh_geom(worldbody, "detail_wall_shelf", "detail/shelf", "habitat/wood", (-180.0, 150.0, 68.0))
    bracket_vertices, bracket_faces = _shelf_bracket_mesh()
    _add_mesh(asset, "detail/shelf_bracket", bracket_vertices, bracket_faces, smoothnormal=False)
    for index, x in enumerate((-235.0, -125.0)):
        _add_mesh_geom(worldbody, f"detail_shelf_bracket_{index}", "detail/shelf_bracket", "habitat/metal", (x, 150.0, 56.0))

    for name, position, size in (
        ("detail_baseboard_back", (0.0, 217.5, 3.0), (300.0, 1.5, 3.0)),
        ("detail_baseboard_left", (-297.5, 0.0, 3.0), (1.5, 220.0, 3.0)),
        ("detail_baseboard_right", (297.5, 0.0, 3.0), (1.5, 220.0, 3.0)),
        ("detail_baseboard_front_left", (-205.0, -217.5, 3.0), (95.0, 1.5, 3.0)),
        ("detail_baseboard_front_right", (205.0, -217.5, 3.0), (95.0, 1.5, 3.0)),
        ("detail_wall_panel_rail_low", (0.0, 217.25, 32.0), (250.0, 0.65, 0.7)),
        ("detail_wall_panel_rail_high", (0.0, 217.25, 94.0), (250.0, 0.65, 0.7)),
    ):
        _add_visual_primitive(worldbody, name, "box", position, size, "habitat/darkwood")
    for index, x in enumerate((-250.0, -125.0, 0.0, 125.0, 250.0)):
        _add_visual_primitive(worldbody, f"detail_wall_panel_stile_{index}", "box", (x, 217.25, 63.0), (0.65, 0.65, 31.0), "habitat/darkwood")
