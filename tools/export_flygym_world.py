from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from export_flygym_retina import export_retina_assets
from flight_assets import (
    FLYBODY_WING_PATTERN_FILENAME,
    add_flight_actuators,
    add_flight_fluid_geoms,
    write_aerodynamics,
)
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
from habitat_assets import add_habitat, write_habitat
from fly_appearance import improve_fly_appearance

FOOD_CENTER = (1.1, 0.0, 0.25)
FOOD_MARKER_RADIUS = 0.25
TASTE_RADIUS = 0.75
FEEDING_ACTUATOR_NAME = "fly/c_head-c_rostrum-pitch-feeding-position"
FEEDING_HAUSTELLUM_ACTUATOR_NAME = "fly/c_rostrum-c_haustellum-pitch-feeding-position"
FEEDING_JOINT_NAME = "fly/c_head-c_rostrum-pitch"
FEEDING_HAUSTELLUM_JOINT_NAME = "fly/c_rostrum-c_haustellum-pitch"
FEEDING_CONTROL_RANGE = (-1.0, 0.0)
FEEDING_ACTUATOR_NAMES = (
    FEEDING_ACTUATOR_NAME,
    FEEDING_HAUSTELLUM_ACTUATOR_NAME,
)
FEEDING_JOINT_NAMES = (
    FEEDING_JOINT_NAME,
    FEEDING_HAUSTELLUM_JOINT_NAME,
)
FEEDING_JOINT_RANGES = {
    FEEDING_JOINT_NAME: (-1.24, 0.183),
    FEEDING_HAUSTELLUM_JOINT_NAME: (-1.59, 0.7),
}
FEEDING_KP = 30.0
FEEDING_ROSTRUM_FULL_EXTENSION_ANGLE = -0.9
FEEDING_HAUSTELLUM_FULL_EXTENSION_ANGLE = 0.65
NATURAL_FEEDING_MATERIAL_NAME = "fly/antennaproboscis"
NEURAL_IO_FILENAME = "flywire_v783_neural_io.json"
POLICY_ARTIFACT_FILENAMES = (
    "flybody_flight_policy_v1.json",
    "flybody_flight_policy_v1.f32le",
    "flybody_flight_policy_fixture_v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/neuromechfly"))
    parser.add_argument("--flygym-repo", type=Path, default=Path("work/upstream/flygym"))
    parser.add_argument("--flybody-data-repo", type=Path, default=Path("work/upstream/flybody-data"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gait-samples", type=int, default=360)
    parser.add_argument(
        "--neural-io",
        type=Path,
        default=Path("assets/neuromechfly") / NEURAL_IO_FILENAME,
    )
    parser.add_argument(
        "--policy-assets",
        type=Path,
        default=Path("assets/neuromechfly"),
        help="directory containing optional pinned FlyBody policy artifacts to preserve",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def compose_world(output: Path, width: int, height: int) -> tuple[mj.MjModel, mj.MjData]:
    fly = NeuroMechFly(name="fly")
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.YAW_PITCH_ROLL,
    )
    fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)
    actuated = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated,
        ActuatorType.POSITION,
        neutral_input=KinematicPosePreset.NEUTRAL,
        kp=50.0,
        ctrlrange=(-3.14, 3.14),
    )
    fly.add_leg_adhesion(gain=20.0)
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.add_vision()
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        (0, 0, 0.8),
        Rotation3D("quat", (1, 0, 0, 0)),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        add_ground_contact_sensors=True,
    )
    world.save_xml_with_assets(output, "fly.xml")
    set_render_resolution(output / "fly.xml", width, height)
    add_world_extensions(output / "fly.xml")

    model = mj.MjModel.from_xml_path(str(output / "fly.xml"))
    data = mj.MjData(model)
    if model.nkey == 0:
        raise RuntimeError("the exported FlyGym model has no neutral keyframe")
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    return model, data


def set_render_resolution(path: Path, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    tree = ET.parse(path)
    root = tree.getroot()
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(width))
    global_visual.set("offheight", str(height))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")


def add_world_extensions(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    left_eye_camera = root.find(".//camera[@name='fly/l_eye_cam_camera']")
    if left_eye_camera is None:
        raise RuntimeError("exported model is missing the left eye camera")
    left_eye_camera.set("pos", "-0.03 0.38 0")
    for geom_name in ("fly/c_rostrum", "fly/c_haustellum"):
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise RuntimeError(f"exported model is missing feeding geom {geom_name}")
        geom.set("material", NATURAL_FEEDING_MATERIAL_NAME)
    for joint_name, (lower, upper) in FEEDING_JOINT_RANGES.items():
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise RuntimeError(f"exported model is missing feeding joint {joint_name}")
        joint.set("limited", "true")
        joint.set("range", f"{lower} {upper}")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("exported model has no worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "food_patch",
            "type": "sphere",
            "pos": " ".join(str(value) for value in FOOD_CENTER),
            "size": str(FOOD_MARKER_RADIUS),
            "rgba": "0.92 0.88 0.76 1",
            "contype": "0",
            "conaffinity": "0",
            "mass": "0",
        },
    )
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for actuator_name, joint_name, full_extension_angle in (
        (
            FEEDING_ACTUATOR_NAME,
            FEEDING_JOINT_NAME,
            FEEDING_ROSTRUM_FULL_EXTENSION_ANGLE,
        ),
        (
            FEEDING_HAUSTELLUM_ACTUATOR_NAME,
            FEEDING_HAUSTELLUM_JOINT_NAME,
            FEEDING_HAUSTELLUM_FULL_EXTENSION_ANGLE,
        ),
    ):
        gain = -FEEDING_KP * full_extension_angle
        ET.SubElement(
            actuator,
            "general",
            {
                "name": actuator_name,
                "joint": joint_name,
                "ctrlrange": " ".join(str(value) for value in FEEDING_CONTROL_RANGE),
                "forcelimited": "true",
                "forcerange": "-30 30",
                "biastype": "affine",
                "gainprm": str(gain),
                "biasprm": f"0 {-FEEDING_KP}",
            },
        )
    add_flight_actuators(root)
    add_flight_fluid_geoms(root, path.parent / "l_wing.stl")
    add_habitat(root)
    improve_fly_appearance(root, path.parent)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")


def object_names(model: mj.MjModel, object_type: mj.mjtObj, count: int) -> list[str]:
    return [mj.mj_id2name(model, object_type, index) or "" for index in range(count)]


def build_metadata(model: mj.MjModel, data: mj.MjData) -> dict:
    actuator_names = object_names(model, mj.mjtObj.mjOBJ_ACTUATOR, model.nu)
    actuators = []
    for index, name in enumerate(actuator_names):
        joint_id = int(model.actuator_trnid[index, 0])
        joint_name = (
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else None
        )
        actuators.append(
            {
                "index": index,
                "name": name,
                "joint_index": joint_id,
                "joint_name": joint_name,
                "control_range": [float(value) for value in model.actuator_ctrlrange[index]],
                "neutral_control": float(data.ctrl[index]),
            }
        )

    sensor_names = object_names(model, mj.mjtObj.mjOBJ_SENSOR, model.nsensor)
    sensors = [
        {
            "index": index,
            "name": name,
            "address": int(model.sensor_adr[index]),
            "dimension": int(model.sensor_dim[index]),
            "type": int(model.sensor_type[index]),
        }
        for index, name in enumerate(sensor_names)
    ]
    return {
        "schema": "flybrain-world-v2",
        "model": "NeuroMechFly",
        "physics": "MuJoCo",
        "timestep_seconds": float(model.opt.timestep),
        "counts": {
            "qpos": int(model.nq),
            "dofs": int(model.nv),
            "bodies": int(model.nbody),
            "joints": int(model.njnt),
            "actuators": int(model.nu),
            "sensors": int(model.nsensor),
            "cameras": int(model.ncam),
        },
        "neutral_qpos": [float(value) for value in data.qpos],
        "neutral_control": [float(value) for value in data.ctrl],
        "actuators": actuators,
        "sensors": sensors,
        "cameras": object_names(model, mj.mjtObj.mjOBJ_CAMERA, model.ncam),
        "environment": {
            "food_center": list(FOOD_CENTER),
            "taste_radius": TASTE_RADIUS,
            "taste_source_body": "fly/c_haustellum",
        },
        "brain_body_interface": {
            "feeding_actuator": FEEDING_ACTUATOR_NAME,
            "feeding_joint": FEEDING_JOINT_NAME,
            "feeding_actuators": list(FEEDING_ACTUATOR_NAMES),
            "feeding_joints": list(FEEDING_JOINT_NAMES),
            "control_range": list(FEEDING_CONTROL_RANGE),
            "full_extension_control": FEEDING_CONTROL_RANGE[0],
            "full_extension_pose": {
                FEEDING_JOINT_NAME: FEEDING_ROSTRUM_FULL_EXTENSION_ANGLE,
                FEEDING_HAUSTELLUM_JOINT_NAME: FEEDING_HAUSTELLUM_FULL_EXTENSION_ANGLE,
            },
            "neural_readout": "contralateral MN9 spikes",
        },
    }


def export_gait(output: Path, sample_count: int) -> None:
    if sample_count < 16:
        raise ValueError("gait sample count must be at least 16")
    steps = PreprogrammedSteps()
    phases = np.linspace(0.0, 2 * np.pi, sample_count, endpoint=False)
    model_dof_order = [2, 0, 1, 3, 4, 5, 6]
    joint_angles = {
        leg: [steps.get_joint_angles(leg, phase)[model_dof_order].tolist() for phase in phases]
        for leg in steps.legs
    }
    adhesion = {
        leg: [steps.get_adhesion_onoff(leg, phase) for phase in phases] for leg in steps.legs
    }
    payload = {
        "schema": "flybrain-gait-v1",
        "source": "FlyGym PreprogrammedSteps single_steps_untethered.pkl",
        "runtime_interpolation": "cyclic-linear",
        "legs": list(steps.legs),
        "joint_order": [
            "coxa_yaw",
            "coxa_pitch",
            "coxa_roll",
            "femur_pitch",
            "femur_roll",
            "tibia_pitch",
            "tarsus_pitch",
        ],
        "sample_count": sample_count,
        "cycle_frequency_hz": steps.step_cycle_frequency_hz,
        "tripod_phase_offsets_rad": [0.0, np.pi, 0.0, np.pi, 0.0, np.pi],
        "neutral_joint_angles": np.concatenate(
            [steps.default_pose[leg * 7 : (leg + 1) * 7][model_dof_order] for leg in range(6)]
        ).tolist(),
        "joint_angles": joint_angles,
        "adhesion": adhesion,
    }
    (output / "tripod_gait.json").write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_policy_artifacts(source_dir: Path) -> dict[str, bytes]:
    artifacts = {}
    missing = []
    for filename in POLICY_ARTIFACT_FILENAMES:
        path = source_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        artifacts[filename] = path.read_bytes()
    if missing:
        raise FileNotFoundError(
            "pinned FlyBody policy artifacts are required for regeneration: "
            + ", ".join(missing)
        )
    return artifacts


def main() -> int:
    args = parse_args()
    repo = args.flygym_repo.resolve()
    flybody_data_repo = args.flybody_data_repo.resolve()
    output = args.output.resolve()
    neural_io_source = args.neural_io.resolve()
    neural_io_bytes = neural_io_source.read_bytes()
    policy_artifacts = read_policy_artifacts(args.policy_assets.resolve())
    neural_io = json.loads(neural_io_bytes)
    if (
        neural_io.get("schema_version") != 1
        or neural_io.get("dataset", {}).get("materialization") != "783"
    ):
        raise ValueError(f"unsupported pinned neural I/O artifact: {neural_io_source}")
    if output == Path(output.anchor) or output == Path.home():
        raise ValueError(f"refusing unsafe output directory: {output}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / NEURAL_IO_FILENAME).write_bytes(neural_io_bytes)
    for filename, payload in policy_artifacts.items():
        (output / filename).write_bytes(payload)
    shutil.copytree(Path(__file__).resolve().parents[1] / "assets/materials", output / "textures")

    model, data = compose_world(output, args.width, args.height)
    export_gait(output, args.gait_samples)
    retina_metadata = export_retina_assets(repo, output / "vision")
    habitat_metadata = write_habitat(output / "habitat.json")
    aerodynamics_metadata = write_aerodynamics(
        output / "aerodynamics.json",
        output / "fly.xml",
        repo,
        flybody_data_repo / FLYBODY_WING_PATTERN_FILENAME,
    )
    metadata = build_metadata(model, data)
    metadata["retina"] = retina_metadata
    metadata["habitat"] = habitat_metadata
    metadata["aerodynamics"] = {
        "schema": aerodynamics_metadata["schema"],
        "model": aerodynamics_metadata["model"],
        "wingbeat": aerodynamics_metadata["wingbeat"],
        "actuators": aerodynamics_metadata["actuators"],
        "limitations": aerodynamics_metadata["limitations"],
    }
    metadata["neural_io"] = {
        "schema_version": neural_io["schema_version"],
        "materialization": neural_io["dataset"]["materialization"],
        "annotation_release": neural_io["dataset"]["annotation_release"],
        "annotation_commit": neural_io["dataset"]["annotation_commit"],
        "annotation_sha256": neural_io["dataset"]["annotation_sha256"],
        "artifact_sha256": hashlib.sha256(neural_io_bytes).hexdigest(),
        "selection_provenance": neural_io["dataset"]["selection_provenance"],
    }
    metadata["policy_artifacts"] = {
        "source_directory": str(args.policy_assets.resolve()),
        "files": {
            filename: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for filename, payload in policy_artifacts.items()
        },
    }
    metadata["source"] = {
        "repository": "https://github.com/NeLy-EPFL/flygym",
        "tag": subprocess.check_output(
            ["git", "-C", str(repo), "describe", "--tags", "--exact-match"],
            text=True,
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "license": "Apache-2.0",
    }
    shutil.copyfile(repo / "LICENSE", output / "LICENSE-FLYGYM")
    metadata["files"] = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["counts"] | {"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
