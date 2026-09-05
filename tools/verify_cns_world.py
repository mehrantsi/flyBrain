"""Run the deterministic MaleCNS/world gate in both output conditions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "flybrain.cns-world-check"
VERIFICATION_SCHEMA = "flybrain.cns-world-verification-v1"
DEFAULT_DURATION_SECONDS = 2.0
DEFAULT_CONTROL_HZ = 500.0
DEFAULT_START_FOOD_DISTANCE = 40.0
DEFAULT_MIN_FORWARD_DISPLACEMENT_MM = 0.5
DEFAULT_MIN_ALTITUDE_RANGE_MM = 2.0
DEFAULT_MIN_LESION_WORLD_DELTA_MM = 0.25
DEFAULT_MAX_BOUNDARY_VIOLATION_MM = 0.0
DEFAULT_MAX_COLLISION_REFLEX_SECONDS = 5.0
CONDITIONS = ("intact", "neural-output-disconnected")
AIRBORNE_MODES = {"TAKEOFF", "CRUISE"}


class VerificationError(RuntimeError):
    """A cns-check report is malformed or a run failed."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise VerificationError(f"{name} must be finite")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerificationError(f"{name} must be a non-negative integer")
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{name} must be an object")
    return value


def _need(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise VerificationError(f"{name} is missing")
    return mapping[name]


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise VerificationError(f"{name} must contain {size} values")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise VerificationError(f"{name} must be a SHA-256 digest")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise VerificationError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _pack_neuron_count(pack: Path) -> int:
    path = pack / "manifest.json"
    try:
        manifest = _object(json.loads(path.read_text()), "CNS pack manifest")
    except FileNotFoundError as error:
        raise VerificationError(f"CNS pack manifest is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise VerificationError(f"CNS pack manifest is not JSON: {path}") from error
    return _integer(_need(manifest, "neuron_count"), "CNS pack neuron_count")


def _load_report(path: Path) -> Mapping[str, Any]:
    try:
        return _object(json.loads(path.read_text()), f"cns-check report {path}")
    except FileNotFoundError as error:
        raise VerificationError(f"cns-check report is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise VerificationError(f"cns-check report is not JSON: {path}") from error


def run_cns_check(
    rust_bin: Path,
    assets: Path,
    cns_pack: Path,
    *,
    duration_seconds: float,
    control_hz: float,
    start_food_distance: float,
    condition: str,
    parameters: Path | None = None,
) -> Mapping[str, Any]:
    """Run one cns-check condition and return its report."""

    if condition not in CONDITIONS:
        raise VerificationError(f"unknown cns-check condition: {condition}")
    with tempfile.TemporaryDirectory(prefix="flybrain-cns-world-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = [
            str(rust_bin),
            "cns-check",
            "--assets",
            str(assets),
            "--pack",
            str(cns_pack),
            "--duration-seconds",
            str(duration_seconds),
            "--control-hz",
            str(control_hz),
            "--start-food-distance",
            str(start_food_distance),
            "--output",
            str(report_path),
        ]
        if parameters is not None:
            command.extend(("--parameters", str(parameters)))
        if condition == "neural-output-disconnected":
            command.append("--disconnect-motor-outputs")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise VerificationError(
                f"cns-check failed for {condition} with exit code {completed.returncode}: {details}"
            )
        return _load_report(report_path)


def _rates(motor: Mapping[str, Any], name: str, index: int) -> tuple[float, float]:
    rates = _vector(_need(motor, name), 2, f"samples[{index}].cns_motor.{name}")
    if min(rates) < 0.0:
        raise VerificationError(f"samples[{index}].cns_motor.{name} must be non-negative")
    return rates


def _sample_activity(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    population_spikes = motor_spikes = 0
    rate_activity = flight_activation = walking_activation = steering_activity = 0.0
    first_motor = first_command = None
    outputs_connected = None
    modes: set[str] = set()
    sample_min = [math.inf] * 3
    sample_max = [-math.inf] * 3
    previous_time = None
    sample_period = 0.0
    reflex_start = None
    maximum_reflex = 0.0
    for index, sample in enumerate(samples):
        time = _number(_need(sample, "time_seconds"), f"samples[{index}].time_seconds")
        if previous_time is not None:
            if time <= previous_time:
                raise VerificationError("sample times must be strictly increasing")
            sample_period = time - previous_time
        previous_time = time
        delta = _integer(
            _need(sample, "population_spike_delta"),
            f"samples[{index}].population_spike_delta",
        )
        population_spikes += delta
        position = _vector(_need(sample, "root_position"), 3, f"samples[{index}].root_position")
        for axis, value in enumerate(position):
            sample_min[axis] = min(sample_min[axis], value)
            sample_max[axis] = max(sample_max[axis], value)
        mode = _need(sample, "flight_mode")
        if not isinstance(mode, str):
            raise VerificationError(f"samples[{index}].flight_mode must be a string")
        modes.add(mode.upper())
        motor = _object(_need(sample, "cns_motor"), f"samples[{index}].cns_motor")
        flight = _rates(motor, "flight_power_hz", index)
        steering_rates = _rates(motor, "wing_steering_hz", index)
        walking = _rates(motor, "walking_hz", index)
        landing = _rates(motor, "landing_hz", index)
        motor_delta = _integer(
            _need(motor, "spike_delta"), f"samples[{index}].cns_motor.spike_delta"
        )
        motor_spikes += motor_delta
        flight_drive = _number(
            _need(motor, "flight_activation"), f"samples[{index}].cns_motor.flight_activation"
        )
        walking_drive = _number(
            _need(motor, "walking_activation"),
            f"samples[{index}].cns_motor.walking_activation",
        )
        steering = _number(_need(motor, "steering"), f"samples[{index}].cns_motor.steering")
        if min(flight_drive, walking_drive) < 0.0:
            raise VerificationError("CNS activations must be non-negative")
        connected = _need(motor, "outputs_connected")
        if not isinstance(connected, bool):
            raise VerificationError(f"samples[{index}].cns_motor.outputs_connected must be boolean")
        if outputs_connected is None:
            outputs_connected = connected
        elif connected is not outputs_connected:
            raise VerificationError("cns_motor.outputs_connected changed within one report")
        rates_total = sum(flight) + sum(steering_rates) + sum(walking) + sum(landing)
        rate_activity += rates_total
        flight_activation += flight_drive
        walking_activation += walking_drive
        steering_activity += abs(steering)
        if (motor_delta or rates_total) and first_motor is None:
            first_motor = index
        if flight_drive + walking_drive + abs(steering) > 0.0 and first_command is None:
            first_command = index
        reflex = _need(sample, "collision_reflex_active")
        if not isinstance(reflex, bool):
            raise VerificationError(f"samples[{index}].collision_reflex_active must be boolean")
        if reflex:
            reflex_start = time if reflex_start is None else reflex_start
        elif reflex_start is not None:
            maximum_reflex = max(maximum_reflex, time - reflex_start)
            reflex_start = None
    if reflex_start is not None and previous_time is not None:
        maximum_reflex = max(maximum_reflex, previous_time - reflex_start + sample_period)
    return {
        "sample_population_spikes": population_spikes,
        "sample_motor_output_spikes": motor_spikes,
        "sample_motor_rate_activity": rate_activity,
        "sample_flight_activation_sum": flight_activation,
        "sample_walking_activation_sum": walking_activation,
        "sample_steering_activity_sum": steering_activity,
        "first_motor_window": first_motor,
        "first_command_window": first_command,
        "sample_outputs_connected": outputs_connected,
        "flight_modes": sorted(modes),
        "sample_minimum_position": sample_min,
        "sample_maximum_position": sample_max,
        "maximum_collision_reflex_seconds": maximum_reflex,
    }


def _room_bounds(value: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise VerificationError("room_bounds_mm must contain lower and upper corners")
    lower = _vector(value[0], 3, "room_bounds_mm[0]")
    upper = _vector(value[1], 3, "room_bounds_mm[1]")
    if any(lower[index] >= upper[index] for index in range(3)):
        raise VerificationError("room_bounds_mm lower corner must be below upper corner")
    return lower, upper


def _world_extrema(
    minimum: Sequence[float],
    maximum: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[float, float, float]:
    altitude_range = maximum[2] - minimum[2]
    clearance = min(
        *(minimum[axis] - lower[axis] for axis in range(3)),
        *(upper[axis] - maximum[axis] for axis in range(3)),
    )
    violation = max(
        0.0,
        *(lower[axis] - minimum[axis] for axis in range(3)),
        *(maximum[axis] - upper[axis] for axis in range(3)),
    )
    return altitude_range, clearance, violation


def evaluate_report(
    report: Mapping[str, Any],
    *,
    condition: str,
    duration_seconds: float,
    control_hz: float,
    min_forward_displacement_mm: float,
    min_altitude_range_mm: float,
    max_boundary_violation_mm: float,
    max_collision_reflex_seconds: float = DEFAULT_MAX_COLLISION_REFLEX_SECONDS,
    expected_neuron_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate one cns-check report without running a binary."""

    if condition not in CONDITIONS:
        raise VerificationError(f"unknown condition: {condition}")
    if report.get("schema") != TRACE_SCHEMA:
        raise VerificationError(f"report schema must be {TRACE_SCHEMA}")
    brain = _object(_need(report, "brain"), "brain")
    _object(_need(report, "initial_state"), "initial_state")
    summary = _object(_need(report, "summary"), "summary")
    raw_samples = _need(report, "samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise VerificationError("report samples must be a non-empty array")
    samples = [_object(sample, f"samples[{index}]") for index, sample in enumerate(raw_samples)]
    requested_duration = _number(_need(report, "duration_seconds"), "duration_seconds")
    requested_control_hz = _number(_need(report, "control_hz"), "control_hz")
    if abs(requested_duration - duration_seconds) > 1e-9:
        raise VerificationError("report duration_seconds does not match the request")
    if abs(requested_control_hz - control_hz) > 1e-9:
        raise VerificationError("report control_hz does not match the request")
    lower, upper = _room_bounds(_need(report, "room_bounds_mm"))
    model = _need(brain, "model")
    neurons = _integer(_need(brain, "neurons"), "brain.neurons")
    sensory_neurons = _integer(_need(brain, "sensory_neurons"), "brain.sensory_neurons")
    connected = _need(brain, "motor_outputs_connected")
    if not isinstance(model, str) or "male-cns" not in model.lower():
        raise VerificationError("brain.model must identify the MaleCNS model")
    if not isinstance(connected, bool):
        raise VerificationError("brain.motor_outputs_connected must be boolean")
    initial_hash = _digest(_need(summary, "initial_state_sha256"), "summary.initial_state_sha256")
    population_spikes = _integer(_need(summary, "population_spikes"), "summary.population_spikes")
    motor_spikes = _integer(_need(summary, "motor_output_spikes"), "summary.motor_output_spikes")
    source = _need(summary, "motor_output_source")
    if not isinstance(source, str):
        raise VerificationError("summary.motor_output_source must be a string")
    initial = _vector(_need(summary, "initial_position_mm"), 3, "summary.initial_position_mm")
    final = _vector(_need(summary, "final_position_mm"), 3, "summary.final_position_mm")
    minimum = _vector(_need(summary, "minimum_position_mm"), 3, "summary.minimum_position_mm")
    maximum = _vector(_need(summary, "maximum_position_mm"), 3, "summary.maximum_position_mm")
    if any(minimum[index] > maximum[index] for index in range(3)):
        raise VerificationError("summary position extrema are invalid")
    activity = _sample_activity(samples)
    if any(
        activity["sample_minimum_position"][axis] < minimum[axis] - 1e-9
        or activity["sample_maximum_position"][axis] > maximum[axis] + 1e-9
        for axis in range(3)
    ):
        raise VerificationError("sample positions exceed summary position extrema")
    altitude_range, clearance, violation = _world_extrema(minimum, maximum, lower, upper)
    forward_distance = _number(
        _need(summary, "forward_flight_distance_mm"), "summary.forward_flight_distance_mm"
    )
    flight_seconds = _number(_need(summary, "flight_seconds"), "summary.flight_seconds")
    if forward_distance < 0.0 or flight_seconds < 0.0:
        raise VerificationError("summary flight metrics must be non-negative")
    expected_connected = condition == "intact"
    activation = (
        activity["sample_flight_activation_sum"]
        + activity["sample_walking_activation_sum"]
        + activity["sample_steering_activity_sum"]
    )
    first_command = activity["first_command_window"]
    first_motor = activity["first_motor_window"]
    checks = {
        "whole_cns_selected": neurons > 0 and sensory_neurons > 0,
        "whole_cns_matches_pack": expected_neuron_count is None or neurons == expected_neuron_count,
        "brain_output_connection_matches_condition": connected is expected_connected,
        "neural_output_source_matches_condition": source
        == ("whole-cns-spikes" if expected_connected else "disconnected"),
        "sample_output_connection_matches_condition": activity["sample_outputs_connected"]
        is expected_connected,
        "whole_cns_spikes_present": population_spikes > 0
        and activity["sample_population_spikes"] > 0,
        "motor_output_spikes_present": motor_spikes > 0
        and activity["sample_motor_output_spikes"] > 0,
        "population_spikes_cover_motor_spikes": population_spikes >= motor_spikes,
        "neural_motor_activity_present": activity["sample_motor_rate_activity"] > 0.0
        and (activation > 0.0 if expected_connected else activation == 0.0),
        "command_follows_cns_activity": not expected_connected
        or (first_command is not None and first_motor is not None and first_command >= first_motor),
        "world_moved": not expected_connected or _distance(initial, final) > 0.0,
        "airborne_mode_observed": not expected_connected
        or bool(AIRBORNE_MODES.intersection(activity["flight_modes"])),
        "airborne_time_observed": not expected_connected or flight_seconds > 0.0,
        "forward_flight_displacement": not expected_connected
        or forward_distance >= min_forward_displacement_mm,
        "altitude_varied": not expected_connected or altitude_range >= min_altitude_range_mm,
        "disconnected_stays_grounded": expected_connected
        or (
            flight_seconds == 0.0
            and forward_distance == 0.0
            and not AIRBORNE_MODES.intersection(activity["flight_modes"])
        ),
        "boundaries_stable": clearance >= 0.0 and violation <= max_boundary_violation_mm,
        "collision_reflex_stable": not expected_connected
        or activity["maximum_collision_reflex_seconds"] <= max_collision_reflex_seconds,
    }
    return {
        "condition": condition,
        "checks": checks,
        "metrics": {
            "model": model,
            "neurons": neurons,
            "sensory_neurons": sensory_neurons,
            "motor_outputs_connected": connected,
            "initial_state_sha256": initial_hash,
            "population_spikes": population_spikes,
            "motor_output_spikes": motor_spikes,
            "motor_output_source": source,
            "initial_position": list(initial),
            "final_position": list(final),
            "world_displacement_mm": _distance(initial, final),
            "forward_flight_distance_mm": forward_distance,
            "altitude_range_mm": altitude_range,
            "flight_seconds": flight_seconds,
            "minimum_boundary_clearance_mm": clearance,
            "maximum_boundary_violation_mm": violation,
            "room_bounds_mm": [list(lower), list(upper)],
            "maximum_collision_reflex_seconds": activity["maximum_collision_reflex_seconds"],
            "sample_activity": {
                key: value
                for key, value in activity.items()
                if not key.startswith("sample_minimum") and not key.startswith("sample_maximum")
            },
        },
        "passed": all(checks.values()),
    }


def compare_intact_and_disconnected(
    intact: Mapping[str, Any], disconnected: Mapping[str, Any], *, min_world_delta_mm: float
) -> dict[str, Any]:
    """Require equal initial state and a changed closed-loop body trajectory."""

    left = _object(intact.get("metrics"), "intact metrics")
    right = _object(disconnected.get("metrics"), "disconnected metrics")
    left_final = _vector(left["final_position"], 3, "intact final position")
    right_final = _vector(right["final_position"], 3, "disconnected final position")
    delta = _distance(left_final, right_final)
    checks = {
        "matched_initial_state": left["initial_state_sha256"] == right["initial_state_sha256"],
        "matched_initial_position": left["initial_position"] == right["initial_position"],
        "intact_outputs_connected": left["motor_outputs_connected"] is True,
        "disconnected_outputs_disabled": right["motor_outputs_connected"] is False,
        "neural_activity_survives_disconnection": right["population_spikes"] > 0
        and right["motor_output_spikes"] > 0
        and right["sample_activity"]["sample_population_spikes"] > 0
        and right["sample_activity"]["sample_motor_output_spikes"] > 0,
        "world_behavior_changes": delta >= min_world_delta_mm,
    }
    return {
        "checks": checks,
        "metrics": {
            "final_position_delta_mm": delta,
            "intact_initial_state_sha256": left["initial_state_sha256"],
            "disconnected_initial_state_sha256": right["initial_state_sha256"],
            "intact_population_spikes": left["population_spikes"],
            "disconnected_population_spikes": right["population_spikes"],
            "intact_motor_output_spikes": left["motor_output_spikes"],
            "disconnected_motor_output_spikes": right["motor_output_spikes"],
        },
        "passed": all(checks.values()),
    }


def verify_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    duration_seconds: float,
    control_hz: float,
    min_forward_displacement_mm: float = DEFAULT_MIN_FORWARD_DISPLACEMENT_MM,
    min_altitude_range_mm: float = DEFAULT_MIN_ALTITUDE_RANGE_MM,
    min_lesion_world_delta_mm: float = DEFAULT_MIN_LESION_WORLD_DELTA_MM,
    max_boundary_violation_mm: float = DEFAULT_MAX_BOUNDARY_VIOLATION_MM,
    max_collision_reflex_seconds: float = DEFAULT_MAX_COLLISION_REFLEX_SECONDS,
    expected_neuron_count: int | None = None,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    failures: list[str] = []
    for condition in CONDITIONS:
        if condition not in reports:
            failures.append(f"missing report for {condition}")
            continue
        try:
            result = evaluate_report(
                reports[condition],
                condition=condition,
                duration_seconds=duration_seconds,
                control_hz=control_hz,
                min_forward_displacement_mm=min_forward_displacement_mm,
                min_altitude_range_mm=min_altitude_range_mm,
                max_boundary_violation_mm=max_boundary_violation_mm,
                max_collision_reflex_seconds=max_collision_reflex_seconds,
                expected_neuron_count=expected_neuron_count,
            )
        except VerificationError as error:
            result = {"condition": condition, "passed": False, "error": str(error)}
        evaluations[condition] = result
        if not result.get("passed", False):
            failures.append(f"trace checks failed for {condition}")
    if all(
        condition in evaluations and evaluations[condition].get("passed")
        for condition in CONDITIONS
    ):
        comparison = compare_intact_and_disconnected(
            evaluations["intact"],
            evaluations["neural-output-disconnected"],
            min_world_delta_mm=min_lesion_world_delta_mm,
        )
    else:
        comparison = {"checks": {}, "metrics": {}, "passed": False}
    if not comparison["passed"]:
        failures.append(
            "neural-output disconnection did not produce the required matched-run change"
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": 1,
        "duration_seconds": duration_seconds,
        "control_hz": control_hz,
        "thresholds": {
            "min_forward_displacement_mm": min_forward_displacement_mm,
            "min_altitude_range_mm": min_altitude_range_mm,
            "min_lesion_world_delta_mm": min_lesion_world_delta_mm,
            "max_boundary_violation_mm": max_boundary_violation_mm,
            "max_collision_reflex_seconds": max_collision_reflex_seconds,
        },
        "expected_neuron_count": expected_neuron_count,
        "evaluations": evaluations,
        "lesion_comparison": comparison,
        "failures": failures,
        "passed": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-bin", type=Path, default=Path("target/release/flybrain-world"))
    parser.add_argument("--assets", type=Path, default=Path("assets/neuromechfly"))
    parser.add_argument("--cns-pack", type=Path, default=Path("outputs/packs/male_cns_v1"))
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--intact-report", type=Path)
    parser.add_argument("--disconnected-report", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--start-food-distance", type=float, default=DEFAULT_START_FOOD_DISTANCE)
    parser.add_argument(
        "--min-forward-displacement-mm", type=float, default=DEFAULT_MIN_FORWARD_DISPLACEMENT_MM
    )
    parser.add_argument(
        "--min-altitude-range-mm", type=float, default=DEFAULT_MIN_ALTITUDE_RANGE_MM
    )
    parser.add_argument(
        "--min-lesion-world-delta-mm", type=float, default=DEFAULT_MIN_LESION_WORLD_DELTA_MM
    )
    parser.add_argument(
        "--max-boundary-violation-mm", type=float, default=DEFAULT_MAX_BOUNDARY_VIOLATION_MM
    )
    parser.add_argument(
        "--max-collision-reflex-seconds",
        type=float,
        default=DEFAULT_MAX_COLLISION_REFLEX_SECONDS,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.intact_report is None) != (args.disconnected_report is None):
        raise SystemExit("--intact-report and --disconnected-report must be supplied together")
    if args.output and args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    for name, value in vars(args).items():
        if name.startswith(("min_", "max_")) and (not math.isfinite(value) or value < 0.0):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and non-negative")
    if (
        args.duration_seconds <= 0.0
        or args.control_hz <= 0.0
        or args.start_food_distance <= 0.0
        or not all(
            math.isfinite(value)
            for value in (args.duration_seconds, args.control_hz, args.start_food_distance)
        )
    ):
        raise SystemExit(
            "duration, control_hz, and start_food_distance must be finite and positive"
        )
    try:
        expected_neuron_count = _pack_neuron_count(args.cns_pack)
        if args.intact_report is not None:
            reports = {
                "intact": _load_report(args.intact_report),
                "neural-output-disconnected": _load_report(args.disconnected_report),
            }
        else:
            reports = {
                condition: run_cns_check(
                    args.rust_bin,
                    args.assets,
                    args.cns_pack,
                    duration_seconds=args.duration_seconds,
                    control_hz=args.control_hz,
                    start_food_distance=args.start_food_distance,
                    condition=condition,
                    parameters=args.parameters,
                )
                for condition in CONDITIONS
            }
        result = verify_reports(
            reports,
            duration_seconds=args.duration_seconds,
            control_hz=args.control_hz,
            min_forward_displacement_mm=args.min_forward_displacement_mm,
            min_altitude_range_mm=args.min_altitude_range_mm,
            min_lesion_world_delta_mm=args.min_lesion_world_delta_mm,
            max_boundary_violation_mm=args.max_boundary_violation_mm,
            max_collision_reflex_seconds=args.max_collision_reflex_seconds,
            expected_neuron_count=expected_neuron_count,
        )
    except VerificationError as error:
        result = {
            "schema": VERIFICATION_SCHEMA,
            "schema_version": 1,
            "duration_seconds": args.duration_seconds,
            "control_hz": args.control_hz,
            "thresholds": {
                "min_forward_displacement_mm": args.min_forward_displacement_mm,
                "min_altitude_range_mm": args.min_altitude_range_mm,
                "min_lesion_world_delta_mm": args.min_lesion_world_delta_mm,
                "max_boundary_violation_mm": args.max_boundary_violation_mm,
                "max_collision_reflex_seconds": args.max_collision_reflex_seconds,
            },
            "expected_neuron_count": None,
            "failures": [str(error)],
            "passed": False,
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x") as output:
                json.dump(result, output, indent=2, sort_keys=True)
                output.write("\n")
        except FileExistsError as error:
            raise SystemExit(f"output already exists: {args.output}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
