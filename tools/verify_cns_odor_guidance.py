"""Verify CNS odor-guidance evidence and its matched disconnection controls."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.verify_cns_foraging import feeding_sample
except ModuleNotFoundError:
    from verify_cns_foraging import feeding_sample

TRACE_SCHEMA = "flybrain.cns-world-check"
VERIFICATION_SCHEMA = "flybrain.cns-odor-guidance-verification-v1"
EXPECTED_NEURON_COUNT = 166_700
MAX_FEED_SAMPLE_GAP_SECONDS = 0.03
MIN_FEEDING_BOUT_SECONDS = 1.0
MIN_CONTROL_AFTER_FIRST_FEED_SECONDS = 1.0
MIN_DEPARTURE_DISTANCE_MM = 10.0
CONDITIONS = (
    "intact",
    "odor-evoked-disconnected",
    "motor-output-disconnected",
)
AIRBORNE_MODES = {"TAKEOFF", "CRUISE"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _samples(report: Any) -> tuple[list[Mapping[str, Any]], bool]:
    raw = report.get("samples") if isinstance(report, Mapping) else None
    if not isinstance(raw, list) or not raw:
        return [], False
    samples = [sample for sample in raw if isinstance(sample, Mapping)]
    return samples, len(samples) == len(raw)


def _timeline(samples: Sequence[Mapping[str, Any]]) -> tuple[list[float], bool]:
    times: list[float] = []
    valid = bool(samples)
    previous = None
    for sample in samples:
        time = _number(sample.get("time_seconds"))
        if time is None or (previous is not None and time <= previous):
            valid = False
        else:
            times.append(time)
            previous = time
    return times, valid and len(times) == len(samples)


def _schema_ok(report: Any) -> bool:
    return (
        isinstance(report, Mapping)
        and report.get("schema") == TRACE_SCHEMA
        and report.get("schema_version") == 1
    )


def _digest(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        return None
    return value.lower()


def _initial_hash(report: Any) -> str | None:
    summary = _mapping(report.get("summary") if isinstance(report, Mapping) else None)
    return _digest(summary.get("initial_state_sha256"))


def _initial_payload(report: Any) -> Mapping[str, Any] | None:
    value = report.get("initial_state") if isinstance(report, Mapping) else None
    return value if isinstance(value, Mapping) and value else None


def _safe_feeding_sample(sample: Mapping[str, Any]) -> bool:
    try:
        quaternion = sample.get("root_quaternion")
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            return False
        values = [_number(value) for value in quaternion]
        if any(value is None for value in values):
            return False
        if abs(sum(value * value for value in values) - 1.0) > 1e-4:
            return False
        up_z = 1.0 - 2.0 * (values[1] ** 2 + values[2] ** 2)
        contacts = _nonnegative_integer(sample.get("contact_count"))
        return up_z > 0.8 and contacts is not None and contacts >= 2 and feeding_sample(dict(sample))
    except (KeyError, TypeError, ValueError):
        return False


def _feeding_bouts(samples: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    bouts: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    previous_valid_time = None
    for sample in samples:
        time = _number(sample.get("time_seconds"))
        valid = time is not None and _safe_feeding_sample(sample)
        if not valid:
            if current:
                bouts.append(current)
                current = []
            previous_valid_time = None
            continue
        if (
            current
            and previous_valid_time is not None
            and (time - previous_valid_time > MAX_FEED_SAMPLE_GAP_SECONDS or time <= previous_valid_time)
        ):
            bouts.append(current)
            current = []
        current.append(sample)
        previous_valid_time = time
    if current:
        bouts.append(current)
    return bouts


def _bout_duration(bout: Sequence[Mapping[str, Any]]) -> float:
    if len(bout) < 2:
        return 0.0
    first = _number(bout[0].get("time_seconds"))
    last = _number(bout[-1].get("time_seconds"))
    if first is None or last is None or last < first:
        return 0.0
    return last - first


def _feeding_metrics(samples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[list[Mapping[str, Any]]]]:
    bouts = _feeding_bouts(samples)
    feeding_samples = [sample for bout in bouts for sample in bout]
    longest = max(bouts, key=lambda bout: (_bout_duration(bout), len(bout)), default=[])
    longest_duration = _bout_duration(longest)
    first = _number(feeding_samples[0].get("time_seconds")) if feeding_samples else None
    last = _number(feeding_samples[-1].get("time_seconds")) if feeding_samples else None
    return (
        {
            "feeding_sample_count": len(feeding_samples),
            "first_feeding_seconds": first,
            "last_feeding_seconds": last,
            "longest_feeding_bout_duration_seconds": longest_duration,
            "longest_feeding_bout_start_seconds": (
                _number(longest[0].get("time_seconds")) if longest else None
            ),
            "longest_feeding_bout_end_seconds": (
                _number(longest[-1].get("time_seconds")) if longest else None
            ),
            "qualifying_feeding_bout": len(longest) >= 2 and longest_duration >= MIN_FEEDING_BOUT_SECONDS,
        },
        bouts,
    )


def _guidance_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = bool(samples)
    active_count = close_count = 0
    for sample in samples:
        guidance = sample.get("odor_guidance")
        if not isinstance(guidance, Mapping):
            complete = False
            continue
        active = guidance.get("active")
        close = guidance.get("close")
        if not isinstance(active, bool) or not isinstance(close, bool):
            complete = False
            continue
        active_count += int(active)
        close_count += int(close)
    return {
        "guidance_telemetry_complete": complete,
        "guidance_active_sample_count": active_count,
        "guidance_close_sample_count": close_count,
        "guidance_inactive": complete and active_count == 0 and close_count == 0,
    }


def _spike_metrics(report: Any, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _mapping(report.get("summary") if isinstance(report, Mapping) else None)
    population = _nonnegative_integer(summary.get("population_spikes"))
    motor = _nonnegative_integer(summary.get("motor_output_spikes"))
    sample_population = 0
    sample_motor = 0
    population_complete = motor_complete = True
    for sample in samples:
        population_delta = _nonnegative_integer(sample.get("population_spike_delta"))
        if population_delta is None:
            population_complete = False
        else:
            sample_population += population_delta
        cns_motor = sample.get("cns_motor")
        if not isinstance(cns_motor, Mapping):
            motor_complete = False
            continue
        motor_delta = _nonnegative_integer(cns_motor.get("spike_delta"))
        if motor_delta is None:
            motor_complete = False
        else:
            sample_motor += motor_delta
    return {
        "population_spikes": population,
        "motor_output_spikes": motor,
        "sample_population_spikes": sample_population,
        "sample_motor_output_spikes": sample_motor,
        "sample_population_spikes_complete": population_complete,
        "sample_motor_output_spikes_complete": motor_complete,
        "neural_spiking": (
            population is not None
            and population > 0
            and population_complete
            and sample_population > 0
        ),
        "motor_spiking": (
            motor is not None
            and motor > 0
            and motor_complete
            and sample_motor > 0
        ),
    }


def _sample_motor_outputs_connected(
    samples: Sequence[Mapping[str, Any]], expected: bool
) -> bool:
    if not samples:
        return False
    for sample in samples:
        motor = sample.get("cns_motor")
        if not isinstance(motor, Mapping) or motor.get("outputs_connected") is not expected:
            return False
    return True


def _departure_after_feeding(
    samples: Sequence[Mapping[str, Any]], bout: Sequence[Mapping[str, Any]]
) -> tuple[bool, float | None]:
    if not bout:
        return False, None
    anchor_time = _number(bout[-1].get("time_seconds"))
    anchor_position = bout[-1].get("root_position")
    if anchor_time is None or not _position(anchor_position):
        return False, None
    for sample in samples:
        time = _number(sample.get("time_seconds"))
        position = sample.get("root_position")
        drive = _number(sample.get("brain_flight_drive"))
        mode = sample.get("flight_mode")
        if (
            time is not None
            and time > anchor_time
            and isinstance(mode, str)
            and mode.upper() in AIRBORNE_MODES
            and drive is not None
            and drive > 0.0
            and _position(position)
            and math.dist(position, anchor_position) > MIN_DEPARTURE_DISTANCE_MM
        ):
            return True, time
    return False, None


def _position(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and all(_number(item) is not None for item in value)
    )


def _condition_configuration(report: Any, condition: str) -> bool:
    if condition not in CONDITIONS or not isinstance(report, Mapping):
        return False
    brain = _mapping(report.get("brain"))
    parameters = _mapping(report.get("parameters"))
    brain_parameters = _mapping(parameters.get("brain"))
    odor_parameters = _mapping(parameters.get("odor_guidance"))
    summary = _mapping(report.get("summary"))
    expected = {
        "intact": {
            "motor": True,
            "olfactory": True,
            "source": "whole-cns-spikes",
            "evoked_positive": True,
        },
        "odor-evoked-disconnected": {
            "motor": True,
            "olfactory": False,
            "source": "whole-cns-spikes",
            "evoked_positive": False,
        },
        "motor-output-disconnected": {
            "motor": False,
            "olfactory": True,
            "source": "disconnected",
            "evoked_positive": True,
        },
    }[condition]
    model = brain.get("model")
    neuron_count = _nonnegative_integer(brain.get("neurons"))
    sensory_neurons = _nonnegative_integer(brain.get("sensory_neurons"))
    baseline = _number(brain_parameters.get("olfactory_baseline_rate_hz"))
    evoked = _number(brain_parameters.get("olfactory_input_rate_hz"))
    return (
        isinstance(model, str)
        and "male-cns" in model.lower()
        and neuron_count == EXPECTED_NEURON_COUNT
        and sensory_neurons is not None
        and sensory_neurons > 0
        and brain.get("motor_outputs_connected") is expected["motor"]
        and brain.get("landing_output_connected") is True
        and brain.get("sensory_inputs_connected") is True
        and brain.get("olfactory_evoked_inputs_connected") is expected["olfactory"]
        and brain.get("odor_guidance_enabled") is True
        and brain_parameters.get("cns_motor_outputs_enabled") is expected["motor"]
        and brain_parameters.get("cns_landing_output_enabled") is True
        and odor_parameters.get("enabled") is True
        and baseline is not None
        and baseline > 0.0
        and evoked is not None
        and ((evoked > 0.0) is expected["evoked_positive"])
        and summary.get("motor_output_source") == expected["source"]
    )


def _no_extension_before_first_taste(samples: Sequence[Mapping[str, Any]]) -> bool:
    if not samples:
        return False
    first_taste = math.inf
    for sample in samples:
        taste = sample.get("taste_active")
        extension = _number(sample.get("feeding_extension"))
        time = _number(sample.get("time_seconds"))
        if not isinstance(taste, bool) or extension is None or time is None:
            return False
        if taste:
            first_taste = min(first_taste, time)
    for sample in samples:
        time = _number(sample.get("time_seconds"))
        extension = _number(sample.get("feeding_extension"))
        if time is None or extension is None:
            return False
        if time < first_taste and extension != 0.0:
            return False
    return True


def _all_extension_zero(samples: Sequence[Mapping[str, Any]]) -> bool:
    return bool(samples) and all(
        _number(sample.get("feeding_extension")) == 0.0 for sample in samples
    )


def _flight_zero(report: Any, samples: Sequence[Mapping[str, Any]]) -> bool:
    summary = _mapping(report.get("summary") if isinstance(report, Mapping) else None)
    flight_seconds = _number(summary.get("flight_seconds"))
    forward_distance = _number(summary.get("forward_flight_distance_mm"))
    if flight_seconds is None or forward_distance is None or flight_seconds != 0.0 or forward_distance != 0.0:
        return False
    for sample in samples:
        mode = sample.get("flight_mode")
        drive = _number(sample.get("brain_flight_drive"))
        steering = _number(sample.get("flight_steering"))
        if (
            not isinstance(mode, str)
            or mode.upper() != "GROUNDED"
            or drive is None
            or drive != 0.0
            or steering is None
            or steering != 0.0
        ):
            return False
    return bool(samples)


def _control_covers_first_feeding(
    report: Any, first_feeding_seconds: float | None
) -> bool:
    if first_feeding_seconds is None:
        return False
    samples, shape_ok = _samples(report)
    times, timeline_ok = _timeline(samples)
    if not shape_ok or not timeline_ok or not times:
        return False
    return (
        times[0] <= first_feeding_seconds
        and times[-1] >= first_feeding_seconds + MIN_CONTROL_AFTER_FIRST_FEED_SECONDS
    )


def _condition_metrics(report: Any) -> tuple[dict[str, Any], list[list[Mapping[str, Any]]]]:
    samples, shape_ok = _samples(report)
    times, timeline_ok = _timeline(samples)
    feeding, bouts = _feeding_metrics(samples)
    guidance = _guidance_metrics(samples)
    spikes = _spike_metrics(report, samples)
    longest = max(bouts, key=lambda bout: (_bout_duration(bout), len(bout)), default=[])
    departure, departure_seconds = _departure_after_feeding(
        samples, longest if feeding["qualifying_feeding_bout"] else []
    )
    summary = _mapping(report.get("summary") if isinstance(report, Mapping) else None)
    reported_duration = _number(summary.get("duration_seconds"))
    metrics = {
        "schema_ok": _schema_ok(report),
        "sample_shape_ok": shape_ok,
        "sample_timeline_ok": timeline_ok,
        "sample_count": len(samples),
        "first_sample_seconds": times[0] if times else None,
        "last_sample_seconds": times[-1] if times else None,
        "reported_duration_seconds": reported_duration,
        **feeding,
        **guidance,
        **spikes,
        "flight_departure_after_feeding": departure,
        "departure_seconds": departure_seconds,
    }
    return metrics, bouts


def analyze(
    intact: Mapping[str, Any],
    odor_control: Mapping[str, Any],
    motor_control: Mapping[str, Any],
) -> dict[str, Any]:
    reports = {
        "intact": intact,
        "odor-evoked-disconnected": odor_control,
        "motor-output-disconnected": motor_control,
    }
    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition, report in reports.items():
        metrics, bouts = _condition_metrics(report)
        condition_metrics[condition] = metrics

    hashes = {condition: _initial_hash(report) for condition, report in reports.items()}
    payloads = {condition: _initial_payload(report) for condition, report in reports.items()}
    intact_metrics = condition_metrics["intact"]
    first_feeding = intact_metrics["first_feeding_seconds"]
    checks: dict[str, bool] = {}
    checks["reports_have_expected_schema"] = all(
        condition_metrics[condition]["schema_ok"] for condition in CONDITIONS
    )
    checks["reports_have_valid_samples"] = all(
        condition_metrics[condition]["sample_shape_ok"]
        and condition_metrics[condition]["sample_timeline_ok"]
        for condition in CONDITIONS
    )
    checks["matching_initial_state_sha256"] = (
        all(value is not None for value in hashes.values())
        and len(set(hashes.values())) == 1
    )
    checks["matching_initial_state_payload"] = (
        all(value is not None for value in payloads.values())
        and payloads["intact"] == payloads["odor-evoked-disconnected"]
        and payloads["intact"] == payloads["motor-output-disconnected"]
    )

    intact_samples, _ = _samples(intact)
    checks["intact_configuration"] = _condition_configuration(
        intact, "intact"
    ) and _sample_motor_outputs_connected(intact_samples, True)
    checks["intact_neural_spiking"] = intact_metrics["neural_spiking"]
    checks["intact_guidance_active"] = (
        intact_metrics["guidance_telemetry_complete"]
        and intact_metrics["guidance_active_sample_count"] > 0
    )
    checks["intact_guidance_close"] = (
        intact_metrics["guidance_telemetry_complete"]
        and intact_metrics["guidance_close_sample_count"] > 0
    )
    checks["intact_contiguous_feeding_bout"] = (
        intact_metrics["qualifying_feeding_bout"]
        and intact_metrics["longest_feeding_bout_duration_seconds"] >= MIN_FEEDING_BOUT_SECONDS
    )
    checks["intact_flight_departure_after_feeding"] = intact_metrics[
        "flight_departure_after_feeding"
    ]

    odor_metrics = condition_metrics["odor-evoked-disconnected"]
    odor_samples, _ = _samples(odor_control)
    checks["odor_control_configuration"] = _condition_configuration(
        odor_control, "odor-evoked-disconnected"
    ) and _sample_motor_outputs_connected(odor_samples, True)
    checks["odor_control_baseline_sensory_and_neural_spiking"] = (
        odor_metrics["neural_spiking"]
        and odor_metrics["motor_spiking"]
        and odor_metrics["sample_population_spikes"] > 0
    )
    checks["odor_control_guidance_inactive"] = odor_metrics["guidance_inactive"]
    checks["odor_control_no_extension_before_first_taste"] = _no_extension_before_first_taste(
        odor_samples
    )

    motor_metrics = condition_metrics["motor-output-disconnected"]
    motor_samples, _ = _samples(motor_control)
    checks["motor_control_configuration"] = _condition_configuration(
        motor_control, "motor-output-disconnected"
    ) and _sample_motor_outputs_connected(motor_samples, False)
    checks["motor_control_brain_spiking_preserved"] = motor_metrics["neural_spiking"]
    checks["motor_control_guidance_inactive"] = motor_metrics["guidance_inactive"]
    checks["motor_control_flight_zero"] = _flight_zero(motor_control, motor_samples)
    checks["motor_control_extension_zero"] = _all_extension_zero(motor_samples)

    checks["odor_control_covers_first_feeding"] = _control_covers_first_feeding(
        odor_control, first_feeding
    )
    checks["motor_control_covers_first_feeding"] = _control_covers_first_feeding(
        motor_control, first_feeding
    )

    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": 1,
        "passed": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "metrics": condition_metrics,
        "thresholds": {
            "max_feed_sample_gap_seconds": MAX_FEED_SAMPLE_GAP_SECONDS,
            "min_feeding_bout_seconds": MIN_FEEDING_BOUT_SECONDS,
            "min_control_after_first_feeding_seconds": MIN_CONTROL_AFTER_FIRST_FEED_SECONDS,
            "min_departure_distance_mm": MIN_DEPARTURE_DISTANCE_MM,
        },
        "initial_state_sha256": hashes,
        "interpretation": "Engineering integration evidence, not animal behavioral validation. "
        "All three conditions must pass; a failed intact food-search run is not replaced by a control or contact-only assay.",
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise SystemExit(f"report is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"report is not valid JSON: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise SystemExit(f"report must be a JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intact-report", type=Path, required=True)
    parser.add_argument("--odor-evoked-disconnected-report", type=Path, required=True)
    parser.add_argument("--motor-output-disconnected-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = analyze(
        _load_json(args.intact_report),
        _load_json(args.odor_evoked_disconnected_report),
        _load_json(args.motor_output_disconnected_report),
    )
    result["reports"] = {
        "intact": str(args.intact_report),
        "odor_evoked_disconnected": str(args.odor_evoked_disconnected_report),
        "motor_output_disconnected": str(args.motor_output_disconnected_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
