from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.verify_cns_odor_guidance import analyze, main


def _sample(
    time: float,
    *,
    mode: str = "GROUNDED",
    foraging: str = "SEARCH",
    behavior: str = "ODOR TAXIS",
    taste: bool = False,
    resource: int | None = None,
    extension: float = 0.0,
    mn9_rate: float = 0.0,
    population_delta: int = 10,
    motor_delta: int = 2,
    guidance_active: bool = False,
    guidance_close: bool = False,
    flight_drive: float = 0.0,
    flight_steering: float = 0.0,
    position: list[float] | None = None,
    motor_connected: bool = True,
) -> dict:
    return {
        "time_seconds": time,
        "root_quaternion": [1.0, 0.0, 0.0, 0.0],
        "contact_count": 6 if mode == "GROUNDED" else 0,
        "root_position": position or [0.0, 0.0, 0.6 if mode == "GROUNDED" else 20.0],
        "flight_mode": mode,
        "foraging_mode": foraging,
        "behavior_mode": behavior,
        "taste_active": taste,
        "tasted_resource": resource,
        "feeding_extension": extension,
        "mn9_rate_hz": mn9_rate,
        "population_spike_delta": population_delta,
        "brain_flight_drive": flight_drive,
        "flight_steering": flight_steering,
        "odor_guidance": {
            "active": guidance_active,
            "close": guidance_close,
            "mean_rate_hz": 20.0 if guidance_active else 0.0,
            "steering": 0.1 if guidance_active else 0.0,
        },
        "cns_motor": {
            "spike_delta": motor_delta,
            "outputs_connected": motor_connected,
        },
    }


def _report(
    *,
    motor_connected: bool = True,
    olfactory_connected: bool = True,
    feed: bool = True,
    duration: float = 3.2,
) -> dict:
    samples = [
        _sample(0.01, guidance_active=True),
        _sample(0.02, guidance_active=True, guidance_close=True),
    ]
    if feed:
        samples.extend(
            _sample(
                1.0 + index * 0.01,
                foraging="FEED",
                behavior="FEED",
                taste=True,
                resource=0,
                extension=0.7,
                mn9_rate=30.0,
            )
            for index in range(101)
        )
        samples.append(
            _sample(
                2.2,
                mode="TAKEOFF",
                foraging="SEARCH",
                behavior="EXPLORE",
                extension=0.0,
                flight_drive=0.5,
                position=[12.0, 0.0, 20.0],
            )
        )
    else:
        samples.extend([_sample(1.0), _sample(2.2)])
    if not motor_connected:
        for sample in samples:
            sample["odor_guidance"] = {
                "active": False,
                "close": False,
                "mean_rate_hz": 0.0,
                "steering": 0.0,
            }
            sample["flight_mode"] = "GROUNDED"
            sample["root_position"] = [0.0, 0.0, 0.6]
            sample["brain_flight_drive"] = 0.0
            sample["flight_steering"] = 0.0
            sample["feeding_extension"] = 0.0
            sample["taste_active"] = False
            sample["tasted_resource"] = None
            sample["foraging_mode"] = "SEARCH"
            sample["behavior_mode"] = "ODOR TAXIS"
            sample["cns_motor"]["outputs_connected"] = False
    if not olfactory_connected:
        for sample in samples:
            sample["odor_guidance"] = {
                "active": False,
                "close": False,
                "mean_rate_hz": 8.0,
                "steering": 0.0,
            }
            sample["feeding_extension"] = 0.0
            sample["taste_active"] = False
            sample["tasted_resource"] = None
            sample["foraging_mode"] = "SEARCH"
            sample["behavior_mode"] = "ODOR TAXIS"
    brain_input = 192.0 if olfactory_connected else 0.0
    return {
        "schema": "flybrain.cns-world-check",
        "schema_version": 1,
        "duration_seconds": duration,
        "control_hz": 500.0,
        "initial_state": {"paired": True, "initial_position_mm": [0.0, 0.0, 0.6]},
        "brain": {
            "model": "male-cns-v1-connectome-partial-engineered-motor-embodiment-v1",
            "neurons": 166700,
            "sensory_neurons": 2324,
            "motor_outputs_connected": motor_connected,
            "landing_output_connected": True,
            "sensory_inputs_connected": True,
            "olfactory_evoked_inputs_connected": olfactory_connected,
            "odor_guidance_enabled": True,
        },
        "parameters": {
            "brain": {
                "cns_motor_outputs_enabled": motor_connected,
                "cns_landing_output_enabled": True,
                "olfactory_baseline_rate_hz": 8.0,
                "olfactory_input_rate_hz": brain_input,
            },
            "odor_guidance": {"enabled": True},
        },
        "summary": {
            "duration_seconds": duration,
            "initial_state_sha256": "a" * 64,
            "population_spikes": 1000,
            "motor_output_spikes": 200,
            "motor_output_source": "whole-cns-spikes" if motor_connected else "disconnected",
            "flight_seconds": 1.0 if feed and motor_connected else 0.0,
            "forward_flight_distance_mm": 12.0 if feed and motor_connected else 0.0,
        },
        "samples": samples,
    }


def _reports() -> tuple[dict, dict, dict]:
    return (
        _report(),
        _report(motor_connected=True, olfactory_connected=False, feed=False),
        _report(motor_connected=False, olfactory_connected=True, feed=False),
    )


def test_accepts_three_condition_gate_and_reports_contiguous_bout() -> None:
    result = analyze(*_reports())

    assert result["passed"]
    assert result["failed_checks"] == []
    metrics = result["metrics"]["intact"]
    assert metrics["first_feeding_seconds"] == 1.0
    assert metrics["last_feeding_seconds"] == 2.0
    assert metrics["longest_feeding_bout_duration_seconds"] == 1.0


def test_rejects_no_food_even_when_guidance_is_active() -> None:
    intact, odor, motor = _reports()
    intact = _report(feed=False)

    result = analyze(intact, odor, motor)

    assert not result["passed"]
    assert "intact_contiguous_feeding_bout" in result["failed_checks"]


def test_rejects_mismatched_odor_control_configuration() -> None:
    intact, odor, motor = _reports()
    odor["brain"]["olfactory_evoked_inputs_connected"] = True

    result = analyze(intact, odor, motor)

    assert not result["checks"]["odor_control_configuration"]
    assert "odor_control_configuration" in result["failed_checks"]


def test_rejects_mismatched_initial_state() -> None:
    intact, odor, motor = _reports()
    motor["summary"]["initial_state_sha256"] = "b" * 64

    result = analyze(intact, odor, motor)

    assert not result["checks"]["matching_initial_state_sha256"]
    assert not result["passed"]


def test_rejects_controls_that_end_before_first_feeding() -> None:
    intact, odor, motor = _reports()
    odor["samples"] = odor["samples"][:1]
    motor["samples"] = motor["samples"][:1]
    odor["summary"]["duration_seconds"] = 0.01
    motor["summary"]["duration_seconds"] = 0.01

    result = analyze(intact, odor, motor)

    assert not result["checks"]["odor_control_covers_first_feeding"]
    assert not result["checks"]["motor_control_covers_first_feeding"]


def test_separated_feeding_touches_do_not_accumulate_duration() -> None:
    intact, odor, motor = _reports()
    intact["samples"] = [
        _sample(
            1.0,
            foraging="FEED",
            behavior="FEED",
            taste=True,
            resource=0,
            extension=0.7,
            mn9_rate=30.0,
        ),
        _sample(
            2.0,
            foraging="FEED",
            behavior="FEED",
            taste=True,
            resource=0,
            extension=0.7,
            mn9_rate=30.0,
        ),
        _sample(
            3.0,
            mode="TAKEOFF",
            flight_drive=0.5,
            position=[12.0, 0.0, 20.0],
        ),
    ]

    result = analyze(intact, odor, motor)

    assert not result["checks"]["intact_contiguous_feeding_bout"]
    assert result["metrics"]["intact"]["longest_feeding_bout_duration_seconds"] == 0.0


def test_invalid_sample_breaks_feeding_bout_even_with_small_timestamp_gap() -> None:
    intact, odor, motor = _reports()
    valid = dict(
        foraging="FEED",
        behavior="FEED",
        taste=True,
        resource=0,
        extension=0.7,
        mn9_rate=30.0,
    )
    feed_before = [_sample(1.0 + index * 0.01, **valid) for index in range(50)]
    feed_after = [_sample(1.51 + index * 0.01, **valid) for index in range(50)]
    intact["samples"] = [
        _sample(0.01, guidance_active=True),
        _sample(0.02, guidance_active=True, guidance_close=True),
        *feed_before,
        _sample(1.50),
        *feed_after,
        _sample(
            2.2,
            mode="TAKEOFF",
            flight_drive=0.5,
            position=[12.0, 0.0, 20.0],
        ),
    ]

    result = analyze(intact, odor, motor)

    assert not result["checks"]["intact_contiguous_feeding_bout"]
    assert result["metrics"]["intact"]["longest_feeding_bout_duration_seconds"] == pytest.approx(0.49)


def test_departure_distance_anchors_to_last_feeding_position() -> None:
    intact, odor, motor = _reports()
    feeding = [sample for sample in intact["samples"] if sample["foraging_mode"] == "FEED"]
    feeding[0]["root_position"] = [20.0, 0.0, 0.6]
    feeding[-1]["root_position"] = [0.0, 0.0, 0.6]
    intact["samples"][-1]["root_position"] = [5.0, 0.0, 8.0]

    result = analyze(intact, odor, motor)

    assert not result["checks"]["intact_flight_departure_after_feeding"]


def test_allows_residual_extension_after_first_taste() -> None:
    intact, odor, motor = _reports()
    odor["samples"][2]["taste_active"] = True
    odor["samples"][2]["tasted_resource"] = 0
    odor["samples"][2]["feeding_extension"] = 0.7
    odor["samples"][3]["feeding_extension"] = 0.2

    result = analyze(intact, odor, motor)

    assert result["checks"]["odor_control_no_extension_before_first_taste"]


def test_rejects_departure_without_ten_millimetres_of_motion() -> None:
    intact, odor, motor = _reports()
    intact["samples"][-1]["root_position"] = [1.0, 0.0, 8.0]

    result = analyze(intact, odor, motor)

    assert not result["checks"]["intact_flight_departure_after_feeding"]


def test_rejects_guidance_claim_in_odor_control() -> None:
    intact, odor, motor = _reports()
    odor["samples"][0]["odor_guidance"]["active"] = True

    result = analyze(intact, odor, motor)

    assert not result["checks"]["odor_control_guidance_inactive"]


def test_rejects_nonzero_flight_or_extension_in_motor_control() -> None:
    intact, odor, motor = _reports()
    motor["samples"][0]["flight_mode"] = "CRUISE"
    motor["samples"][0]["brain_flight_drive"] = 0.1
    motor["samples"][0]["feeding_extension"] = 0.2

    result = analyze(intact, odor, motor)

    assert not result["checks"]["motor_control_flight_zero"]
    assert not result["checks"]["motor_control_extension_zero"]


def test_old_reports_without_new_flags_fail_gracefully() -> None:
    intact, odor, motor = _reports()
    for report in (intact, odor, motor):
        report["brain"].pop("olfactory_evoked_inputs_connected")
        report["brain"].pop("odor_guidance_enabled")
        for sample in report["samples"]:
            sample.pop("odor_guidance")

    result = analyze(intact, odor, motor)

    assert not result["passed"]
    assert result["failed_checks"]


@pytest.mark.parametrize("quaternion,contacts", [([0.0, 1.0, 0.0, 0.0], 6), ([1.0, 0.0, 0.0, 0.0], 1), ([2.0, 0.0, 0.0, 0.0], 6)])
def test_unstable_or_unsupported_food_contact_is_not_feeding(quaternion: list[float], contacts: int) -> None:
    intact, odor, motor = _reports()
    for sample in intact["samples"]:
        if sample["behavior_mode"] == "FEED":
            sample["root_quaternion"] = quaternion
            sample["contact_count"] = contacts
    result = analyze(intact, odor, motor)
    assert not result["passed"]


def test_cli_writes_result_and_refuses_overwrite(tmp_path: Path) -> None:
    reports = _reports()
    paths = []
    for name, report in zip(("intact", "odor", "motor"), reports):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    output = tmp_path / "verification.json"
    args = [
        "--intact-report",
        str(paths[0]),
        "--odor-evoked-disconnected-report",
        str(paths[1]),
        "--motor-output-disconnected-report",
        str(paths[2]),
        "--output",
        str(output),
    ]

    assert main(args) == 0
    assert json.loads(output.read_text())["passed"] is True
    with pytest.raises(SystemExit):
        main(args)
