from __future__ import annotations

import json
from pathlib import Path

from tools.verify_cns_foraging import analyze, main


def _sample(
    time: float,
    *,
    flight_mode: str,
    foraging_mode: str,
    behavior_mode: str,
    taste_active: bool = False,
    tasted_resource: int | None = None,
    contact_count: int = 0,
    activation: float = 0.0,
    landing_drive: float = 0.0,
    feeding_extension: float = 0.0,
    mn9_spikes: int = 0,
) -> dict:
    return {
        "time_seconds": time,
        "root_position": [0.0, 0.0, 2.0 if flight_mode != "GROUNDED" else 0.6],
        "flight_mode": flight_mode,
        "foraging_mode": foraging_mode,
        "behavior_mode": behavior_mode,
        "taste_active": taste_active,
        "tasted_resource": tasted_resource,
        "contact_count": contact_count,
        "feeding_extension": feeding_extension,
        "mn9_spike_delta": mn9_spikes,
        "mn9_rate_hz": float(mn9_spikes) * 50.0,
        "brain_flight_drive": activation,
        "brain_landing_drive": landing_drive,
        "population_spike_delta": 10,
        "cns_motor": {
            "spike_delta": 2,
            "flight_activation": activation,
            "outputs_connected": True,
        },
    }


def _report(*, landing_connected: bool = True, taste: bool = True) -> dict:
    samples = [
        _sample(
            0.01,
            flight_mode="GROUNDED",
            foraging_mode="APPROACH",
            behavior_mode="ODOR TAXIS",
        ),
        _sample(
            0.02,
            flight_mode="CRUISE",
            foraging_mode="APPROACH",
            behavior_mode="ODOR TAXIS",
            activation=0.9,
        ),
    ]
    if landing_connected:
        samples.extend(
            [
                _sample(
                    0.03,
                    flight_mode="LANDING",
                    foraging_mode="DESCEND",
                    behavior_mode="ODOR TAXIS",
                    activation=0.9,
                    landing_drive=0.04,
                ),
                _sample(
                    0.04,
                    flight_mode="GROUNDED",
                    foraging_mode="DESCEND",
                    behavior_mode="ODOR TAXIS",
                    contact_count=2,
                    activation=0.9,
                ),
            ]
        )
    if taste:
        samples.extend(
            [
                _sample(
                    0.05,
                    flight_mode="GROUNDED",
                    foraging_mode="FEED",
                    behavior_mode="FEED",
                    taste_active=True,
                    tasted_resource=0,
                    contact_count=4,
                    mn9_spikes=1,
                    feeding_extension=0.7,
                ),
                _sample(
                    0.06,
                    flight_mode="GROUNDED",
                    foraging_mode="POST-MEAL",
                    behavior_mode="POST-MEAL",
                    taste_active=True,
                    tasted_resource=0,
                    contact_count=4,
                    feeding_extension=0.5,
                ),
                _sample(
                    0.07,
                    flight_mode="TAKEOFF",
                    foraging_mode="SEARCH",
                    behavior_mode="EXPLORE",
                    activation=0.9,
                    feeding_extension=0.2,
                ),
            ]
        )
        samples[-1]["root_position"] = [20.0, 0.0, 20.0]
    if not landing_connected:
        samples.append(
            _sample(
                0.08,
                flight_mode="CRUISE",
                foraging_mode="APPROACH",
                behavior_mode="ODOR TAXIS",
                activation=0.9,
            )
        )
    brain = {
        "model": "male-cns-v1-connectome-partial-engineered-motor-embodiment-v1",
        "neurons": 166700,
        "sensory_neurons": 2324,
        "motor_outputs_connected": True,
        "landing_output_connected": landing_connected,
    }
    return {
        "schema": "flybrain.cns-world-check",
        "schema_version": 1,
        "brain": brain,
        "initial_state": {},
        "summary": {
            "initial_state_sha256": "a" * 64,
            "population_spikes": 100,
            "motor_output_spikes": 20,
            "flight_seconds": 1.0 if landing_connected else 0.5,
        },
        "samples": samples,
    }


def test_report_gate_accepts_all_qualified_transitions() -> None:
    result = analyze(_report(), _report(landing_connected=False, taste=False))

    assert result["passed"]
    assert all(result["checks"].values())


def test_no_feeding_is_an_explicit_failed_gate() -> None:
    result = analyze(_report(taste=False), _report(landing_connected=False, taste=False))

    assert not result["passed"]
    assert not result["checks"]["food_contact_with_neural_feeding"]
    assert "food_contact_with_neural_feeding" in result["failed_checks"]


def test_matched_initial_state_is_required() -> None:
    intact = _report()
    disconnected = _report(landing_connected=False, taste=False)
    disconnected["summary"]["initial_state_sha256"] = "b" * 64

    result = analyze(intact, disconnected)

    assert not result["checks"]["matched_initial_state"]
    assert not result["passed"]


def test_landing_disconnection_requires_zero_drive_without_descend() -> None:
    disconnected = _report(landing_connected=False, taste=False)
    disconnected["samples"][1]["brain_landing_drive"] = 0.01
    disconnected["samples"][1]["foraging_mode"] = "DESCEND"
    result = analyze(_report(), disconnected)

    assert not result["checks"]["disconnected_landing_readout_removes_descent"]


def test_pre_taste_extension_fails_even_when_feeding_is_present() -> None:
    intact = _report()
    intact["samples"][0]["feeding_extension"] = 0.2
    result = analyze(intact, _report(landing_connected=False, taste=False))

    assert not result["checks"]["no_extension_before_food_contact"]


def test_postmeal_extension_is_not_feeding() -> None:
    intact = _report()
    for sample in intact["samples"]:
        if sample["behavior_mode"] == "FEED":
            sample["behavior_mode"] = "POST-MEAL"
    result = analyze(intact, _report(landing_connected=False, taste=False))
    assert not result["checks"]["food_contact_with_neural_feeding"]


def test_food_context_without_mn9_activity_is_not_neural_feeding() -> None:
    intact = _report()
    for sample in intact["samples"]:
        sample["mn9_rate_hz"] = 0.0
    result = analyze(intact, _report(landing_connected=False, taste=False))
    assert not result["checks"]["food_contact_with_neural_feeding"]


def test_control_must_cover_the_intact_landing_time() -> None:
    disconnected = _report(landing_connected=False, taste=False)
    disconnected["samples"] = disconnected["samples"][:2]
    result = analyze(_report(), disconnected)
    assert not result["checks"]["control_covers_first_landing"]


def test_cli_writes_report_and_returns_nonzero_for_missing_feeding(tmp_path: Path) -> None:
    intact_path = tmp_path / "intact.json"
    disconnected_path = tmp_path / "landing-disconnected.json"
    output_path = tmp_path / "verification.json"
    intact_path.write_text(json.dumps(_report(taste=False)))
    disconnected_path.write_text(json.dumps(_report(landing_connected=False, taste=False)))

    exit_code = main(
        [
            "--intact-report",
            str(intact_path),
            "--landing-disconnected-report",
            str(disconnected_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert json.loads(output_path.read_text())["passed"] is False
