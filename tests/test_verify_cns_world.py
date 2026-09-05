from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from tools.verify_cns_world import CONDITIONS, run_cns_check, verify_reports


def _sample(
    time: float,
    position: list[float],
    *,
    connected: bool,
    flight: bool,
    reflex: bool = False,
    population_delta: int = 10,
    motor_delta: int = 2,
) -> dict:
    return {
        "time_seconds": time,
        "population_spike_delta": population_delta,
        "root_position": position,
        "flight_mode": "CRUISE" if flight else "GROUNDED",
        "collision_reflex_active": reflex,
        "cns_motor": {
            "spike_delta": motor_delta,
            "flight_power_hz": [20.0, 18.0],
            "wing_steering_hz": [5.0, 4.0],
            "walking_hz": [1.0, 1.0],
            "landing_hz": [0.0, 0.0],
            "flight_activation": 0.5 if flight and connected else 0.0,
            "walking_activation": 0.0,
            "steering": 0.01 if flight and connected else 0.0,
            "outputs_connected": connected,
        },
    }


def _report(*, connected: bool, final: list[float]) -> dict:
    samples = [
        _sample(0.01, [0.0, 0.0, 0.0], connected=connected, flight=False),
        _sample(0.02, final, connected=connected, flight=connected),
    ]
    return {
        "schema": "flybrain.cns-world-check",
        "duration_seconds": 2.0,
        "control_hz": 500.0,
        "brain": {
            "model": "male-cns-v1-connectome-motor-output-embodiment-v1",
            "neurons": 166700,
            "sensory_neurons": 2,
            "motor_outputs_connected": connected,
        },
        "initial_state": {
            "initial_position_mm": [0.0, 0.0, 0.0],
            "pack_arrays": {"row_ptr.npy": "a" * 64},
            "io_sha256": "b" * 64,
            "parameters": {"brain": {"cns_motor_outputs_enabled": True}},
        },
        "room_bounds_mm": [[-300.0, -220.0, 0.0], [300.0, 220.0, 220.0]],
        "summary": {
            "duration_seconds": 1.998,
            "initial_state_sha256": "c" * 64,
            "population_spikes": 30,
            "motor_output_spikes": 7,
            "motor_output_source": "whole-cns-spikes" if connected else "disconnected",
            "forward_flight_distance_mm": 5.0 if connected else 0.0,
            "initial_position_mm": [0.0, 0.0, 0.0],
            "final_position_mm": final,
            "minimum_position_mm": [0.0, 0.0, 0.0],
            "maximum_position_mm": final,
            "path_length_mm": 5.0 if connected else 0.0,
            "flight_seconds": 1.0 if connected else 0.0,
            "feeding_seconds": 0.0,
            "maximum_speed_mm_s": 20.0,
            "maximum_abs_pitch_deg": 3.0,
        },
        "samples": samples,
    }


def _reports() -> dict[str, dict]:
    return {
        CONDITIONS[0]: _report(connected=True, final=[5.0, 0.0, 8.0]),
        CONDITIONS[1]: _report(connected=False, final=[0.0, 0.0, 0.0]),
    }


def test_gate_accepts_neural_motion_and_output_lesion() -> None:
    result = verify_reports(
        _reports(), duration_seconds=2.0, control_hz=500.0, expected_neuron_count=166700
    )

    assert result["passed"]
    assert result["lesion_comparison"]["passed"]
    assert result["lesion_comparison"]["checks"]["matched_initial_state"]
    assert result["lesion_comparison"]["metrics"]["final_position_delta_mm"] > 0.25


def test_gate_rejects_unmatched_initial_state() -> None:
    reports = _reports()
    reports[CONDITIONS[1]]["summary"]["initial_state_sha256"] = "d" * 64
    result = verify_reports(reports, duration_seconds=2.0, control_hz=500.0)

    assert not result["passed"]
    assert not result["lesion_comparison"]["checks"]["matched_initial_state"]


def test_gate_rejects_grounded_or_mode_only_motion() -> None:
    reports = _reports()
    reports[CONDITIONS[0]]["summary"]["forward_flight_distance_mm"] = 0.0
    result = verify_reports(reports, duration_seconds=2.0, control_hz=500.0)

    assert not result["passed"]
    assert not result["evaluations"][CONDITIONS[0]]["checks"]["forward_flight_displacement"]


def test_gate_rejects_disconnected_physical_flight() -> None:
    reports = _reports()
    disconnected = reports[CONDITIONS[1]]
    disconnected["summary"]["flight_seconds"] = 1.0
    disconnected["summary"]["forward_flight_distance_mm"] = 1.0
    disconnected["samples"][1]["flight_mode"] = "CRUISE"
    result = verify_reports(reports, duration_seconds=2.0, control_hz=500.0)

    assert not result["passed"]
    assert not result["evaluations"][CONDITIONS[1]]["checks"]["disconnected_stays_grounded"]


def test_gate_rejects_a_reflex_that_stays_latched() -> None:
    reports = _reports()
    reports[CONDITIONS[0]]["samples"][0]["collision_reflex_active"] = True
    reports[CONDITIONS[0]]["samples"][1]["collision_reflex_active"] = True
    reports[CONDITIONS[0]]["samples"][1]["time_seconds"] = 5.02
    result = verify_reports(reports, duration_seconds=2.0, control_hz=500.0)

    assert not result["passed"]
    assert not result["evaluations"][CONDITIONS[0]]["checks"]["collision_reflex_stable"]


def test_cli_invocation_has_no_seed_and_uses_output_lesion_flag() -> None:
    report = _report(connected=False, final=[0.0, 0.0, 0.0])

    def run(command, **_kwargs):
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        return CompletedProcess(command, 0, "", "")

    with patch(
        "tools.verify_cns_world.subprocess.run",
        side_effect=run,
    ) as run:
        run_cns_check(
            Path("target/release/flybrain-world"),
            Path("assets/neuromechfly"),
            Path("outputs/packs/male_cns_v1"),
            duration_seconds=2.0,
            control_hz=500.0,
            start_food_distance=2.5,
            condition="neural-output-disconnected",
        )

    command = run.call_args.args[0]
    assert command[:2] == ["target/release/flybrain-world", "cns-check"]
    assert "--seed" not in command
    assert "--disconnect-motor-outputs" in command


def test_gate_does_not_require_equal_closed_loop_sensory_hashes() -> None:
    reports = _reports()
    reports[CONDITIONS[0]]["sensory_trace_sha256"] = "a" * 64
    reports[CONDITIONS[1]]["sensory_trace_sha256"] = "b" * 64
    result = verify_reports(reports, duration_seconds=2.0, control_hz=500.0)

    assert result["passed"]
