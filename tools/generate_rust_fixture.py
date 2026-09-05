"""Generate and validate the small cross-language tick-parity fixture."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters
from flybrain.reference import ReferenceSimulator

SCHEMA = "flybrain.tick-fixture"
SCHEMA_VERSION = 1
DEFAULT_PATH = _ROOT / "fixtures" / "tiny-parity-v1.json"
STATE_ATOL_MV = 1e-10
TIME_ATOL_MS = 1e-12

_DYNAMICS_PARAMETER_KEYS = (
    "dt_ms",
    "resting_mv",
    "reset_mv",
    "threshold_mv",
    "membrane_tau_ms",
    "synapse_tau_ms",
    "refractory_ms",
    "delay_ms",
    "synapse_weight_mv",
)
_RUST_PARAMETER_KEYS = (*_DYNAMICS_PARAMETER_KEYS, "external_weight_mv")


def _canonical_case() -> tuple[PackedConnectome, ModelParameters, np.ndarray, np.ndarray, int]:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[10, 20],
        row_ptr=[0, 1, 2],
        destinations=[1, 0],
        signed_counts=[50, -3],
    )
    parameters = ModelParameters(delay_ms=0.2, refractory_ms=0.3)
    initial_v = np.asarray([-44.0, -52.0], dtype=np.float64)
    initial_g = np.zeros(connectome.neuron_count, dtype=np.float64)
    steps = 15
    return connectome, parameters, initial_v, initial_g, steps


def make_fixture() -> dict[str, Any]:
    """Return the canonical fixture and its NumPy float64 golden output."""

    connectome, parameters, initial_v, initial_g, steps = _canonical_case()
    result = ReferenceSimulator(
        connectome,
        parameters,
        initial_v=initial_v,
        initial_g=initial_g,
    ).run(steps=steps, record=True, record_state=True)
    assert result is not None
    spike_ticks, spike_neurons = np.nonzero(result.spikes)

    model_parameters = {key: getattr(parameters, key) for key in _DYNAMICS_PARAMETER_KEYS}
    model_parameters["external_weight_mv"] = parameters.poisson_weight_mv
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "case_id": "recurrent_cycle_end_v1",
        "parameters": model_parameters,
        "network": {
            "neuron_count": connectome.neuron_count,
            "edge_count": connectome.edge_count,
            "neuron_ids_u64": connectome.neuron_ids.tolist(),
            "row_ptr_u32": connectome.row_ptr.tolist(),
            "destinations_u32": connectome.destinations.tolist(),
            "signed_counts_i16": connectome.signed_counts.tolist(),
        },
        "initial_state": {
            "v_mv": initial_v.tolist(),
            "g_mv": initial_g.tolist(),
        },
        "overrides": {
            "silenced_sources": [],
            "zero_refractory": [],
        },
        "stimulus": {
            "external_counts": [],
            "source_spikes": [],
        },
        "run": {
            "steps": steps,
            "record_slots": ["end"],
            "state_layout": "tick_major",
        },
        "acceptance": {
            "state_abs_tol_mv": STATE_ATOL_MV,
            "time_abs_tol_ms": TIME_ATOL_MS,
            "spikes": "exact",
        },
        "expected": {
            "times_ms": result.times_ms.tolist(),
            "spike_events": [
                {"tick": int(tick), "neuron": int(neuron)}
                for tick, neuron in zip(spike_ticks, spike_neurons, strict=True)
            ],
            "v_end_mv": result.voltage.tolist(),
            "g_end_mv": result.conductance.tolist(),
        },
    }


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _validate_indices(values: Any, neuron_count: int, label: str) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"{label} must be a JSON array")
    result = [int(value) for value in values]
    if result != sorted(set(result)):
        raise ValueError(f"{label} must be sorted and unique")
    if any(index < 0 or index >= neuron_count for index in result):
        raise ValueError(f"{label} contains an out-of-range index")
    return result


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate schema, replay the NumPy baseline, and check its golden output."""

    _require_keys(
        fixture,
        {
            "schema",
            "schema_version",
            "case_id",
            "parameters",
            "network",
            "initial_state",
            "overrides",
            "stimulus",
            "run",
            "acceptance",
            "expected",
        },
        "fixture",
    )
    if fixture["schema"] != SCHEMA or fixture["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported fixture schema or version")
    if fixture["case_id"] != "recurrent_cycle_end_v1":
        raise ValueError("unexpected fixture case_id")

    raw_parameters = fixture["parameters"]
    if not isinstance(raw_parameters, Mapping) or set(raw_parameters) != set(_RUST_PARAMETER_KEYS):
        raise ValueError("parameters must contain the Python and Rust model parameter fields")
    parameters = ModelParameters(**{key: raw_parameters[key] for key in _DYNAMICS_PARAMETER_KEYS})
    if float(raw_parameters["external_weight_mv"]) != parameters.poisson_weight_mv:
        raise ValueError("external_weight_mv must equal the published direct-input weight")

    raw_network = fixture["network"]
    if not isinstance(raw_network, Mapping):
        raise TypeError("network must be an object")
    _require_keys(
        raw_network,
        {
            "neuron_count",
            "edge_count",
            "neuron_ids_u64",
            "row_ptr_u32",
            "destinations_u32",
            "signed_counts_i16",
        },
        "network",
    )
    neuron_count = int(raw_network["neuron_count"])
    edge_count = int(raw_network["edge_count"])
    neuron_ids = np.asarray(raw_network["neuron_ids_u64"], dtype=np.uint64)
    row_ptr = np.asarray(raw_network["row_ptr_u32"], dtype=np.uint32)
    destinations = np.asarray(raw_network["destinations_u32"], dtype=np.uint32)
    signed_counts = np.asarray(raw_network["signed_counts_i16"], dtype=np.int16)
    if neuron_ids.shape != (neuron_count,):
        raise ValueError("neuron_ids_u64 has the wrong shape")
    if row_ptr.shape != (neuron_count + 1,):
        raise ValueError("row_ptr_u32 has the wrong shape")
    if destinations.shape != (edge_count,) or signed_counts.shape != (edge_count,):
        raise ValueError("CSR edge arrays have the wrong shape")
    connectome = PackedConnectome.from_arrays(neuron_ids, row_ptr, destinations, signed_counts)

    raw_initial = fixture["initial_state"]
    if not isinstance(raw_initial, Mapping):
        raise TypeError("initial_state must be an object")
    _require_keys(raw_initial, {"v_mv", "g_mv"}, "initial_state")
    initial_v = np.asarray(raw_initial["v_mv"], dtype=np.float64)
    initial_g = np.asarray(raw_initial["g_mv"], dtype=np.float64)
    if initial_v.shape != (neuron_count,) or initial_g.shape != (neuron_count,):
        raise ValueError("initial state vectors must have one value per neuron")
    if not np.all(np.isfinite(initial_v)) or not np.all(np.isfinite(initial_g)):
        raise ValueError("initial state must be finite")

    raw_overrides = fixture["overrides"]
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("overrides must be an object")
    _require_keys(raw_overrides, {"silenced_sources", "zero_refractory"}, "overrides")
    _validate_indices(raw_overrides["silenced_sources"], neuron_count, "silenced_sources")
    _validate_indices(raw_overrides["zero_refractory"], neuron_count, "zero_refractory")

    raw_stimulus = fixture["stimulus"]
    if not isinstance(raw_stimulus, Mapping):
        raise TypeError("stimulus must be an object")
    _require_keys(raw_stimulus, {"external_counts", "source_spikes"}, "stimulus")
    if raw_stimulus["external_counts"] or raw_stimulus["source_spikes"]:
        raise ValueError("tiny-parity-v1 intentionally has no external stimulus")

    raw_run = fixture["run"]
    if not isinstance(raw_run, Mapping):
        raise TypeError("run must be an object")
    _require_keys(raw_run, {"steps", "record_slots", "state_layout"}, "run")
    steps = int(raw_run["steps"])
    if steps != 15 or raw_run["record_slots"] != ["end"]:
        raise ValueError("unexpected run configuration")
    if raw_run["state_layout"] != "tick_major":
        raise ValueError("state_layout must be tick_major")

    raw_acceptance = fixture["acceptance"]
    if not isinstance(raw_acceptance, Mapping):
        raise TypeError("acceptance must be an object")
    state_atol = float(raw_acceptance["state_abs_tol_mv"])
    time_atol = float(raw_acceptance["time_abs_tol_ms"])
    if state_atol != STATE_ATOL_MV or time_atol != TIME_ATOL_MS:
        raise ValueError("unexpected acceptance tolerances")
    if raw_acceptance["spikes"] != "exact":
        raise ValueError("spikes must use exact comparison")

    raw_expected = fixture["expected"]
    if not isinstance(raw_expected, Mapping):
        raise TypeError("expected must be an object")
    _require_keys(raw_expected, {"times_ms", "spike_events", "v_end_mv", "g_end_mv"}, "expected")
    expected_times = np.asarray(raw_expected["times_ms"], dtype=np.float64)
    expected_v = np.asarray(raw_expected["v_end_mv"], dtype=np.float64)
    expected_g = np.asarray(raw_expected["g_end_mv"], dtype=np.float64)
    if expected_times.shape != (steps,):
        raise ValueError("expected times have the wrong shape")
    if expected_v.shape != (steps, neuron_count) or expected_g.shape != (
        steps,
        neuron_count,
    ):
        raise ValueError("expected state matrices have the wrong shape")
    if not np.all(np.isfinite(expected_times)):
        raise ValueError("expected times must be finite")
    expected_events = raw_expected["spike_events"]
    if not isinstance(expected_events, list):
        raise TypeError("spike_events must be an array")
    event_pairs = [(int(event["tick"]), int(event["neuron"])) for event in expected_events]
    if event_pairs != sorted(event_pairs):
        raise ValueError("spike_events must be sorted")
    if any(
        tick < 0 or tick >= steps or neuron < 0 or neuron >= neuron_count
        for tick, neuron in event_pairs
    ):
        raise ValueError("spike_events contains an out-of-range value")

    result = ReferenceSimulator(
        connectome,
        parameters,
        initial_v=initial_v,
        initial_g=initial_g,
        silenced_sources=raw_overrides["silenced_sources"],
        activated=raw_overrides["zero_refractory"],
    ).run(steps=steps, record=True, record_state=True)
    assert result is not None and result.voltage is not None and result.conductance is not None
    np.testing.assert_allclose(
        expected_times,
        np.arange(steps, dtype=np.float64) * parameters.dt_ms,
        rtol=0,
        atol=time_atol,
    )
    np.testing.assert_allclose(result.times_ms, expected_times, rtol=0, atol=time_atol)
    actual_pairs = list(zip(*np.nonzero(result.spikes), strict=True))
    if actual_pairs != event_pairs:
        raise AssertionError(f"spike events differ: expected {event_pairs}, actual {actual_pairs}")
    np.testing.assert_allclose(result.voltage, expected_v, rtol=0, atol=state_atol)
    np.testing.assert_allclose(result.conductance, expected_g, rtol=0, atol=state_atol)


def load_fixture(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("fixture root must be a JSON object")
    return value


def write_fixture(path: str | Path = DEFAULT_PATH) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(make_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--check", action="store_true", help="validate instead of writing")
    args = parser.parse_args()
    if args.check:
        validate_fixture(load_fixture(args.output))
        print(f"validated {args.output}")
    else:
        write_fixture(args.output)
        validate_fixture(load_fixture(args.output))
        print(f"generated and validated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
