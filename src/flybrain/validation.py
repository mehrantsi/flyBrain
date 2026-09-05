from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import mlx.core as mx
import numpy as np

from flybrain.connectome import PackedConnectome
from flybrain.mlx_engine import MLXEngine, PropagationBackend
from flybrain.protocols import RIGHT_SUGAR_GRN_IDS, indices_for_flywire_ids
from flybrain.reference import ReferenceSimulator
from flybrain.stimulus import ExternalEvents, PoissonStimulus


@dataclass(frozen=True, slots=True)
class EngineValidation:
    materialization: str
    steps: int
    biological_ms: float
    stimulus_rate_hz: float
    seed: int
    propagation: str
    stimulus_target_count: int
    missing_stimulus_ids: tuple[int, ...]
    reference_seconds: float
    accelerated_seconds: float
    speedup: float
    accelerated_realtime_factor: float
    peak_memory_bytes: int
    reference_spike_count: int
    accelerated_spike_count: int
    spike_count_equal: bool
    per_neuron_spike_counts_equal: bool
    mismatched_spike_count_neuron_ids: tuple[int, ...]
    tick_exact_spikes: bool
    mismatched_ticks: int
    mismatched_spike_events: int
    mismatched_neuron_ids: tuple[int, ...]
    maximum_mismatched_spike_shift_ms: float | None
    maximum_voltage_error_mv: float
    maximum_conductance_error_mv: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_engine(
    connectome: PackedConnectome,
    *,
    steps: int = 1000,
    rate_hz: float = 150.0,
    seed: int = 20260816,
    propagation: PropagationBackend = "metal",
) -> EngineValidation:
    if steps <= 0:
        raise ValueError("steps must be positive")
    available_ids = {int(flywire_id) for flywire_id in connectome.neuron_ids}
    missing_ids = tuple(
        flywire_id for flywire_id in RIGHT_SUGAR_GRN_IDS if flywire_id not in available_ids
    )
    targets = indices_for_flywire_ids(
        connectome,
        RIGHT_SUGAR_GRN_IDS,
        allow_missing=True,
    )
    if targets.size == 0:
        raise ValueError("none of the published sugar-neuron IDs exist in this pack")
    stimulus = PoissonStimulus(
        targets,
        rate_hz,
        dt_ms=0.1,
        seed=seed,
    )
    events = [stimulus.next_events() for _ in range(steps)]

    reference = ReferenceSimulator(connectome, activated=targets)
    expected_spikes: list[np.ndarray] = []
    reference_counts = np.zeros(connectome.neuron_count, dtype=np.int32)
    reference_count = 0
    started = perf_counter()
    for event in events:
        spikes = reference.step(_event_mapping(event))
        spike_indices = np.flatnonzero(spikes)
        expected_spikes.append(spike_indices)
        reference_counts[spike_indices] += 1
        reference_count += int(np.count_nonzero(spikes))
    reference_seconds = perf_counter() - started

    mx.reset_peak_memory()
    accelerated = MLXEngine(
        connectome,
        propagation=propagation,
        zero_refractory=targets,
    )
    mismatched_ticks = 0
    mismatched_events = 0
    mismatched_neurons: set[int] = set()
    actual_spikes_by_tick: list[np.ndarray] = []
    accelerated_count = 0
    started = perf_counter()
    for step, event in enumerate(events):
        spikes = accelerated.step(event).spikes
        assert spikes is not None
        actual = np.flatnonzero(spikes)
        actual_spikes_by_tick.append(actual)
        accelerated_count += int(actual.size)
        if not np.array_equal(actual, expected_spikes[step]):
            mismatched_ticks += 1
            different = np.setxor1d(actual, expected_spikes[step], assume_unique=True)
            mismatched_events += int(different.size)
            mismatched_neurons.update(int(index) for index in different)
    mx.synchronize()
    accelerated_seconds = perf_counter() - started

    voltage = np.asarray(accelerated.voltage_mv, dtype=np.float32)
    conductance = np.asarray(accelerated.conductance_mv, dtype=np.float32)
    accelerated_counts = np.asarray(accelerated.spike_counts, dtype=np.int32)
    mismatched_count_indices = np.flatnonzero(reference_counts != accelerated_counts)
    maximum_shift = _maximum_spike_shift_ms(
        expected_spikes,
        actual_spikes_by_tick,
        mismatched_neurons,
        dt_ms=0.1,
    )
    return EngineValidation(
        materialization=str(connectome.manifest.get("materialization", "unknown")),
        steps=steps,
        biological_ms=steps * 0.1,
        stimulus_rate_hz=rate_hz,
        seed=seed,
        propagation=propagation,
        stimulus_target_count=int(targets.size),
        missing_stimulus_ids=missing_ids,
        reference_seconds=reference_seconds,
        accelerated_seconds=accelerated_seconds,
        speedup=reference_seconds / accelerated_seconds,
        accelerated_realtime_factor=accelerated_seconds / (steps * 0.1 / 1000.0),
        peak_memory_bytes=int(mx.get_peak_memory()),
        reference_spike_count=reference_count,
        accelerated_spike_count=accelerated_count,
        spike_count_equal=reference_count == accelerated_count,
        per_neuron_spike_counts_equal=mismatched_count_indices.size == 0,
        mismatched_spike_count_neuron_ids=tuple(
            int(connectome.neuron_ids[index]) for index in mismatched_count_indices
        ),
        tick_exact_spikes=mismatched_ticks == 0,
        mismatched_ticks=mismatched_ticks,
        mismatched_spike_events=mismatched_events,
        mismatched_neuron_ids=tuple(
            int(connectome.neuron_ids[index]) for index in sorted(mismatched_neurons)
        ),
        maximum_mismatched_spike_shift_ms=maximum_shift,
        maximum_voltage_error_mv=float(np.max(np.abs(reference.v - voltage))),
        maximum_conductance_error_mv=float(np.max(np.abs(reference.g - conductance))),
    )


def _event_mapping(event: ExternalEvents) -> dict[int, int]:
    return {
        int(index): int(count) for index, count in zip(event.indices, event.counts, strict=True)
    }


def _maximum_spike_shift_ms(
    expected_by_tick: list[np.ndarray],
    actual_by_tick: list[np.ndarray],
    neuron_indices: set[int],
    *,
    dt_ms: float,
) -> float | None:
    if not neuron_indices:
        return 0.0
    maximum_steps = 0
    for neuron in neuron_indices:
        expected = np.asarray(
            [step for step, indices in enumerate(expected_by_tick) if neuron in indices],
            dtype=np.int64,
        )
        actual = np.asarray(
            [step for step, indices in enumerate(actual_by_tick) if neuron in indices],
            dtype=np.int64,
        )
        if expected.shape != actual.shape:
            return None
        maximum_steps = max(maximum_steps, int(np.max(np.abs(expected - actual), initial=0)))
    return maximum_steps * dt_ms


__all__ = ["EngineValidation", "validate_engine"]
