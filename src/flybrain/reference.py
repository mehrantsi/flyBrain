"""A small, deterministic NumPy implementation of the FlyBrain model.

The production model is written in Brian2.  This module intentionally keeps the
same event ordering, but does not depend on Brian2 (which makes it useful as a
correctness oracle for the other backends):

``state update -> threshold -> synaptic/input events -> reset``.

Values in this module are plain floating point numbers in millivolts and
milliseconds.  In particular, ``g`` is a voltage-valued conductance variable,
as it is in the published Brian model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import Any

import numpy as np
import numpy.typing as npt

from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ExternalEvent = npt.ArrayLike | Mapping[int | float, Any]


def _as_index_mask(value: Sequence[int] | npt.ArrayLike | None, size: int, name: str) -> BoolArray:
    """Turn indices or a boolean mask into a validated mask."""

    mask = np.zeros(size, dtype=bool)
    if value is None:
        return mask

    array = np.asarray(value)
    if array.ndim == 1 and array.size == size and array.dtype.kind == "b":
        return array.astype(bool, copy=True)

    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index sequence or boolean mask")
    if array.size == 0:
        return mask
    if not np.issubdtype(array.dtype, np.integer):
        # A list such as [1.0, 2.0] is harmless, but silently truncating a
        # non-integral index is not.
        integral = np.equal(array, np.floor(array))
        if not np.all(integral):
            raise ValueError(f"{name} indices must be integers")
    indices = array.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError(f"{name} contains an index outside [0, {size})")
    mask[indices] = True
    return mask


def _duration_steps(duration_ms: float, dt_ms: float) -> int:
    if not np.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("duration_ms must be finite and non-negative")
    steps = round(float(duration_ms) / dt_ms)
    if not isclose(steps * dt_ms, float(duration_ms), abs_tol=1e-9):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    return int(steps)


def _steps_array(
    value: float | Sequence[float] | npt.ArrayLike,
    size: int,
    dt_ms: float,
    name: str,
) -> npt.NDArray[np.int64]:
    """Convert a scalar/vector of refractory durations to integer steps."""

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(size, float(values), dtype=float)
    elif values.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have one entry per neuron")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    steps = np.rint(values / dt_ms).astype(np.int64)
    if not np.all(np.isclose(steps * dt_ms, values, atol=1e-9)):
        raise ValueError(f"{name} must be an integer multiple of dt_ms")
    return steps


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Recorded output from :meth:`ReferenceSimulator.run`.

    ``spikes`` is a boolean array indexed as ``[step, neuron]``.  ``voltage``
    and ``conductance`` are state snapshots after each step when
    ``record_state=True`` was requested; otherwise they are ``None``.  The
    snapshots include events and resets performed during that step.
    """

    times_ms: FloatArray
    spikes: BoolArray
    voltage: FloatArray | None = None
    conductance: FloatArray | None = None

    @property
    def v(self) -> FloatArray | None:
        return self.voltage

    @property
    def g(self) -> FloatArray | None:
        return self.conductance

    @property
    def spike_times_ms(self) -> FloatArray:
        """Return one timestamp per spike, in step-major order."""

        return np.repeat(self.times_ms, np.count_nonzero(self.spikes, axis=1))

    @property
    def spike_neurons(self) -> npt.NDArray[np.int64]:
        """Return one neuron index per spike, in step-major order."""

        return np.nonzero(self.spikes)[1]

    @property
    def spike_indices(self) -> npt.NDArray[np.int64]:
        return self.spike_neurons

    @property
    def spike_trains(self) -> dict[int, FloatArray]:
        """Return Brian-like per-neuron spike time arrays."""

        return {
            neuron: self.times_ms[self.spikes[:, neuron]].copy()
            for neuron in range(self.spikes.shape[1])
            if np.any(self.spikes[:, neuron])
        }

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return np.asarray(self.spikes, dtype=dtype)


class ReferenceSimulator:
    """Deterministic NumPy simulator for a :class:`PackedConnectome`.

    Parameters
    ----------
    connectome:
        Source-major CSR connectivity.  Each signed count is multiplied by
        ``parameters.synapse_weight_mv`` on a presynaptic spike.
    parameters:
        Model constants.  The published defaults use a 0.1 ms timestep.
    activated, poisson_targets:
        Neurons that receive external/Poisson-style input.  Their refractory
        period is set to zero, matching ``poi`` in the Brian implementation.
        Either name may be used; supplying both is allowed and takes their
        union.
    silenced, silenced_sources:
        Presynaptic neurons whose outgoing synapses are disabled.  Silencing
        does not prevent those neurons from integrating or spiking.
    refractory_ms, refractory_steps:
        Optional scalar/vector override for the refractory period.  The
        ``*_steps`` form is convenient for tests and must contain integers.
    initial_v, initial_g:
        Optional initial state vectors.  Defaults are ``resting_mv`` and zero.

    Notes
    -----
    The threshold is strict (``v > threshold``).  Incoming events are applied
    after threshold evaluation and before reset, so an input landing on a
    neuron that spikes in the same step is cleared by that reset, just as in
    Brian's default schedule.
    """

    def __init__(
        self,
        connectome: PackedConnectome,
        parameters: ModelParameters | None = None,
        *,
        activated: Sequence[int] | npt.ArrayLike | None = None,
        poisson_targets: Sequence[int] | npt.ArrayLike | None = None,
        silenced: Sequence[int] | npt.ArrayLike | None = None,
        silenced_sources: Sequence[int] | npt.ArrayLike | None = None,
        refractory_ms: float | Sequence[float] | npt.ArrayLike | None = None,
        refractory_steps: int | Sequence[int] | npt.ArrayLike | None = None,
        initial_v: float | Sequence[float] | npt.ArrayLike | None = None,
        initial_g: float | Sequence[float] | npt.ArrayLike | None = None,
        external_weight_mv: float | npt.ArrayLike | None = None,
    ) -> None:
        if not isinstance(connectome, PackedConnectome):
            raise TypeError("connectome must be a PackedConnectome")
        self.connectome = connectome
        self.parameters = ModelParameters() if parameters is None else parameters
        self._n = connectome.neuron_count

        if refractory_ms is not None and refractory_steps is not None:
            raise ValueError("supply at most one of refractory_ms and refractory_steps")
        if refractory_steps is not None:
            values = np.asarray(refractory_steps)
            if values.ndim == 0:
                if not np.isfinite(values) or not np.equal(values, np.rint(values)):
                    raise ValueError("refractory_steps must contain integers")
                values = np.full(self._n, int(values), dtype=np.int64)
            elif values.shape == (self._n,):
                if not np.all(np.equal(values, np.rint(values))):
                    raise ValueError("refractory_steps must contain integers")
                values = values.astype(np.int64, copy=True)
            else:
                raise ValueError("refractory_steps must be a scalar or have one entry per neuron")
            if np.any(values < 0):
                raise ValueError("refractory_steps cannot be negative")
            self._refractory_period_steps = values
        elif refractory_ms is not None:
            self._refractory_period_steps = _steps_array(
                refractory_ms, self._n, self.parameters.dt_ms, "refractory_ms"
            )
        else:
            self._refractory_period_steps = np.full(
                self._n, self.parameters.refractory_steps, dtype=np.int64
            )

        activated_mask = _as_index_mask(activated, self._n, "activated")
        activated_mask |= _as_index_mask(poisson_targets, self._n, "poisson_targets")
        self._activated = activated_mask
        self._refractory_period_steps[activated_mask] = 0

        silenced_mask = _as_index_mask(silenced, self._n, "silenced")
        silenced_mask |= _as_index_mask(silenced_sources, self._n, "silenced_sources")
        self._silenced_sources = silenced_mask

        if external_weight_mv is None:
            self._external_weight_mv: float | FloatArray = float(self.parameters.poisson_weight_mv)
        else:
            weight = np.asarray(external_weight_mv, dtype=float)
            if weight.ndim == 0:
                self._external_weight_mv = float(weight)
            elif weight.shape == (self._n,):
                self._external_weight_mv = weight.copy()
            else:
                raise ValueError("external_weight_mv must be a scalar or one value per neuron")
        if isinstance(self._external_weight_mv, float) and not np.isfinite(
            self._external_weight_mv
        ):
            raise ValueError("external_weight_mv must be finite")
        if isinstance(self._external_weight_mv, np.ndarray) and not np.all(
            np.isfinite(self._external_weight_mv)
        ):
            raise ValueError("external_weight_mv must be finite")

        self._membrane_decay = float(
            np.exp(-self.parameters.dt_ms / self.parameters.membrane_tau_ms)
        )
        self._synapse_decay = float(np.exp(-self.parameters.dt_ms / self.parameters.synapse_tau_ms))
        if isclose(self.parameters.membrane_tau_ms, self.parameters.synapse_tau_ms):
            self._coupling = float(
                (self.parameters.dt_ms / self.parameters.membrane_tau_ms) * self._membrane_decay
            )
        else:
            self._coupling = float(
                self.parameters.synapse_tau_ms
                / (self.parameters.membrane_tau_ms - self.parameters.synapse_tau_ms)
                * (self._membrane_decay - self._synapse_decay)
            )

        # One slot is needed even when delay is zero.  A spike writes to the
        # current slot for zero delay and to a future slot otherwise.
        self._ring = np.zeros((self.parameters.delay_steps + 1, self._n), dtype=float)
        self._initial_v = self._state_vector(initial_v, self.parameters.resting_mv, "initial_v")
        self._initial_g = self._state_vector(initial_g, 0.0, "initial_g")
        self.v = self._initial_v.copy()
        self.g = self._initial_g.copy()
        self._refractory_remaining = np.zeros(self._n, dtype=np.int64)
        self._step_index = 0
        self.last_spikes = np.zeros(self._n, dtype=bool)

    def _state_vector(
        self, value: float | Sequence[float] | npt.ArrayLike | None, default: float, name: str
    ) -> FloatArray:
        if value is None:
            return np.full(self._n, default, dtype=float)
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(self._n, float(array), dtype=float)
        elif array.shape != (self._n,):
            raise ValueError(f"{name} must be a scalar or have one value per neuron")
        else:
            array = array.copy()
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values")
        return array

    @property
    def neuron_count(self) -> int:
        return self._n

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def time_ms(self) -> float:
        return self._step_index * self.parameters.dt_ms

    @property
    def refractory_remaining_steps(self) -> npt.NDArray[np.int64]:
        return self._refractory_remaining.copy()

    @property
    def refractory_remaining_ms(self) -> FloatArray:
        return self._refractory_remaining.astype(float) * self.parameters.dt_ms

    @property
    def refractory_steps(self) -> npt.NDArray[np.int64]:
        return self._refractory_period_steps.copy()

    @property
    def activated(self) -> BoolArray:
        return self._activated.copy()

    @property
    def poisson_targets(self) -> BoolArray:
        return self._activated.copy()

    @property
    def silenced_sources(self) -> BoolArray:
        return self._silenced_sources.copy()

    @silenced_sources.setter
    def silenced_sources(self, value: Sequence[int] | npt.ArrayLike) -> None:
        self._silenced_sources = _as_index_mask(value, self._n, "silenced_sources")

    @property
    def pending_synaptic(self) -> FloatArray:
        """Copy of the delayed-event ring, useful for diagnostics and tests."""

        return self._ring.copy()

    def reset(
        self,
        *,
        v: float | Sequence[float] | npt.ArrayLike | None = None,
        g: float | Sequence[float] | npt.ArrayLike | None = None,
    ) -> None:
        """Reset state and pending events while retaining model configuration."""

        self.v = (
            self._state_vector(v, self.parameters.resting_mv, "v")
            if v is not None
            else self._initial_v.copy()
        )
        self.g = self._state_vector(g, 0.0, "g") if g is not None else self._initial_g.copy()
        self._ring.fill(0.0)
        self._refractory_remaining.fill(0)
        self._step_index = 0
        self.last_spikes.fill(False)

    def _normalise_external_counts(self, counts: Any) -> FloatArray:
        """Convert one step of explicit inputs to a dense count vector."""

        if counts is None:
            return np.zeros(self._n, dtype=float)
        if isinstance(counts, Mapping):
            result = np.zeros(self._n, dtype=float)
            for target, count in counts.items():
                if not isinstance(target, (int, np.integer)):
                    raise TypeError("external event target indices must be integers")
                target_int = int(target)
                if target_int < 0 or target_int >= self._n:
                    raise IndexError(
                        f"external event target {target_int} is outside [0, {self._n})"
                    )
                result[target_int] += float(count)
            return result

        array = np.asarray(counts, dtype=float)
        if array.ndim == 0:
            if self._n != 1:
                raise ValueError("a scalar external count is only valid for a one-neuron network")
            array = array.reshape(1)
        if array.shape != (self._n,):
            raise ValueError("external counts must have one value per neuron")
        if not np.all(np.isfinite(array)):
            raise ValueError("external counts must be finite")
        return array

    def _apply_external_counts(self, counts: Any) -> None:
        dense = self._normalise_external_counts(counts)
        if not np.any(dense):
            return
        if isinstance(self._external_weight_mv, np.ndarray):
            self.v += dense * self._external_weight_mv
        else:
            self.v += dense * self._external_weight_mv

    def _push_spikes(self, spikes: BoolArray, slot: int) -> None:
        """Push source-major CSR events into one delay-ring slot."""

        sources = np.flatnonzero(spikes & ~self._silenced_sources)
        if sources.size == 0 or self.connectome.edge_count == 0:
            return
        destination_counts = self._ring[slot]
        row_ptr = self.connectome.row_ptr
        destinations = self.connectome.destinations
        signed_counts = self.connectome.signed_counts
        weight = self.parameters.synapse_weight_mv
        for source in sources:
            start = int(row_ptr[source])
            stop = int(row_ptr[source + 1])
            if start == stop:
                continue
            np.add.at(
                destination_counts, destinations[start:stop], signed_counts[start:stop] * weight
            )

    def step(
        self,
        external_counts: Any = None,
        *,
        external_events: Any = None,
        input_counts: Any = None,
    ) -> BoolArray:
        """Advance exactly one timestep and return the spikes from that step.

        ``external_counts`` (or either alias) is a dense count vector or a
        ``{target: count}`` mapping.  Inputs are direct voltage increments of
        ``count * parameters.poisson_weight_mv`` and are applied after the
        threshold test, matching Brian's PoissonInput scheduling.
        """

        supplied = [value is not None for value in (external_counts, external_events, input_counts)]
        if sum(supplied) > 1:
            raise ValueError("supply only one of external_counts, external_events, or input_counts")
        if external_events is not None:
            external_counts = external_events
        elif input_counts is not None:
            external_counts = input_counts

        free = self._refractory_remaining == 0
        if np.any(free):
            old_v = self.v[free]
            old_g = self.g[free]
            self.g[free] = old_g * self._synapse_decay
            self.v[free] = (
                self.parameters.resting_mv
                + (old_v - self.parameters.resting_mv) * self._membrane_decay
                + old_g * self._coupling
            )

        # The state updater runs before thresholding.  Refractory neurons do
        # not threshold, even if an input from an earlier event raised v.
        spikes = (self.v > self.parameters.threshold_mv) & free

        current_slot = self._step_index % self._ring.shape[0]
        self._push_spikes(
            spikes, (self._step_index + self.parameters.delay_steps) % self._ring.shape[0]
        )

        # Events are delivered after threshold evaluation.  For zero delay the
        # push above writes to current_slot, giving the same-step Brian order.
        delayed = self._ring[current_slot].copy()
        self._ring[current_slot].fill(0.0)
        self.g += delayed
        self._apply_external_counts(external_counts)

        # Brian reset is after synapses.  Consequently, a simultaneous input
        # and spike is discarded when g is reset here.
        if np.any(spikes):
            self.v[spikes] = self.parameters.reset_mv
            self.g[spikes] = 0.0
            # The spike itself occurs at the current clock time.  A period of
            # N clock intervals therefore blocks the following N-1 update
            # slots; the Nth slot is exactly the first allowed time.
            self._refractory_remaining[spikes] = np.maximum(
                self._refractory_period_steps[spikes] - 1, 0
            )

        # Existing refractory intervals consume one clock tick.  Newly set
        # intervals are intentionally left untouched until the next step.
        existing = ~spikes & (self._refractory_remaining > 0)
        self._refractory_remaining[existing] -= 1

        self.last_spikes = spikes.copy()
        self._step_index += 1
        return spikes.copy()

    def _event_schedule(self, events: Any, steps: int) -> Callable[[int], Any]:
        """Build a per-step accessor for the supported deterministic inputs."""

        if events is None:
            return lambda _step: None
        if callable(events):
            return lambda step: events(step)
        if isinstance(events, Mapping):
            by_step: dict[int, Any] = {}
            for key, value in events.items():
                if isinstance(key, (np.integer, int)):
                    event_step = int(key)
                elif isinstance(key, (float, np.floating)):
                    event_step = round(float(key) / self.parameters.dt_ms)
                    if not isclose(event_step * self.parameters.dt_ms, float(key), abs_tol=1e-9):
                        raise ValueError("external event times must be integer multiples of dt_ms")
                else:
                    raise TypeError(
                        "external event schedule keys must be step indices or times in ms"
                    )
                if event_step < 0 or event_step >= steps:
                    raise IndexError(f"external event step {event_step} is outside [0, {steps})")
                by_step[event_step] = value
            return lambda step: by_step.get(step)

        array = np.asarray(events, dtype=object)
        if array.ndim == 1 and array.size == self._n:
            # A vector is a constant input vector for every simulated step.
            dense = np.asarray(events, dtype=float).copy()
            return lambda _step: dense
        if array.ndim == 2 and array.shape == (steps, self._n):
            return lambda step: array[step]
        if array.ndim == 1 and self._n == 1 and array.size == steps:
            return lambda step: array[step]
        if isinstance(events, Sequence) and len(events) == steps:
            return lambda step: events[step]
        raise ValueError(
            "external events must be a per-neuron vector, a (steps, neurons) array, "
            "a step-indexed mapping, or a callable"
        )

    def run(
        self,
        duration_ms: float | None = None,
        *,
        steps: int | None = None,
        external_events: ExternalEvent | Callable[[int], Any] | None = None,
        external_counts: ExternalEvent | Callable[[int], Any] | None = None,
        record: bool = False,
        record_state: bool = False,
    ) -> SimulationResult | None:
        """Advance for a fixed duration.

        ``duration_ms`` and ``steps`` are interchangeable.  The optional
        external schedule is indexed by zero-based step; mapping keys may also
        be times in milliseconds.  With ``record=True`` a
        :class:`SimulationResult` is returned.  ``record_state`` additionally
        stores voltage and g snapshots.  Without recording, the simulator is
        advanced in-place and ``None`` is returned.
        """

        if duration_ms is not None and steps is not None:
            raise ValueError("supply at most one of duration_ms and steps")
        if steps is None:
            if duration_ms is None:
                raise ValueError("one of duration_ms or steps is required")
            steps = _duration_steps(float(duration_ms), self.parameters.dt_ms)
        else:
            if not isinstance(steps, (int, np.integer)) or int(steps) < 0:
                raise ValueError("steps must be a non-negative integer")
            steps = int(steps)

        if external_events is not None and external_counts is not None:
            raise ValueError("supply only one of external_events and external_counts")
        events = external_events if external_events is not None else external_counts
        schedule = self._event_schedule(events, steps)
        # Recording state snapshots is itself a recording request; callers do
        # not need to spell both flags when they only want trajectories.
        record = bool(record or record_state)
        keep_state = bool(record_state or record)
        spike_log = np.empty((steps, self._n), dtype=bool) if record else None
        times = np.empty(steps, dtype=float) if record else None
        voltage = np.empty((steps, self._n), dtype=float) if record_state else None
        conductance = np.empty((steps, self._n), dtype=float) if record_state else None

        for offset in range(steps):
            spikes = self.step(schedule(offset))
            if record:
                assert spike_log is not None and times is not None
                spike_log[offset] = spikes
                # Spikes are observed at the current Brian clock time.  The
                # state transition just performed advances the simulator to
                # the next clock boundary, hence the offset timestamp here.
                times[offset] = (self._step_index - 1) * self.parameters.dt_ms
            if keep_state and record_state:
                assert voltage is not None and conductance is not None
                voltage[offset] = self.v
                conductance[offset] = self.g

        if not record:
            return None
        assert spike_log is not None and times is not None
        return SimulationResult(times, spike_log, voltage, conductance)


# A short alias is useful to callers that refer to this as the reference
# engine rather than a simulator.
ReferenceEngine = ReferenceSimulator
NumpyReference = ReferenceSimulator


def simulate(
    connectome: PackedConnectome,
    duration_ms: float,
    *,
    parameters: ModelParameters | None = None,
    external_events: ExternalEvent | Callable[[int], Any] | None = None,
    activated: Sequence[int] | npt.ArrayLike | None = None,
    silenced_sources: Sequence[int] | npt.ArrayLike | None = None,
    record_state: bool = False,
) -> SimulationResult:
    """Convenience wrapper returning a recorded deterministic simulation."""

    simulator = ReferenceSimulator(
        connectome,
        parameters,
        activated=activated,
        silenced_sources=silenced_sources,
    )
    result = simulator.run(
        duration_ms,
        external_events=external_events,
        record=True,
        record_state=record_state,
    )
    assert result is not None
    return result


simulate_reference = simulate


__all__ = [
    "NumpyReference",
    "ReferenceEngine",
    "ReferenceSimulator",
    "SimulationResult",
    "simulate",
    "simulate_reference",
]
