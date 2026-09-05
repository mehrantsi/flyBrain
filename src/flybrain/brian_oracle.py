"""Small Brian2 reference/oracle network used for tick-level parity checks.

Brian2 is an optional dependency of the project.  Importing this module does
not require Brian2 to be installed; constructing an oracle does.  The oracle
uses a :class:`~flybrain.connectome.PackedConnectome` directly and keeps the
deterministic input path separate from the network's own threshold spikes.
Both paths execute their pre-synaptic code in Brian2's ``synapses`` slot.

The state monitors deliberately cover every scheduler boundary that is useful
when comparing a tick engine with Brian2.  A monitor records the value at the
beginning of its scheduler slot, after all lower-order objects in that slot
have run.  Thus, for example, ``g_by_slot_mv["synapses"]`` contains the
post-synaptic conductance after delayed pre-events, while
``g_by_slot_mv["groups"]`` contains the value before those events.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters

try:  # Brian2 is intentionally optional.
    from brian2 import (  # type: ignore[import-not-found]
        Clock,
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        ms,
        mV,
    )

    _BRIAN2_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised in no-extra-deps envs
    Clock = Network = NeuronGroup = SpikeGeneratorGroup = SpikeMonitor = None
    StateMonitor = Synapses = mV = ms = None
    _BRIAN2_IMPORT_ERROR = exc


DEFAULT_RECORD_SLOTS = (
    "start",
    "groups",
    "thresholds",
    "synapses",
    "resets",
    "end",
)
"""Brian2 scheduler slots at which state can be sampled."""

SYNAPSES_SCHEDULE_SLOT = "synapses"
"""The slot used by both network and deterministic-input pre pathways."""

RESOLVED_RESET = "v = reset_mv; g = 0 * mV"
"""Reset used by the oracle.

The published upstream string also contains ``w = 0``.  There is no ``w``
state variable in that model.  Brian2 compiles that statement as a local
scalar assignment and it has no effect on the neuron state; it is therefore
omitted here.  The unit on the ``g`` assignment is retained intentionally.
"""

UPSTREAM_RESET = "v = reset_mv; w = 0; g = 0 * mV"
"""The reset string in ``work/upstream/Drosophila_brain_model/model.py``."""


class Brian2UnavailableError(ImportError):
    """Raised when an oracle is requested without the optional Brian2 extra."""


def brian2_available() -> bool:
    """Return whether Brian2 can be imported in this Python environment."""

    return _BRIAN2_IMPORT_ERROR is None


def _require_brian2() -> None:
    if _BRIAN2_IMPORT_ERROR is not None:
        raise Brian2UnavailableError(
            "Brian2 is required for the oracle; install the optional 'reference' extra"
        ) from _BRIAN2_IMPORT_ERROR


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """One deterministic spike injected into a packed-connectome source.

    ``time_ms`` must be non-negative and an exact multiple of the model's
    ``dt_ms``.  The event causes all outgoing edges of ``neuron_index`` to be
    delivered through a Brian2 ``Synapses`` object at the normal ``synapses``
    scheduler slot, including the configured propagation delay.
    """

    neuron_index: int
    time_ms: float


@dataclass(frozen=True, slots=True)
class DeterministicEventSchedule:
    """Normalized deterministic events for :class:`Brian2Oracle`.

    Events are sorted by time and then source index.  Repeated events for one
    source at one time are rejected because ``SpikeGeneratorGroup`` represents
    one spike per neuron per clock tick.
    """

    indices: npt.NDArray[np.int64]
    times_ms: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        times_ms = np.asarray(self.times_ms, dtype=np.float64).reshape(-1)
        if indices.size != times_ms.size:
            raise ValueError("event indices and times must have equal lengths")
        if not np.all(np.isfinite(times_ms)):
            raise ValueError("event times must be finite")
        if np.any(times_ms < 0):
            raise ValueError("event times must be non-negative")
        order = np.lexsort((indices, times_ms))
        indices = indices[order]
        times_ms = times_ms[order]
        if indices.size:
            duplicate = (indices[1:] == indices[:-1]) & np.isclose(
                times_ms[1:], times_ms[:-1], atol=1e-12, rtol=0
            )
            if np.any(duplicate):
                raise ValueError("a source cannot have duplicate events at one time")
        # Frozen dataclasses do not freeze ndarray contents, so store private
        # copies to prevent accidental aliasing from a caller's input arrays.
        object.__setattr__(self, "indices", indices.copy())
        object.__setattr__(self, "times_ms", times_ms.copy())

    @property
    def neuron_indices(self) -> npt.NDArray[np.int64]:
        """Alias useful when reading a trace or constructing a schedule."""

        return self.indices

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True, slots=True)
class Brian2Trace:
    """Tick-level Brian2 observations.

    Arrays are shaped ``(tick, neuron)``.  Brian2's regular monitors emit one
    sample at times ``0, dt, ..., duration - dt``; consequently ``times_ms``
    has one fewer entry than the requested duration in milliseconds divided by
    ``dt_ms`` only when a zero-duration run is requested (where it is empty).
    Spike arrays hold threshold-event times, not monitor sample indices.
    """

    times_ms: npt.NDArray[np.float64]
    v_by_slot_mv: Mapping[str, npt.NDArray[np.float64]]
    g_by_slot_mv: Mapping[str, npt.NDArray[np.float64]]
    spike_indices: npt.NDArray[np.int64]
    spike_times_ms: npt.NDArray[np.float64]

    @property
    def tick_times_ms(self) -> npt.NDArray[np.float64]:
        """Alias for the regular state-monitor times."""

        return self.times_ms

    @property
    def v_mv(self) -> npt.NDArray[np.float64]:
        """Post-reset, post-synapse values (the ``end`` scheduler slot)."""

        return self.v_by_slot_mv["end"]

    @property
    def g_mv(self) -> npt.NDArray[np.float64]:
        """Post-reset, post-synapse values (the ``end`` scheduler slot)."""

        return self.g_by_slot_mv["end"]

    @property
    def spikes(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return ``(indices, times_ms)`` for threshold events."""

        return self.spike_indices, self.spike_times_ms


def _as_ms_float(value: Any) -> float:
    """Convert a scalar milliseconds value or Brian2 time quantity to float."""

    # Plain Python/NumPy scalars are already interpreted as milliseconds by
    # this adapter.  Dividing a plain scalar by Brian2's ``ms`` would instead
    # interpret it as SI seconds and silently multiply it by 1000.
    if _BRIAN2_IMPORT_ERROR is None and hasattr(value, "dim"):
        try:
            return float(value / ms)
        except (TypeError, ValueError, AttributeError):
            pass
    return float(value)


def _validate_grid(value_ms: float, dt_ms: float, label: str) -> float:
    if not np.isfinite(value_ms) or value_ms < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    steps = round(value_ms / dt_ms)
    if not np.isclose(steps * dt_ms, value_ms, atol=1e-9, rtol=0):
        raise ValueError(f"{label} must be an integer multiple of dt_ms")
    return float(steps * dt_ms)


def _normalize_schedule(
    schedule: DeterministicEventSchedule
    | Mapping[int, Iterable[float]]
    | Sequence[ScheduledEvent | Sequence[float]]
    | npt.ArrayLike
    | None,
) -> DeterministicEventSchedule:
    if schedule is None:
        return DeterministicEventSchedule(np.empty(0, dtype=np.int64), np.empty(0))
    if isinstance(schedule, DeterministicEventSchedule):
        return schedule
    if isinstance(schedule, Mapping):
        indices: list[int] = []
        times: list[float] = []
        for index, event_times in schedule.items():
            for time_ms in event_times:
                indices.append(int(index))
                times.append(_as_ms_float(time_ms))
        return DeterministicEventSchedule(indices, times)
    if isinstance(schedule, np.ndarray):
        pairs = np.asarray(schedule)
    else:
        items = list(schedule)
        if not items:
            return DeterministicEventSchedule(np.empty(0, dtype=np.int64), np.empty(0))
        if isinstance(items[0], ScheduledEvent):
            return DeterministicEventSchedule(
                [item.neuron_index for item in items],  # type: ignore[union-attr]
                [item.time_ms for item in items],  # type: ignore[union-attr]
            )
        pairs = np.asarray(items)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("event schedule must be (neuron_index, time_ms) pairs")
    return DeterministicEventSchedule(
        np.asarray(pairs[:, 0], dtype=np.int64),
        np.asarray([_as_ms_float(item) for item in pairs[:, 1]], dtype=np.float64),
    )


def _normalize_indices(
    values: Iterable[int] | None, neuron_count: int, label: str
) -> npt.NDArray[np.int64]:
    if values is None:
        return np.empty(0, dtype=np.int64)
    result = np.asarray(list(values), dtype=np.int64).reshape(-1)
    if result.size and (np.any(result < 0) or np.any(result >= neuron_count)):
        raise ValueError(f"{label} contains an index outside the connectome")
    return np.unique(result)


def _packed_edges(
    connectome: PackedConnectome,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    row_ptr = np.asarray(connectome.row_ptr, dtype=np.int64)
    source = np.repeat(np.arange(connectome.neuron_count, dtype=np.int64), np.diff(row_ptr))
    destination = np.asarray(connectome.destinations, dtype=np.int64)
    return source, destination


class Brian2Oracle:
    """Build and run a small deterministic Brian2 network.

    Parameters
    ----------
    connectome:
        Packed CSR connectivity.  ``signed_counts`` are converted to volt
        weights by multiplying by ``parameters.synapse_weight_mv``.
    parameters:
        Model constants.  The default ``dt_ms`` is 0.1 ms and the network uses
        Brian2's ``method="linear"`` state updater.
    event_schedule:
        Explicit ``(neuron_index, time_ms)`` events, a mapping from source to
        times, :class:`ScheduledEvent` objects, or a
        :class:`DeterministicEventSchedule`.  No stochastic input is created.
    silenced_sources:
        Source indices whose outgoing edge weights are set to zero in both the
        network and deterministic-input synapses.
    zero_refractory:
        Target neuron indices whose per-neuron refractory variable is set to
        zero, matching the upstream input-neuron override.
    record_slots:
        Any subset of :data:`DEFAULT_RECORD_SLOTS`.  Include ``"end"`` (the
        default) to use the convenient ``trace.v_mv``/``trace.g_mv`` aliases.
    """

    def __init__(
        self,
        connectome: PackedConnectome,
        parameters: ModelParameters | None = None,
        *,
        event_schedule: DeterministicEventSchedule
        | Mapping[int, Iterable[float]]
        | Sequence[ScheduledEvent | Sequence[float]]
        | npt.ArrayLike
        | None = None,
        silenced_sources: Iterable[int] | None = None,
        zero_refractory: Iterable[int] | None = None,
        initial_v_mv: float | npt.ArrayLike | None = None,
        initial_g_mv: float | npt.ArrayLike | None = None,
        record_slots: Sequence[str] = DEFAULT_RECORD_SLOTS,
    ) -> None:
        _require_brian2()
        connectome.validate()
        self.connectome = connectome
        self.parameters = parameters or ModelParameters()
        self.event_schedule = _normalize_schedule(event_schedule)
        self.silenced_sources = _normalize_indices(
            silenced_sources, connectome.neuron_count, "silenced_sources"
        )
        self.zero_refractory = _normalize_indices(
            zero_refractory, connectome.neuron_count, "zero_refractory"
        )
        slots = tuple(record_slots)
        unknown_slots = set(slots).difference(DEFAULT_RECORD_SLOTS)
        if unknown_slots:
            raise ValueError(f"unknown Brian2 record slot(s): {sorted(unknown_slots)}")
        if len(set(slots)) != len(slots):
            raise ValueError("record_slots must not contain duplicates")
        if "end" not in slots:
            raise ValueError("record_slots must include 'end' for the trace aliases")
        self.record_slots = slots

        self.clock = Clock(dt=self.parameters.dt_ms * ms)
        self._edge_source, self._edge_destination = _packed_edges(connectome)
        self._edge_weight_mv = connectome.signed_counts.astype(np.float64) * float(
            self.parameters.synapse_weight_mv
        )
        if self.silenced_sources.size:
            silenced = np.isin(self._edge_source, self.silenced_sources)
            self._edge_weight_mv = self._edge_weight_mv.copy()
            self._edge_weight_mv[silenced] = 0.0

        self.neurons = self._build_neurons(initial_v_mv, initial_g_mv)
        self.synapses = self._build_network_synapses()
        self.event_group, self.event_synapses = self._build_event_synapses()
        self.spike_monitor = SpikeMonitor(self.neurons, name="oracle_spikes")
        self.state_monitors = {
            slot: StateMonitor(
                self.neurons,
                ("v", "g"),
                record=True,
                when=slot,
                order=10_000,
                name=f"oracle_state_{slot}",
            )
            for slot in self.record_slots
        }
        objects: list[Any] = [self.neurons, self.synapses, self.spike_monitor]
        if self.event_group is not None:
            objects.extend([self.event_group, self.event_synapses])
        objects.extend(self.state_monitors.values())
        self.network = Network(*objects, name="oracle_network")

    def _build_neurons(
        self, initial_v_mv: float | npt.ArrayLike | None, initial_g_mv: float | npt.ArrayLike | None
    ) -> Any:
        p = self.parameters
        namespace = {
            "resting_mv": p.resting_mv * mV,
            "reset_mv": p.reset_mv * mV,
            "threshold_mv": p.threshold_mv * mV,
            "membrane_tau_ms": p.membrane_tau_ms * ms,
            "synapse_tau_ms": p.synapse_tau_ms * ms,
        }
        equations = """
        dv/dt = (resting_mv - v + g) / membrane_tau_ms : volt (unless refractory)
        dg/dt = -g / synapse_tau_ms : volt (unless refractory)
        rfc : second
        """
        neurons = NeuronGroup(
            self.connectome.neuron_count,
            equations,
            method="linear",
            threshold="v > threshold_mv",
            reset=RESOLVED_RESET,
            refractory="rfc",
            namespace=namespace,
            clock=self.clock,
            name="oracle_neurons",
        )
        neurons.v = self._initial_values(initial_v_mv, p.resting_mv, "initial_v_mv") * mV
        neurons.g = self._initial_values(initial_g_mv, 0.0, "initial_g_mv") * mV
        neurons.rfc = p.refractory_ms * ms
        if self.zero_refractory.size:
            neurons.rfc[self.zero_refractory] = 0 * ms
        return neurons

    def _initial_values(
        self, values: float | npt.ArrayLike | None, default: float, label: str
    ) -> float | npt.NDArray[np.float64]:
        if values is None:
            return default
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return float(array)
        array = array.reshape(-1)
        if array.size != self.connectome.neuron_count:
            raise ValueError(f"{label} must be scalar or have one value per neuron")
        return array

    def _build_network_synapses(self) -> Any:
        p = self.parameters
        synapses = Synapses(
            self.neurons,
            self.neurons,
            "w : volt",
            on_pre="g_post += w",
            delay=p.delay_ms * ms,
            clock=self.clock,
            name="oracle_synapses",
        )
        synapses.connect(i=self._edge_source, j=self._edge_destination)
        synapses.w = self._edge_weight_mv * mV
        return synapses

    def _build_event_synapses(self) -> tuple[Any | None, Any | None]:
        if len(self.event_schedule) == 0:
            return None, None
        schedule = self.event_schedule
        if np.any(schedule.indices < 0) or np.any(schedule.indices >= self.connectome.neuron_count):
            raise ValueError("event schedule contains an index outside the connectome")
        _validate_grid_array(schedule.times_ms, self.parameters.dt_ms, "event times")
        event_group = SpikeGeneratorGroup(
            self.connectome.neuron_count,
            indices=schedule.indices,
            times=schedule.times_ms * ms,
            clock=self.clock,
            name="oracle_deterministic_sources",
        )
        event_synapses = Synapses(
            event_group,
            self.neurons,
            "w : volt",
            on_pre="g_post += w",
            delay=self.parameters.delay_ms * ms,
            clock=self.clock,
            name="oracle_deterministic_synapses",
        )
        event_synapses.connect(i=self._edge_source, j=self._edge_destination)
        event_synapses.w = self._edge_weight_mv * mV
        return event_group, event_synapses

    def run(self, duration_ms: float) -> Brian2Trace:
        """Run once and return regular state snapshots plus threshold events."""

        duration = _validate_grid(_as_ms_float(duration_ms), self.parameters.dt_ms, "duration_ms")
        if len(self.event_schedule) and np.any(self.event_schedule.times_ms > duration + 1e-9):
            raise ValueError("event schedule contains an event after duration_ms")
        if getattr(self, "_has_run", False):
            raise RuntimeError("a Brian2Oracle instance can only be run once")
        self._has_run = True
        self.network.run(duration * ms)

        monitors = self.state_monitors
        if monitors:
            times_ms = np.asarray(monitors[self.record_slots[0]].t / ms, dtype=np.float64)
        else:  # ``end`` is required, so this is defensive only.
            times_ms = np.empty(0, dtype=np.float64)
        v_by_slot = {
            slot: np.asarray(mon.v / mV, dtype=np.float64).T.copy()
            for slot, mon in monitors.items()
        }
        g_by_slot = {
            slot: np.asarray(mon.g / mV, dtype=np.float64).T.copy()
            for slot, mon in monitors.items()
        }
        spike_indices = np.asarray(self.spike_monitor.i[:], dtype=np.int64).copy()
        spike_times_ms = np.asarray(self.spike_monitor.t[:] / ms, dtype=np.float64).copy()
        return Brian2Trace(
            times_ms=times_ms.copy(),
            v_by_slot_mv=v_by_slot,
            g_by_slot_mv=g_by_slot,
            spike_indices=spike_indices,
            spike_times_ms=spike_times_ms,
        )


def _validate_grid_array(values_ms: npt.ArrayLike, dt_ms: float, label: str) -> None:
    values = np.asarray(values_ms, dtype=np.float64)
    steps = np.rint(values / dt_ms)
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0)
        or np.any(~np.isclose(steps * dt_ms, values, atol=1e-9, rtol=0))
    ):
        raise ValueError(f"{label} must be finite, non-negative, and aligned to dt_ms")


def run_brian2_oracle(
    connectome: PackedConnectome,
    duration_ms: float,
    parameters: ModelParameters | None = None,
    **kwargs: Any,
) -> Brian2Trace:
    """Convenience wrapper that builds and runs :class:`Brian2Oracle`."""

    return Brian2Oracle(connectome, parameters, **kwargs).run(duration_ms)


# A short name is handy in notebooks and keeps the adapter discoverable without
# changing flybrain.__init__ (the optional import must remain opt-in).
run_oracle = run_brian2_oracle


__all__ = [
    "DEFAULT_RECORD_SLOTS",
    "RESOLVED_RESET",
    "SYNAPSES_SCHEDULE_SLOT",
    "UPSTREAM_RESET",
    "Brian2Oracle",
    "Brian2Trace",
    "Brian2UnavailableError",
    "DeterministicEventSchedule",
    "ScheduledEvent",
    "brian2_available",
    "run_brian2_oracle",
    "run_oracle",
]
