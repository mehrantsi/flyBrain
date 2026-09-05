from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters
from flybrain.stimulus import ExternalEvents, PoissonStimulus

PropagationBackend = Literal["scatter", "metal"]


_PROPAGATE_SOURCE = r"""
    uint source = thread_position_in_grid.x;
    if (source >= delayed_spikes_shape[0]) {
        return;
    }
    if (delayed_spikes[source] == 0 || silenced_sources[source] != 0) {
        return;
    }

    uint begin = row_ptr[source];
    uint end = row_ptr[source + 1];
    for (uint edge = begin; edge < end; ++edge) {
        uint destination = destinations[edge];
        int count = int(signed_counts[edge]);
        atomic_fetch_add_explicit(
            &arrivals[destination], count, memory_order_relaxed
        );
    }
"""


def _make_propagation_kernel() -> object:
    return mx.fast.metal_kernel(
        name="flybrain_propagate_csr",
        input_names=[
            "row_ptr",
            "destinations",
            "signed_counts",
            "delayed_spikes",
            "silenced_sources",
        ],
        output_names=["arrivals"],
        source=_PROPAGATE_SOURCE,
        atomic_outputs=True,
    )


@dataclass(frozen=True, slots=True)
class MLXStep:
    index: int
    spikes: npt.NDArray[np.bool_] | None
    voltage_mv: npt.NDArray[np.float32] | None = None
    conductance_mv: npt.NDArray[np.float32] | None = None


class MLXEngine:
    def __init__(
        self,
        connectome: PackedConnectome,
        parameters: ModelParameters | None = None,
        *,
        propagation: PropagationBackend = "metal",
        silenced_sources: npt.ArrayLike | None = None,
        zero_refractory: npt.ArrayLike | None = None,
    ) -> None:
        if propagation not in ("scatter", "metal"):
            raise ValueError(f"unknown propagation backend: {propagation}")

        self.connectome = connectome
        self.parameters = parameters or ModelParameters()
        self.propagation = propagation
        self._step_index = 0
        self._ring_size = max(1, self.parameters.delay_steps)

        neuron_count = connectome.neuron_count
        silenced = _index_mask(neuron_count, silenced_sources)
        zero_rfc = _index_mask(neuron_count, zero_refractory)

        self.row_ptr = mx.array(np.asarray(connectome.row_ptr))
        self.destinations = mx.array(np.asarray(connectome.destinations))
        self.signed_counts = mx.array(np.asarray(connectome.signed_counts))
        self.silenced_sources = mx.array(silenced.astype(np.uint8))
        refractory_lengths = np.full(neuron_count, self.parameters.refractory_steps, dtype=np.int32)
        refractory_lengths[zero_rfc] = 0
        self.refractory_lengths = mx.array(refractory_lengths)

        self._sources: mx.array | None = None
        if propagation == "scatter":
            degrees = np.diff(np.asarray(connectome.row_ptr, dtype=np.int64))
            self._sources = mx.array(np.repeat(np.arange(neuron_count, dtype=np.uint32), degrees))
            self._propagation_kernel = None
        else:
            self._propagation_kernel = _make_propagation_kernel()

        self.voltage_mv = mx.full((neuron_count,), self.parameters.resting_mv, dtype=mx.float32)
        self.conductance_mv = mx.zeros((neuron_count,), dtype=mx.float32)
        self.refractory_remaining = mx.zeros((neuron_count,), dtype=mx.int32)
        self.spike_ring = mx.zeros((self._ring_size, neuron_count), dtype=mx.uint8)
        self.spike_counts = mx.zeros((neuron_count,), dtype=mx.int32)
        mx.eval(
            self.voltage_mv,
            self.conductance_mv,
            self.refractory_remaining,
            self.spike_ring,
            self.spike_counts,
        )

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(
        self,
        external_event_counts: npt.ArrayLike | ExternalEvents | None = None,
        *,
        record_state: bool = False,
        copy_spikes: bool = True,
    ) -> MLXStep:
        parameters = self.parameters
        neuron_count = self.connectome.neuron_count

        self.refractory_remaining = mx.maximum(self.refractory_remaining - 1, 0)
        can_update = self.refractory_remaining == 0

        old_g = self.conductance_mv
        integrated_g = old_g * parameters.synapse_decay
        integrated_v = (
            parameters.resting_mv
            + parameters.membrane_decay * (self.voltage_mv - parameters.resting_mv)
            + parameters.membrane_synapse_coupling * old_g
        )
        self.voltage_mv = mx.where(can_update, integrated_v, self.voltage_mv)
        self.conductance_mv = mx.where(can_update, integrated_g, old_g)

        spikes = can_update & (self.voltage_mv > parameters.threshold_mv)

        if parameters.delay_steps == 0:
            delayed_spikes = spikes.astype(mx.uint8)
        else:
            slot = self._step_index % self._ring_size
            delayed_spikes = self.spike_ring[slot]

        arrivals = self._propagate(delayed_spikes)
        self.conductance_mv = (
            self.conductance_mv + arrivals.astype(mx.float32) * parameters.synapse_weight_mv
        )

        if isinstance(external_event_counts, ExternalEvents):
            if external_event_counts.indices.size:
                if int(external_event_counts.indices.max()) >= neuron_count:
                    raise ValueError("external event index is outside the connectome")
                self.voltage_mv = self.voltage_mv.at[mx.array(external_event_counts.indices)].add(
                    mx.array(external_event_counts.counts).astype(mx.float32)
                    * parameters.poisson_weight_mv
                )
        elif external_event_counts is not None:
            external = np.asarray(external_event_counts, dtype=np.int32)
            if external.shape != (neuron_count,):
                raise ValueError("external_event_counts must have one entry per neuron")
            self.voltage_mv = (
                self.voltage_mv
                + mx.array(external).astype(mx.float32) * parameters.poisson_weight_mv
            )

        self.voltage_mv = mx.where(spikes, parameters.reset_mv, self.voltage_mv)
        self.conductance_mv = mx.where(spikes, 0.0, self.conductance_mv)
        self.refractory_remaining = mx.where(
            spikes, self.refractory_lengths, self.refractory_remaining
        )

        if parameters.delay_steps > 0:
            self.spike_ring = mx.slice_update(
                self.spike_ring,
                spikes.astype(mx.uint8)[None, :],
                start_indices=mx.array([slot, 0], dtype=mx.int32),
                axes=(0, 1),
            )
        self.spike_counts = self.spike_counts + spikes.astype(mx.int32)

        mx.eval(
            spikes,
            self.voltage_mv,
            self.conductance_mv,
            self.refractory_remaining,
            self.spike_ring,
            self.spike_counts,
        )
        spike_array = np.asarray(spikes, dtype=np.bool_) if copy_spikes else None
        result = MLXStep(
            index=self._step_index,
            spikes=spike_array,
            voltage_mv=(
                np.asarray(self.voltage_mv, dtype=np.float32).copy() if record_state else None
            ),
            conductance_mv=(
                np.asarray(self.conductance_mv, dtype=np.float32).copy() if record_state else None
            ),
        )
        self._step_index += 1
        return result

    def run_counts(
        self,
        steps: int,
        stimulus: PoissonStimulus | None = None,
    ) -> npt.NDArray[np.int32]:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        for _ in range(steps):
            events = None if stimulus is None else stimulus.next_events()
            self.step(events, copy_spikes=False)
        return np.asarray(self.spike_counts, dtype=np.int32).copy()

    def run(
        self,
        steps: int,
        external_events: npt.ArrayLike | None = None,
        *,
        record_state: bool = False,
    ) -> list[MLXStep]:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        events: npt.NDArray[np.int32] | None = None
        if external_events is not None:
            events = np.asarray(external_events, dtype=np.int32)
            expected = (steps, self.connectome.neuron_count)
            if events.shape != expected:
                raise ValueError(f"external_events must have shape {expected}")

        return [
            self.step(None if events is None else events[index], record_state=record_state)
            for index in range(steps)
        ]

    def _propagate(self, delayed_spikes: mx.array) -> mx.array:
        if self.propagation == "scatter":
            assert self._sources is not None
            active_edges = delayed_spikes[self._sources].astype(mx.bool_) & (
                self.silenced_sources[self._sources] == 0
            )
            contributions = mx.where(
                active_edges,
                self.signed_counts.astype(mx.int32),
                mx.zeros(self.signed_counts.shape, dtype=mx.int32),
            )
            return (
                mx.zeros((self.connectome.neuron_count,), dtype=mx.int32)
                .at[self.destinations]
                .add(contributions)
            )

        assert self._propagation_kernel is not None
        threadgroup_width = min(256, self.connectome.neuron_count)
        outputs = self._propagation_kernel(
            inputs=[
                self.row_ptr,
                self.destinations,
                self.signed_counts,
                delayed_spikes,
                self.silenced_sources,
            ],
            grid=(self.connectome.neuron_count, 1, 1),
            threadgroup=(threadgroup_width, 1, 1),
            output_shapes=[(self.connectome.neuron_count,)],
            output_dtypes=[mx.int32],
            init_value=0,
        )
        return outputs[0]


def _index_mask(size: int, indices: npt.ArrayLike | None) -> npt.NDArray[np.bool_]:
    mask = np.zeros(size, dtype=np.bool_)
    if indices is None:
        return mask
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim != 1:
        raise ValueError("neuron index lists must be one-dimensional")
    if index_array.size and (int(index_array.min()) < 0 or int(index_array.max()) >= size):
        raise ValueError("neuron index is outside the connectome")
    mask[index_array] = True
    return mask
