from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ExternalEvents:
    indices: npt.NDArray[np.int32]
    counts: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int32)
        counts = np.asarray(self.counts, dtype=np.int32)
        if indices.ndim != 1 or counts.ndim != 1:
            raise ValueError("external event arrays must be one-dimensional")
        if indices.shape != counts.shape:
            raise ValueError("external event indices and counts must have equal lengths")
        if counts.size and np.any(counts <= 0):
            raise ValueError("external event counts must be positive")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "counts", counts)

    @classmethod
    def empty(cls) -> ExternalEvents:
        return cls(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))


class PoissonStimulus:
    def __init__(
        self,
        indices: npt.ArrayLike,
        rates_hz: float | npt.ArrayLike,
        *,
        dt_ms: float = 0.1,
        seed: int = 0,
    ) -> None:
        targets = np.asarray(indices, dtype=np.int32)
        if targets.ndim != 1:
            raise ValueError("stimulus indices must be one-dimensional")
        if targets.size and int(targets.min()) < 0:
            raise ValueError("stimulus indices cannot be negative")

        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), targets.shape)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("stimulus rates must be finite and non-negative")
        probabilities = rates * dt_ms / 1000.0
        if np.any(probabilities > 1):
            raise ValueError("rate * timestep cannot exceed one for N=1 PoissonInput")

        self.indices = targets
        self.probabilities = np.asarray(probabilities)
        self._rng = np.random.default_rng(seed)

    def next_events(self) -> ExternalEvents:
        fired = self._rng.random(self.indices.size) < self.probabilities
        if not np.any(fired):
            return ExternalEvents.empty()
        return ExternalEvents(
            indices=self.indices[fired],
            counts=np.ones(int(fired.sum()), dtype=np.int32),
        )


class CounterStimulus:
    def __init__(
        self,
        indices: npt.ArrayLike,
        rates_hz: float | npt.ArrayLike,
        *,
        dt_ms: float = 0.1,
        seed: int = 0,
    ) -> None:
        targets = np.asarray(indices, dtype=np.int32)
        if targets.ndim != 1:
            raise ValueError("stimulus indices must be one-dimensional")
        if targets.size and int(targets.min()) < 0:
            raise ValueError("stimulus indices cannot be negative")
        if seed < 0 or seed > (1 << 64) - 1:
            raise ValueError("seed must fit in uint64")
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), targets.shape)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("stimulus rates must be finite and non-negative")
        probabilities = rates * dt_ms / 1000.0
        if np.any(probabilities > 1):
            raise ValueError("rate * timestep cannot exceed one for N=1 input")

        self.indices = targets
        self.probabilities = np.asarray(probabilities)
        self.seed = int(seed)
        self._tick = 0

    def next_events(self) -> ExternalEvents:
        fired = np.fromiter(
            (
                _counter_uniform(self.seed, self._tick, lane) < probability
                for lane, probability in enumerate(self.probabilities)
            ),
            dtype=np.bool_,
            count=self.indices.size,
        )
        self._tick += 1
        if not np.any(fired):
            return ExternalEvents.empty()
        return ExternalEvents(
            indices=self.indices[fired],
            counts=np.ones(int(fired.sum()), dtype=np.int32),
        )


def _counter_uniform(seed: int, tick: int, lane: int) -> float:
    mask = (1 << 64) - 1
    value = seed ^ ((tick * 0x9E3779B97F4A7C15) & mask) ^ ((lane * 0xBF58476D1CE4E5B9) & mask)
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return (value >> 11) * (1.0 / (1 << 53))
