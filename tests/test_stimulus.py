from __future__ import annotations

import numpy as np
import pytest

from flybrain.stimulus import CounterStimulus, ExternalEvents, PoissonStimulus


def test_external_events_validate_shapes_and_counts() -> None:
    events = ExternalEvents(np.array([2, 5]), np.array([1, 3]))

    assert events.indices.dtype == np.int32
    assert events.counts.tolist() == [1, 3]

    with pytest.raises(ValueError, match="positive"):
        ExternalEvents(np.array([2]), np.array([0]))


def test_poisson_stimulus_is_seeded() -> None:
    first = PoissonStimulus([1, 4, 7], 1000.0, dt_ms=0.1, seed=42)
    second = PoissonStimulus([1, 4, 7], 1000.0, dt_ms=0.1, seed=42)

    for _ in range(20):
        expected = first.next_events()
        actual = second.next_events()
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.counts, expected.counts)


def test_impossible_n1_poisson_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed one"):
        PoissonStimulus([0], 10_001.0, dt_ms=0.1)
    for stimulus in (PoissonStimulus, CounterStimulus):
        with pytest.raises(ValueError, match="finite"):
            stimulus([0], np.nan)


def test_counter_stimulus_is_repeatable_and_distinct_by_tick() -> None:
    first = CounterStimulus([2, 4], 150.0, seed=7)
    second = CounterStimulus([2, 4], 150.0, seed=7)

    first_events = [first.next_events() for _ in range(100)]
    second_events = [second.next_events() for _ in range(100)]

    assert [event.indices.tolist() for event in first_events] == [
        event.indices.tolist() for event in second_events
    ]
    assert sum(event.indices.size for event in first_events) > 0
