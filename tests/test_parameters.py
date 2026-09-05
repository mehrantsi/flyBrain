from __future__ import annotations

import pytest

from flybrain.parameters import ModelParameters


def test_published_parameters_have_integer_timing() -> None:
    parameters = ModelParameters()

    assert parameters.delay_steps == 18
    assert parameters.refractory_steps == 22
    assert parameters.poisson_weight_mv == pytest.approx(68.75)
    assert 0 < parameters.membrane_synapse_coupling < 1


def test_non_integral_delay_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        ModelParameters(delay_ms=1.85)
