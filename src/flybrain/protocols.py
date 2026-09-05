from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from flybrain.connectome import PackedConnectome

RIGHT_SUGAR_GRN_IDS = (
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
)


def indices_for_flywire_ids(
    connectome: PackedConnectome,
    flywire_ids: Sequence[int],
    *,
    allow_missing: bool = False,
) -> npt.NDArray[np.int32]:
    lookup = {int(flywire_id): index for index, flywire_id in enumerate(connectome.neuron_ids)}
    missing = [int(flywire_id) for flywire_id in flywire_ids if int(flywire_id) not in lookup]
    if missing and not allow_missing:
        raise KeyError(f"FlyWire IDs are absent from this pack: {missing[:5]}")
    return np.asarray(
        [lookup[int(flywire_id)] for flywire_id in flywire_ids if int(flywire_id) in lookup],
        dtype=np.int32,
    )


__all__ = ["RIGHT_SUGAR_GRN_IDS", "indices_for_flywire_ids"]
