import operator
from pathlib import Path

import numpy as np
import pytest

base = (Path(__file__).parent / "benchmarks").resolve()

diagnostics = {
    "crameri_2012": [
        (
            lambda data: np.abs(
                np.subtract(data["max_topography"], data["max_topography_analytical"])
            ).max(),
            operator.le,
            0.05,
        )
    ],
    "gerya_2003": [
        (
            lambda data: np.abs(1 - np.asarray(data["block_area"])).max(),
            operator.le,
            0.035,
        )
    ],
    "robey_2019": [
        (lambda data: abs(max(data["rms_velocity"]) - 284.82), operator.le, 0.01),
        (lambda data: abs(max(data["entrainment"]) - 0.9302), operator.le, 1e-4),
    ],
    "schmalholz_2011": [
        (
            lambda data: abs(
                np.asarray(data["normalised_time"])[
                    np.asarray(data["slab_necking"]) <= 0.2
                ].min()
                - 0.845
            ),
            operator.le,
            5e-3,
        )
    ],
    "schmeling_2008": [
        (
            lambda data: abs(
                np.asarray(data["output_time"])[
                    np.asarray(data["slab_tip_depth"]) >= 600
                ].min()
                - 44.6
            ),
            operator.le,
            0.1,
        )
    ],
    "tosi_2015": [
        (lambda data: abs(data["avg_temperature"][-1] - 0.5275), operator.le, 1e-4),
        (lambda data: abs(data["nusselt_top"][-1] - 6.64), operator.le, 0.03),
        (lambda data: abs(data["nusselt_bottom"][-1] - 6.65), operator.le, 0.04),
        (lambda data: abs(data["rms_velocity"][-1] - 79.1), operator.le, 0.1),
        (lambda data: abs(data["min_visc"][-1] - 1.92e-4), operator.le, 1e-6),
        (lambda data: abs(data["max_visc"][-1] - 1.92), operator.le, 0.06),
    ],
    "trim_2023": [
        (lambda data: abs(data["rms_velocity"][-1] - 157), operator.le, 3.5),
    ],
    "van_keken_1997_isothermal": [
        (lambda data: abs(max(data["rms_velocity"]) - 3.1e-3), operator.le, 1e-4),
        (lambda data: abs(max(data["entrainment"]) - 0.802), operator.le, 3e-3),
    ],
    "van_keken_1997_thermochemical": [
        (
            lambda data: abs(
                np.asarray(data["rms_velocity"])[
                    (np.asarray(data["output_time"]) >= 0.0214)
                    & (np.asarray(data["output_time"]) <= 0.0218)
                ].max()
                - 487
            ),
            operator.le,
            30,
        )
    ],
    "woidt_1978": [
        (lambda data: abs(max(data["rms_velocity"]) - 1.82e-11), operator.le, 1e-13),
        (lambda data: abs(max(data["entrainment"]) - 0.249), operator.le, 1e-3),
    ],
}


@pytest.mark.parametrize("bench_name,bench_diagnostics", diagnostics.items())
def test_multi_material(bench_name, bench_diagnostics):
    diag_file = np.load(base / bench_name / "output_0_reference.npz", allow_pickle=True)
    for diag_function, diag_operator, diag_threshold in bench_diagnostics:
        diag_data = diag_file["diag_fields"][()]
        assert diag_operator(diag_function(diag_data), diag_threshold)
