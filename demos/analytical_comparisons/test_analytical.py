import pytest
import analytical
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

enabled_cases = [
    "smooth/cylindrical/free_slip",
]

params = {
    f"{l1}/{l2}/{l3}": v3 for l1, v1 in analytical.cases.items()
    for l2, v2 in v1.items()
    for l3, v3 in v2.items()
    if f"{l1}/{l2}/{l3}" in enabled_cases
}

configs = []
for name, conf in params.items():
    # these two keys don't form a part of the parameter matrix
    conf = conf.copy()
    conf.pop("cores")
    conf.pop("levels")

    for combination in itertools.product(*conf.values()):
        configs.append((name, dict(zip(conf.keys(), combination))))


def idfn(val):
    if isinstance(val, dict):
        return "-".join([f"{k}{v}" for k, v in val.items()])


@pytest.mark.parametrize("name,config", configs, ids=idfn)
def test_analytical(name, config):
    levels = analytical.get_case(analytical.cases, name)["levels"]

    b = Path(__file__).parent.resolve()

    dats = [
        pd.read_csv(b / "errors-{}-levels{}-{}.dat".format(
            name.replace("/", "_"),
            level,
            idfn(config)
        ), sep=" ", header=None)
        for level in levels
    ]

    dat = pd.concat(dats)
    dat.columns = ["l2error_u", "l2error_p", "l2anal_u", "l2anal_p"]
    dat.insert(0, "level", levels)
    dat = dat.set_index("level")

    # use the highest resolution analytical solutions as the reference
    ref = dat.iloc[-1][["l2anal_u", "l2anal_p"]].rename(index=lambda s: s.replace("anal", "error"))
    errs = dat[["l2error_u", "l2error_p"]] / ref
    errs = errs.reset_index(drop=True)  # drop resolution label

    convergence = np.log2(errs.shift() / errs).drop(index=0)
    expected = pd.Series([4.0, 2.0], index=["l2error_u", "l2error_p"])

    assert np.allclose(convergence, expected, rtol=1e-2)
