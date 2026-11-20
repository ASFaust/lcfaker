import numpy as np
import pytest

from curvefaker import data
from curvefaker.heldout import HeldOutError


def test_loader_refuses_heldout():
    with pytest.raises(HeldOutError):
        data.load("taskset/FixedTextRNNClassification_imdb_patch32_GRU128_bs128/adam4p")
    with pytest.raises(HeldOutError):
        data.load_synetune("fcnet/lcbench_airlines")


def test_every_family_assigned():
    for i in data.ids():
        assert data.split_of(data.family_of(i)) in ("fit", "select")


def test_splits_disjoint():
    fit, sel = set(data.ids("fit")), set(data.ids("select"))
    assert fit and sel and not fit & sel
    assert {data.family_of(i) for i in fit}.isdisjoint({data.family_of(i) for i in sel})


def test_jahs_loads():
    ids = data.ids(sources=["jahs"])
    if not ids:
        pytest.skip("JAHS not prepared (scripts/convert_jahs.py, scripts/prepare_jahs.py)")
    assert len(ids) == 81 and data.split_of("jahs") == "fit"
    t = data.load(ids[0])
    assert t.y.ndim == 3 and t.T == 200 and t.y.shape[0] == len(t.hps)
    assert np.isfinite(t.y[:, 0]).all()            # first replica is always a full curve
    assert t.X().shape[0] == t.n and 0 <= t.X().min() and t.X().max() <= 1
