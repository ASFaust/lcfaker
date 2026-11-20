import numpy as np

from curvefaker import curves


def test_normalise_keeps_padding_and_caps_divergence():
    y = np.full((3, 2, 5), np.nan)
    y[:, 0] = [[10, 8, 6, 4, 2], [10, 10, 10, 10, 10], [10, np.inf, np.nan, 50, 50]]
    z, info = curves.normalise(y)
    assert np.isnan(z[:, 1]).all()                       # padded replica stays NaN
    assert z[0, 0, -1] == 0 and z[0, 0, 0] == 1           # best -> 0, bad start -> 1
    assert (z[2, 0, 1:] == curves.CAP).all()              # non-finite / huge -> cap
    m = curves.mean_curve(z)
    assert list(curves.classify(m)) == [0, 1, 3]          # improving, dead, diverged
