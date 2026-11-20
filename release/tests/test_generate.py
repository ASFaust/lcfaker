import numpy as np

from lcfaker import generate


def test_shapes_and_ranges():
    tasks = generate(3, seed=0)
    assert len(tasks) == 3
    for t in tasks:
        n, T = t.y.shape
        assert t.X.shape[0] == n and 1 <= t.X.shape[1] <= 10
        assert np.isfinite(t.y).all()
        assert t.y.min() >= -0.5 and t.y.max() <= 2.0
        assert (t.X >= 0).all() and (t.X <= 1).all()


def test_deterministic():
    a, b = generate(2, seed=7), generate(2, seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.y, y.y)


def test_source_weights():
    tasks = generate(4, seed=1, source_weights={"jahs": 1.0})
    assert all(t.source == "jahs" for t in tasks)
