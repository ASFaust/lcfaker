"""The standalone generator reproduces the development faker (curvefaker.faker_bpl, version 16) exactly.
Runs only in the development repository (needs curvefaker and its fitted records)."""
import numpy as np
import pytest

curvefaker = pytest.importorskip("curvefaker")


def test_same_tasks_as_development_faker():
    from curvefaker import faker_bpl
    from lcfaker import generate
    dev = faker_bpl.faker(6, np.random.default_rng(42), version=16)
    rel = generate(6, seed=42)
    for (y, X), t in zip(dev, rel):
        np.testing.assert_array_equal(X, t.X)
        np.testing.assert_allclose(y[:, 0], t.y, rtol=1e-6, atol=1e-6)
