"""The batched GPU prior: function-based config maps, any curve length, shapes and units."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lcfaker.torch_prior import BplTasks  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tasks():
    g = torch.Generator(device=DEV)
    g.manual_seed(0)
    P = BplTasks(16, 10, DEV, generator=g)
    P.resample(torch.arange(16), torch.randint(1, 11, (16,)))
    return P, g


def test_shapes_and_finite(tasks):
    P, g = tasks
    x = torch.rand(16, 32, 10, device=DEV, generator=g)
    T = torch.randint(4, 301, (16,))
    y = P.curves(x, T, 300)
    assert y.shape == (16, 32, 300) and torch.isfinite(y).all()
    u = P.curves(x, T, 300, scale="unit")
    assert u.min() >= 0 and u.max() <= 2


def test_same_config_same_curve(tasks):
    P, g = tasks
    x = torch.rand(16, 20, 10, device=DEV, generator=g)
    a = P.curves(x, torch.full((16,), 50), 50, noise=False)
    b = P.curves(x[:, [5, 3]], torch.full((16,), 50), 50, noise=False)     # other configs, other order
    assert torch.allclose(a[:, [5, 3]], b)


def test_any_length_and_past_T(tasks):
    P, g = tasks
    x = torch.rand(16, 8, 10, device=DEV, generator=g)
    T = torch.full((16,), 40)
    a = P.curves(x, T, 60, noise=False)
    b = P.curves(x, T * 10, 400, noise=False)          # same budget fractions at points 10k
    assert torch.allclose(a[..., 39], a[..., -1])       # past T: the value at T
    c = P.curves(x, torch.arange(30, 46), 60)            # with noise, per-row T
    for r in range(16):
        t = 30 + r
        assert torch.equal(c[r, :, t:], c[r, :, t - 1:t].expand(-1, 60 - t))   # past T: the observation at T
        assert not torch.equal(c[r, :, t - 2], c[r, :, t - 1])                 # before T: fresh noise
    assert torch.allclose(a[..., 19], b[..., 199], rtol=1e-4, atol=1e-4)
    assert torch.allclose(a[..., 39], b[..., 399], rtol=1e-4, atol=1e-4)


def test_resample_changes_only_given_rows(tasks):
    P, g = tasks
    x = torch.rand(16, 8, 10, device=DEV, generator=g)
    T = torch.full((16,), 50)
    a = P.curves(x, T, 50, noise=False)
    P.resample(torch.tensor([0, 1]), torch.tensor([3, 10]))
    b = P.curves(x, T, 50, noise=False)
    assert torch.allclose(a[2:], b[2:]) and not torch.allclose(a[:2], b[:2])
    assert len(P.dims(0)) == 3 and P.dims(0)[0] == "log_learning_rate"
