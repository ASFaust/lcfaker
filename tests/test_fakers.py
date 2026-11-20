import numpy as np
import torch

from curvefaker import fakers
from curvefaker.measures import curves as _curves  # noqa: F401  (import check)


def _draw(cls, seed=0, E=8, d=4, M=64, T=30):
    g = torch.Generator()
    g.manual_seed(seed)
    pr = cls(E, 10, "cpu", generator=g)
    pr.resample(torch.arange(E), torch.full((E,), d))
    x = torch.rand(E, M, 10, generator=g)
    x[:, :, d:] = 0
    return pr, x, pr.curves(x, torch.full((E,), T), T)


def test_no_components_is_ifbo():
    _, _, a = _draw(fakers.PriorTasks)
    _, _, b = _draw(fakers.make_faker())
    assert torch.equal(a, b)


def test_dead_region_flat_and_sized():
    F = fakers.make_faker(dead=dict(p_task=1.0, frac=(0.3, 0.3)))
    pr, x, y = _draw(F, M=2000)
    s = (x * pr.dead_w[:, None]).sum(-1)
    dead = s > pr.dead_c[:, None]
    frac = dead.float().mean(1).numpy()
    assert np.allclose(frac, 0.3, atol=0.06), frac
    rng = (y.max(-1).values - y.min(-1).values)
    assert (rng[dead] < 0.05).all()
    assert torch.allclose(y[dead].mean(-1), (1 - pr.y0)[:, None].expand_as(dead)[dead], atol=0.01)


def test_noise_rows_cover_both_priors():
    F = fakers.make_faker(noise={})
    pr, _, y = _draw(F, E=256)
    assert pr.row_noise.min() >= 0.25 - 1e-6 and pr.row_noise.max() <= 1.0 + 1e-6
    assert (pr.row_smooth == 0).float().mean() > 0.3 and (pr.row_smooth > 0).any()
    assert torch.isfinite(y).all() and (y >= 0).all() and (y <= 1).all()


def test_noisefix_faker_matches_scalar_prior():
    class P(fakers.PriorTasks):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.noise_scale, self.noise_smooth = 0.25, 1.0
    _, _, a = _draw(P)
    _, _, b = _draw(fakers.get("noisefix"))
    assert torch.allclose(a, b, atol=1e-6)


def test_dead_faker_keeps_ifbo_curves_outside_region():
    _, _, a = _draw(fakers.PriorTasks, M=500)
    pr, x, b = _draw(fakers.get("dead"), M=500)
    dead = pr.dead_on[:, None] & ((x * pr.dead_w[:, None]).sum(-1) > pr.dead_c[:, None])
    assert dead.any() and (~dead).any()
    assert torch.equal(a[~dead], b[~dead])
