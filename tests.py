"""The checks from HYPOTHESES.md that need no trained model: truth, projections, invisible control, injection,
correction, M3."""

from functools import partial

import numpy as np
import pytest
import torch

from experiments.common import visited
from experiments.controlled import check_injection
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.models import Net, world_model
from invariantlens.physics import A_MAX, N, balance, initial_state, step, totals, uniform_disc
from invariantlens.symmetry import average, certificate, corrected, fix, invisible, rotate, signals

TRUTHS = {"base": step, "D x 1.5": partial(step, depth=1.5), "damping": partial(step, damping=0.5)}
PROJECTIONS = {
    "S": lambda f: lambda s, a: f(s, a) - average(f)(s, a),
    "M": lambda f: lambda s, a: f(s, a) - fix(f(s, a)),
    "V": lambda f: lambda s, a: f(s, a) - invisible(f)(s, a),
}


@pytest.fixture(scope="module")
def bank():
    return make_bank(20)


def orbit_sample(n=20, seed=0):
    """n random (s, a) with all 16 rotations of each, so means over them are under a C16-invariant measure."""
    rng = np.random.default_rng(seed)
    s = np.stack([initial_state(rng) for _ in range(n)])
    return rotate(s), rotate(uniform_disc(rng, A_MAX, n))


def random_field(seed):
    """A smooth error field with no structure: a small random tanh network on (s, a)."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(0, 0.3, (4 * N + 2, 32)), rng.normal(0, 0.1, (32, 4 * N))
    return lambda s, a: (np.tanh(np.concatenate([s.reshape(*s.shape[:-3], -1), a], -1) @ w1) @ w2).reshape(s.shape)


def inner(f, h, s, a):
    return (f(s, a) * h(s, a)).sum((-3, -2, -1)).mean()


@pytest.mark.parametrize("truth", TRUTHS.values(), ids=TRUTHS)
def test_balance_laws(truth):
    s, a = orbit_sample()
    scale = 1 + np.linalg.norm(totals(s), axis=-1).sum(-1)
    assert (np.abs(totals(truth(s, a)) - balance(s, a)).max((-2, -1)) <= 1e-12 * scale).all()


def test_truth_on_bank(bank):
    """Returns agree across each orbit and the truth always picks the best candidate, so its certificate is zero."""
    np.testing.assert_allclose(bank.returns, np.broadcast_to(bank.returns[:, :1], bank.returns.shape), rtol=1e-9)
    choices = bank.returns.argmax(-1)
    assert (choices == bank.best[:, None]).all() and not certificate(choices).any()


@pytest.mark.parametrize("name", PROJECTIONS)
def test_projection(name):
    """Idempotent and self-adjoint under a C16-invariant measure."""
    P, (s, a), f, h = PROJECTIONS[name], orbit_sample(), random_field(1), random_field(2)
    np.testing.assert_allclose(P(P(f))(s, a), P(f)(s, a), rtol=0, atol=1e-12)
    np.testing.assert_allclose(inner(P(f), h, s, a), inner(f, P(h), s, a), rtol=0, atol=1e-12)


def test_projections_commute():
    S, M, (s, a), f = PROJECTIONS["S"], PROJECTIONS["M"], orbit_sample(), random_field(1)
    np.testing.assert_allclose(S(M(f))(s, a), M(S(f))(s, a), rtol=0, atol=1e-12)


def test_invisible_control():
    """The simulator with D = 1.1 as a model: its error rotates correctly and keeps the balance, so none is visible."""
    s, a = orbit_sample()

    def e(s, a):
        return step(s, a, depth=1.1) - step(s, a)

    assert np.linalg.norm(PROJECTIONS["V"](e)(s, a)) / np.linalg.norm(e(s, a)) < 1e-10


def test_correction(bank):
    """The corrected model's error is the invisible part, both its signals vanish and it never disagrees on an orbit."""
    s, a, r = *orbit_sample(), random_field(3)

    def model(s, a):
        return step(s, a) + r(s, a)

    fixed = corrected(model)
    np.testing.assert_allclose(fixed(s, a) - step(s, a), invisible(r)(s, a), rtol=0, atol=1e-10)
    assert all(signal.max() < 1e-12 for signal in signals(fixed, s, a))
    assert not certificate(reward(rollout(fixed, bank.state, bank.actions), bank.goal).argmax(-1)).any()


def test_injection(bank):
    """Both controlled fields have visible norm alpha^2 and obey the defect identity; the balance one rotates right."""
    check_injection(visited(bank)[:2], m=50)


def test_m3_rotates_correctly():
    """Untrained, float64: M3's symmetry signal vanishes but its balance signal doesn't; M1's symmetry signal is on."""
    torch.manual_seed(0)
    s, a = orbit_sample()
    sym3, bal3 = signals(world_model(Net(True, [1.0] * 5)), s, a)
    sym1, _ = signals(world_model(Net(False, [1.0] * 5)), s, a)
    assert sym3.max() < 1e-12 < bal3.min() and sym1.min() > 1e-6


def test_fast_forward_matches():
    """The evaluation forward pass (first layer per disc) equals the training one (per pair) to float64 rounding."""
    torch.manual_seed(0)
    s, a = (torch.from_numpy(x.reshape(-1, *x.shape[2:])) for x in orbit_sample())
    for equivariant in (False, True):
        net = Net(equivariant, [1.0] * 5).double()
        torch.testing.assert_close(net.eval()(s, a), net.train()(s, a), rtol=0, atol=1e-12)
