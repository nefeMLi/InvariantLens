"""The checks from HYPOTHESES.md that need no trained model."""

from functools import partial

import numpy as np
import pytest
import torch

from experiments.common import visited
from experiments.controlled import check_injection
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.models import Net, world_model
from invariantlens.physics import A_MAX, N, balance, initial_state, step, totals, uniform_disc
from invariantlens.symmetry import (
    average,
    canonical,
    certificate,
    corrected,
    error_parts,
    fix,
    invisible,
    rotate,
    signals,
)

TRUTHS = {"base": step, "D x 1.5": partial(step, depth=1.5), "damping": partial(step, damping=0.5)}


def breaks_rotation(f):
    """The part of f that breaks rotation, S f."""
    return lambda s, a: f(s, a) - average(f)(s, a)


def breaks_balance(f):
    """The part of f that breaks the balance, M f."""
    return lambda s, a: f(s, a) - fix(f(s, a))


def visible(f):
    """The part of f the checks can see, V f."""
    return lambda s, a: f(s, a) - invisible(f)(s, a)


PROJECTIONS = {"S": breaks_rotation, "M": breaks_balance, "V": visible}


@pytest.fixture(scope="module")
def bank():
    return make_bank(20)


def orbit_sample(n=20, seed=0):
    """n random (s, a) in all 16 rotations, so that means over them don't depend on the angle."""
    rng = np.random.default_rng(seed)
    s = np.stack([initial_state(rng) for _ in range(n)])
    return rotate(s), rotate(uniform_disc(rng, A_MAX, n))


def random_field(seed):
    """A smooth field with no structure: a small random tanh network on (s, a)."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.3, (4 * N + 2, 32))
    w2 = rng.normal(0, 0.1, (32, 4 * N))

    def field(s, a):
        inputs = np.concatenate([s.reshape(*s.shape[:-3], -1), a], -1)
        return (np.tanh(inputs @ w1) @ w2).reshape(s.shape)

    return field


def inner(f, h, s, a):
    return (f(s, a) * h(s, a)).sum((-3, -2, -1)).mean()


@pytest.mark.parametrize("truth", TRUTHS.values(), ids=TRUTHS)
def test_balance_laws(truth):
    s, a = orbit_sample()
    scale = 1 + np.linalg.norm(totals(s), axis=-1).sum(-1)
    assert (np.abs(totals(truth(s, a)) - balance(s, a)).max((-2, -1)) <= 1e-12 * scale).all()


def test_truth_on_bank(bank):
    """The truth agrees across each orbit and always picks the best candidate, so its certificate is zero."""
    first = np.broadcast_to(bank.returns[:, :1], bank.returns.shape)
    np.testing.assert_allclose(bank.returns, first, rtol=1e-9)
    choices = bank.returns.argmax(-1)
    assert (choices == bank.best[:, None]).all()
    assert not certificate(choices).any()


@pytest.mark.parametrize("name", PROJECTIONS)
def test_projection(name):
    """Each projection is idempotent and self-adjoint."""
    P = PROJECTIONS[name]
    s, a = orbit_sample()
    f, h = random_field(1), random_field(2)
    np.testing.assert_allclose(P(P(f))(s, a), P(f)(s, a), rtol=0, atol=1e-12)
    np.testing.assert_allclose(inner(P(f), h, s, a), inner(f, P(h), s, a), rtol=0, atol=1e-12)


def test_projections_commute():
    s, a = orbit_sample()
    f = random_field(1)
    np.testing.assert_allclose(
        breaks_rotation(breaks_balance(f))(s, a), breaks_balance(breaks_rotation(f))(s, a), rtol=0, atol=1e-12
    )


def test_invisible_control():
    """The simulator with D = 1.1, used as a model: its error turns with the scene and keeps the balance."""
    s, a = orbit_sample()

    def e(s, a):
        return step(s, a, depth=1.1) - step(s, a)

    assert np.linalg.norm(visible(e)(s, a)) / np.linalg.norm(e(s, a)) < 1e-10


def test_correction(bank):
    """The corrected model's error is the invisible part, it passes both checks, and it never disagrees with itself
    across an orbit."""
    s, a = orbit_sample()
    r = random_field(3)

    def model(s, a):
        return step(s, a) + r(s, a)

    fixed = corrected(model)
    np.testing.assert_allclose(fixed(s, a) - step(s, a), invisible(r)(s, a), rtol=0, atol=1e-10)
    for signal in signals(fixed, s, a):
        assert signal.max() < 1e-12
    choices = reward(rollout(fixed, bank.state, bank.actions), bank.goal).argmax(-1)
    assert not certificate(choices).any()


def test_injection(bank):
    """Both controlled fields have visible norm alpha^2 and satisfy the defect identity."""
    check_injection(visited(bank)[:2], m=50)


def test_m3_turns_with_the_scene():
    """Untrained and in float64, M3 passes the rotation check but not the balance check, and M1 fails both."""
    torch.manual_seed(0)
    s, a = orbit_sample()
    sym3, bal3 = signals(world_model(Net(True, [1.0] * 5)), s, a)
    sym1, _ = signals(world_model(Net(False, [1.0] * 5)), s, a)
    assert sym3.max() < 1e-12 < bal3.min()
    assert sym1.min() > 1e-6


def test_fast_forward_matches():
    """The faster forward pass used in evaluation gives the same answer as the one used in training."""
    torch.manual_seed(0)
    s, a = (torch.from_numpy(x.reshape(-1, *x.shape[2:])) for x in orbit_sample())
    for equivariant in (False, True):
        net = Net(equivariant, [1.0] * 5).double()
        torch.testing.assert_close(net.eval()(s, a), net.train()(s, a), rtol=0, atol=1e-12)


def test_error_parts():
    """All of a constant drift's error is visible, and none of a change in the physics."""
    s, a = orbit_sample()

    def drifting(s, a):
        return step(s, a) + 0.01 * np.array([1.0, 0.0])

    error, seen = error_parts(drifting, s, a, step(s, a))
    np.testing.assert_allclose(seen, error, rtol=1e-8)
    error, seen = error_parts(step, s, a, step(s, a, depth=1.1))
    assert seen.max() < 1e-20 < error.min()


def test_frame_shifts():
    """Moving the scene, or giving every disc and the goal the same drift, changes nothing the physics can see."""
    plain = make_bank(3, margin=1e-6)
    for shift in ({"shift": 10.0}, {"drift": 2.0}):
        np.testing.assert_allclose(make_bank(3, margin=1e-6, **shift).returns, plain.returns, rtol=1e-9, atol=0)


def test_canonical():
    """Asked in one standard pose, even M1 turns with the scene."""
    torch.manual_seed(0)
    s, a = orbit_sample()
    symmetry, _ = signals(canonical(world_model(Net(False, [1.0] * 5))), s, a)
    assert symmetry.max() < 1e-10
