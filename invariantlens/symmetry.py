"""The audit: rotation average, balance fix, the invisible part of an error, the correction, the canonical pose and the
certificate."""

import numpy as np

from invariantlens.physics import ROTATIONS, N, balance, rotation, totals


def rotate(v):
    """All 16 rotations of v, whose last axis holds 2-vectors, stacked on a new first axis."""
    return np.einsum("gij,...j->g...i", ROTATIONS, v)


def unrotate(y):
    """The mean over g of R_g^T y[g]: outputs at the 16 rotations of an input, turned back and averaged."""
    return np.einsum("gji,g...j->...i", ROTATIONS, y) / len(ROTATIONS)


def average(f):
    """P_G f: f on the 16 rotations of (s, a), each output rotated back, then averaged."""
    return lambda s, a: unrotate(f(rotate(s), rotate(a)))


def fix(y, target=0.0):
    """Shift every disc by the same amounts so the totals of y equal target. With target 0 this is I - M."""
    return y - (totals(y) - target)[..., :, None, :] / N


def invisible(e):
    """(I - S)(I - M)e = P_G(I - M)e, the part of an error field the audit cannot see."""
    return average(lambda s, a: fix(e(s, a)))


def corrected(model):
    """The model with its balance fixed, then averaged over the 16 rotations. Its error is the invisible part."""
    return average(lambda s, a: fix(model(s, a), balance(s, a)))


def canonical(model):
    """The model asked once, in a standard pose: the scene turned about the origin until the agent faces the target
    along +x, and the answer turned back. It turns with the scene exactly, at one call instead of 16, but keeps the
    error of the one pose it asks about instead of averaging it away."""

    def posed(s, a):
        d = s[..., 0, 1, :] - s[..., 0, 0, :]
        R = rotation(-np.arctan2(d[..., 1], d[..., 0]))
        y = model(np.einsum("...ij,...kmj->...kmi", R, s), np.einsum("...ij,...j->...i", R, a))
        return np.einsum("...ji,...kmj->...kmi", R, y)

    return posed


def signals(model, s, a):
    """The symmetry and balance signals at each (s, a), from the model alone."""
    ys = model(rotate(s), rotate(a))
    y = ys[0]  # rotation 0 is the identity, so this is the model's own answer
    d, r = y - unrotate(ys), totals(y) - balance(s, a)
    return np.sqrt(np.square(d).sum((-3, -2, -1))), np.sqrt(np.square(r).sum((-2, -1)))


def surprise(model, s, a, s1):
    """Per transition (s, a, s'): the squared error |F(s, a) - s'|^2, and its squared visible part
    |F(s, a) - F_c(s, a)|^2, which needs no s'. Summed over transitions, their ratio is the share of the error that
    the correction can remove."""
    y = model(s, a)
    return np.square(y - s1).sum((-3, -2, -1)), np.square(y - corrected(model)(s, a)).sum((-3, -2, -1))


def certificate(choices):
    """16 - max_k n_k: a lower bound on wrong decisions per orbit, from the choices (..., 16) alone."""
    return choices.shape[-1] - (choices[..., :, None] == choices[..., None, :]).sum(-1).max(-1)
