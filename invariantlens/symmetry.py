"""The audit: rotation average, balance fix, the invisible part of an error, the correction and the certificate."""

import numpy as np

from invariantlens.physics import ROTATIONS, N, balance, totals


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
    """Shift every disc's position and velocity by the same amounts so the totals of y equal target; I - M on errors."""
    return y - (totals(y) - target)[..., :, None, :] / N


def invisible(e):
    """(I - S)(I - M)e = P_G(I - M)e, the part of an error field the audit cannot see."""
    return average(lambda s, a: fix(e(s, a)))


def corrected(model):
    """P_G(F^ - Me): the model with its balance fixed, then averaged over rotations. Its error is invisible(e)."""
    return average(lambda s, a: fix(model(s, a), balance(s, a)))


def signals(model, s, a):
    """Symmetry signal |S F^| and balance signal |totals(F^) - balance| at each (s, a); neither needs the truth.
    The first of the 16 rotations is the identity, so F^(s, a) itself comes with the average."""
    ys = model(rotate(s), rotate(a))
    y = ys[0]
    d, r = y - unrotate(ys), totals(y) - balance(s, a)
    return np.sqrt(np.square(d).sum((-3, -2, -1))), np.sqrt(np.square(r).sum((-2, -1)))


def certificate(choices):
    """16 - max_k n_k: a lower bound on wrong decisions per orbit, from the choices (..., 16) alone."""
    return choices.shape[-1] - (choices[..., :, None] == choices[..., None, :]).sum(-1).max(-1)
