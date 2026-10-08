"""Rotations, the balance fix, the corrected model and the label-free bound on wrong decisions.

A model maps (s, a) to the next state. S and M below are the two projections from the README: S keeps the part of a
model's output that breaks rotation, M the part that breaks the balance of position and momentum."""

import numpy as np

from invariantlens.physics import ROTATIONS, N, balance, rotation, totals


def rotate(v):
    """v in all 16 rotations, stacked on a new first axis. The last axis of v holds 2-vectors."""
    return np.einsum("gij,...j->g...i", ROTATIONS, v)


def unrotate(y):
    """Turn y[g] back by rotation g and average over g."""
    return np.einsum("gji,g...j->...i", ROTATIONS, y) / len(ROTATIONS)


def average(f):
    """f asked in all 16 rotations, with the answers turned back and averaged (I - S)."""

    def averaged(s, a):
        return unrotate(f(rotate(s), rotate(a)))

    return averaged


def fix(y, target=0.0):
    """Shift every disc by the same amount so that the totals of y equal target. With target 0 this is I - M."""
    return y - (totals(y) - target)[..., :, None, :] / N


def invisible(e):
    """The part of an error the checks can't see, (I - S)(I - M)e."""
    return average(lambda s, a: fix(e(s, a)))


def corrected(model):
    """The model with its balance fixed and averaged over the 16 rotations. Its error is the invisible part."""
    return average(lambda s, a: fix(model(s, a), balance(s, a)))


def canonical(model):
    """The model asked once, with the scene turned about the origin until the agent faces the target along +x.

    It turns with the scene exactly, like the corrected model, but keeps the error of the one pose it asks about
    instead of averaging it away."""

    def posed(s, a):
        d = s[..., 0, 1, :] - s[..., 0, 0, :]
        R = rotation(-np.arctan2(d[..., 1], d[..., 0]))
        y = model(np.einsum("...ij,...kmj->...kmi", R, s), np.einsum("...ij,...j->...i", R, a))
        return np.einsum("...ji,...kmj->...kmi", R, y)

    return posed


def signals(model, s, a):
    """How much the model breaks rotation, and how much it breaks the balance, at each (s, a)."""
    ys = model(rotate(s), rotate(a))
    y = ys[0]  # the first rotation is the identity
    symmetry = np.sqrt(np.square(y - unrotate(ys)).sum((-3, -2, -1)))
    balance_error = np.sqrt(np.square(totals(y) - balance(s, a)).sum((-2, -1)))
    return symmetry, balance_error


def error_parts(model, s, a, s1):
    """The squared error of each prediction, and the squared part of it the correction removes."""
    y = model(s, a)
    error = np.square(y - s1).sum((-3, -2, -1))
    visible = np.square(y - corrected(model)(s, a)).sum((-3, -2, -1))
    return error, visible


def certificate(choices):
    """At least this many of the 16 choices in each orbit are wrong: 16 minus the count of the most common choice."""
    same = choices[..., :, None] == choices[..., None, :]
    return choices.shape[-1] - same.sum(-1).max(-1)
