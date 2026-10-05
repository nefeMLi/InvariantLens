"""The true simulator: five unit-mass discs, Morse pair forces and a held push on the agent."""

import numpy as np
from scipy.spatial.distance import pdist

N = 5  # disc 0 is the agent, disc 1 the target
DEPTH, WIDTH, R0 = 1.0, 2.0, 1.0  # Morse D, a, r0
A_MAX = 2.0
H, SUBSTEPS = 0.01, 10
DT = H * SUBSTEPS
SPEED = 0.3


def polar(radius, angle):
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], -1)


def uniform_disc(rng, radius, size):
    return polar(radius * np.sqrt(rng.uniform(size=size)), rng.uniform(0, 2 * np.pi, size))


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


ROTATIONS = rotation(2 * np.pi * np.arange(16) / 16)  # C16, acting on every trailing 2-vector as v @ R.T


def initial_state(rng, speed=SPEED):
    """(2, N, 2) positions and velocities: target and distractors uniform within radius 2, agent 1 to 1.5 from the
    target, every pair at least 0.8 apart."""
    while True:
        x = uniform_disc(rng, 2.0, N - 1)
        x = np.vstack([x[0] + polar(rng.uniform(1.0, 1.5), rng.uniform(0, 2 * np.pi)), x])
        if pdist(x).min() >= 0.8:
            return np.stack([x, rng.normal(0, speed, (N, 2))])


def forces(x, v, depth, damping):
    """Net Morse force on each disc plus central damping on relative velocity; equal and opposite within each pair."""
    d = x[..., :, None, :] - x[..., None, :, :]
    r = np.linalg.norm(d, axis=-1) + np.where(np.eye(N, dtype=bool), np.inf, 0.0)
    u = d / r[..., None]
    e = np.exp(-WIDTH * (r - R0))
    f = 2 * depth * WIDTH * e * (e - 1)
    if damping:
        f = f - damping * np.einsum("...ijk,...ijk->...ij", v[..., :, None, :] - v[..., None, :, :], u)
    return (f[..., None] * u).sum(-2)


def step(s, a, depth=DEPTH, damping=0.0):
    """State after one decision step of velocity Verlet, force a on the agent; damping sees the half-step velocity."""
    x, v = s[..., 0, :, :], s[..., 1, :, :]
    push = np.where(np.arange(N)[:, None] == 0, a[..., None, :], 0.0)
    f = forces(x, v, depth, damping)
    for _ in range(SUBSTEPS):
        half = v + H / 2 * (f + push)
        x = x + H * half
        f = forces(x, half, depth, damping)
        v = half + H / 2 * (f + push)
    return np.stack([x, v], -3)


def totals(s):
    """Total position X and momentum P as (..., 2, 2); all masses are 1."""
    return s.sum(-2)


def balance(s, a):
    """Totals one decision step later, as the balance laws fix them: X + DT P + DT^2/2 a and P + DT a."""
    x, p = np.unstack(totals(s), axis=-2)
    return np.stack([x + DT * p + DT**2 / 2 * a, p + DT * a], -2)
