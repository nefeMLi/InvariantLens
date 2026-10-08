"""The true simulator: five discs with Morse forces between them, and a push on the agent."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

N = 5  # disc 0 is the agent, disc 1 the target
DEPTH, WIDTH, R0 = 1.0, 2.0, 1.0  # Morse D, a and r0
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


# C16, the 16 rotations by multiples of 22.5 degrees. A 2-vector v turns as v @ R.T
ROTATIONS = rotation(2 * np.pi * np.arange(16) / 16)


def initial_state(rng, speed=SPEED, crowd=0.0):
    """Positions and velocities, shape (2, N, 2). The agent starts 1 to 1.5 from the target and no two discs closer
    than 0.8, except that with crowd > 0 the first distractor starts that close to the target."""
    while True:
        x = uniform_disc(rng, 2.0, N - 1)
        agent = x[0] + polar(rng.uniform(1.0, 1.5), rng.uniform(0, 2 * np.pi))
        x = np.vstack([agent, x])
        if crowd:
            x[2] = x[1] + polar(crowd, rng.uniform(0, 2 * np.pi))

        distance = squareform(pdist(x))
        np.fill_diagonal(distance, np.inf)
        if crowd:
            distance[1, 2] = distance[2, 1] = np.inf
        if distance.min() >= 0.8:
            return np.stack([x, rng.normal(0, speed, (N, 2))])


def forces(x, v, depth, damping):
    """Morse force on each disc, plus damping along the line between each pair."""
    d = x[..., :, None, :] - x[..., None, :, :]
    r = np.linalg.norm(d, axis=-1) + np.where(np.eye(N, dtype=bool), np.inf, 0.0)
    u = d / r[..., None]
    e = np.exp(-WIDTH * (r - R0))
    f = 2 * depth * WIDTH * e * (e - 1)
    if damping:
        closing = np.einsum("...ijk,...ijk->...ij", v[..., :, None, :] - v[..., None, :, :], u)
        f = f - damping * closing
    return (f[..., None] * u).sum(-2)


def step(s, a, depth=DEPTH, damping=0.0):
    """One decision step: SUBSTEPS of velocity Verlet with the push a held on the agent."""
    x, v = s[..., 0, :, :], s[..., 1, :, :]
    push = np.where(np.arange(N)[:, None] == 0, a[..., None, :], 0.0)
    f = forces(x, v, depth, damping)
    for _ in range(SUBSTEPS):
        half = v + H / 2 * (f + push)
        x = x + H * half
        f = forces(x, half, depth, damping)  # damping sees the half-step velocity
        v = half + H / 2 * (f + push)
    return np.stack([x, v], -3)


def totals(s):
    """Sum of positions and of velocities, shape (..., 2, 2). All masses are 1."""
    return s.sum(-2)


def balance(s, a):
    """The totals one decision step later. Pair forces cancel, so only the push changes them."""
    x, p = np.unstack(totals(s), axis=-2)
    return np.stack([x + DT * p + DT**2 / 2 * a, p + DT * a], -2)
