"""Goals, candidate action sequences and the decision bank over C16 orbits."""

from dataclasses import dataclass
from itertools import count

import numpy as np

from invariantlens.physics import A_MAX, DT, ROTATIONS, SPEED, initial_state, polar, step

HORIZON = 10
CONE = np.pi / 3  # the goal lies within this angle of the agent-to-target heading, so pushing is the right idea
SPREAD = 0.6  # sd of the candidates' heading offset; the one knob the go/no-go may change
MARGIN = 0.05


def situation(seed, speed=SPEED, spread=SPREAD, shift=0.0, drift=0.0, crowd=0.0):
    """A start state, a goal ahead of the target and five candidate pushes. `shift` moves the whole scene along x,
    `drift` gives every disc and the goal the same extra velocity along x, and `crowd` starts a distractor that close
    to the target. Shift and drift change nothing the physics can see."""
    rng = np.random.default_rng(seed)
    s = initial_state(rng, speed, crowd)
    d = s[0, 1] - s[0, 0]
    heading = np.arctan2(d[1], d[0])
    goal = s[0, 1] + polar(rng.uniform(0.5, 1.5), heading + rng.uniform(-CONE, CONE))
    offset, amplitude = rng.normal(0, spread, (5, 1)), rng.uniform(0, 0.5, (5, 1))
    phase, force = rng.uniform(0, 2 * np.pi, (5, 1)), rng.uniform(0.5, 1.0, (5, 1)) * A_MAX
    angle = heading + offset + amplitude * np.sin(2 * np.pi * np.arange(HORIZON) / HORIZON + phase)
    s[0] += (shift, 0.0)
    s[1] += (drift, 0.0)
    return s, goal + (shift + drift * HORIZON * DT, 0.0), polar(force, angle)


def rollout(model, s, actions):
    """States visited under each candidate: s (..., 2, N, 2), actions (..., K, T, 2) -> (..., K, T + 1, 2, N, 2)."""
    states = [np.broadcast_to(s[..., None, :, :, :], actions.shape[:-2] + s.shape[-3:])]
    for t in range(actions.shape[-2]):
        states.append(model(states[-1], actions[..., t, :]))
    return np.stack(states, -4)


def reward(states, goal):
    """Minus the final target-to-goal distance, per candidate."""
    return -np.linalg.norm(states[..., -1, 0, 1, :] - goal[..., None, :], axis=-1)


@dataclass(frozen=True)
class Bank:
    """Accepted situations over their C16 orbits; axis 1 is the rotation, identity first."""

    seeds: np.ndarray  # (n,) accepted seeds; the gaps are the rejections
    state: np.ndarray  # (n, 16, 2, N, 2)
    goal: np.ndarray  # (n, 16, 2)
    actions: np.ndarray  # (n, 16, 5, HORIZON, 2)
    returns: np.ndarray  # (n, 16, 5) under the truth
    best: np.ndarray  # (n,) true best candidate of the unrotated situation


def make_bank(
    size=500, truth=step, speed=SPEED, spread=SPREAD, shift=0.0, drift=0.0, crowd=0.0, margin=MARGIN, start=0
):
    """The first `size` seeds from `start` whose best push beats the runner-up by the margin, each in all 16
    rotations."""
    seeds, kept = [], []
    for seed in count(start):
        s, goal, actions = situation(seed, speed, spread, shift, drift, crowd)
        second, first = np.sort(reward(rollout(truth, s, actions), goal))[-2:]
        if (first - second) / max(abs(first), 1e-3) > margin:
            seeds.append(seed)
            kept.append((s, goal, actions))
        if len(kept) == size:
            break
    s, goal, actions = (np.einsum("gij,n...j->ng...i", ROTATIONS, np.stack(v)) for v in zip(*kept))
    returns = reward(rollout(truth, s, actions), goal)
    return Bank(np.array(seeds), s, goal, actions, returns, returns[:, 0].argmax(-1))
