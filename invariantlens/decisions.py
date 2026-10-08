"""Situations, candidate pushes, and the bank of decisions in all 16 rotations."""

from dataclasses import dataclass
from itertools import count

import numpy as np

from invariantlens.physics import A_MAX, DT, ROTATIONS, SPEED, initial_state, polar, step

HORIZON = 10
CONE = np.pi / 3  # the goal is within 60 degrees of the agent-to-target direction
SPREAD = 0.6  # sd of each candidate's heading offset
MARGIN = 0.05


def situation(seed, speed=SPEED, shift=0.0, drift=0.0, crowd=0.0):
    """A start state, a goal ahead of the target, and five candidate pushes of HORIZON steps.

    shift moves the whole scene along x and drift gives every disc, and the goal, the same extra velocity along x.
    Neither changes anything the physics can see. crowd starts one distractor that close to the target."""
    rng = np.random.default_rng(seed)
    s = initial_state(rng, speed, crowd)
    d = s[0, 1] - s[0, 0]
    heading = np.arctan2(d[1], d[0])
    goal = s[0, 1] + polar(rng.uniform(0.5, 1.5), heading + rng.uniform(-CONE, CONE))

    offset = rng.normal(0, SPREAD, (5, 1))
    amplitude = rng.uniform(0, 0.5, (5, 1))
    phase = rng.uniform(0, 2 * np.pi, (5, 1))
    force = rng.uniform(0.5, 1.0, (5, 1)) * A_MAX
    angle = heading + offset + amplitude * np.sin(2 * np.pi * np.arange(HORIZON) / HORIZON + phase)

    s[0] += (shift, 0.0)
    s[1] += (drift, 0.0)
    goal = goal + (shift + drift * HORIZON * DT, 0.0)
    return s, goal, polar(force, angle)


def rollout(model, s, actions):
    """The states each candidate visits: (..., K, T + 1, 2, N, 2) from s (..., 2, N, 2) and actions (..., K, T, 2)."""
    states = [np.broadcast_to(s[..., None, :, :, :], actions.shape[:-2] + s.shape[-3:])]
    for t in range(actions.shape[-2]):
        states.append(model(states[-1], actions[..., t, :]))
    return np.stack(states, -4)


def reward(states, goal):
    """Minus the distance from the target to the goal at the end, for each candidate."""
    return -np.linalg.norm(states[..., -1, 0, 1, :] - goal[..., None, :], axis=-1)


@dataclass(frozen=True)
class Bank:
    """Situations in all 16 rotations. Axis 1 is the rotation, with the identity first."""

    seeds: np.ndarray  # (n,)
    state: np.ndarray  # (n, 16, 2, N, 2)
    goal: np.ndarray  # (n, 16, 2)
    actions: np.ndarray  # (n, 16, 5, HORIZON, 2)
    returns: np.ndarray  # (n, 16, 5), from the true simulator
    best: np.ndarray  # (n,), the best candidate


def make_bank(size=500, truth=step, speed=SPEED, shift=0.0, drift=0.0, crowd=0.0, margin=MARGIN, start=0):
    """The first size seeds from start whose best push beats the second best by margin, each in all 16 rotations."""
    seeds, kept = [], []
    for seed in count(start):
        s, goal, actions = situation(seed, speed, shift, drift, crowd)
        second, first = np.sort(reward(rollout(truth, s, actions), goal))[-2:]
        if (first - second) / max(abs(first), 1e-3) > margin:
            seeds.append(seed)
            kept.append((s, goal, actions))
        if len(kept) == size:
            break

    states, goals, actions = (np.stack(x) for x in zip(*kept))
    states, goals, actions = (np.einsum("gij,n...j->ng...i", ROTATIONS, x) for x in (states, goals, actions))
    returns = reward(rollout(truth, states, actions), goals)
    return Bank(np.array(seeds), states, goals, actions, returns, returns[:, 0].argmax(-1))
