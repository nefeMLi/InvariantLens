"""H3 to H7: the trained models, their label-free signals, the corrected models, the ensemble and the orbit vote."""

import argparse
import json
import re
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.common import (
    ROOT,
    auroc,
    bootstrap,
    cached,
    check_truth,
    choose,
    chunks,
    fit,
    load,
    mean_square,
    save_figure,
    saved_or_run,
    score,
    spawned,
    visited,
)
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.models import KINDS
from invariantlens.physics import ROTATIONS, SPEED, balance, rotation, step
from invariantlens.symmetry import certificate, corrected, fix, rotate, signals, unrotate

SEEDS = range(5)
NOISE = 0.02  # sd of the noise on observed positions and velocities
SHIFTS = {
    "none": {},
    "D x 1.5": {"truth": partial(step, depth=1.5)},
    "damping": {"truth": partial(step, damping=0.5)},
    "speed x 2": {"speed": 2 * SPEED},
    "input noise": {},  # the unshifted bank, seen through noisy readings
}
CONTROLS = {"far 2.5": {"shift": 2.5}, "far 5": {"shift": 5.0}}  # H7: the same physics, away from the origin
ALL = SHIFTS | CONTROLS  # H3 uses SHIFTS only
DONE = ROOT / "results" / "learned"  # one file per finished evaluation; delete it after changing the code
COLUMNS = ("wrong", "consistent", "visible", "auroc_symmetry", "auroc_balance", "auroc_ensemble", "coverage")


def kept(fn, name, *args):
    """fn(*args), which returns (rows, returns), saved under results/learned/ as soon as it finishes. Spaces and
    brackets in the name become dashes in the file name."""

    def arrays():
        rows, returns = fn(*args)
        return {"rows": json.dumps(rows), "returns": returns}

    job = cached(DONE / f"{re.sub(r'[^\w.]+', '-', name).strip('-')}.npz", arrays)
    return json.loads(job["rows"].item()), job["returns"]


def read_job(name):
    """The rows and returns of an evaluation that has already run."""
    return kept(None, name)


def noise(shift, shape, seed=0):
    """Sensor noise for the input-noise shift, the same draw for every model. Zero for every other shift."""
    return NOISE * np.random.default_rng(seed).normal(size=shape) if shift == "input noise" else np.zeros(shape)


def observed(bank, shift):
    """The bank as the models see it: one noisy reading per situation, turned with each of its 16 copies."""
    reading = noise(shift, bank.state[:, 0].shape)
    return replace(bank, state=bank.state + np.moveaxis(rotate(reading), 0, 1))


def fractions(model, mu):
    """The visible share of the one-step error under mu, and its three pieces."""
    s, a, s1 = map(rotate, mu)
    y = model(s, a)
    e = y - s1
    me = y - fix(y, balance(s, a))
    whole = mean_square(e)
    pe, pm = unrotate(e), unrotate(me)
    both = mean_square(me) - mean_square(pm)
    return {
        "error": whole,
        "visible": 1 - mean_square(pe - pm) / whole,
        "symmetry_only": (whole - mean_square(pe) - both) / whole,
        "balance_only": mean_square(pm) / whole,
        "both": both / whole,
    }


def evaluate(kind, seed, shift, bank, mu, device):
    """One model under one shift: its decisions, signals, AUROCs and visible share, and for M1 and M2 the decisions
    of the corrected model."""
    model = load(kind, seed, device)
    seen = observed(bank, shift)
    returns, symmetry, imbalance = [], [], []
    for part in chunks(seen):
        states = rollout(model, seen.state[part], seen.actions[part])
        returns.append(reward(states, seen.goal[part]))
        sym, bal = signals(model, states[..., :-1, :, :, :], seen.actions[part])
        symmetry.append(sym.mean((-2, -1)))
        imbalance.append(bal.mean((-2, -1)))
    returns = np.concatenate(returns)
    symmetry, imbalance = np.concatenate(symmetry), np.concatenate(imbalance)
    choices = returns.argmax(-1)
    wrong = choices != bank.best[:, None]

    if shift == "input noise":
        # the checks only see the model's error at the states it observes
        s, a, _ = mu
        s = s + noise(shift, s.shape, seed=1)
        mu = s, a, step(s, a)

    row = {"kind": kind, "seed": seed, "shift": shift}
    row.update(score(choices, bank))
    row.update(fractions(model, mu))
    row["auroc_symmetry"] = auroc(symmetry, wrong)
    row["auroc_balance"] = auroc(imbalance, wrong)
    rows = [row]
    if kind != "M3":
        fixed = {"kind": f"{kind}c", "seed": seed, "shift": shift}
        fixed.update(score(choose(corrected(model), seen), bank))
        rows.append(fixed)
    return rows, returns


def pooled(banks, returns):
    """The ensemble M4, the mean of the five M1s with their spread as its signal, and every M1's and M2's orbit
    vote."""
    rows = []
    for shift, bank in banks.items():
        members = np.stack([returns["M1", seed, shift] for seed in SEEDS])
        choices = members.mean(0).argmax(-1)
        row = {"kind": "M4", "shift": shift}
        row.update(score(choices, bank))
        row["auroc_ensemble"] = auroc(members.var(0).mean(-1), choices != bank.best[:, None])
        rows.append(row)
        for kind in ("M1", "M2"):
            for seed in SEEDS:
                vote = returns[kind, seed, shift].mean(1).argmax(-1)
                row = {"kind": f"{kind} vote", "seed": seed, "shift": shift}
                row.update(score(np.repeat(vote[:, None], 16, 1), bank))
                rows.append(row)
    return rows


def continuous(kind, seed, bank, mu, device, angles=100, points=2000):
    """H5: how much the model breaks rotation at 100 random angles, next to the 16 that are checked."""
    model = load(kind, seed, device)
    s, a = mu[0][:points], mu[1][:points]
    y = model(s, a)
    row = {"kind": kind, "seed": seed, "shift": "SO(2)"}
    groups = {"C16": ROTATIONS, "SO(2)": rotation(np.random.default_rng(seed).uniform(0, 2 * np.pi, angles))}
    for name, R in groups.items():
        state, goal, actions = (
            np.einsum("gij,n...j->ng...i", R, x[:, 0]) for x in (bank.state, bank.goal, bank.actions)
        )
        choices = choose(model, replace(bank, state=state, goal=goal, actions=actions))
        row[f"{name} disagreement"] = float(certificate(choices).mean() / len(R))
        defects = [mean_square(model(s @ r.T, a @ r.T) - y @ r.T) for r in R]
        row[f"{name} defect"] = float(np.mean(defects))
    return [row], np.zeros(0)  # nothing to return, but saved the same way as evaluate


def pick(rows, kinds, shifts):
    return [r for r in rows if r["kind"] in kinds and r["shift"] in shifts]


def mean_of(rows, kind, shift, key):
    return np.nanmean([r[key] for r in pick(rows, [kind], [shift])], 0)


def value(rows, kind, shift, seed, key):
    return next(r[key] for r in pick(rows, [kind], [shift]) if r.get("seed") == seed)


def weighted(rows, key):
    """A share like consistent or coverage, pooled over models by how many mistakes each made."""
    mistakes = np.float64(sum(r["n_wrong"] for r in rows))  # a float, so no mistakes gives nan rather than an error
    return sum(r[key] * r["n_wrong"] for r in rows if r["n_wrong"]) / mistakes


def spearman(v):
    """Spearman correlation between the two columns of v, over its finite rows."""
    v = v.reshape(-1, 2)
    v = v[np.isfinite(v).all(-1)]
    return spearmanr(v[:, 0], v[:, 1]).statistic


def print_means(rows):
    print("\nLearned models, mean over seeds")
    keys = (*COLUMNS, "C16 disagreement", "SO(2) disagreement", "C16 defect", "SO(2) defect")
    for kind in sorted({r["kind"] for r in rows}):
        for shift in [*ALL, "SO(2)"]:
            mine = pick(rows, [kind], [shift])
            shown = [key for key in keys if mine and mine[0].get(key) is not None]
            if shown:
                means = ", ".join(f"{key} {mean_of(mine, kind, shift, key):.3g}" for key in shown)
                print(f"  {kind:8} {shift:12} {means}")


def report_h3(rows):
    units = pick(rows, KINDS, SHIFTS)
    for key in ("auroc_symmetry", "auroc_balance"):
        undefined = sum(not np.isfinite(r[key]) for r in units)
        print(f"H3 {key}: Spearman with the visible fraction, resampling trained models; {undefined} undefined")
        groups = {"pooled": KINDS, "M1": ("M1",), "M2": ("M2",), "M3": ("M3",)}
        for name, kinds in groups.items():
            # one cluster per trained model, holding its five shifts
            clusters = []
            for kind in kinds:
                for seed in SEEDS:
                    clusters.append([(r["visible"], r[key]) for r in pick(units, [kind], SHIFTS) if r["seed"] == seed])
            clusters = np.array(clusters)
            if np.isfinite(clusters[..., 1]).any():
                mean, low, high = bootstrap(clusters, spearman)
                print(f"   {name:6} {mean:+.2f} [{low:+.2f}, {high:+.2f}]")


def report_h4(rows):
    for kind in ("M1", "M2"):
        for shift in ALL:
            before = mean_of(rows, kind, shift, "by_situation") / 16
            for name in (f"{kind}c", f"{kind} vote"):
                after = mean_of(rows, name, shift, "by_situation") / 16
                change, low, high = bootstrap(after - before)
                by_seed = [value(rows, name, shift, s, "wrong") - value(rows, kind, shift, s, "wrong") for s in SEEDS]
                print(
                    f"H4 {name:8} {shift:12} change in wrong rate {change:+.2%} [{low:+.2%}, {high:+.2%}] over"
                    f" situations for these five models; by seed from {min(by_seed):+.2%} to {max(by_seed):+.2%}"
                )
            left = sum(r["n_wrong"] for r in pick(rows, [f"{kind}c"], [shift]))
            removed = mean_of(rows, kind, shift, "visible")
            print(f"   visible error removed by the correction {removed:.1%}; {left} wrong decisions left after it")


def report_h6(rows):
    none = {kind: pick(rows, [kind], ["none"]) for kind in ("M1", "M2", "M1 vote", "M2 vote")}
    consistent = weighted(none["M1"], "consistent")
    print(f"H6a M1 share of wrong decisions in consistent orbits, no shift: {consistent:.2f} (below 0.5 predicted)")
    for kind in ("M1", "M2"):
        wrong = sum(r["n_wrong"] for r in none[kind])
        after = sum(r["n_wrong"] for r in none[f"{kind} vote"])
        removed = (wrong - after) / (weighted(none[kind], "coverage") * wrong)  # over the certificate's total
        print(
            f"H6b {kind} wrong decisions the vote removes, over the certificate: {removed:.2f} (0.5 or more predicted)"
        )
    for kind in ("M1", "M2"):
        shares = [value(rows, kind, "none", s, "symmetry_only") + value(rows, kind, "none", s, "both") for s in SEEDS]
        shares = " ".join(f"{x:.3f}" for x in shares)
        print(f"{kind} rotation-breaking share of one-step error by seed, no shift: {shares}")


def report_h7(rows):
    for shift in CONTROLS:
        far = pick(rows, ("M1", "M2"), [shift])
        # the mean over the 10 models, with an interval over models
        auc, low, high = bootstrap([r["auroc_symmetry"] for r in far])
        mistakes = sum(r["n_wrong"] for r in far)
        m3 = [sum(r["n_wrong"] for r in pick(rows, ["M3"], [s])) for s in (shift, "none")]
        note = "" if mistakes >= 100 else f"; only {mistakes} mistakes, so H7b and H7c are inconclusive"
        ensemble = mean_of(rows, "M4", shift, "auroc_ensemble")
        print(
            f"H7 {shift}: M1 and M2 symmetry AUROC {auc:.2f} [{low:.2f}, {high:.2f}] (above 0.7 predicted), consistent"
            f" share {weighted(far, 'consistent'):.2f} (below 0.5 predicted){note}; M3 wrong decisions {m3[0]}"
            f" against {m3[1]} unshifted (equal expected); ensemble AUROC {ensemble:.2f} (above 0.56 predicted)"
        )


def report(rows):
    print_means(rows)
    report_h3(rows)
    report_h4(rows)
    report_h6(rows)
    report_h7(rows)


def plot(rows):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for kind in KINDS:
        mine = pick(rows, [kind], SHIFTS)
        x = [r["visible"] for r in mine]
        size = [6 + 2 * np.sqrt(r["n_wrong"]) for r in mine]  # bigger points rest on more mistakes
        for key, marker, label in (("auroc_symmetry", "o", "symmetry"), ("auroc_balance", "x", "balance")):
            y = [r[key] for r in mine]
            if np.isfinite(y).any():
                left.scatter(x, y, s=size, marker=marker, label=f"{kind}, {label} signal")
    left.axhline(0.5, color="black", lw=1, ls=":")
    left.set(xlabel="visible fraction of one-step error", ylabel="AUROC for wrong decisions")
    left.legend(fontsize=8)

    x = np.arange(len(ALL))
    for i, name in enumerate(("M1", "M1c", "M1 vote", "M2", "M2c", "M2 vote")):
        # side by side, so that equal values stay visible
        right.plot(x + 0.1 * (i - 2.5), [mean_of(rows, name, shift, "wrong") for shift in ALL], "o", label=name)
    right.set_xticks(x, list(ALL), rotation=30)
    right.set(ylabel="wrong-decision rate")
    right.legend(fontsize=8)
    fig.suptitle("What the label-free signals see, and what correcting the visible error buys")
    save_figure(fig, "learned")


def run(situations, device, workers):
    """Train any model that isn't saved yet, then evaluate each one as soon as it's ready, reusing finished
    evaluations. Training runs on the CPU in float32 and evaluation on device in float64."""
    with spawned(workers) as pool:
        trained = {(kind, seed): pool.submit(fit, kind, seed) for kind in KINDS for seed in SEEDS}
        built = {shift: pool.submit(make_bank, situations, **kw) for shift, kw in ALL.items() if shift != "input noise"}
        banks = {shift: job.result() for shift, job in built.items()}
        banks["input noise"] = banks["none"]
        for bank in banks.values():
            check_truth(bank)
        for shift in CONTROLS:
            # H7a: moving the scene changes nothing the physics can see
            assert np.allclose(banks[shift].returns, banks["none"].returns, rtol=1e-9, atol=0), "the truth moved"
        mus = {shift: visited(bank, ALL[shift].get("truth", step)) for shift, bank in banks.items()}
        jobs = {}
        for (kind, seed), done in trained.items():
            done.result()
            for shift in ALL:
                task = kind, seed, shift, banks[shift], mus[shift], device
                jobs[kind, seed, shift] = pool.submit(kept, evaluate, f"{kind}_{seed}_{shift}", *task)
            task = kind, seed, banks["none"], mus["none"], device
            jobs[kind, seed, "SO(2)"] = pool.submit(kept, continuous, f"{kind}_{seed}_SO(2)", *task)
        results = {key: job.result() for key, job in jobs.items()}
    rows = [row for out in results.values() for row in out[0]]
    returns = {key: out[1] for key, out in results.items() if key[2] != "SO(2)"}
    return rows + pooled(banks, returns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--situations", type=int, default=500)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="for evaluation")
    parser.add_argument("--workers", type=int, help="processes; default one per core, fewer on one GPU")
    args = parser.parse_args()
    rows = saved_or_run("learned", args.figures_only, run, args.situations, args.device, args.workers)
    report(rows)
    plot(rows)
