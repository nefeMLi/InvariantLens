"""H3, H4 and H5: trained models and their label-free signals, the corrected models, the ensemble and the orbit vote."""

import argparse
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
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
    check_truth,
    choose,
    chunks,
    mean_square,
    read_results,
    save_figure,
    score,
    visited,
    write_results,
)
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.models import KINDS, Net, train, transitions, world_model
from invariantlens.physics import ROTATIONS, SPEED, balance, rotation, step
from invariantlens.symmetry import certificate, corrected, fix, rotate, signals, unrotate

SEEDS = range(5)
NOISE = 0.02  # sd of the input noise on positions and velocities
SHIFTS = {
    "none": {},
    "D x 1.5": {"truth": partial(step, depth=1.5)},
    "damping": {"truth": partial(step, damping=0.5)},
    "speed x 2": {"speed": 2 * SPEED},
    "input noise": {},  # the unshifted bank, seen through noisy inputs
}
CONTROLS = {"far 2.5": {"shift": 2.5}, "far 5": {"shift": 5.0}}  # H7: same physics, scene moved away from the origin
ALL = SHIFTS | CONTROLS  # H3 stays on SHIFTS
MODELS = ROOT / "results" / "models"
DONE = ROOT / "results" / "learned"  # one file per finished evaluation; delete the folder after changing the code
COLUMNS = ("wrong", "consistent", "visible", "auroc_symmetry", "auroc_balance", "auroc_ensemble", "coverage")


def kept(fn, name, *args):
    """fn(*args), saved to results/learned/<name>.pkl as soon as it finishes, so an interrupted run resumes."""
    path = DONE / f"{name}.pkl"
    if not path.exists():
        DONE.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")  # renamed only once complete, so a killed run leaves no broken file
        part.write_bytes(pickle.dumps(fn(*args)))
        part.replace(path)
    return pickle.loads(path.read_bytes())


def fit(kind, seed):
    """Train one model, unless it was trained before."""
    path = MODELS / f"{kind}_{seed}.pt"
    if not path.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(train(kind, seed, transitions(5000, 0), transitions(500, 1)).state_dict(), path)


def load(kind, seed, device):
    net = Net(kind == "M3", torch.zeros(5))
    net.load_state_dict(torch.load(MODELS / f"{kind}_{seed}.pt"))
    return world_model(net, device)


def noise(shift, shape, seed=0):
    """Isotropic sensor noise, one draw per observation and the same for every model; zero without the noise shift."""
    return NOISE * np.random.default_rng(seed).normal(size=shape) if shift == "input noise" else np.zeros(shape)


def fractions(model, mu):
    """The visible share of the one-step error under mu and its three pieces, from one pass over the rotated points."""
    s, a, s1 = map(rotate, mu)
    y = model(s, a)
    e, me = y - s1, y - fix(y, balance(s, a))
    whole, pe, pm = mean_square(e), unrotate(e), unrotate(me)
    both = mean_square(me) - mean_square(pm)
    return {
        "error": whole,
        "visible": 1 - mean_square(pe - pm) / whole,
        "symmetry_only": (whole - mean_square(pe) - both) / whole,
        "balance_only": mean_square(pm) / whole,
        "both": both / whole,
    }


def evaluate(kind, seed, shift, bank, mu, device):
    """One model under one shift: its decisions, signals, AUROCs and visible share, and for M1 and M2 the correction."""
    model = load(kind, seed, device)
    # one noisy observation per situation, turned with each rotated copy; the model's own rollouts aren't noised again
    seen = replace(bank, state=bank.state + np.moveaxis(rotate(noise(shift, bank.state[:, 0].shape)), 0, 1))
    returns, found = [], []
    for p in chunks(seen):
        states = rollout(model, seen.state[p], seen.actions[p])
        returns.append(reward(states, seen.goal[p]))
        found.append([x.mean((-2, -1)) for x in signals(model, states[..., :-1, :, :, :], seen.actions[p])])
    returns = np.concatenate(returns)
    (sym, bal), choices = map(np.concatenate, zip(*found)), returns.argmax(-1)
    wrong, unit = choices != bank.best[:, None], {"kind": kind, "seed": seed, "shift": shift}
    if shift == "input noise":  # the audit only sees the model's error at the states it observes
        s, a, _ = mu
        s = s + noise(shift, s.shape, seed=1)
        mu = s, a, step(s, a)
    rows = [
        unit
        | score(choices, bank)
        | fractions(model, mu)
        | {"auroc_symmetry": auroc(sym, wrong), "auroc_balance": auroc(bal, wrong)}
    ]
    if kind != "M3":
        rows.append(unit | {"kind": f"{kind}c"} | score(choose(corrected(model), seen), bank))
    return rows, returns


def pooled(banks, returns):
    """M4, the mean prediction of the five M1 seeds, with its ensemble signal; and the orbit vote of every M1 and M2."""
    rows = []
    for shift, bank in banks.items():
        members = np.stack([returns["M1", seed, shift] for seed in SEEDS])
        choices = members.mean(0).argmax(-1)
        spread = auroc(members.var(0).mean(-1), choices != bank.best[:, None])
        rows.append({"kind": "M4", "shift": shift} | score(choices, bank) | {"auroc_ensemble": spread})
        for kind in ("M1", "M2"):
            for seed in SEEDS:
                vote = returns[kind, seed, shift].mean(1).argmax(-1)
                rows.append(
                    {"kind": f"{kind} vote", "seed": seed, "shift": shift}
                    | score(np.repeat(vote[:, None], 16, 1), bank)
                )
    return rows


def continuous(kind, seed, bank, mu, device, angles=100, points=2000):
    """H5: decision disagreement and one-step defect at 100 random angles, next to the 16 audit angles."""
    model, s, a = load(kind, seed, device), mu[0][:points], mu[1][:points]
    y, row = model(s, a), {"kind": kind, "seed": seed, "shift": "SO(2)"}
    for name, R in {
        "C16": ROTATIONS,
        "SO(2)": rotation(np.random.default_rng(seed).uniform(0, 2 * np.pi, angles)),
    }.items():
        state, goal, actions = (
            np.einsum("gij,n...j->ng...i", R, x[:, 0]) for x in (bank.state, bank.goal, bank.actions)
        )
        choices = choose(model, replace(bank, state=state, goal=goal, actions=actions))
        row[f"{name} disagreement"] = float(certificate(choices).mean() / len(R))
        row[f"{name} defect"] = float(np.mean([mean_square(model(s @ r.T, a @ r.T) - y @ r.T) for r in R]))
    return row


def pick(rows, kinds, shifts):
    return [r for r in rows if r["kind"] in kinds and r["shift"] in shifts]


def mean_of(rows, kind, shift, key):
    return np.nanmean([r[key] for r in pick(rows, [kind], [shift])], 0)


def value(rows, kind, shift, seed, key):
    return next(r[key] for r in pick(rows, [kind], [shift]) if r.get("seed") == seed)


def weighted(rows, key):
    """A share like 'consistent' or 'coverage', pooled over models by how many mistakes each made."""
    mistakes = np.float64(sum(r["n_wrong"] for r in rows))  # a float, so no mistakes gives nan instead of an error
    return sum(r[key] * r["n_wrong"] for r in rows if r["n_wrong"]) / mistakes


def spearman(v):
    """Spearman correlation of the two columns of v (..., 2), over its finite rows."""
    v = v.reshape(-1, 2)
    return spearmanr(*v[np.isfinite(v).all(-1)].T).statistic


def report(rows) -> None:
    print("\nLearned models, mean over seeds")
    keys = (*COLUMNS, "C16 disagreement", "SO(2) disagreement", "C16 defect", "SO(2) defect")
    for kind in sorted({r["kind"] for r in rows}):
        for shift in [*ALL, "SO(2)"]:
            mine = pick(rows, [kind], [shift])
            shown = [k for k in keys if mine and mine[0].get(k) is not None]
            if shown:
                print(f"  {kind:8} {shift:12} " + ", ".join(f"{k} {mean_of(mine, kind, shift, k):.3g}" for k in shown))
    units = pick(rows, KINDS, SHIFTS)
    for key in ("auroc_symmetry", "auroc_balance"):
        undefined = sum(not np.isfinite(r[key]) for r in units)
        print(f"H3 {key}: Spearman with the visible fraction, resampling trained models; {undefined} undefined")
        for name, kinds in ({"pooled": KINDS} | {kind: (kind,) for kind in KINDS}).items():
            # one cluster per trained model, holding its five shifts
            clusters = np.array(
                [
                    [(r["visible"], r[key]) for r in pick(units, [k], SHIFTS) if r["seed"] == s]
                    for k in kinds
                    for s in SEEDS
                ]
            )
            if np.isfinite(clusters[..., 1]).any():
                print(f"   {name:6} " + "{:+.2f} [{:+.2f}, {:+.2f}]".format(*bootstrap(clusters, spearman)))
    for kind in ("M1", "M2"):
        for shift in ALL:
            before = mean_of(rows, kind, shift, "by_situation") / 16
            for name in (f"{kind}c", f"{kind} vote"):
                change, lo, hi = bootstrap(mean_of(rows, name, shift, "by_situation") / 16 - before)
                seeds = [value(rows, name, shift, s, "wrong") - value(rows, kind, shift, s, "wrong") for s in SEEDS]
                print(
                    f"H4 {name:8} {shift:12} change in wrong rate {change:+.2%} [{lo:+.2%}, {hi:+.2%}] over situations"
                    f" for these five models; by seed from {min(seeds):+.2%} to {max(seeds):+.2%}"
                )
            left = sum(r["n_wrong"] for r in pick(rows, [f"{kind}c"], [shift]))  # none of these can be flagged
            removed = mean_of(rows, kind, shift, "visible")
            print(f"   visible error removed by the correction {removed:.1%}; {left} wrong decisions left after it")
    none = {kind: pick(rows, [kind], ["none"]) for kind in ("M1", "M2", "M1 vote", "M2 vote")}
    consistent = weighted(none["M1"], "consistent")
    print(f"H6a M1 share of wrong decisions in consistent orbits, no shift: {consistent:.2f} (below 0.5 predicted)")
    for kind in ("M1", "M2"):
        wrong, after = (sum(r["n_wrong"] for r in none[k]) for k in (kind, f"{kind} vote"))
        removed = (wrong - after) / (weighted(none[kind], "coverage") * wrong)  # over the certificate's total
        print(
            f"H6b {kind} wrong decisions the vote removes, over the certificate: {removed:.2f} (0.5 or more predicted)"
        )
    for kind in ("M1", "M2"):
        breaking = [value(rows, kind, "none", s, "symmetry_only") + value(rows, kind, "none", s, "both") for s in SEEDS]
        print(
            f"{kind} rotation-breaking share of one-step error by seed, no shift: "
            + " ".join(f"{x:.3f}" for x in breaking)
        )
    for shift in CONTROLS:
        far = pick(rows, ("M1", "M2"), [shift])
        auc, lo, hi = bootstrap([r["auroc_symmetry"] for r in far])  # mean over the 10 models, interval over models
        mistakes = sum(r["n_wrong"] for r in far)
        m3 = [sum(r["n_wrong"] for r in pick(rows, ["M3"], [s])) for s in (shift, "none")]
        verdict = "" if mistakes >= 100 else f"; only {mistakes} mistakes, so H7b and H7c are inconclusive"
        print(
            f"H7 {shift}: M1 and M2 symmetry AUROC {auc:.2f} [{lo:.2f}, {hi:.2f}] (above 0.7 predicted), consistent"
            f" share {weighted(far, 'consistent'):.2f} (below 0.5 predicted){verdict}; M3 wrong decisions {m3[0]}"
            f" against {m3[1]} unshifted (equal expected); ensemble AUROC"
            f" {mean_of(rows, 'M4', shift, 'auroc_ensemble'):.2f} (above 0.56 predicted)"
        )


def plot(rows) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for kind in KINDS:
        mine = pick(rows, [kind], SHIFTS)
        x, size = [r["visible"] for r in mine], [6 + 2 * np.sqrt(r["n_wrong"]) for r in mine]  # bigger = more mistakes
        for key, marker in (("auroc_symmetry", "o"), ("auroc_balance", "x")):
            y = [r[key] for r in mine]
            if np.isfinite(y).any():
                left.scatter(x, y, s=size, marker=marker, label=f"{kind}, {key[6:]} signal")
    left.axhline(0.5, color="black", lw=1, ls=":")
    left.set(xlabel="visible fraction of one-step error", ylabel="AUROC for wrong decisions")
    left.legend(fontsize=8)
    x = np.arange(len(ALL))
    for i, name in enumerate(("M1", "M1c", "M1 vote", "M2", "M2c", "M2 vote")):  # side by side so equal values show
        right.plot(x + 0.1 * (i - 2.5), [mean_of(rows, name, shift, "wrong") for shift in ALL], "o", label=name)
    right.set_xticks(x, list(ALL), rotation=30)
    right.set(ylabel="wrong-decision rate")
    right.legend(fontsize=8)
    fig.suptitle("What the label-free signals see, and what correcting the visible error buys")
    save_figure(fig, "learned")


def run(situations, device, workers):
    """Train what isn't saved yet and evaluate each model as soon as it's ready, reusing finished jobs. Training runs
    on the CPU in float32; evaluation runs on device in float64."""
    # one core per worker; spawned, not forked: CUDA cannot start in a forked process (the Linux default before 3.14)
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, spawn, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        trained = {(kind, seed): pool.submit(fit, kind, seed) for kind in KINDS for seed in SEEDS}
        built = {shift: pool.submit(make_bank, situations, **kw) for shift, kw in ALL.items() if shift != "input noise"}
        banks = {shift: job.result() for shift, job in built.items()} | {"input noise": built["none"].result()}
        for bank in banks.values():
            check_truth(bank)
        for shift in CONTROLS:  # H7a: moving the scene changes nothing the physics can see
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
    rows = [row for key, out in results.items() for row in ([out] if key[2] == "SO(2)" else out[0])]
    return rows + pooled(banks, {key: out[1] for key, out in results.items() if key[2] != "SO(2)"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--situations", type=int, default=500)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="for evaluation")
    parser.add_argument("--workers", type=int, help="processes; default one per core, use fewer on one GPU")
    args = parser.parse_args()
    if args.figures_only:
        rows = read_results("learned")
    else:
        rows = run(args.situations, args.device, args.workers)
        write_results(rows, "learned")
    report(rows)
    plot(rows)
