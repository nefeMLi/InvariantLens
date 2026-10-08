"""H8 and H9: where does a symmetry that a world model learned from data break outside that data, and can averaging
over the symmetry repair what breaks?

Four ways out of the training data, all with the physics unchanged. In two the model meets familiar physics in a new
frame: the scene far from the origin, or every disc drifting at the same speed. In the other two it meets physics it
never saw: much faster discs, or two discs starting closer than any pair in training. Asking the model once in a
standard pose is reported next to the average, to tell symmetry apart from averaging. The dev seeds were used to
design this and the test seeds once."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import ROOT, cached, check_truth, load, regret, save_figure, saved_or_run, spawned
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.physics import SPEED, balance, step
from invariantlens.symmetry import canonical, corrected, error_parts, fix

KINDS, SEEDS = ("M1", "M2"), range(5)  # M2 learned the symmetry from rotated data; M3 has it built in
SPLITS = {"dev": 1_000_000, "test": 2_000_000}  # the first seed of each split
SIZES = {"dev": 30, "test": 100}  # situations per bank

# bank settings, what the model meets, and the prediction each shift belongs to
SHIFTS = {
    "far 10": ({"shift": 10.0}, "frame", "H8"),
    "speed x 5": ({"speed": 5 * SPEED}, "physics", "H8"),
    "drift 2": ({"drift": 2.0}, "frame", "H9"),
    "crowded": ({"crowd": 0.35}, "physics", "H9"),
}
RULES = {"raw": "the model", "posed": "one standard pose", "averaged": "average of 16 poses"}
MARKERS = {"M1": "o", "M2": "s"}
COSTLY = 0.05  # a decision is costly if it ends this much further from the goal than the best candidate would
BROKEN, KEPT = 0.5, 0.1  # visible shares that count as the symmetry broken, and as kept
ENOUGH = 20  # costly decisions needed before the change in them counts
CACHE = ROOT / "results" / "diagnose"


def evaluate(kind, seed, bank, device):
    """Predicted returns under each decision rule, and the one-step error and its visible part along the true rollouts
    of every candidate."""
    model = load(kind, seed, device)

    # both corrections fix the balance first, so they differ only in one pose against the average of 16
    def balanced(s, a):
        return fix(model(s, a), balance(s, a))

    rules = {"raw": model, "posed": canonical(balanced), "averaged": corrected(model)}
    out = {}
    for name, rule in rules.items():
        out[name] = reward(rollout(rule, bank.state, bank.actions), bank.goal)

    states = rollout(step, bank.state[:, 0], bank.actions[:, 0])
    s = states[..., :-1, :, :, :].reshape(-1, *states.shape[-3:])
    a = bank.actions[:, 0].reshape(-1, 2)
    out["error"], out["visible"] = error_parts(model, s, a, step(s, a))
    return out


def summarise(split, kind, seed, name, bank, out):
    """One row per model and shift: the visible share, and the costly decisions and regret under each rule."""
    row = {"split": split, "kind": kind, "seed": seed, "shift": name, "meets": SHIFTS[name][1]}
    row["share"] = float(out["visible"].sum() / out["error"].sum())
    row["error"] = float(np.sqrt(out["error"].mean()))
    losses = {rule: regret(bank.returns, out[rule].argmax(-1)) for rule in RULES}
    row.update({f"costly_{rule}": int((loss > COSTLY).sum()) for rule, loss in losses.items()})
    row.update({f"regret_{rule}": float(loss.mean()) for rule, loss in losses.items()})
    row["decisions"] = int(losses["raw"].size)
    return row


def run(split, size, device, workers):
    start = SPLITS[split]
    with spawned(workers) as pool:
        built = {
            name: pool.submit(make_bank, size, margin=1e-6, start=start, **kw) for name, (kw, _, _) in SHIFTS.items()
        }
        plain = pool.submit(make_bank, size, margin=1e-6, start=start).result()
        banks = {name: job.result() for name, job in built.items()}
        for name, bank in banks.items():
            check_truth(bank)
            if SHIFTS[name][1] == "frame":
                # a new frame changes nothing the physics can see
                assert np.allclose(bank.returns, plain.returns, rtol=1e-9, atol=0), f"{name} moved the truth"
        jobs = {}
        for name, bank in banks.items():
            for kind in KINDS:
                for seed in SEEDS:
                    path = CACHE / f"{split}_{size}_{kind}_{seed}_{name.replace(' ', '')}.npz"
                    jobs[kind, seed, name] = pool.submit(cached, path, evaluate, kind, seed, bank, device)
        return [summarise(split, *key, banks[key[2]], job.result()) for key, job in jobs.items()]


def costly(rows, rule):
    return sum(r[f"costly_{rule}"] for r in rows)


def verdict(rows, name):
    """Whether one shift went as predicted. In a new frame the symmetry should break and averaging repair the
    decisions; facing new physics the symmetry should hold and averaging not help."""
    mine = [r for r in rows if r["shift"] == name]
    meets = SHIFTS[name][1]
    share = np.mean([r["share"] for r in mine])
    raw, averaged = costly(mine, "raw"), costly(mine, "averaged")
    removed = 1 - averaged / raw if raw else float("nan")
    if meets == "frame":
        held = share >= BROKEN and (raw < ENOUGH or removed >= 0.5)
    else:
        held = share <= KEPT and (raw < ENOUGH or removed < 0.2)

    by_kind = ", ".join(f"{k} {np.mean([r['share'] for r in mine if r['kind'] == k]):.2f}" for k in KINDS)
    by_rule = ", ".join(f"{label} {costly(mine, rule)}" for rule, label in RULES.items())
    note = "" if raw >= ENOUGH else f", fewer than {ENOUGH} so only the share counts"
    decisions = sum(r["decisions"] for r in mine)
    line = f"{name:9} (new {meets}): visible share {share:.2f} ({by_kind}); costly decisions of {decisions}: {by_rule}"
    return held, line + note


def report(rows):
    print("M1 and M2, five seeds each")
    for hypothesis in ("H8", "H9"):
        results = [verdict(rows, name) for name, (_, _, h) in SHIFTS.items() if h == hypothesis]
        for _, line in results:
            print(f"  {line}")
        print(f"{hypothesis} held: {all(held for held, _ in results)}")


def plot(rows):
    fig, (share, bars) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = {"frame": "tab:blue", "physics": "tab:red"}
    for i, name in enumerate(SHIFTS):
        for r in rows:
            if r["shift"] != name:
                continue
            # spread the ten models out a little around each shift
            x = i + (KINDS.index(r["kind"]) - 0.5) * 0.3 + (r["seed"] - 2) * 0.04
            share.scatter(x, r["share"], s=14, color=colors[r["meets"]], marker=MARKERS[r["kind"]])
    # the thresholds fixed before the test run
    for level, label, x in ((BROKEN, "broken at or above 0.5", 0.5), (KEPT, "kept at or below 0.1", 2.5)):
        share.axhline(level, color="black", lw=1, ls=":")
        share.text(x, level + 0.01, label, ha="center", va="bottom", fontsize=8)  # in a gap between shifts
    share.set_xticks(range(len(SHIFTS)), SHIFTS)
    share.set(
        ylabel="share of the error that breaks the symmetry",
        ylim=(-0.03, 1.06),
        title="Blue: new frame. Red: new physics. Circles M1, squares M2",
    )

    width = 0.8 / len(RULES)
    for j, (rule, label) in enumerate(RULES.items()):
        counts = [costly([r for r in rows if r["shift"] == name], rule) for name in SHIFTS]
        drawn = bars.bar(np.arange(len(SHIFTS)) + (j - 1) * width, counts, width, label=label)
        bars.bar_label(drawn, fontsize=7)  # so that zeros show as zeros, not as missing bars
    bars.set_xticks(range(len(SHIFTS)), SHIFTS)
    decisions = sum(r["decisions"] for r in rows if r["shift"] == next(iter(SHIFTS)))
    bars.set(ylabel=f"costly decisions out of {decisions:,}", title="How the agent decides: M1 and M2 together")
    bars.legend()
    fig.suptitle("Where the learned symmetry broke, and what averaging over it repaired")
    save_figure(fig, f"diagnose_{rows[0]['split']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--situations", type=int, help="per bank; default 30 on dev, 100 on test")
    parser.add_argument("--device", default="cpu", help="cuda for a GPU, then with one or two workers")
    parser.add_argument("--workers", type=int, help="processes; default one per core")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    size = args.situations or SIZES[args.split]
    rows = saved_or_run(f"diagnose_{args.split}", args.figures_only, run, args.split, size, args.device, args.workers)
    report(rows)
    plot(rows)
