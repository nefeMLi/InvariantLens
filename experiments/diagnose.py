"""H8 and H9: when does a symmetry a world model learned from data break outside that data, and can averaging over
the symmetry at test time repair what breaks?

Four ways out of the training data, with the physics unchanged. In two the model meets familiar physics in a new
frame: the scene far from the origin, or every disc drifting at the same speed. In the other two it meets physics it
never saw: much faster discs, or two discs starting closer than any pair in training. The prediction is that the
learned symmetry breaks in the first kind, where averaging repairs it, and holds in the second, where nothing
computed from the model can help. Asking the model once in a standard pose is reported beside the average, to tell
symmetry from averaging. Dev seeds were used to design this; test seeds are used once."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common import ROOT, cached, check_truth, read_results, regret, save_figure, write_results
from experiments.learned import load
from invariantlens.decisions import make_bank, reward, rollout
from invariantlens.physics import SPEED, balance, step
from invariantlens.symmetry import canonical, corrected, fix, surprise

KINDS, SEEDS = ("M1", "M2"), range(5)  # M2 learned the symmetry from rotated data; M3 has it built in
SPLITS = {"dev": 1_000_000, "test": 2_000_000}  # first seed of each split
SIZES = {"dev": 30, "test": 100}  # situations per bank
SHIFTS = {  # bank settings, what the model meets, and the prediction it belongs to
    "far 10": ({"shift": 10.0}, "frame", "H8"),
    "speed x 5": ({"speed": 5 * SPEED}, "physics", "H8"),
    "drift 2": ({"drift": 2.0}, "frame", "H9"),
    "crowded": ({"crowd": 0.35}, "physics", "H9"),
}
RULES = {"raw": "the model", "posed": "one standard pose", "averaged": "average of 16 poses"}  # how the agent decides
COSTLY = 0.05  # a decision is costly if it ends this much further from the goal than the best candidate would
BROKEN, KEPT = 0.5, 0.1  # visible shares that count as the symmetry broken, and as kept
ENOUGH = 20  # costly decisions needed before the change in them counts
CACHE = ROOT / "results" / "diagnose"


def evaluate(kind, seed, bank, device):
    """Predicted returns under each decision rule, and the squared one-step error and its visible part along the true
    rollouts of every candidate. Both corrections fix the balance first, so they differ only in pose against average."""
    model = load(kind, seed, device)

    def balanced(s, a):
        return fix(model(s, a), balance(s, a))

    rules = {"raw": model, "posed": canonical(balanced), "averaged": corrected(model)}
    states = rollout(step, bank.state[:, 0], bank.actions[:, 0])
    s, a = states[..., :-1, :, :, :].reshape(-1, *states.shape[-3:]), bank.actions[:, 0].reshape(-1, 2)
    error, visible = surprise(model, s, a, step(s, a))
    returns = {name: reward(rollout(rule, bank.state, bank.actions), bank.goal) for name, rule in rules.items()}
    return returns | {"error": error, "visible": visible}


def run(split, size, device, workers):
    start = SPLITS[split]
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, spawn, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        built = {
            name: pool.submit(make_bank, size, margin=1e-6, start=start, **kw) for name, (kw, _, _) in SHIFTS.items()
        }
        plain = pool.submit(make_bank, size, margin=1e-6, start=start)
        banks = {name: job.result() for name, job in built.items()}
        for name, bank in banks.items():
            check_truth(bank)
            if SHIFTS[name][1] == "frame":  # a frame shift changes nothing the physics can see
                assert np.allclose(bank.returns, plain.result().returns, rtol=1e-9, atol=0), f"{name} moved the truth"
        jobs = {
            (kind, seed, name): pool.submit(
                cached,
                CACHE / f"{split}_{size}_{kind}_{seed}_{name.replace(' ', '')}.npz",
                evaluate,
                kind,
                seed,
                bank,
                device,
            )
            for name, bank in banks.items()
            for kind in KINDS
            for seed in SEEDS
        }
        rows = []
        for (kind, seed, name), job in jobs.items():
            out = job.result()
            loss = {rule: regret(banks[name].returns, out[rule].argmax(-1)) for rule in RULES}
            rows.append(
                {"split": split, "kind": kind, "seed": seed, "shift": name, "meets": SHIFTS[name][1]}
                | {
                    "share": float(out["visible"].sum() / out["error"].sum()),
                    "error": float(np.sqrt(out["error"].mean())),
                }
                | {f"costly_{rule}": int((x > COSTLY).sum()) for rule, x in loss.items()}
                | {f"regret_{rule}": float(x.mean()) for rule, x in loss.items()}
                | {"decisions": int(loss["raw"].size)}
            )
    return rows


def costly(rows, rule):
    return sum(r[f"costly_{rule}"] for r in rows)


def verdict(rows, name):
    """Whether one shift behaves as predicted: in a new frame the symmetry breaks and averaging repairs the decisions;
    facing new physics the symmetry holds and averaging doesn't."""
    mine = [r for r in rows if r["shift"] == name]
    share = np.mean([r["share"] for r in mine])
    raw, averaged = costly(mine, "raw"), costly(mine, "averaged")
    removed = 1 - averaged / raw if raw else float("nan")
    if SHIFTS[name][1] == "frame":
        held = share >= BROKEN and (raw < ENOUGH or removed >= 0.5)
    else:
        held = share <= KEPT and (raw < ENOUGH or removed < 0.2)
    by_kind = ", ".join(f"{k} {np.mean([r['share'] for r in mine if r['kind'] == k]):.2f}" for k in KINDS)
    counted = "" if raw >= ENOUGH else f", fewer than {ENOUGH} so only the share counts"
    rules = ", ".join(f"{label} {costly(mine, rule)}" for rule, label in RULES.items())
    return held, (
        f"{name:9} (new {SHIFTS[name][1]}): visible share {share:.2f} ({by_kind});"
        f" costly decisions of {sum(r['decisions'] for r in mine)}: {rules}{counted}"
    )


def report(rows) -> None:
    print("M1 and M2, five seeds each")
    for hypothesis in ("H8", "H9"):
        results = [verdict(rows, name) for name, (_, _, h) in SHIFTS.items() if h == hypothesis]
        for _, line in results:
            print(f"  {line}")
        print(f"{hypothesis} held: {all(held for held, _ in results)}")


def plot(rows) -> None:
    fig, (share, bars) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = {"frame": "tab:blue", "physics": "tab:red"}
    for i, name in enumerate(SHIFTS):
        for r in (r for r in rows if r["shift"] == name):
            x = i + (KINDS.index(r["kind"]) - 0.5) * 0.3 + (r["seed"] - 2) * 0.04
            share.scatter(x, r["share"], s=14, color=colors[r["meets"]], marker="os"[KINDS.index(r["kind"])])
    for level in (KEPT, BROKEN):
        share.axhline(level, color="black", lw=1, ls=":")
    share.set_xticks(range(len(SHIFTS)), SHIFTS)
    share.set(
        ylabel="share of the error that breaks the symmetry",
        ylim=(-0.03, 1.03),
        title="Blue: new frame. Red: new physics. Circles M1, squares M2",
    )
    width = 0.8 / len(RULES)
    for j, (rule, label) in enumerate(RULES.items()):
        counts = [costly([r for r in rows if r["shift"] == name], rule) for name in SHIFTS]
        bars.bar(np.arange(len(SHIFTS)) + (j - 1) * width, counts, width, label=label)
    bars.set_xticks(range(len(SHIFTS)), SHIFTS)
    bars.set(ylabel="costly decisions, M1 and M2", title="What averaging over the symmetry repairs")
    bars.legend()
    fig.suptitle("A learned symmetry breaks in a new frame, where averaging repairs it, and holds for new physics")
    save_figure(fig, f"diagnose_{rows[0]['split']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--situations", type=int, help="per bank; default 30 on dev, 100 on test")
    parser.add_argument("--device", default="cpu", help="cuda for a GPU; then use one or two workers")
    parser.add_argument("--workers", type=int, help="processes; default one per core")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    name = f"diagnose_{args.split}"
    if args.figures_only:
        rows = read_results(name)
    else:
        rows = run(args.split, args.situations or SIZES[args.split], args.device, args.workers)
        write_results(rows, name)
    report(rows)
    plot(rows)
