"""Exploratory, written after all the results were in: two label-free warnings the pre-registration left out, and a
check of what the noisy-input condition tests.

The margin is the gap between the model's best and second-best predicted return, so a small one means a close call.
The orbit flag marks a copy whose choice differs from the most common choice in its orbit. Needs the saved jobs in
results/learned/ from a learned run."""

import pickle

import numpy as np

from experiments.common import auroc, bootstrap, choose, score
from experiments.learned import ALL, DONE, KINDS, SEEDS, observed, spearman
from invariantlens.decisions import make_bank
from invariantlens.physics import step

if __name__ == "__main__":
    banks = {shift: make_bank(**kw) for shift, kw in ALL.items() if shift != "input noise"}
    banks["input noise"] = banks["none"]
    units = {}
    for shift in ALL:
        for kind in KINDS:
            aucs, wrong, flag = [], [], []
            for seed in SEEDS:
                rows, returns = pickle.loads((DONE / f"{kind}_{seed}_{shift}.pkl").read_bytes())
                units[kind, seed, shift] = rows[0]
                top = np.sort(returns, -1)
                margin = top[..., -1] - top[..., -2]
                choices = returns.argmax(-1)
                common = np.array([np.bincount(c, minlength=5).argmax() for c in choices])
                wrong.append(choices != banks[shift].best[:, None])
                flag.append(choices != common[:, None])
                aucs.append(auroc(margin.max() - margin, wrong[-1]))  # the closer the call, the louder the warning
            wrong, flag = np.concatenate(wrong), np.concatenate(flag)
            hits = (wrong & flag).sum()
            print(
                f"{kind} {shift:12} {wrong.sum():5d} wrong; margin AUROC {np.nanmean(aucs):.3f}; orbit flag marks"
                f" {flag.sum()}, {hits} of them wrong, {hits / max(wrong.sum(), 1):.0%} of all mistakes"
            )
    truth = score(choose(step, observed(banks["none"], "input noise")), banks["none"])["n_wrong"]
    print(f"\nThe true simulator on the noisy inputs: {truth} wrong, so those mistakes aren't the models'")
    for name, last in (("without the noisy inputs", []), ("with far 5 in their place", ["far 5"])):
        for key in ("auroc_symmetry", "auroc_balance"):
            shifts = ["none", "D x 1.5", "damping", "speed x 2", *last]
            clusters = np.array(
                [[(units[k, s, x]["visible"], units[k, s, x][key]) for x in shifts] for k in KINDS for s in SEEDS]
            )
            print(f"H3 {key} {name}: " + "{:+.2f} [{:+.2f}, {:+.2f}]".format(*bootstrap(clusters, spearman)))
