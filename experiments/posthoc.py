"""Exploratory analyses, written after all the version 1 results were in.

Two label-free warnings that the predictions left out: the margin, the gap between a model's best and second-best
predicted return, which is small for a close call; and the orbit flag, which marks a copy whose choice differs from
the most common choice in its orbit. Also a check of what the noisy-input condition really tests. Needs the saved
evaluations in results/learned/."""

import numpy as np

from experiments.common import auroc, bootstrap, choose, score
from experiments.learned import ALL, KINDS, SEEDS, observed, read_job, spearman
from invariantlens.decisions import make_bank
from invariantlens.physics import step


def margin_and_flag(bank, returns):
    """Which decisions are wrong, the margin's AUROC for them, and which copies the orbit flag marks."""
    top = np.sort(returns, -1)
    margin = top[..., -1] - top[..., -2]
    choices = returns.argmax(-1)
    common = np.array([np.bincount(c, minlength=5).argmax() for c in choices])
    wrong = choices != bank.best[:, None]
    flag = choices != common[:, None]
    # the closer the call, the louder the warning
    return wrong, auroc(margin.max() - margin, wrong), flag


if __name__ == "__main__":
    banks = {shift: make_bank(**settings) for shift, settings in ALL.items() if shift != "input noise"}
    banks["input noise"] = banks["none"]

    units = {}
    for shift in ALL:
        for kind in KINDS:
            aucs, wrong, flag = [], [], []
            for seed in SEEDS:
                rows, returns = read_job(f"{kind}_{seed}_{shift}")
                units[kind, seed, shift] = rows[0]
                w, auc, f = margin_and_flag(banks[shift], returns)
                wrong.append(w)
                aucs.append(auc)
                flag.append(f)
            wrong, flag = np.concatenate(wrong), np.concatenate(flag)
            hits = (wrong & flag).sum()
            print(
                f"{kind} {shift:12} {wrong.sum():5d} wrong; margin AUROC {np.nanmean(aucs):.3f}; orbit flag marks"
                f" {flag.sum()}, {hits} of them wrong, {hits / max(wrong.sum(), 1):.0%} of all mistakes"
            )

    noisy = observed(banks["none"], "input noise")
    truth = score(choose(step, noisy), banks["none"])["n_wrong"]
    print(f"\nThe true simulator on the noisy inputs: {truth} wrong, so those mistakes aren't the models'")

    for name, extra in (("without the noisy inputs", []), ("with far 5 in their place", ["far 5"])):
        shifts = ["none", "D x 1.5", "damping", "speed x 2", *extra]
        for key in ("auroc_symmetry", "auroc_balance"):
            clusters = []
            for kind in KINDS:
                for seed in SEEDS:
                    clusters.append([(units[kind, seed, x]["visible"], units[kind, seed, x][key]) for x in shifts])
            mean, low, high = bootstrap(np.array(clusters), spearman)
            print(f"H3 {key} {name}: {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
