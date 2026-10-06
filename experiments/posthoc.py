"""Exploratory, written after all the results were in: two label-free warnings the pre-registration left out.

The margin is the gap between the model's best and second-best predicted return, so a small one means a close call.
The orbit flag marks a copy whose choice differs from the most common choice in its orbit. Needs the saved jobs in
results/learned/ from a learned run."""

import pickle

import numpy as np

from experiments.common import auroc
from experiments.learned import ALL, DONE, KINDS, SEEDS
from invariantlens.decisions import make_bank

if __name__ == "__main__":
    best = {shift: make_bank(**kw).best for shift, kw in ALL.items() if shift != "input noise"}
    best["input noise"] = best["none"]
    for shift in ALL:
        for kind in KINDS:
            aucs, wrong, flag = [], [], []
            for seed in SEEDS:
                returns = pickle.loads((DONE / f"{kind}_{seed}_{shift}.pkl").read_bytes())[1]
                top = np.sort(returns, -1)
                margin = top[..., -1] - top[..., -2]
                choices = returns.argmax(-1)
                common = np.array([np.bincount(c, minlength=5).argmax() for c in choices])
                wrong.append(choices != best[shift][:, None])
                flag.append(choices != common[:, None])
                aucs.append(auroc(margin.max() - margin, wrong[-1]))  # the closer the call, the louder the warning
            wrong, flag = np.concatenate(wrong), np.concatenate(flag)
            hits = (wrong & flag).sum()
            print(
                f"{kind} {shift:12} {wrong.sum():5d} wrong; margin AUROC {np.nanmean(aucs):.3f}; orbit flag marks"
                f" {flag.sum()}, {hits} of them wrong, {hits / max(wrong.sum(), 1):.0%} of all mistakes"
            )
