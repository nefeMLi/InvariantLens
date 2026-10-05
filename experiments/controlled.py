"""H2: equal-size error moved from the invisible part to the visible part, and what that does to decisions."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    bootstrap,
    check_truth,
    choose,
    mean_square,
    parallel,
    read_results,
    save_figure,
    score,
    slope,
    visited,
    write_results,
)
from invariantlens.decisions import make_bank
from invariantlens.physics import ROTATIONS, N, step
from invariantlens.symmetry import average, invisible, rotate

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPSILONS = [0.01, 0.03, 0.1]  # of the RMS one-step state change
DRAWS, FEATURES = 20, 64
FIELDS = ("visible", "balance")  # balance: the visible part rotates correctly, so only the balance check sees it
OUTCOMES = {"wrong": "wrong decisions", "consistent": "share in consistent orbits", "coverage": "certificate coverage"}


def fourier_field(seed):
    """Random Fourier features over (x, v, a) with Gaussian output weights: a smooth field with no structure."""
    rng = np.random.default_rng(seed)
    freq, phase = rng.normal(size=(4 * N + 2, FEATURES)), rng.uniform(0, 2 * np.pi, FEATURES)
    weight = rng.normal(size=(FEATURES, 4 * N))

    def r(s, a):
        return (np.cos(np.concatenate([s.reshape(*s.shape[:-3], -1), a], -1) @ freq + phase) @ weight).reshape(s.shape)

    return r


def continuum(draw, mu, field="visible"):
    """alpha -> e_alpha = sqrt(1 - alpha^2) u + alpha w with u, w of unit norm under mu, and |Sw|^2 / |w|^2.
    w is V r2, or for the balance field (I - S)M r2 = V P_G r2, which rotates correctly.
    Equivariant fields have the same norm at every rotation, so only |r2|^2 needs the rotated points."""
    s, a = mu
    r1, r2 = fourier_field(2 * draw), fourier_field(2 * draw + 1)
    if field == "balance":
        r = average(r2)
        whole = symmetric = mean_square(r(s, a))
    else:
        r = r2
        whole, symmetric = mean_square(r2(rotate(s), rotate(a))), mean_square(average(r2)(s, a))
    hidden = mean_square(invisible(r2)(s, a))  # (I - V) P_G = I - V, so this is the same for both fields
    nu, nw = np.sqrt(mean_square(invisible(r1)(s, a))), np.sqrt(whole - hidden)

    def e(alpha):
        cu, cw = np.sqrt(1 - alpha**2) / nu, alpha / nw
        part = invisible(lambda s, a: cu * r1(s, a) - cw * r2(s, a))  # (I - V) is linear, and (I - V)w = 0
        return lambda s, a: cw * r(s, a) + part(s, a)

    return e, (whole - symmetric) / (whole - hidden)


def check_injection(mu, m=100):
    """The injection gate on m points of mu and their rotations, for both fields: |V e_alpha|^2 = alpha^2, the
    orbit-averaged defect equals twice the orbit mean of |S e|^2, and the balance field breaks no rotation."""
    base = tuple(x[:m] for x in mu)
    s, a = map(rotate, base)
    for field in FIELDS:
        e, share = continuum(0, base, field)
        assert field != "balance" or abs(share) < 1e-10, "the balance field breaks rotation"
        for alpha in ALPHAS:
            f = e(alpha)
            y = f(s, a)
            assert abs(mean_square(y - invisible(f)(s, a)) - alpha**2) < 1e-10, "visible norm is not alpha^2"
            moved = y[(np.arange(16)[:, None] + np.arange(16)) % 16]  # f at g h x, from y[h] = f(h x)
            defect = mean_square(moved - np.einsum("gij,h...j->gh...i", ROTATIONS, y))
            assert abs(defect - 2 * mean_square(y - average(f)(s, a))) < 1e-10, "defect identity fails"


def perturbed(field, size):
    return lambda s, a: step(s, a) + size * field(s, a)


def condition(field, draw, eps, bank, mu, rms):
    """Every alpha for one field draw at one error size."""
    e, share = continuum(draw, mu, field)
    return [
        {"field": field, "draw": draw, "eps": eps, "alpha": alpha, "sym_share": alpha**2 * share}
        | score(choose(perturbed(e(alpha), eps * rms), bank), bank)
        for alpha in ALPHAS
    ]


def table(rows, key):
    """key as a (draw, alpha) array."""
    by = {(r["draw"], r["alpha"]): r[key] for r in rows}
    return np.array([[by[d, alpha] for alpha in ALPHAS] for d in sorted({r["draw"] for r in rows})])


def report(rows) -> None:
    for field in FIELDS:
        print(f"\nControlled continuum, {field} field")
        for eps in sorted({r["eps"] for r in rows if r["field"] == field}):
            mine = [r for r in rows if r["field"] == field and r["eps"] == eps]
            rates = " ".join(f"{np.mean([r['wrong'] for r in mine if r['alpha'] == alpha]):.1%}" for alpha in ALPHAS)
            print(f"  eps {eps:.0%}: {sum(r['n_wrong'] for r in mine)} wrong decisions; mean rate by alpha {rates}")
            # the balance field's error rotates correctly, so all its mistakes come in whole orbits
            for key, label in OUTCOMES.items() if field == "visible" else [("wrong", OUTCOMES["wrong"])]:
                mean, lo, hi = bootstrap([slope(np.square(ALPHAS), y) for y in table(mine, key)])
                print(f"    slope of {label} on alpha^2: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            zero = [0] if field == "visible" else ALPHAS
            bound = sum(r["coverage"] * r["n_wrong"] for r in mine if r["alpha"] in zero and r["n_wrong"])
            print(f"    certificate at alpha in {zero}: {bound:.0f} (0 expected)")


def plot(rows) -> None:
    """Visible field: every draw in grey and the mean in blue. Balance field: its mean wrong-decision rate in orange;
    its mistakes always come as whole orbits, so the other two rows would be flat."""
    epsilons = sorted({r["eps"] for r in rows})
    fig, axes = plt.subplots(3, len(epsilons), figsize=(3.4 * len(epsilons), 8), sharex=True, layout="constrained")
    axes = np.asarray(axes).reshape(3, -1)
    x = np.square(ALPHAS)
    lines = {"visible": ("tab:blue", "visible part breaks rotation"), "balance": ("tab:orange", "balance check only")}
    for col, eps in enumerate(epsilons):
        for row, (key, label) in enumerate(OUTCOMES.items()):
            ax = axes[row, col]
            for field, (colour, name) in lines.items():
                mine = [r for r in rows if r["field"] == field and r["eps"] == eps]
                if mine and (field == "visible" or key == "wrong"):
                    curves = table(mine, key)
                    if field == "visible":
                        ax.plot(x, curves.T, color="grey", lw=0.6, alpha=0.5)
                    ax.plot(x, np.nanmean(curves, 0), "o-", color=colour, lw=2, ms=4, label=name)
            ax.set(ylabel=label if col == 0 else None, title=f"error {eps:.0%}" if row == 0 else None)
        axes[-1, col].set_xlabel("visible share α²")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Equal one-step error, moved from invisible to visible")
    save_figure(fig, "controlled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, nargs="+", default=EPSILONS, help="the go/no-go runs 0.03 alone")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS)
    parser.add_argument("--situations", type=int, default=500)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.figures_only:
        rows = read_results("controlled")
    else:
        bank = make_bank(args.situations)
        check_truth(bank)
        s, a, s1 = visited(bank)
        mu = s, a
        check_injection(mu)
        rms = np.sqrt(mean_square(s1 - s))
        tasks = [(f, d, eps, bank, mu, rms) for f in args.fields for eps in args.eps for d in range(args.draws)]
        rows = [row for part in parallel(condition, tasks) for row in part]
        write_results(rows, "controlled")
    report(rows)
    plot(rows)
