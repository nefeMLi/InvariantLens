"""H2: an error of fixed size, moved from the invisible part to the visible part, and what that does to decisions."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    bootstrap,
    check_truth,
    choose,
    mean_square,
    parallel,
    save_figure,
    saved_or_run,
    score,
    slope,
    visited,
)
from invariantlens.decisions import make_bank
from invariantlens.physics import ROTATIONS, N, step
from invariantlens.symmetry import average, invisible, rotate

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPSILONS = [0.01, 0.03, 0.1]  # error sizes, as a share of the RMS one-step change
DRAWS, FEATURES = 20, 64
# in the balance field the visible part turns with the scene, so only the balance check sees it
FIELDS = ("visible", "balance")
OUTCOMES = {"wrong": "wrong decisions", "consistent": "share in consistent orbits", "coverage": "certificate coverage"}


def fourier_field(seed):
    """A smooth random field over (x, v, a): random Fourier features with Gaussian output weights."""
    rng = np.random.default_rng(seed)
    freq = rng.normal(size=(4 * N + 2, FEATURES))
    phase = rng.uniform(0, 2 * np.pi, FEATURES)
    weight = rng.normal(size=(FEATURES, 4 * N))

    def field(s, a):
        inputs = np.concatenate([s.reshape(*s.shape[:-3], -1), a], -1)
        return (np.cos(inputs @ freq + phase) @ weight).reshape(s.shape)

    return field


def continuum(draw, mu, field="visible"):
    """The errors e(alpha) = sqrt(1 - alpha^2) u + alpha w for one draw, with u invisible, w visible and both of unit
    norm under mu, and the share of w that breaks rotation. For the balance field w = (I - S)M r2."""
    s, a = mu
    r1, r2 = fourier_field(2 * draw), fourier_field(2 * draw + 1)
    if field == "balance":
        r = average(r2)
        whole = symmetric = mean_square(r(s, a))
    else:
        # only r2 itself needs the rotated points; the averaged fields have the same norm at every rotation
        r = r2
        whole = mean_square(r2(rotate(s), rotate(a)))
        symmetric = mean_square(average(r2)(s, a))
    hidden = mean_square(invisible(r2)(s, a))  # the same for both fields
    norm_u = np.sqrt(mean_square(invisible(r1)(s, a)))
    norm_w = np.sqrt(whole - hidden)

    def e(alpha):
        cu, cw = np.sqrt(1 - alpha**2) / norm_u, alpha / norm_w
        # invisible() is linear: this adds u, and takes the invisible part of r2 back out of r, which leaves w
        part = invisible(lambda s, a: cu * r1(s, a) - cw * r2(s, a))
        return lambda s, a: cw * r(s, a) + part(s, a)

    return e, (whole - symmetric) / (whole - hidden)


def check_injection(mu, m=100):
    """The injection check from HYPOTHESES.md, for both fields, on m points of mu in all 16 rotations."""
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
    def model(s, a):
        return step(s, a) + size * field(s, a)

    return model


def condition(field, draw, eps, bank, mu, rms):
    """Every alpha for one field and draw at one error size."""
    e, share = continuum(draw, mu, field)
    rows = []
    for alpha in ALPHAS:
        choices = choose(perturbed(e(alpha), eps * rms), bank)
        row = {"field": field, "draw": draw, "eps": eps, "alpha": alpha, "sym_share": alpha**2 * share}
        row.update(score(choices, bank))
        rows.append(row)
    return rows


def run(situations, fields, epsilons, draws):
    """Every field, error size, draw and alpha, on one bank."""
    bank = make_bank(situations)
    check_truth(bank)
    s, a, s1 = visited(bank)
    mu = s, a
    check_injection(mu)
    rms = np.sqrt(mean_square(s1 - s))
    tasks = [(f, d, eps, bank, mu, rms) for f in fields for eps in epsilons for d in range(draws)]
    return [row for part in parallel(condition, tasks) for row in part]


def table(rows, key):
    """rows[key] as an array of draws by alphas."""
    by = {(r["draw"], r["alpha"]): r[key] for r in rows}
    draws = sorted({r["draw"] for r in rows})
    return np.array([[by[d, alpha] for alpha in ALPHAS] for d in draws])


def report(rows):
    for field in FIELDS:
        print(f"\nControlled continuum, {field} field")
        share = np.mean([r["sym_share"] for r in rows if r["field"] == field and r["alpha"] == 1])
        print(f"  rotation-breaking share of the visible part: {share:.1%}")

        for eps in sorted({r["eps"] for r in rows if r["field"] == field}):
            mine = [r for r in rows if r["field"] == field and r["eps"] == eps]
            rates = [np.mean([r["wrong"] for r in mine if r["alpha"] == alpha]) for alpha in ALPHAS]
            rates = " ".join(f"{rate:.1%}" for rate in rates)
            print(f"  eps {eps:.0%}: {sum(r['n_wrong'] for r in mine)} wrong decisions; mean rate by alpha {rates}")

            # the balance field turns with the scene, so all its mistakes come in whole orbits
            outcomes = OUTCOMES if field == "visible" else {"wrong": OUTCOMES["wrong"]}
            for key, label in outcomes.items():
                mean, low, high = bootstrap([slope(np.square(ALPHAS), y) for y in table(mine, key)])
                print(f"    slope of {label} on alpha^2: {mean:+.3f} [{low:+.3f}, {high:+.3f}]")

            zero = [0] if field == "visible" else ALPHAS
            bound = sum(r["coverage"] * r["n_wrong"] for r in mine if r["alpha"] in zero and r["n_wrong"])
            print(f"    certificate at alpha in {zero}: {bound:.0f} (0 expected)")


def plot(rows):
    epsilons = sorted({r["eps"] for r in rows})
    fig, axes = plt.subplots(
        3, len(epsilons), figsize=(3.4 * len(epsilons), 8), sharex=True, layout="constrained", squeeze=False
    )
    x = np.square(ALPHAS)
    for col, eps in enumerate(epsilons):
        visible = [r for r in rows if r["field"] == "visible" and r["eps"] == eps]
        balance = [r for r in rows if r["field"] == "balance" and r["eps"] == eps]
        for row, (key, label) in enumerate(OUTCOMES.items()):
            ax = axes[row, col]
            curves = table(visible, key)
            ax.plot(x, curves.T, color="grey", lw=0.6, alpha=0.5)
            ax.plot(x, np.nanmean(curves, 0), "o-", label="visible part breaks rotation")
            ax.set(ylabel=label if col == 0 else None, title=f"error {eps:.0%}" if row == 0 else None)
        if balance:
            # its mistakes always come in whole orbits, so only its rate of wrong decisions is worth drawing
            axes[0, col].plot(x, np.nanmean(table(balance, "wrong"), 0), "o-", label="balance check only")
        axes[-1, col].set_xlabel("visible share α²")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Equal one-step error, moved from invisible to visible")
    save_figure(fig, "controlled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, nargs="+", default=EPSILONS, help="0.03 alone is the go/no-go run")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS)
    parser.add_argument("--situations", type=int, default=500)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    rows = saved_or_run("controlled", args.figures_only, run, args.situations, args.fields, args.eps, args.draws)
    report(rows)
    plot(rows)
