"""Shared bank checks, scoring, statistics, results files and figures for the experiments."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from invariantlens.decisions import reward, rollout
from invariantlens.physics import step
from invariantlens.symmetry import certificate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESAMPLES = 1000


def check_truth(bank) -> None:
    """The truth gate: returns agree across each orbit to 1e-9 relative, and the truth always picks the best."""
    assert np.all(np.abs(bank.returns - bank.returns[:, :1]) <= 1e-9 * np.abs(bank.returns[:, :1])), "orbits disagree"
    assert (bank.returns.argmax(-1) == bank.best[:, None]).all(), "the truth misses the best candidate"


def visited(bank, truth=step):
    """(s, a, F(s, a)) along every true candidate rollout, unrotated; mu is (s, a) in all 16 rotations."""
    states = rollout(truth, bank.state[:, 0], bank.actions[:, 0])
    s, s1 = (x.reshape(-1, *states.shape[-3:]) for x in (states[..., :-1, :, :, :], states[..., 1:, :, :, :]))
    return s, bank.actions[:, 0].reshape(-1, 2), s1


def mean_square(x):
    return np.square(x).sum((-3, -2, -1)).mean()


def chunks(bank, size=50):
    """Slices over the situations, so that rollouts of all their candidates fit in memory."""
    return [slice(i, i + size) for i in range(0, len(bank.best), size)]


def choose(model, bank):
    """The model's choice on every bank entry, (n, 16)."""
    returns = [reward(rollout(model, bank.state[p], bank.actions[p]), bank.goal[p]) for p in chunks(bank)]
    return np.concatenate(returns).argmax(-1)


def score(choices, bank) -> dict:
    """Wrong-decision rate, normalised regret, and how the wrong decisions sit in their orbits."""
    wrong, r, bound = choices != bank.best[:, None], bank.returns, certificate(choices)
    chosen = np.take_along_axis(r, choices[..., None], -1)[..., 0]
    n = int(wrong.sum())
    return {
        "wrong": float(wrong.mean()),
        "n_wrong": n,
        "regret": float(((r.max(-1) - chosen) / (r.max(-1) - r.min(-1))).mean()),
        "consistent": float(wrong[bound == 0].sum() / n) if n else float("nan"),  # share in orbits that never disagree
        "coverage": float(bound.sum() / n) if n else float("nan"),  # certificate over the true count
        "by_situation": wrong.sum(1).tolist(),  # for bootstraps over situations
    }


def auroc(signal, wrong) -> float:
    """How often a wrong decision gets a higher signal than a right one (ties count half); nan if undefined."""
    signal, wrong = np.ravel(signal), np.ravel(wrong)
    n = int(wrong.sum())
    if signal.max() < 1e-12 or n in (0, wrong.size):
        return float("nan")
    return float((rankdata(signal)[wrong].sum() - n * (n + 1) / 2) / (n * (wrong.size - n)))


def regret(returns, choices):
    """How much further from the goal each choice ends than the best candidate would."""
    return returns.max(-1) - np.take_along_axis(returns, choices[..., None], -1)[..., 0]


def slope(x, y) -> float:
    """Least-squares slope of y on x over the finite y; nan with fewer than two."""
    ok = np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 1 else float("nan")


def bootstrap(values, statistic=np.mean, seed=0) -> tuple[float, float, float]:
    """The statistic over the rows of values, with a 95% interval from resampling rows; all-nan rows are dropped."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v).reshape(len(v), -1).any(-1)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    draws = [statistic(v[i]) for i in np.random.default_rng(seed).integers(0, len(v), (RESAMPLES, len(v)))]
    return float(statistic(v)), *np.nanpercentile(draws, [2.5, 97.5]).tolist()


def cached(path, fn, *args):
    """fn(*args), a dict of arrays, saved to path as soon as it finishes, so an interrupted run resumes. Stored as plain
    arrays, so loading runs no code."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = fn(*args)
        part = path.with_suffix(".part")  # renamed only once complete, so a killed run leaves no broken file
        with part.open("wb") as f:  # a file object, so numpy doesn't add .npz to the name
            np.savez_compressed(f, **arrays)
        part.replace(path)
    with np.load(path) as saved:
        return dict(saved)


def parallel(fn, tasks):
    """fn(*task) for every task, across processes, in task order."""
    with ProcessPoolExecutor() as pool:
        yield from (future.result() for future in [pool.submit(fn, *task) for task in tasks])


def write_results(rows: list[dict], name: str) -> None:
    """Save rows to results/<name>.parquet; a key a row doesn't have is stored as null."""
    (ROOT / "results").mkdir(exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    pq.write_table(
        pa.Table.from_pylist([{k: row.get(k) for k in keys} for row in rows]), ROOT / "results" / f"{name}.parquet"
    )


def read_results(name: str) -> list[dict]:
    return pq.read_table(ROOT / "results" / f"{name}.parquet").to_pylist()


def save_figure(fig, name: str) -> None:
    """figures/<name>.svg, identical across reruns."""
    (ROOT / "figures").mkdir(exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(ROOT / "figures" / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
