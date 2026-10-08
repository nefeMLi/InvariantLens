"""Helpers shared by the experiments: models, checks, scoring, statistics, caching, results files and figures."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata

from invariantlens.decisions import reward, rollout
from invariantlens.models import Net, train, transitions, world_model
from invariantlens.physics import step
from invariantlens.symmetry import certificate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"
RESAMPLES = 1000


def fit(kind, seed):
    """Train one model, unless it is already saved."""
    path = MODELS / f"{kind}_{seed}.pt"
    if not path.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(train(kind, seed, transitions(5000, 0), transitions(500, 1)).state_dict(), path)


def load(kind, seed, device):
    net = Net(kind == "M3", torch.zeros(5))
    net.load_state_dict(torch.load(MODELS / f"{kind}_{seed}.pt"))
    return world_model(net, device)


def check_truth(bank):
    """The true simulator must agree with itself across every orbit, and always pick the best candidate."""
    first = bank.returns[:, :1]
    assert np.all(np.abs(bank.returns - first) <= 1e-9 * np.abs(first)), "orbits disagree"
    assert (bank.returns.argmax(-1) == bank.best[:, None]).all(), "the truth misses the best candidate"


def visited(bank, truth=step):
    """(s, a, s') at every step of every candidate's true rollout, unrotated."""
    states = rollout(truth, bank.state[:, 0], bank.actions[:, 0])
    s = states[..., :-1, :, :, :].reshape(-1, *states.shape[-3:])
    s1 = states[..., 1:, :, :, :].reshape(-1, *states.shape[-3:])
    return s, bank.actions[:, 0].reshape(-1, 2), s1


def mean_square(x):
    return np.square(x).sum((-3, -2, -1)).mean()


def chunks(bank, size=50):
    """Slices of the bank small enough for the rollouts of all their candidates to fit in memory."""
    return [slice(i, i + size) for i in range(0, len(bank.best), size)]


def choose(model, bank):
    """The candidate the model picks in every situation and rotation, shape (n, 16)."""
    returns = []
    for part in chunks(bank):
        states = rollout(model, bank.state[part], bank.actions[part])
        returns.append(reward(states, bank.goal[part]))
    return np.concatenate(returns).argmax(-1)


def score(choices, bank):
    """How often the choices are wrong, their regret, and how the wrong ones sit in their orbits."""
    wrong = choices != bank.best[:, None]
    bound = certificate(choices)
    r = bank.returns
    chosen = np.take_along_axis(r, choices[..., None], -1)[..., 0]
    n = int(wrong.sum())
    return {
        "wrong": float(wrong.mean()),
        "n_wrong": n,
        "regret": float(((r.max(-1) - chosen) / (r.max(-1) - r.min(-1))).mean()),
        # share of the wrong decisions that sit in orbits where all 16 choices agree
        "consistent": float(wrong[bound == 0].sum() / n) if n else float("nan"),
        # how much of the true number of wrong decisions the certificate finds
        "coverage": float(bound.sum() / n) if n else float("nan"),
        "by_situation": wrong.sum(1).tolist(),
    }


def auroc(signal, wrong):
    """How often a wrong decision gets a higher signal than a right one, ties counting half. nan if undefined."""
    signal, wrong = np.ravel(signal), np.ravel(wrong)
    n = int(wrong.sum())
    if signal.max() < 1e-12 or n in (0, wrong.size):
        return float("nan")
    return float((rankdata(signal)[wrong].sum() - n * (n + 1) / 2) / (n * (wrong.size - n)))


def regret(returns, choices):
    """How much further from the goal each choice ends than the best candidate would."""
    chosen = np.take_along_axis(returns, choices[..., None], -1)[..., 0]
    return returns.max(-1) - chosen


def slope(x, y):
    """Least-squares slope of y on x over the finite values of y."""
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def bootstrap(values, statistic=np.mean, seed=0):
    """The statistic and a 95% interval from resampling the rows of values. Rows that are all nan are dropped."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v).reshape(len(v), -1).any(-1)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    picks = np.random.default_rng(seed).integers(0, len(v), (RESAMPLES, len(v)))
    draws = [statistic(v[i]) for i in picks]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(statistic(v)), float(low), float(high)


def cached(path, fn, *args):
    """fn(*args), a dict of arrays, saved to path when it finishes so that a stopped run can pick up again."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = fn(*args)
        # write to a temporary name first, so a killed run never leaves half a file
        part = path.with_suffix(".part")
        with part.open("wb") as f:
            np.savez_compressed(f, **arrays)
        part.replace(path)
    with np.load(path) as saved:
        return dict(saved)


def spawned(workers=None):
    """A pool of spawned processes with one thread each. CUDA can't start in a forked process."""
    spawn = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(workers, spawn, initializer=torch.set_num_threads, initargs=(1,))


def parallel(fn, tasks):
    """fn(*task) for each task, run across processes, yielded in order."""
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(fn, *task) for task in tasks]
        for future in futures:
            yield future.result()


def write_results(rows, name):
    """Save rows to results/<name>.parquet. A key that a row doesn't have is saved as null."""
    (ROOT / "results").mkdir(exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    table = pa.Table.from_pylist([{key: row.get(key) for key in keys} for row in rows])
    pq.write_table(table, ROOT / "results" / f"{name}.parquet")


def read_results(name):
    return pq.read_table(ROOT / "results" / f"{name}.parquet").to_pylist()


def saved_or_run(name, figures_only, run, *args):
    """The rows saved in results/<name>.parquet if figures_only, otherwise run(*args), saved there."""
    if figures_only:
        return read_results(name)
    rows = run(*args)
    write_results(rows, name)
    return rows


def save_figure(fig, name):
    """Save figures/<name>.svg, byte for byte the same on every run."""
    (ROOT / "figures").mkdir(exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(ROOT / "figures" / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
