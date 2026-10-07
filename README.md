# InvariantLens

[![tests](https://github.com/nefeMLi/InvariantLens/actions/workflows/tests.yml/badge.svg)](https://github.com/nefeMLi/InvariantLens/actions/workflows/tests.yml)

A world model predicts what happens next, and an agent uses those predictions to decide what to do. The physics it
models has a symmetry: turn the scene and the outcome turns with it. A network that isn't built with that symmetry
can learn it from data, but does what it learned still hold when the agent ends up somewhere the data never went?
Where it breaks, the error shows up when you ask the model the same question from different angles, and averaging
over the angles removes it, with no labels and no retraining. Where it holds, the error follows the rules and
nothing computed from the model can find it. This project asks **where a learned symmetry breaks outside the data,
and when averaging over it repairs the agent's decisions.**

Version 1 built the tools: an exact split of a world model's error into the part that breaks the rules (visible) and
the part that doesn't (invisible), a way to subtract the visible part, and tests of what each does to decisions.
Version 2 uses them on the question above, with predictions fixed on development data and tested once on new data.

## In short

- **Version 1: the visible part can be measured and removed without labels.** Far from its training data, most of
  a world model's error turned visible (65 to 88%), and subtracting it cut wrong decisions from 0.39% to 0.08% for
  one model and from 0.28% to none for another.
- **Version 1: a change in the physics is invisible to every check on the model.** Its error follows the rules, so
  the checks, like an ensemble, flag the bad decisions barely better than chance (AUROC 0.50 to 0.60).
- **Version 2: where the learned symmetry breaks.** On development data the symmetry broke when the scene was
  moved far away, a new frame for familiar physics, and averaging removed all 21 costly decisions. It held when the
  discs moved five times faster, physics the model never saw, and averaging removed 1 of 49. The prediction, to be
  tested on new data: a learned symmetry breaks in a new frame, where averaging repairs it, and holds for new
  physics, where nothing computed from the model can help.

![What the label-free signals see, and what correcting the visible error buys](figures/learned.svg)

*Left: one point per trained model and condition, sized by how many mistakes its AUROC rests on. Right: wrong
decisions before and after the correction and the vote.*

## How I tested it

The test bed is a small 2D push task where I know the exact answer. Five discs attract and repel each other. The agent
pushes one of them to knock a target disc towards a goal, picking the best of five candidate pushes by the world
model's predictions. Since I can simulate every candidate for real, I know every time it picks wrong. Every situation
also comes in 16 copies, rotated by multiples of 22.5°, all with the same right answer. If the model picks different
pushes for the copies, some of those picks must be wrong, and that check needs no labels.

- **Controlled.** I take the true simulator, add an error of fixed size, and slide it from fully invisible to fully
  visible. I do this for an error that breaks rotation and for one that only breaks the momentum rules.
- **Learned.** I train three kinds of network, five copies each. M1 sees absolute coordinates, M2 is M1 trained on
  randomly rotated data, and M3 turns with the scene by construction. The five M1s also form an ensemble. I test them
  on the original situations, under changes they weren't trained for (stronger forces, friction between discs, faster
  starts, noisy observations), and with the whole scene moved away from the origin, which changes nothing in the
  physics but takes M1 and M2 off their training data.

I wrote my predictions down before training any model, in [HYPOTHESES.md](HYPOTHESES.md). The file was first
committed on 4 October, after the controlled run and the training, so for those its log, not the git history, is the
record of what I had seen. The log also says which later predictions were committed only after their results.

## What I found

**Visible or invisible, error of the same size does the same damage.** As a rotation-breaking error slides from
invisible to visible, the wrong-decision rate stays flat (about 2.9% at 3% error, slope +0.002 [−0.002, +0.006]). Only
the pattern changes. Invisible error makes the same mistake in all 16 copies, where no check can catch it; with just 6%
of it visible, two thirds or more of the mistakes scatter, and fully visible error is counted almost entirely by the
label-free bound. Error that only breaks the momentum rules even helps (26.5% to 21.0% wrong at 10% error, slope
−0.056 [−0.074, −0.035]): it moves every disc by the same offset, and with forces that depend only on relative
positions that's a harmless change of reference frame. With walls or gravity it wouldn't be.

![Equal one-step error, moved from invisible to visible](figures/controlled.svg)

*Rows: wrong decisions, the share of them that come as whole orbits, and the share the bound catches. Grey lines are
the 20 random errors.*

**Under a physics change or a noisy sensor, the checks are blind, as they have to be.** The models are nearly perfect
on the situations they trained on (M1: 5 wrong decisions out of 40,000), but pick wrong 16% of the time with 1.5 times
stronger forces and 11 to 12% with friction. These changes keep the rules, so their error is invisible: under 0.2% of
it shows, 96 to 100% of the mistakes come as whole orbits, and the bound catches at most 2%. The ensemble is just as
blind, since ensembles notice unfamiliar inputs, not familiar ones whose outcome has changed. Under noisy observations
even the true simulator makes 128 wrong decisions, as many as M1 and M2, so those mistakes aren't the model's at all.

**Far from the training data, the error turns visible and the correction works.** Moving every situation 5 units from
the origin changes nothing for the true physics or for M3 (112 mistakes, as before). M1 and M2 read absolute
coordinates, so 65 to 88% of their error turns visible: all 267 of their mistakes scatter across their orbits, the
bound counts 90 to 97% of them, and the correction removes most. But the size of the symmetry signal doesn't single
out the bad decisions (AUROC 0.53 [0.49, 0.57]), because it is high everywhere out there. Flagging each copy that
chose differently from the rest of its orbit does: it catches 82% of the mistakes, and 218 of its 246 flags are right.

**The model's margin looked like the best warning, but under the physics changes that came from the bank.** AUROC
for flagging wrong decisions:

| Condition | Symmetry and balance signals | Ensemble | Predicted margin |
|---|---|---|---|
| Original situations | 0.86 to 0.90 for M1 and M2 (1 or 2 mistakes each); 0.93 for M3's balance signal (about 22) | no mistakes | 0.996 to 1.000 |
| Stronger forces | 0.57 to 0.59 | 0.50 | 0.82 to 0.83 |
| Friction | 0.57 | 0.56 | 0.84 to 0.85 |
| Noisy observations | 0.54 to 0.60 | 0.53 | 0.87 to 0.89 |
| Far from the origin | 0.53 to 0.62 (M1 and M2) | 0.37 (7 mistakes) | 0.97 to 1.00 |

The bank only keeps situations whose best push wins by at least 5%, judged by the shifted truth, and that choice
does the work. On a later bank that keeps near-ties, every model made the same decisions under 1.5 times stronger
forces whether corrected or not, and the margin flagged the costly ones at chance (AUROC 0.48). Nothing computed from
the model alone can see a change in the physics.

## The predictions, scored

| Prediction | Result |
|---|---|
| H1: the identities hold, and M3 is exactly symmetric | held |
| H2a: as the error turns visible, mistakes scatter across orbits | held, but close to guaranteed by H1 |
| H2b: the wrong-decision rate changes too | failed at all three error sizes |
| H2c: the same, for error that only breaks the momentum rules | held: the rate falls |
| H3: the visible share of a model's error explains how well the signals work | failed: Spearman +0.06 [−0.12, +0.20] and +0.18 [−0.02, +0.39] |
| H4: the correction removes wrong decisions on the original situations | failed: there were about 5 to remove |
| H5: rotation breaks as much at 100 random angles as at the 16 checked | the same at both (4.9·10⁻⁵ for M1, 3.3·10⁻⁵ for M2) |
| H6a, H6b: M1's mistakes scatter, and the vote removes them | held, on 5 and 6 mistakes |
| Noisy observations, four predictions | held, but committed after the results |
| H7a: moving the scene changes nothing for the truth or M3 | held |
| H7b: far away, the symmetry signal flags mistakes (AUROC above 0.7) | failed: 0.53 |
| H7c, H7d: far away, mistakes scatter and the correction removes them | held at 5 units; at 2.5 (13 mistakes) H7c can't tell and H7d failed |
| H7e: the ensemble does better far away than under physics changes | failed, on 7 mistakes |

The failures have one cause in common: a decision flips because it is a close call, not because the error is large.
H3 also looked supported at first, but the support came from a bug: my first noise
test gave each rotated copy its own noise, so the vote was averaging 16 observations. I fixed it after seeing the
results, logged it, and kept the old numbers in the git history.

## What I'd push back on, if I were reviewing this

- **The physics-shift result is close to guaranteed.** A change that keeps the rules must put its error in the
  invisible part; the experiment only measures how much it matters.
- **The margin wasn't pre-registered,** and its good scores under the physics changes came from how the bank was
  built. The label-free bound also relies on that bank: it assumes one clearly best push, which a real agent couldn't
  check.
- **It's a toy world.** Five discs in 2D, a known reward, models that see states, not images. The momentum rules need
  open space and equal, known masses; with gravity or contact, as in MuJoCo, only the rotation check applies.
- **The evidence on the original situations is thin.** The models make 5 to 112 mistakes per 40,000 decisions there.
- **Some comparisons are weaker than they look.** The controlled errors are random smooth fields, and real model
  errors can look different. M3 differs from M1 in more than symmetry. "Speed × 2" was a mild shift: only 4% of the
  faster starts go beyond the speeds seen in training.
- **The pieces are known.** Splitting an error by symmetry, disagreement under rotation, and the margin as a confidence
  score all exist already. What's new is putting them side by side on decisions.

## Version 2: where does a learned symmetry break?

M1 and M2 learned that the physics turns with the scene from data. Outside the data, networks become close to linear
along each direction (Xu et al., 2021). A quantity the physics ignores, like where the scene is, gets a small,
arbitrary dependence in training that grows outside the data with nothing to make it turn with the scene, so the
learned symmetry breaks and averaging cancels the error. A quantity the physics uses, like how fast two discs close
in, is learned from data that look the same from every angle, so the network's guess stays nearly symmetric even
where it is wrong, and the error is invisible. In short: **the symmetry breaks in a new frame and holds for new
physics.**

The test uses four ways out of the training data, with the physics unchanged, on 100 new situations each:

| Shift | What the model meets | Prediction |
|---|---|---|
| Scene 10 units from the origin | familiar physics, new frame | breaks, averaging repairs (H8, seen on development data) |
| Starting speeds five times the usual | new physics | holds, averaging doesn't repair (H8, seen on development data) |
| Every disc drifting at the same extra speed | familiar physics, new frame | breaks, averaging repairs (H9, unseen) |
| One disc starting closer to the target than any pair in training | new physics | holds, averaging doesn't repair (H9, unseen) |

Drift is the sharpest test: its velocities are as far outside training as the fast discs', so if it isn't repaired,
the explanation is wrong and large velocities themselves give invisible error. Two comparisons are reported beside
the predictions: whether M2, which learned the symmetry from turned data, keeps it better than M1; and whether asking
the model once in a standard pose, which makes it exactly symmetric at a sixteenth of the cost, repairs as much as
averaging. If it doesn't, the repair comes from cancelling the broken part, not from being symmetric. Thresholds and
the full reasoning are in [HYPOTHESES.md](HYPOTHESES.md), committed before the test run. Results to follow.

## Why the split is exact

Write F for the true one-step map (state and push in, next state out), F̂ for the model, and e = F̂ − F for its error.
Errors are measured in mean square over the states and pushes seen in the true rollouts, in all 16 rotations, so the
measure looks the same from every angle.

Two operations do the work. 𝒫_G asks the model about all 16 rotations of a situation, turns each answer back, and
averages them; S = I − 𝒫_G is what breaks rotation. M takes the average position error and average velocity error
over the discs and gives that average to every disc. Both are orthogonal projections, and they commute, because
turning every disc the same way and averaging over discs can be done in either order. So the error splits into four
pieces that don't overlap:

| | Follows the momentum rules | Breaks them |
|---|---|---|
| **Turns with the scene** | (I − S)(I − M)e: invisible | (I − S)Me: seen by the momentum check only |
| **Breaks rotation** | S(I − M)e: seen by the rotation check only | SMe: seen by both |

**The visible part needs no ground truth.** The true physics turns with the scene, so Se = SF̂. The discs have equal
mass, pair forces are equal and opposite, and the push is the only outside force, so over one decision step Δt the
totals P = Σvᵢ and X = Σxᵢ follow P(F(s, a)) = P(s) + Δt·a and X(F(s, a)) = X(s) + Δt·P(s) + (Δt²/2)·a, exactly
for velocity Verlet too. So Me also comes from F̂, s and a alone.

**The visible part can be removed.** The corrected model 𝒫_G(F̂ − Me) shifts every disc by the same amount so both
rules hold, then averages over the 16 rotations, at the cost of 16 model calls per step. Its error is exactly the
invisible piece. Neither step makes the error bigger on average over the 16 rotations, though the averaging can at a
single one.

**A label-free bound on wrong decisions.** The right answer is the same in all 16 rotations, so if the model's
choices have counts n₁, …, n₅, at least 16 − max n_k of them are wrong. A model whose error turns with the scene makes
the same choice on all 16, so for it the bound is zero and its mistakes come as whole orbits.

## Details

Every setting is frozen in [HYPOTHESES.md](HYPOTHESES.md). In brief: five unit-mass discs with Morse forces in open
space, integrated in float64 with velocity Verlet; 500 situations, each in 16 rotations, with 5 candidate pushes of 10
decision steps, kept only if the best push wins by 5%; controlled errors from random Fourier features at 1%, 3% and
10% of a typical one-step change, 20 draws each; message-passing networks of three layers of width 64, parameter
counts within 1% of each other, trained for 200 epochs on 50,000 random transitions and evaluated in float64; input
noise of standard deviation 0.02, one draw per situation, turned with each copy; and scenes moved 2.5 or 5 units along
x, where training positions reach about 3 to 4 units out.

## Background

Splitting a function into the part that respects a symmetry and the part that breaks it goes back to Elesedy and
Zaidi (ICML 2021), and Wang et al. (NeurIPS 2023) studied what goes wrong when the assumed symmetry is wrong.
Disagreement under transformed inputs is a known label-free uncertainty signal (Ayhan and Berens, MIDL 2018). M3
follows the EGNN of Satorras, Hoogeboom and Welling (ICML 2021), and the ensemble baseline is Lakshminarayanan,
Pritzel and Blundell (NeurIPS 2017); Ovadia et al. (NeurIPS 2019) showed how such uncertainty holds up under dataset
shift. The margin is the decision-making version of the maximum-softmax baseline of Hendrycks and Gimpel (ICLR 2017).

Symmetry learned from data is known to be unreliable under distribution shift (Moskalev et al., ICML TAG-ML 2023),
and Gruver et al. (ICLR 2023) measure how much a trained network breaks a symmetry; the visible share is the
finite-group version of that measure. Xu et al. (ICLR 2021) showed that networks extrapolate close to linearly.
Averaging or canonicalising a trained model over a group at test time makes it exactly symmetric (Puny et al., ICLR
2022; Kim et al., NeurIPS 2023; Mondal et al., NeurIPS 2023); version 2 asks which extrapolation errors that removes.
Enforcing conservation laws on a model's predictions at inference, as the balance fix does, follows Hansen et al.
(ICML 2023).

## Running it

Python 3.14:

```sh
pip install -r requirements.txt
pytest tests.py
python -m experiments.controlled    # --eps 0.03 runs only the go/no-go size, --fields one kind of error
python -m experiments.learned       # trains 15 models on the CPU, then evaluates them: the slow part
python -m experiments.posthoc       # the unplanned analyses, from the saved evaluations
python -m experiments.diagnose --split test   # version 2; --split dev is the development data
```

The trained models and the saved evaluations are in the [v1 release](https://github.com/nefeMLi/InvariantLens/releases/tag/v1)
rather than the repo. Unzip it in the repo root and `posthoc.py` runs straight away, while `learned.py` skips the
training and every finished evaluation:

```sh
curl -LO https://github.com/nefeMLi/InvariantLens/releases/download/v1/results.zip && unzip results.zip
```

The controlled experiment took about 6.5 hours on 4 CPU cores. Training takes 3 to 4 hours per model on one core,
with the 15 in parallel, and the evaluation about 3 hours on one T4 GPU (`--device` to choose; use a CUDA build of the
same PyTorch version). Version 2 takes about an hour on 4 CPU cores. An interrupted run picks up where it stopped; delete `results/learned/`
or `results/diagnose/` after changing the code.
`--figures-only` redraws the figures from `results/*.parquet`.

The tests check every identity above, and every check HYPOTHESES.md says must pass first. The code is in
`invariantlens/` (the simulator, the decision bank, the projections, the correction and the standard pose, and the
networks), and the
experiments are in `experiments/`. A quick example:

```python
import numpy as np
from invariantlens.physics import initial_state, step, uniform_disc
from invariantlens.symmetry import corrected, signals

rng = np.random.default_rng(0)
s, a = initial_state(rng), uniform_disc(rng, 2.0, ())


def model(s, a):  # the true simulator plus a small constant drift to the right
    return step(s, a) + 0.01 * np.array([1.0, 0.0])


symmetry, balance = signals(model, s, a)  # computed without the truth
print(f"{symmetry:.4f} {balance:.4f}")  # 0.0316 0.0707: the drift breaks both rules
print(np.abs(corrected(model)(s, a) - step(s, a)).max())  # 4.4e-16: all of this error was visible
```

## License

MIT, see [LICENSE](LICENSE).
