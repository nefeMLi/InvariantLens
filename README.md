# InvariantLens

[![tests](https://github.com/nefeMLi/InvariantLens/actions/workflows/tests.yml/badge.svg)](https://github.com/nefeMLi/InvariantLens/actions/workflows/tests.yml)

A world model predicts what happens next, and an agent uses those predictions to decide what to do. When the model
is wrong, can anything tell, without access to the right answer? I wanted to know how much of a world model's error
can be seen from the model alone, what that visible part does to the agent's decisions, and what is left once you
remove it.

The idea is simple. The true physics follows rules I know in advance. Turn the whole scene and the outcome turns with
it, and the total momentum and centre of mass change by amounts fixed by the push. Any part of the model's error that
breaks those rules can be measured from the model alone, by asking it the same question in different ways. Any part
that follows the rules can't be seen without the true simulator. The split between the two is exact, and the visible
part can even be subtracted.

I wrote down what I expected before training any model, in [HYPOTHESES.md](HYPOTHESES.md). Three predictions and a
few changes were added later, after the first experiment and before the trained models were evaluated. The log at
the end of that file says what I had seen at each point.

## What I tested

The test bed is a small 2D push task where I know the exact answer. Five discs attract and repel each other. The
agent pushes one of them, and has to knock a target disc towards a goal. In each situation it picks the best of five
candidate pushes using the world model's predictions. Since I can simulate every candidate for real, I know every
time it picks wrong.

Every situation also comes in 16 copies, rotated by multiples of 22.5°. The right answer is the same in all 16. That
gives a check that needs no labels: if the model picks different pushes for rotated copies of the same situation,
some of those picks must be wrong.

There are two experiments.

- **Controlled.** I take the true simulator and add an error of fixed size, then slide that error from fully
  invisible to fully visible. If decisions get worse along the way, visible error is the harmful kind; if not, it
  isn't. I do this with two kinds of visible error: one that breaks the rotation rule, and one that only breaks the
  momentum and centre-of-mass rules.
- **Learned.** I train three kinds of network as world models, five copies each, and test them on the original
  situations and under four changes they weren't trained for: stronger forces, added friction between discs, faster
  starting speeds, and noise on what they observe. One network (M1) sees absolute coordinates, one (M2) is the same
  but trained on randomly rotated data, and one (M3) is built so it always turns with the scene. Five copies of M1
  together make an ensemble, the usual baseline for uncertainty.

## What I found

**Visible and invisible error of the same size do the same damage, but only one of them shows.** When the visible
part breaks the rotation rule, the wrong-decision rate doesn't move as the error slides from invisible to visible
(at 3% error it stays at about 2.9%, slope +0.002 [−0.002, +0.006]). What changes is how the mistakes look.
Invisible error makes the same mistake in all 16 rotations, so no consistency check can catch it. Even a small
visible share scatters the mistakes: with 6% of the error visible, only a quarter to a third of them still come as
whole orbits. Once all of the error is visible, the label-free check counts essentially every wrong decision.

**Error that only breaks the momentum rules is less harmful.** With the second kind of visible error, decisions get
better as more error moves into it: at 10% error the wrong-decision rate falls from 26.5% to 21.0% (slope −0.056
[−0.074, −0.035]), and at 3% from 2.8% to 2.2%. My guess at why, which I haven't tested: this kind of error shifts
every disc by the same amount. That barely changes the forces between them, and it shifts all five candidates'
outcomes alike, so it mostly cancels when the agent compares them.

![Equal one-step error, moved from invisible to visible](figures/controlled.svg)

**When the physics changes, the trained models fail and every label-free signal is close to blind.** On the
situations they were trained for, the models are nearly perfect: M1 made 5 wrong decisions out of 40,000. With 1.5
times stronger forces between the discs, every model picks wrong 16% of the time, and with added friction, 11 to
12%. The theory already says why. A change in physics that still respects rotations and momentum adds only invisible
error. The measurements agree: less than 0.2% of the error is visible, 96 to 100% of the mistakes come as whole
orbits, and the label-free check catches at most 2% of them. Both signals score an AUROC of 0.57 to 0.59, barely
above chance. The ensemble does no better (0.50 and 0.56), because all five copies learned the same old physics.

**Where there is visible error, the signals do follow it.** Across models, copies and conditions, a signal is better
at flagging wrong decisions when more of the model's error is visible: Spearman +0.68 [+0.62, +0.73] for the rotation
signal and +0.58 [+0.50, +0.67] for the momentum signal. The catch is in the figure. Point size shows how many wrong
decisions each score rests on, and the highest scores rest on one or two.

![What the label-free signals see, and what correcting the visible error buys](figures/learned.svg)

**Removing the visible error fixes what little it breaks.** The corrected models drop 53 to 57% of the one-step error,
and make no wrong decisions at all on the original situations, or with faster starting speeds. But the models only
made a handful of mistakes there to begin with, so the improvement (−0.01% [−0.05%, +0.00%]) doesn't clear the bar I
set in advance. Under the physics changes, where the mistakes actually are, the correction changes nothing, because
there is nothing visible to remove. Averaging the model's predictions over the 16 rotations before choosing does the
same, except under input noise. There it averages 16 differently noised views of the scene and cuts wrong decisions
from 1.4% to 0.4%.

**The check works beyond its 16 angles.** At 100 random angles, decisions disagree and the error breaks rotation
about as often as at the 16 checked ones, and M3 stays exactly symmetric at all of them.

## Things that didn't work out the way I expected

- I expected visible and invisible error of the same size to do different amounts of damage. When the visible part
  breaks rotation, they don't, at any error size. Only the pattern of the mistakes differs.
- I expected that pattern to show most clearly at the smallest error. By the measure I fixed in advance it showed
  least there. At 1% error many runs make no mistakes at all when the error is invisible, so their fitted slope
  misses the steepest part of the curve. The figure shows the drop is just as large as at 3%.
- I expected removing the visible error to remove wrong decisions. It removes every one it can reach, but on the
  original situations there were almost none, and under the physics changes the errors that count are the ones it
  can't see.
- I expected training on rotated data (M2) to cut the part of the error that breaks rotation. It came out at 38 to
  58% of the one-step error, against 50 to 61% for M1.
- My first version of the input-noise test added fresh noise at every step of the model's own predictions, which
  swamped all the models into identical decisions. I moved the noise to what the model observes at the start, before
  the full run.
- I expected the evaluation to take hours on my laptop. In float64 it would have taken about a day on 12 cores, so I
  ran it on a GPU instead.

## What I'd push back on, if I were reviewing this

I tried to be honest about these. Some are limits of the setup, some are limits of what the results can say.

- **The main learned result is close to guaranteed.** A change in physics that keeps the rules intact must put its
  error in the invisible part; that follows from the maths. The experiment shows how large the effect is and that an
  ensemble fails too, not that it happens.
- **It's a toy world.** Five discs in 2D, a known reward, and models that see states, not images. I haven't run the
  same check on a standard benchmark such as a MuJoCo environment. That's the obvious next step, though there only
  the rotation half of the check would apply: gravity and contact forces break the momentum rules.
- **There's no positive control yet.** I haven't tested a change that breaks the rules, such as adding walls, to show
  the signals light up when the error does become visible.
- **On the original situations the evidence is thin.** The models make 5 to 112 mistakes per 40,000 decisions there,
  so everything measured on that bank rests on a handful of them. Most of the evidence comes from the changed
  conditions.
- **Some predictions came after part of the data.** The balance-only experiment and the two predictions about how
  trained models' mistakes are spread were written after the first experiment and a small code test. They are marked
  as such in HYPOTHESES.md, but they deserve less weight than the original ones.
- **The controlled errors are random smooth fields.** Real model errors can look different.
- **The explanation for the balance-only result is untested.** It fits, but I haven't checked it directly.
- **Only part of the symmetry is checked.** The check covers 16 rotations, not every angle (random angles behave the
  same, but without a guarantee), and only two conservation laws, which need open space and equal, known masses.
- **M3 differs from M1 in more than symmetry.** Its outputs can only point along the gaps between discs, its own
  velocity and the push. If it does better or worse, symmetry isn't the only possible reason.
- **The label-free bound assumes one clearly best push.** My bank guarantees that; a real agent couldn't check it.
- **The pieces are known.** Splitting an error by symmetry and using disagreement as a signal both exist already.
  What's new here is the link to decisions.
- **Invisible doesn't mean harmless, and visible doesn't mean wrong at that particular state.**

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

**The visible part needs no ground truth.** The true physics turns with the scene, so Se = SF̂. It also follows two
rules exactly. The discs have equal mass, the forces between two discs are equal and opposite, and the push is the
only outside force, so over each small integration step h the total momentum P = Σvᵢ grows by h·a and the position
total X = Σxᵢ grows by h(P + h·a/2). Over one decision step Δt that gives P(F(s, a)) = P(s) + Δt·a and
X(F(s, a)) = X(s) + Δt·P(s) + (Δt²/2)·a. So Me only needs the model's own predictions, and every part of the
visible error comes from F̂, s and a alone.

**The visible part can be removed.** The corrected model 𝒫_G(F̂ − Me) first shifts every disc by the same amount so
both rules hold, then averages over the 16 rotations. Its error is exactly the invisible piece. The shift never makes
the error bigger; the averaging never makes it bigger on average over the 16 rotations, though it can at a single
one. It costs 16 model calls per step.

**A label-free bound on wrong decisions.** The right answer has the same label in all 16 rotations. If the model's
choices over the 16 have label counts n₁, …, n₅, at most the largest count can be right, so at least 16 − max n_k
are wrong. A model whose error turns with the scene makes the same choice on all 16, so for it the bound is always
zero: its mistakes, if any, come as whole orbits.

## Details

- **Physics.** Five unit-mass discs in open 2D space with Morse forces between them, integrated in float64 with
  velocity Verlet (10 steps of 0.01 per decision). The push on the agent has strength at most 2.
- **Decisions.** 500 situations, each in 16 rotations, with 5 candidate pushes of 10 decision steps. The goal lies
  ahead of the target, so a good push has to hit the target well. A situation is kept only if its best push wins by
  5%, so the right answer is unique.
- **Controlled experiment.** Errors built from random Fourier features, scaled to 1%, 3% and 10% of a typical
  one-step change, 20 random draws each, with the visible share going from 0 to 1 in five steps.
- **Learned models.** Message-passing networks with three layers of width 64 and parameter counts within 1% of each
  other, trained for 200 epochs on 50,000 random transitions. Everything is evaluated in float64.

## Background

Splitting a function into the part that respects a symmetry and the part that breaks it goes back to Elesedy and
Zaidi (ICML 2021), and Wang et al. (NeurIPS 2023) studied what goes wrong when the assumed symmetry is wrong.
Disagreement under transformed inputs is a known label-free uncertainty signal (Ayhan and Berens, MIDL 2018). M3
follows the EGNN of Satorras, Hoogeboom and Welling (ICML 2021), and the ensemble baseline is Lakshminarayanan,
Pritzel and Blundell (NeurIPS 2017). What this project adds is the link to decisions: which part of a world model's
error can be seen, removed or bounded without ground truth, and what each part does to an agent's choices.

## Running it

Python 3.14:

```sh
pip install -r requirements.txt
pytest tests.py
python -m experiments.controlled    # --eps 0.03 runs only the go/no-go size, --fields one kind of error
python -m experiments.learned       # trains 15 models on the CPU, then evaluates them: the slow part
```

Run times: the controlled experiment took about 6.5 hours on 4 CPU cores (the first kind of error alone took about
an hour on 12).
Training takes 3 to 4 hours per model on one core, with the 15 in parallel. The evaluation took about 3 hours on one
T4 GPU, and would take about a day on 12 CPU cores. It uses a GPU when PyTorch sees one (`--device` to choose; use a
CUDA build of the same PyTorch version), and `--workers` sets how many processes share it.

Results go to `results/` and figures to `figures/`. Trained models are saved in `results/models/` and each finished
evaluation in `results/learned/`, so an interrupted run picks up where it stopped (delete `results/learned/` after
changing the code). `--figures-only` redraws the figures from saved results.

The tests check the momentum rules, that the true simulator agrees with itself across every rotation, that the
projections behave as the maths says, that a model whose error is invisible shows no visible error, that the
controlled errors have the visible share they should, that the corrected model's error is exactly the invisible
piece, and that M3 is exactly symmetric in float64.

The code is in `invariantlens/` (the simulator, the decision bank, the projections and the correction, and the
networks), and the two experiments are in `experiments/`. A quick example:

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
