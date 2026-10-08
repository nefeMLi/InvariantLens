# Hypotheses

Written before any model was trained or any decision scored. Anything already seen is logged as an observation, not
a prediction. First written 2026-09-29, changed 2026-09-30 and 2026-10-04; the log at the end says what was seen
before each change, and lines a change touched are marked with its date.

## The question

A world model's one-step error splits exactly into two parts. One breaks rules the true simulator always follows:
turn the scene and the answer turns with it (C₁₆ equivariance), and total momentum and centre of mass change by known
amounts. This part can be measured from the model alone. The other part follows the rules, so it is invisible
without the true simulator.

1. At a fixed error size, does moving error from the invisible to the visible part change how, or how often,
   decisions go wrong?
2. The visible part can be subtracted. How many wrong decisions go away when it is?
3. For trained models, does the visible share of the error explain how well label-free signals catch wrong
   decisions?

The split itself is known (Elesedy & Zaidi, 2021), and so is disagreement under rotation as a label-free signal.
What is new is the link to decisions.

## Definitions

| Symbol | Meaning |
|---|---|
| F, F̂, e | true and learned step maps, (s, a) → state one decision step later; e = F̂ − F |
| 𝒫_G, S | average over the 16 rotations, (1/16) Σ_g ρ(g)⁻¹ f(gs, ga); S = I − 𝒫_G |
| X, P | Σᵢ xᵢ and Σᵢ vᵢ, total position and momentum (all masses are 1) |
| M | (1/N) B*B with B e = (Σᵢ e_x,i, Σᵢ e_v,i): spreads the mean position and velocity error over every disc |
| V, 𝒩 | V = S + M − SM, the visible part; 𝒩 = ker S ∩ ker M, the invisible part |
| F̂_c | 𝒫_G(F̂ − Me), the corrected model |

Equal masses, equal and opposite pair forces and the push as the only outside force give two exact balance laws:
P(F(s, a)) = P(s) + Δt·a and X(F(s, a)) = X(s) + Δt·P(s) + (Δt²/2)·a. So Se = SF̂ and Me depend only on F̂, s and a,
and Ve needs no ground truth. F̂_c − F = (I − S)(I − M)e exactly, and both signals of F̂_c are zero. The README proves
this; tests.py checks it. Since S and M commute, the error has four orthogonal pieces:

| | Follows the balance laws | Breaks them |
|---|---|---|
| **Turns with the scene** | (I − S)(I − M)e: invisible | (I − S)Me: balance signal only |
| **Breaks rotation** | S(I − M)e: symmetry signal only | SMe: both |

## Frozen system

| Item | Value |
|---|---|
| Discs | N = 5, mass 1, open 2D space. Disc 0 is the agent, disc 1 the target |
| Pair potential | Morse, φ(r) = D(1 − e^(−a(r − r₀)))² − D, with D = 1, a = 2, r₀ = 1 |
| Action | outside force on the agent, ‖a‖ ≤ a_max = 2, held for one decision step |
| Integrator | velocity Verlet, 10 substeps of h = 0.01 per decision step Δt = 0.1, float64 |
| Initial state | target and distractors uniform in a disc of radius 2; agent U[1.0, 1.5] from the target at a uniform angle; all pairs ≥ 0.8 apart, by rejection; velocities N(0, 0.3²) per component |
| Goal | target plus a step of length U[0.5, 1.5], within ±60° of the agent-to-target direction |
| Horizon, reward | 10 decision steps (T = 1); R = −‖x_target(T) − goal‖ |
| Group | C₁₆, rotations by multiples of 22.5° about the origin, on positions, velocities, goal and action |
| Bank | 500 kept situations × 16 rotations × 5 candidates; seeds 0, 1, 2, … in order, rejections logged |
| Candidates | all drawn alike: strength m_k ~ U[0.5, 1]·a_max, direction θ_t = θ₀ + δ_k + b_k sin(2πt/T + c_k), θ₀ the agent-to-target direction, δ_k ~ N(0, 0.6²), b_k ~ U[0, 0.5], c_k ~ U[0, 2π); a rotated situation gets the rotated originals |
| Margin | keep a situation only if (R₍₁₎ − R₍₂₎) / max(\|R₍₁₎\|, 10⁻³) > 0.05 |

| Model | Description |
|---|---|
| M1 | message-passing network on absolute coordinates, 3 layers, width 64; does not turn with the scene |
| M2 | M1 trained with a random continuous rotation per sample |
| M3 | EGNN-style, non-symmetric messages; turns with the scene exactly, balance laws not built in |
| M4 | the five M1 seeds as an ensemble |
| M1c, M2c | M1 and M2 corrected at test time as F̂_c; no retraining |

All predict (Δx, Δv) for every disc, with parameter counts within 10% and one input/output scale per quantity.
Training: 5,000 trajectories of 10 steps with uniform random actions (plus 500 trajectories for validation); Adam, learning rate
10⁻³ with cosine decay, batch 256, 200 epochs, best validation epoch kept; 5 seeds; float32. Every diagnostic in
float64.

**Controlled fields.** Random Fourier features over (x, v, a): 64 features, N(0, I) frequencies, uniform phases,
Gaussian weights. u = (I − V)r₁, w = V r₂, both of unit norm under μ, the (state, action) pairs of all true candidate
rollouts in all 16 rotations. e_α = √(1 − α²) u + α w for α ∈ {0, 0.25, 0.5, 0.75, 1}, and F̂_α = F + ε e_α with
ε = 1%, 3% and 10% of the RMS one-step change under μ. 20 draws of (r₁, r₂). Every α has the same one-step error;
rollout error may differ, and that is part of what is measured. The rotation-breaking share α²‖Sw‖² is reported.

## Analysis choices

| Choice | Value |
|---|---|
| Decision, wrong decision | the candidate with the highest predicted reward; wrong if it is not the true best |
| Certificate | per orbit, 16 − max_k n_k, n_k counting how often candidate k was chosen |
| Consistent share, coverage | share of wrong decisions in orbits where all 16 rotations agree; total certificate over total wrong |
| Symmetry, balance signal | ‖SF̂‖ and the size of both balance residuals, averaged over all steps of the 5 candidate rollouts |
| Ensemble signal | variance of predicted returns across M4's members, averaged over the 5 candidates |
| Orbit vote | predicted returns averaged over the 16 rotations before choosing; M1 and M2 |
| Visible fraction | ‖Ve‖² / ‖e‖² under μ with its three pieces; raw units, positions and velocities weighted equally (stated 2026-10-04) |
| Uncertainty | bootstrap, 1,000 resamples, clustered by situation; over field draws for H2, over trained models for H3 (changed 2026-10-04) |
| Undefined AUROC | a signal that is always zero has no AUROC; reported as undefined, never as 0.5 |

## Checks that must pass first

- **Truth:** the true simulator picks the best candidate everywhere, returns agree across orbits to 10⁻⁹, and both
  balance laws hold to 10⁻¹²(1 + ‖X‖ + ‖P‖) under the base dynamics and both dynamics shifts.
- **Projections:** S, M and V are idempotent and self-adjoint, and SM = MS, to 10⁻¹².
- **Invisible control:** the simulator with D = 1.1, used as a model, has ‖Ve‖ / ‖e‖ < 10⁻¹⁰.
- **Injection:** ‖Ve_α‖² = α², and the orbit-averaged defect equals twice the orbit mean of ‖Se‖², to 10⁻¹⁰.
- **Correction:** F̂_c − F = (I − S)(I − M)e to 10⁻¹⁰, and F̂_c's signals are zero to 10⁻¹².

## Predictions

**H1, the identities hold.** Every check passes; at α = 0 all 16 rotations choose alike; M3's symmetry signal is zero
to float64 precision while its balance signal need not be. *Confidence:* high; a failure means a bug.

**H2, equal-size visible and invisible error do different damage (primary).** Error that turns with the scene makes
the model turn with it, so its mistakes come in whole orbits (exact). For small rotation-breaking error, the change
in each return gap is linear in the error, and averaged over the orbit it vanishes, so those mistakes scatter within
orbits, where the certificate sees them.

- **H2a, pattern.** The consistent share falls and coverage rises with α², clearest at ε = 1%. Readout: per-draw
  least-squares slope on α², averaged, 95% interval over draws, at each ε. *Wrong if,* at the smallest ε with at
  least 100 wrong decisions, either interval contains zero or has the wrong sign. *Confidence:* moderate.
- **H2b, rate.** The wrong-decision rate changes with α². Same readout. *Wrong if* the intervals contain zero at all
  three ε, which would still be worth reporting. *Confidence:* low, no guess at the sign.

**H3, the visible fraction explains the signals.** Across (model, seed, shift) units, each signal's AUROC for wrong
decisions rises with the visible fraction: Spearman interval above zero, with certificate tightness beside it and the
ensemble as reference. *Wrong if* the interval contains zero or lies below it. *Confidence:* moderate.

**H4, correction removes wrong decisions.** M1c and M2c make fewer wrong decisions than M1 and M2, reported with an
interval next to the share of one-step error removed, the orbit vote beside it, and the count of wrong decisions left
(which neither signal can flag). *Wrong if* the interval contains zero or lies above it; a large error drop with no
decision gain is reported as a result. *Confidence:* moderate, since rollout error need not fall.

**H5, C₁₆ to SO(2).** The symmetry signal and decision disagreement at 100 random angles, reported descriptively.

## Shifts, stopping and cutting

No retraining; each shift gets its own bank with the same seeds and acceptance rule.

| Shift | Truth changes | Still follows the audited rules | Theory says |
|---|---|---|---|
| D × 1.5 | yes | yes | the shift's own error is invisible |
| Pair damping −c((vᵢ − vⱼ)·r̂ᵢⱼ) r̂ᵢⱼ, c = 0.5, half-step velocity | yes | yes | same |
| Initial speeds × 2 | no | – | extrapolation; the visible part may grow |
| Input noise σ = 0.02 on the observed starting state only (changed 2026-10-04, fixed 2026-10-05) | no | – | measured |

Masses are never shifted. Go/no-go: before training, run the continuum at ε = 3%; if the wrong-decision rate is below
2% or above 80% at every α, change the spread of δ_k once, on the true simulator only. If time runs short, cut the
SO(2) test, then input noise, then damping. Never cut the checks, the certificate, the corrected models, the
intervals, the undefined-AUROC counts or the limitations.

## Log

**2026-09-30, candidate design** (true simulator only, no model). The first design had four full-force pushes, one
random candidate and the goal at any angle. The random candidate was best in 231 of 500 situations (82% when the goal
lay behind the target, 6% when ahead); pushing did worse than not pushing on average. So the bank rewarded avoiding
contact. Changed once, to the design above: best-candidate counts 91, 116, 90, 109, 94 of 500; the agent reaches the
target in 97.6% of rollouts; not pushing wins 1.8% of situations; 632 of 1,132 seeds rejected; orbit agreement
3·10⁻¹³. The corrected models, orbit vote, H2a and the four-piece table were added the same day.

**2026-09-30, centre of mass.** The audit first covered momentum only, so mean-position errors counted as invisible
although the simulator fixes that mean. The centre-of-mass law was added; both laws hold to 10⁻¹⁵ on 5,000 states.

**2026-10-02, controlled run.** Go/no-go at ε = 3%: 2.8–3.0% wrong across α, so the spread stayed. All checks passed.

| ε | wrong rate, α = 0 → 1 | consistent-share slope | coverage slope |
|---|---|---|---|
| 1% | 0.2% → 0.2% | −0.41 [−0.58, −0.23] | +0.47 [+0.27, +0.66] |
| 3% | 2.8% → 3.0% | −0.72 [−0.74, −0.69] | +0.87 [+0.86, +0.89] |
| 10% | 26.5% → 27.7% | −0.70 [−0.71, −0.69] | +0.87 [+0.86, +0.88] |

H2a held but was close to guaranteed: the share is 1 at α = 0 by H1 and already 0.23–0.34 at α² = 0.06. H2b failed
at all three ε. At α = 1 coverage was 0.99–1.00. The fields' visible part was 97–98% rotation-breaking and about 2%
balance-only, so H2 said little about the balance check.

**2026-10-03, training.** All 15 models trained as specified. A first evaluation run was stopped before writing any
result.

**2026-10-04, smoke test** on banks of 2 and 4 situations, to test the code; its numbers aren't reported. Under input
noise all models made identical decisions, because the noise was redrawn at every rollout step. On 100 states, M2's
rotation-breaking share came out close to M1's, so I make no prediction about it. M1's share was seen at the same
time, and H6a's premise that most of M1's error breaks rotation is informed by it.

**2026-10-04, changes.** Input noise falls only on the observed starting state, the same draws for every model. The
H3 interval resamples trained models, each with its five shifts, with per-type correlations also reported; the
pooled interval decides. The H4 interval covers these five models, with the range over seeds beside it. Evaluation
may run on a GPU in float64; results can differ only in the last digits.

## New predictions, 2026-10-04

Informed by the controlled run and the smoke test above; by no full learned result.

**H2c, balance-only fields.** Rerun the continuum with w = (I − S)M r₂ at unit norm, which turns with the scene and
only the balance check sees. Same draws, α and ε. The certificate is zero at every α (a check). Does the
wrong-decision rate change with α²? Same readout as H2b. *Wrong if* the intervals contain zero at all three ε, which
would still be worth reporting. *Confidence:* low, no guess at the sign.

**H6a, learned mistakes scatter.** On the unshifted bank, pooled over seeds, less than half of M1's wrong decisions
fall in orbits where all 16 rotations agree. *Wrong if* the share is 0.5 or more. *Confidence:* moderate.

**H6b, the vote fixes what the certificate sees.** On the unshifted bank, pooled over seeds, the orbit vote removes at
least half as many wrong decisions as the certificate counts, for both M1 and M2. *Wrong if* the ratio is below 0.5
for either. *Confidence:* moderate; the vote averages returns rather than counting choices.

## After the learned results, 2026-10-05

**Input-noise fix.** Found after the learned results were in, while checking why the orbit vote helped so much under
input noise. The 2026-10-04 version drew separate noise for each of the 16 rotated copies of a situation. But the vote
and the certificate are meant to work from one observation, turned 16 ways, so the copies should share one draw,
rotated with the scene. With separate draws the vote was averaging 16 independent observations, and even M3 disagreed
across rotations. Now each situation gets one draw, rotated with it. Under noise, the visible fraction is now the
model's own error at the states it observes, since that is all a label-free check can see; the observation error
itself is outside the audit. Only the 15 input-noise evaluations were rerun. The first results under noise (the vote
cutting wrong decisions from 1.4% to 0.4%, and certificate coverage near 1 for every model, M3 included) are kept in
the README for comparison, and in `results/learned.parquet` at commit 09428e3. No prediction was changed. The
per-copy draws also broke a premise of the split: the points μ is measured on were no longer the same from every
rotation, so the orthogonal pieces of the visible fraction weren't guaranteed. One rotated draw restores that.

**What the fix should give.** Written while the rerun was running, before its results were seen. I only committed
it after the results came back, though, so there is no timestamp to prove the order.

- **M3 under noise** makes the same choice in all 16 copies of every situation, so its consistent share is 1 and its
  certificate 0, apart from exact ties. This is an identity, so it also checks the fix.
- **The orbit vote** gains almost nothing for any model under noise: rotating one observation adds no new
  information. For M3 the gain is exactly zero.
- **M1's and M2's mistakes under noise** come mostly as whole orbits: consistent share above 0.5.
- **The visible fraction under noise** now describes the model's own error at what it observes, so it should look
  like the unshifted bank's (0.5 to 0.6 for M1 and M2), while the signals' AUROCs fall towards chance. The error that
  causes these mistakes, the observation error, is outside the audit.

## Positive control, 2026-10-05

Written before this run, and committed before starting it, but after the noise rerun had made H3 fail. So far the
learned results only show the signals failing; this test is meant to show them working where they should.

Each situation is moved 2.5 or 5 units from the origin before it's rotated, so its 16 copies sit on a circle around
the origin. The true physics only depends on where the discs are relative to each other, so it doesn't change, and
neither does M3, which only sees relative quantities. M1 and M2 read absolute coordinates, and the far copies sit
where they never trained (training positions reach about 3 to 4 units out). Their error there should break rotation,
which the audit can see. This is a covariate shift (unfamiliar inputs, same physics), while the physics shifts were
concept shifts (familiar inputs, new physics).

- **H7a, check.** The truth's returns match the unshifted bank's to 10⁻⁹, and M3 makes the same number of mistakes
  as on the unshifted bank.
- **H7b.** Pooled over M1 and M2, the symmetry signal's AUROC is above 0.7 at each offset. *Wrong if* 0.7 or below.
- **H7c.** Less than half of M1's and M2's wrong decisions fall in orbits where all 16 copies agree. *Wrong if* 0.5
  or more.
- **H7d.** M1c and M2c make fewer wrong decisions than M1 and M2, with an interval below zero. *Wrong if* it contains
  zero or lies above it.
- **H7e.** The ensemble's AUROC is higher than under the physics shifts (0.50 and 0.56).

*Confidence:* moderate for H7b and H7c, low for H7d: far from the data, M1's symmetric error may be large too, and the
correction can't remove that. These runs are not added to H3, which stays on the five shifts it was written for.

**H7, made precise the same day, while the run was in progress and before any of its results were seen.**

- In H7b, "pooled" means the mean of the 10 per-model AUROCs (M1 and M2, five seeds each), with a 95% bootstrap
  interval over those 10 models. The 0.7 threshold applies to the mean.
- H7e: *wrong if* the ensemble's AUROC is 0.56 or below (the highest it reached under the physics shifts) at either
  offset.
- If M1 and M2 make fewer than 100 wrong decisions in total at an offset, H7b and H7c count as inconclusive there.

## Results log, 2026-10-06

**Input-noise rerun (2026-10-05).** All four predictions written for it held: M3 chose alike in all 16 copies, the
vote gained almost nothing, M1's and M2's mistakes came as whole orbits (0.97 and 0.96), and the visible fraction
looked like the unshifted one while the AUROCs fell to about 0.55. With the fixed noise, H3 fails: the Spearman
intervals are +0.06 [−0.12, +0.20] for the symmetry signal and +0.18 [−0.02, +0.39] for the balance signal. Most of
its earlier support came from the noise bug.

**H7.** The run's results were written at 02:07 on 6 October, after both H7 commits (14:17 and 14:30 on the 5th). It
ran on Kaggle's CPU rather than its GPU, which is why it took 11.5 hours. The truth check passed, and M3 made 112
mistakes far away, the same as unshifted, so H7a held. At 2.5 units M1 and M2 made only 13 mistakes, so H7b and H7c
are inconclusive there and H7d failed. At 5 units (267 mistakes) H7b failed (AUROC 0.53 [0.49, 0.57]), H7c held
(consistent share 0.00) and H7d held (−0.30% [−0.56%, −0.10%] for M1c, −0.28% [−0.53%, −0.09%] for M2c). H7e failed
(0.37), but the ensemble made only 7 mistakes there, so that number says little.

**Not pre-registered.** After all of this I looked at two label-free warnings I hadn't planned: the model's own
predicted margin between its two best candidates, and a flag on copies that disagree with their orbit. They are
computed by `experiments/posthoc.py` from the saved results and reported as exploratory.

**Also not pre-registered: what the noise condition tests.** Given the same noisy readings, the true simulator makes
128 wrong decisions, as many as M1 and M2 (128 to 135). So the mistakes under noise come from the observation, not
the model, and that condition can't test H3. Without it, H3's Spearman is +0.72 [+0.59, +0.82] for the symmetry
signal and +0.63 [+0.43, +0.81] for the balance signal. With far 5 in its place they are +0.11 [−0.13, +0.33] and
+0.42 [+0.27, +0.60]. H3 still counts as failed. Both checks are in `experiments/posthoc.py`.

## Version 2: where does a learned symmetry break, and can averaging repair it? (2026-10-07)

Written after all of the above and after exploring on dev seeds (below), before any test seed was used. Situations
from seed 1,000,000 on are dev, from 2,000,000 on test; version 1 used seeds from 0. Run by `experiments/diagnose.py`.

**The question.** M1 and M2 were never told that the physics turns with the scene; they learned it from data, M2
from data turned on purpose. When such a model is asked about situations it never trained on, does the symmetry it
learned still hold? Where it breaks, the error shows up in the rotation check, and averaging over the 16 rotations,
the correction 𝒫_G(F̂ − Me), removes it. Where it holds, the error is invisible, and nothing computed from the model
can help. So: where does a learned symmetry break outside the data, and when does averaging repair the decisions?

**Background.** Symmetry learned from data is known to be unreliable under distribution shift (Moskalev et al.,
2023), but not where it breaks. Averaging or canonicalising a model over a group at test time makes it exactly
symmetric (Puny et al., 2022; Kim et al., 2023; Mondal et al., 2023), but those papers ask about accuracy and sample
efficiency, not about which extrapolation errors it removes. Gruver et al. (2023) measure how much a trained network
breaks a symmetry; the visible share is the finite-group version of that measure.

**What the dev seeds showed.** With the physics unchanged and only the situations new, two kinds of novelty behaved
in opposite ways. With the scene 10 units from the origin, 53 to 98% of the one-step error was visible (two M1s and
an M2), and the correction cut costly decisions from 21 to 0. With starting speeds five times the usual, 1% of it
was, and the correction cut costly decisions from 49 to 48. Starting speeds × 3 and pushes twice the training
maximum made no costly decisions, so they say nothing about repair. On the way, three ideas were tried and dropped.
Nothing computed from the model alone can detect a change in the physics: on a bank that keeps near-ties, every
model, corrected or not, made the same decisions under D × 1.5, and the predicted margin flagged the costly ones at
chance (AUROC 0.48), so the 0.82 in version 1 came from the bank's 5% filter, which picked situations using the
shifted truth. A diagnosis of physics changes from observed transitions worked, but its main half follows from the
maths. And the five-model ensemble also removes the costly decisions far from the origin.

**The proposed explanation.** Outside their data, networks become close to linear along each direction (Xu et al.,
2021, for ReLU networks; ours use SiLU, which also turns linear far from zero). A quantity the physics ignores, like
where the scene is, gets a small, arbitrary dependence in training, and outside the data that dependence grows with
nothing to make it turn with the scene: the learned symmetry breaks, the check sees it and averaging cancels it. A
quantity the physics uses, like how fast two discs close in, is learned from data that look the same from every
angle, so the network's guess at it stays nearly symmetric even where the guess is wrong: the error is invisible.
In short, the symmetry breaks in a new frame and holds for new physics.

**The test.** Two new kinds of novelty, chosen to tell this explanation from the obvious alternative ("large
velocities give invisible error"), and not looked at on the dev seeds beyond checking that the code runs:

- **Drift:** every disc and the goal get the same extra velocity of 2 along x. The physics ignores it (the truth's
  returns match the plain bank's to 10⁻⁹, checked), but the velocities are as far outside training as at five
  times the speed (99% of training speeds are below 1.51). A new frame.
- **Crowded:** one distractor starts 0.35 from the target, where the closest pair in 50,000 training states was
  0.50, so the repulsion is stronger than anything in training. Familiar positions and speeds, new physics.

**Fixed for the test run.** M1 and M2, five seeds each; M3 has the symmetry built in, so it has nothing to break.
Banks of 100 situations in all 16 rotations, near-ties kept (margin 10⁻⁶). Shifts: far 10 (scene moved 10 along x),
speed × 5 (starting speeds N(0, 1.5²)), drift 2 and crowded 0.35. The visible share is Σ‖F̂ − F̂_c‖² over
Σ‖F̂ − F‖², one step at a time along the true rollouts of every candidate, averaged over the 10 models. A decision
is costly if it ends more than 0.05 further from the goal than the best candidate would. A shift counts as broken
and repaired if its visible share is 0.5 or more and the correction removes at least half of its costly decisions,
and as kept and not repaired if the share is 0.1 or less and the correction removes less than a fifth. The part
about costly decisions counts only if the models make at least 20 of them, pooled.

**H8, replication.** Far 10 is broken and repaired; speed × 5 is kept and not repaired. *Wrong if* either fails.
*Confidence:* high; these are the dev results on new situations.

**H9, the test.** Drift is broken and repaired; crowded is kept and not repaired. *Wrong if* either fails.
*Confidence:* moderate. If drift is not repaired, the explanation is wrong, and large velocities themselves give
invisible error. If crowded is repaired, new physics can break a learned symmetry too, and the line runs somewhere
else.

**Reported beside them, not predicted.**

- **M1 against M2.** Whether learning the symmetry from turned data (M2) makes it hold any better outside the data,
  shift by shift.
- **One pose against the average.** The model asked once, with the scene turned until the agent faces the target
  along +x (the canonicalisation of Mondal et al.), balance fixed as in the correction. It is exactly symmetric at
  one call instead of 16, but it keeps the error of the one pose it asks about instead of averaging it away. If
  averaging repairs far 10 and one pose doesn't, the repair comes from cancelling the broken part, not from being
  symmetric. The maths makes that likely on average but doesn't guarantee it, so it is reported, not predicted.


## Results log, 2026-10-08

**The test run.** Predictions committed and pushed at 14:26 on 7 October; the test seeds ran afterwards on Kaggle's
CPU, and the results were downloaded on 8 October. 100 situations per shift, 16,000 decisions per shift over the 10
models. The checks passed: far 10 and drift 2 left the truth's returns unchanged.

| Shift | Visible share (models) | Costly decisions: model, one pose, average | Mean regret: model, average |
|---|---|---|---|
| far 10 | 0.86 (0.55 to 1.02) | 335, 80, 0 | 3.8·10⁻³, 0.3·10⁻³ |
| speed × 5 | 0.103 (0.07 to 0.15) | 157, 160, 176 | 1.93·10⁻³, 1.96·10⁻³ |
| drift 2 | 0.67 (0.54 to 0.79) | 0, 0, 0 | 0.09·10⁻³, 0.06·10⁻³ |
| crowded | 0.04 (0.02 to 0.11) | 158, 144, 112 | 1.98·10⁻³, 1.38·10⁻³ |

**H8 failed.** Far 10 was broken and repaired. Speed × 5 was not repaired (157 to 176), but its visible share, 0.103,
was just above the bound of 0.10.

**H9 failed.** Drift was broken (0.67), and made no costly decisions, so only its share counted. Crowded was kept
(0.04), but the correction removed 29% of its costly decisions, more than the fifth allowed.

**Reported beside them.** One pose against the average: at far 10 the standard pose left 80 costly decisions and
the average none, so the repair comes from averaging, not from being symmetric. M1 against M2: the shares were the
same within a few hundredths on every shift (0.85 and 0.88, 0.11 and 0.09, 0.69 and 0.65, 0.04 and 0.05), so
learning the symmetry from turned data didn't make it hold any better outside the data.

**Not pre-registered, computed after the results.**

- The split itself held for every model: on both frame shifts each of the 10 models had a share of 0.54 or more, on
  both physics shifts 0.15 or less.
- Intervals, resampling the 100 situations, for the change in mean regret the correction made, in units of 10⁻³:
  −3.4 [−5.6, −1.9] at far 10, +0.02 [−0.22, +0.34] at speed × 5, −0.035 [−0.076, −0.005] at drift 2 and −0.60
  [−1.19, −0.16] at crowded. So averaging did repair part of crowded, and of drift's small errors.
- The decision part of both predictions rested on very few situations. The corrected models choose alike in all 16
  rotations, so their costly decisions come in whole orbits of 16; the 112 at crowded are 7 situations. And the
  raw models' costly decisions under the physics shifts came from 1 to 4 situations per model.
- The bound of 0.10 for speed × 5 came from three models on 30 dev situations, where the share was 0.01. On 100
  situations it was ten times that: a few extreme starting speeds dominate the error, so the dev estimate was too
  small a sample.
