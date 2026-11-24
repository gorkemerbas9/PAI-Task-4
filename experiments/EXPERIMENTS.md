# PAI Task 4 — DPPO fine-tuning of a diffusion policy: experiment log

This file is the running log of the work: what was implemented, which hypotheses were tested,
how, and what came out. Raw results are in `results.jsonl` (one line per run) and
`runs/<name>.json` (full learning curve); training logs are in `logs/<name>.log`.

## 0. Setup and reference numbers

* **Hardware used for experiments:** cloud Linux container, 2 CPU cores, PyTorch 2.x (CPU).
  Two experiments run in parallel with `num_threads=1` each (measured: 2 × ~58 it/min
  vs. 1 × ~80 it/min with 2 threads). **Absolute iteration counts here are lower than on a
  modern laptop**, so a 30-min run here is a pessimistic proxy for a 30-min checker run.
* **Metric:** official `utils.evaluate` on `scenes_local.npz`, 2000 episodes, seed 0
  (standard error ≈ 0.003 according to the handout).
* **Pretrained policy:** local score **0.4185**, success 0.984, 86.9 steps to goal
  (task page: 0.417 on the hidden scenes). Baseline to beat: **0.600**.
* **Rough ceiling:** start–goal distance ≈ 8.2 (range 7.1–9.4). With |u| ≤ 1 and v ≤ 3 a straight
  run takes ≈ 42 steps → 0.99^41 ≈ 0.66; detours around the obstacles reduce this, so ≈ 0.63–0.65
  is a realistic upper bound. Baseline 0.600 ≈ 51 steps with almost no crashes.
* **Profiling of one iteration (default config, 1 thread):** rollout of 64 envs × 16 chunks
  (all 10 denoising steps) ≈ 0.13 s; critic update ≈ 0.1 s; actor update (25 minibatches of
  2048 pairs) ≈ 1.2 s. → **Training is bound by actor gradient steps; environment samples are
  almost free.** This shapes several hypotheses below.

## 1. Implementation (TODO 1–3)

* `sample_with_chain`: runs the K = 10 reverse steps; for transition j uses utils-index
  k = K−1−j, mean `step_mean`, std `step_std(k, min_std)`, records chain (B, K+1, 8) and
  per-step Gaussian log-probs (B, K).
* `ValueFunction`: MLP on `featurize(obs)`, 3 hidden layers × 256, Mish.
* `compute_gae` (course script §12.4.4, Schulman et al. 2016), per chunk t:
  δ_t = r_t + d_t (1 − term_t) V(o'_t) − V(o_t), A_t = δ_t + d_t λ (1 − done_t) A_{t+1},
  with d_t = γ^{#executed steps} the chunk's discount. Termination (goal/crash) does not
  bootstrap; truncation does bootstrap but does not pass on the next episode's advantage.
  Targets: A_t + V(o_t).
* `ppo_loss`: per (env step, denoising step) pair, log-prob of x_out under
  N(μ_θ(x_in, k, o), σ_k² I) (σ_k clamped by min_std), ratio vs. rollout log-prob, clipped
  objective −min(r A, clip(r, 1±ε) A). All denoising steps of a chunk share its advantage.
* Smoke test (default config, 2 threads): local monitor score 0.42 → 0.467 within 1.5 min. ✔

Extensions added to `Config` for experiments (defaults reproduce the template): linear
annealing of `min_std` and of the actor LR, `target_kl` early stopping (KL estimator
E[w − 1 − log w] from the course script, Remark 12.12), separate `critic_epochs`, a `log`
callback for the harness.

## 2. Hypotheses

| id | hypothesis | rationale |
|---|---|---|
| H1 | The default actor LR (3e-5) is too small for a 30-min budget; 1e-4 learns faster. | Compute-bound; each gradient step should move more. |
| H2 | A larger PPO clip (0.1) lets each iteration move further. | Same as H1, but via the trust region. |
| H3 | Fine-tuning only the last 5 steps with the early steps frozen (DPPO) is cheaper per iteration (half the pairs) and more stable. | DPPO paper; but early steps decide the route → may limit route adaptation. |
| H4 | More exploration noise (min_std 0.2) helps discover faster routes; too much noise biases the trained policy away from the evaluated one. | Hint in supplementary §6. |
| H5 | Fewer epochs (2) → more fresh data per unit compute, better time-efficiency. | Env samples are cheap relative to gradient steps. |
| H6 | Annealing min_std towards the official noise late in training reduces the train/eval mismatch and lets the policy cut margins safely. | Eval uses std 0 on the last step. |

## 3. Results

(filled in as experiments finish)

### Batch 1 (pair 1) — 30-min runs, 1 thread each, cloud
Note: a bug in `queue.sh` dropped the run name, so `budget_minutes=15` was read as the name
and **these runs used the default 30-min budget** (fixed afterwards; it is the budget that matters anyway).

| run | change vs. default | iters | local score | success | steps |
|---|---|---|---|---|---|
| A_default | – | 1515 | **0.5607** | 0.970 | 55.7 |
| B_lr1e-4 | actor_lr 1e-4 | 1507 | 0.5578 | 0.946 | 53.7 |

Monitoring curves (200 episodes) rise roughly linearly over the whole 30 min and have not saturated.
**H1 (higher LR) not supported**: 3× LR gives the same score (faster, but more crashes). With
clip = 0.05 and 5 epochs, the PPO clip probably limits how far each iteration moves, regardless of LR.

**Hardware check:** on the user's computer (Linux VM, 4 cores) the template runs at ≈ 53 it/min with
4 threads (≈ 41 with 2), against ≈ 58 it/min per 1-thread run here. So 30-min results here are a fair
(slightly optimistic) proxy for the checker run.

### Diagnostic: where is the score lost? (`analyze.py`, `route_oracle.py`)

| policy | score | success | steps | obstacle / wall crashes | mean max speed |
|---|---|---|---|---|---|
| pretrained | 0.4185 | 0.984 | 86.9 | 1.5% / 0.1% | 1.54 |
| A_default (30 min) | 0.5607 | 0.970 | 55.7 | 2.0% / 1.0% | 2.13 |

* Route choice: marginal route frequencies stay ≈ 1/3 each, but this is because each route is the
  fastest in 1/3 of the scenes. Running 24 episodes in each of 200 local scenes shows the fine-tuned
  policy takes **its own best route in 85% of episodes**; an oracle route choice would only add ≈ +0.012.
  → Route selection is mostly learned within 30 min.
* Speed is the main lever: the goal may be entered at any speed, so the time-optimal behaviour is
  essentially full throttle (|u| = 1) to v = 3 while steering around the obstacles. The policy only reaches
  2.1 on average. Failures cost ≈ 0.017.
* → The bottleneck is **how fast PPO can move the policy within 30 min**: the next experiments target the
  size of each update (clip, epochs) and the cost per iteration (ft_steps).

### Batch 1 (pairs 2–3) — 15-min screening runs
| run | change vs. default | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| C_lr1e-4_clip0.1 | lr 1e-4, clip 0.1 | 711 | 0.5271 | 0.940 | 58.8 | 3.9% / 2.1% |
| D_lr1e-4_ft5_freeze | lr 1e-4, ft_steps 5, freeze_early | 1047 | **0.5410** | 0.965 | 58.7 | 2.6% / 0.9% |

* **H2 (larger clip) not supported**: more crashes, slower iterations (more of the minibatch stays unclipped
  → same compute, but noisier updates).
* **H3 (DPPO: only last 5 steps fine-tuned, early ones frozen) promising**: half the (step, denoising-step)
  pairs → ~40% more iterations in the same time, and fewer crashes. Route choice still adapts
  (route frequencies change), so freezing the early steps does not block route selection.
| E_lr1e-4_minstd0.2 | lr 1e-4, min_std 0.2 | 751 | **0.5568** | 0.979 | 57.3 | 1.7% / 0.4% |
| F_lr1e-4_ep2 | lr 1e-4, epochs 2 | 1497 | **0.5579** | 0.936 | 52.6 | 3.5% / 2.9% |

* **H4 (more exploration noise) supported**, in an unexpected way: min_std 0.2 learns faster *and* crashes
  least. Training under extra noise forces the policy to keep safety margins that are only needed because
  of the hidden obstacle radius; at evaluation (less noise) it is then very safe.
* **H5 (fewer epochs) supported**: 2 epochs → 2× iterations, fastest policy (52.6 steps), but most crashes.
* Both reach in 15 min what the default reaches in 30 min (≈ 0.56).
* Next hypothesis **H7**: combine the two (epochs 2 + min_std ≥ 0.2) → speed of F with safety of E.
  Batch 2 was cut after its first pair (A15 reference, D2) in favour of these runs.

### Batch 2 (cut short) — 15-min reference and DPPO-freeze variant
| run | change vs. default | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| **A15_default** (reference) | – (15 min) | 741 | 0.5110 | 0.970 | 65.0 | 2.5% / 0.5% |
| D2_ft3_freeze_lr1e-4 | lr 1e-4, ft_steps 3, freeze | 1324 | 0.5281 | 0.951 | 59.7 | 4.5% / 0.4% |

15-min ranking so far: default 0.511 < C (clip 0.1) 0.527 ≈ D2 (ft3) 0.528 < D (ft5) 0.541 < **E (min_std 0.2) 0.557 ≈ F (epochs 2) 0.558**.
D2 skews its routes towards the gap (53%) and crashes more there: with only 3 of 10 steps trainable,
the route is mostly fixed by the frozen early steps and the late steps squeeze through the gap with
small margins. → freezing fewer steps is not the way; drop the DPPO-freeze line for now.

### Batch 3 — combining fewer epochs with more noise (H7), 15-min runs
| run | change vs. default | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| G1 | lr 1e-4, epochs 2, min_std 0.2 | 1469 | 0.5851 | 0.961 | 50.5 | 3.5% / 0.4% |
| **G2** | lr 1e-4, epochs 2, min_std 0.3 | 1470 | **0.5973** | 0.970 | 49.4 | 2.8% / 0.3% |

**H7 strongly supported** (+0.086 over the 15-min default; the effects are super-additive).

**Why more noise helps so much — interpretation (H8).** The PPO clip constrains the probability ratio
N(x; μ', σ²)/N(x; μ, σ²). For a Gaussian, a ratio within 1 ± ε allows a mean shift of order ε·σ along the
sampled noise direction. With the official noise the last two denoising steps have σ = 0.144 and 0,
so with min_std = 0.1 the final step lets the action mean move only by ≈ 0.05 · 0.1 per update and pair:
the clip, not the LR, is the brake (consistent with H1 failing). Raising min_std enlarges the trust
region in action space proportionally, and in addition the training noise forces safety margins.
min_std 0.2 changes the last two steps (σ = 0.144, 0); 0.3 also the third-to-last (0.243).
Prediction: larger min_std keeps helping up to a point where the train/eval mismatch dominates → sweep 0.4, 0.5.
| G3 | lr 1e-4, **epochs 1**, min_std 0.2 | 2254 | **0.5963** | 0.957 | 48.2 | 3.2% / 1.1% |
| G4 | lr 3e-4, epochs 2, min_std 0.2 | 1434 | 0.5752 | 0.942 | 50.3 | 4.5% / 1.3% |

* epochs 1 > epochs 2 at min_std 0.2 (0.596 vs 0.585): one pass over fresh data per iteration is the most
  time-efficient here, again at the price of more crashes (4.3%).
* A higher LR (3e-4) hurts again → LR is not the brake; the trust region/noise is.

### Batch 4 — min_std sweep at epochs 2 (H8), 15-min runs
| run | min_std | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| F | 0.1 | 1497 | 0.5579 | 0.936 | 52.6 | 3.5% / 2.9% |
| G1 | 0.2 | 1469 | 0.5851 | 0.961 | 50.5 | 3.5% / 0.4% |
| G2 | 0.3 | 1470 | **0.5973** | 0.970 | 49.4 | 2.8% / 0.3% |
| H1 | 0.4 | 1441 | 0.5945 | 0.959 | 48.7 | 3.8% / 0.3% |
| H2 | 0.5 | 1426 | 0.5858 | 0.952 | 49.4 | 4.5% / 0.3% |

H8 confirmed: an optimum around min_std 0.3–0.4. Beyond it, crashes rise again — the trained
denoising steps (σ ≥ 0.4 on the last three/four steps) differ too much from the official sampler's
(0.243, 0.144, 0), i.e. the train/eval mismatch starts to dominate.

### Batch 5 — epochs 1 × min_std 0.3/0.4, 15-min runs
| run | change vs. default | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| **I1** | lr 1e-4, epochs 1, min_std 0.3 | 2223 | **0.6092** | 0.977 | 48.1 | 1.8% / 0.4% |
| **I2** | lr 1e-4, epochs 1, min_std 0.4 | 2253 | **0.6105** | 0.981 | 48.3 | 1.9% / 0.0% |

**Both beat the 0.600 baseline after only 15 minutes**, with crash rates back near the pretrained level.
At epochs 1, the larger noise does not cost safety (in contrast to epochs 2) — fewer passes over the
same data means each update stays closer to the data-collecting policy.
Next: confirm at the real 30-min budget with 3 seeds each (training variance in deep RL is high).

### Batch 6 — 30-min confirmation, 3 seeds (in progress)
| run | config | iters | local score | success | steps | crashes obst/wall |
|---|---|---|---|---|---|---|
| K_std0.3_s0 | lr 1e-4, epochs 1, min_std 0.3, seed 0 | 4543 | **0.6260** | 0.982 | 45.9 | 1.5% / 0.3% |
| K_std0.4_s0 | lr 1e-4, epochs 1, min_std 0.4, seed 0 | 4575 | **0.6258** | 0.979 | 45.6 | 2.1% / 0.0% |
