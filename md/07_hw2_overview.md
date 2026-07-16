# HW2 Phase 1 — Assignment Overview: Flow Matching & DDIM (100 pts)

Same repo as HW1. New method file + one new sampler for the old method.

## What each question tests

### Q1 (0 pts, optional) — 1D playground + overfit-single-batch for FM
Same debugging strategy as HW1. Worth 30 min; FM converges visibly faster in 1D.

### Q2 (25 pts) — Implement & train Flow Matching
- **Concept:** conditional straight-line paths x_t = (1−t)·noise + t·data, target
  velocity v = data − noise, MSE regression; Euler sampler. (Full derivation:
  notes/06, Parts C–E.)
- **Constraint:** "roughly the same architecture, iterations, batch size as your
  DDPM" — this makes Q5's comparison scientifically fair (same capacity+compute).
- **Deliverables:** loss curve, model size, batch, iterations, GPU-hours,
  16-grid, KID<0.005 @1k samples. Sampling steps: ~100 Euler steps.
- **Traps:** (1) time convention flips vs DDPM (0=noise here); pick one and be
  consistent between compute_loss and sample. (2) t is continuous [0,1] — feed
  t*1000 to the sinusoidal embedding, else all embeddings collapse into a tiny
  frequency band and conditioning is mush. (3) same (B,1,1,1) broadcast care.
- **Difficulty:** ★★ given HW1 infra (the loss is 4 lines; the sampler 6).

### Q3 (15 pts) — Kale debugging, FM edition
- (a) FM converged loss >> DDPM converged loss — expected? YES. Different
  regression targets: DDPM predicts unit-variance ε (per-pixel loss floor well
  below 1 after training); FM predicts v = x₁−x₀ with Var = Var(data)+1 ≈ 2·
  larger scale, AND the loss floor includes the irreducible variance of the
  average arrow E[Var(v|x_t)], which is large at small t where many pairs share
  x_t. Losses across different objectives are NOT comparable; compare KID.
- (b) FM loss spikes/NaN early with DDPM's hyperparameters. Cause candidates:
  fp16 overflow under AMP (target scale ~2x, gradient spikes trip the scaler),
  no/too-loose gradient clipping, LR tuned for eps-scale outputs. Fixes (name 2):
  gradient clipping (norm 1.0), lower LR or warmup, bf16/fp32 fallback, loss/
  target normalization.
- (c) Converged loss, 200 Euler steps, samples = colorful abstract mush.
  Debug path: the loss curve tests training only — bug is in the SAMPLER.
  Checklist: integration direction (starting at data-side t and walking wrong
  way), Δt sign/size (x += v·(1/N)), t fed to network not matching training
  scale (t*1000 vs t), starting from randn but integrating t from 1 instead of
  0, and forgetting unnormalize/clamp on save. Recommend: sample with N=1000
  tiny steps (isolates integrator vs field), visualize trajectory x_t at
  t=0,0.25,...,1, and overfit-single-batch then sample (should reproduce batch).

### Q4 (20 pts) — DDIM sampling for the HW1 DDPM (no retraining)
- **Concept:** DDPM's training only constrains the marginals q(x_t|x_0); a whole
  FAMILY of non-Markovian reverse processes shares them (Song et al. 2021).
  The eta=0 member is deterministic:
      x_{t_prev} = sqrt(ab_prev)·x0_hat + sqrt(1−ab_prev)·eps_hat   [HW Eq. 1]
  "Re-noise the predicted clean image to the LOWER noise level, reusing the
  SAME predicted noise instead of fresh randomness."
- Given our HW1 reverse_process already computes x0_hat explicitly, DDIM is a
  different recombination line — the striding machinery (ts subsequence,
  t_prev=-1 ⇒ ab_prev=1 ⇒ return x0_hat) is already built and shared.
- **Deliverables:** 16-grid @100 DDIM steps; KID @1k samples/100 steps vs
  DDPM-1000 and FM.
- **Traps:** using eps_hat inconsistently with the clamped x0_hat (recompute
  eps from clamped x0 for self-consistency); confusing DDIM striding with HW1
  Q7's stochastic striding (respaced ancestral) — DDIM has NO noise injection.
- **Difficulty:** ★★ math understanding, ★ code.

### Q5 (25 pts) — Sampling steps ablation (the payoff table)
KID for FM-Euler and DDIM at steps ∈ {1, 5, 10, 50, 100, 200, 1000}, 1k samples
each = 14 eval runs (+ compare vs HW1 DDPM-1000 baseline). All inference-only.
- Expected shape of results: at 1 step both are poor (FM: average-arrow blur;
  DDIM: single jump = its x0_hat, blurry "average face"); DDIM typically usable
  by ~50-100 steps; FM usable by ~10-50 (straighter marginal paths); both ≈
  DDPM-1000 at high step counts; ancestral DDPM at 100 steps is worse than
  DDIM at 100 (noise re-injection amplifies model error at big strides).
- **Trap:** budget the wall-clock — 14 × 1k samples; batch as large as VRAM
  allows; reuse the fidelity cache for the real-dataset side.

### Q6 (10 pts) — Track selection (Fidelity / Controllability / Speed)
Personal choice; graded on reasoning. Given this codebase and results, Speed
track composes naturally with FM+DDIM results (distillation, consistency);
Controllability composes with the attribute CSV we already have (conditioning
on the 40 attributes). Decide after seeing the Q5 table.

### Q7 (5 pts) — Reflection + resources (cite AI assistance).

## Time & difficulty

| Q | time | difficulty |
|---|------|-----------|
| Q2 FM implement+train | 2-3 h code+test, ~15 h GPU (unattended) | ★★ |
| Q3 debugging | 1 h | ★★ |
| Q4 DDIM | 1-2 h | ★★ |
| Q5 ablation | 2 h + ~3-4 h GPU inference | ★★ |
| Q6+Q7 | 1 h | ★ |

## Dependency graph
```
HW1 DDPM checkpoint ──► Q4 DDIM ──┐
notes/06 (FM math) ──► Q2 FM ─────┼──► Q5 ablation table ──► Q6 track ──► Q7
                        └── Q3 debugging (conceptual, after Q2 works)
```

## GPU plan (2×A6000 shared with HW1 runs)
1. HW1 eps run (GPU0) → stop when KID<0.005 at a checkpoint (est. 30-50k steps).
2. HW1 x0 run (GPU1) → stop at the SAME step count for a fair Q6(HW1) comparison.
3. FM training (Q2) takes the freed GPU0 (~15 h at same config).
4. Q4/Q5 evals are inference-only, run on whichever GPU is free.
