# Rebuild-From-Scratch Checklist

*The goal: close every reference and reimplement DDPM → DDIM → FM → Reflow →
DPM-Solver from a blank file. Each item names (1) what to build, (2) the one
formula you must be able to re-derive first, (3) the self-test that proves your
code is right WITHOUT training anything. Do them in order — each layer's test
uses the previous layer.*

Derivations: `04/05` (DDPM), `06` (FM). Working reference code: `cmu-10799-diffusion/src/methods/`, `kaist/*/…_solved.ipynb`.

## Level 0 — Muscle memory (no code)
Re-derive on paper, from nothing:
- [ ] Var rules → sum of independent Gaussians (05 §A1–A4)
- [ ] x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε by two-step merge + induction (05 §B2)
- [ ] Posterior μ̃, β̃ by completing the square (05 §C3)
- [ ] μ̃ rewritten with ε; why loss = MSE on ε (05 §D3–D4)
- [ ] FM: why regressing x₁−x₀ learns the marginal field (06 §D2)
- [ ] ELBO: master identity log p = ELBO + KL, telescoping, KL→MSE (09)
- [ ] DDIM re-noising family (η dial) and Solver-1 ≡ DDIM (10 §1.2, §5)
If any step needs the notes → repeat tomorrow. These seven are the whole edifice.

## Level 1 — Schedule + forward (30 min)
Build: β linspace(1e-4, 0.02, 1000) → α, ᾱ buffers (fp64 cumprod, store fp32);
`extract(buf, t, ndim)` gather+reshape helper; `forward_process(x0, t, noise)`.
- [ ] Self-test 1: ᾱ[0]=0.9999, ᾱ[-1]≈4.0e-5.
- [ ] Self-test 2: x_t at t=999 over a big batch has mean≈0, std≈1.
- [ ] Bug to remember: (B,) coefficients MUST become (B,1,1,1) before
      multiplying (B,C,H,W) — centralize in `extract`, never inline.

## Level 2 — Loss (15 min)
Build: `compute_loss`: t~randint(0,T), ε~randn, MSE(net(x_t,t), ε).
- [ ] Self-test: with a zero-output net, loss = 1.000 ± 0.05 (it's E‖ε‖²).
      (For x₀-target: loss = Var(data); for FM velocity-target: Var(data)+1.)

## Level 3 — Posterior + ancestral sampling (1 h)
Build: posterior coefficient buffers (β̃, coef_x0, coef_xt);
`_predict_x0` (invert forward + clamp [−1,1]); `reverse_process` (posterior
mean + √β̃·z, z=0 on last step); `sample` loop from randn.
- [ ] Self-test 1: your (x_t,x₀)-form posterior mean == Eq.11 ε-form to ~1e-6
      for random inputs, all t (this catches almost every algebra slip).
- [ ] Self-test 2: β̃₁ ≈ 1e-5-ish (≈0); β̃_t < β_t everywhere.
- [ ] Bugs to remember: noise added at the final step (Q5a); missing
      unnormalize+clamp when saving (Q5c); sampling with raw instead of EMA
      weights; t indexing 0-based in code vs 1-based in the paper.

## Level 4 — U-Net (2 h)
Build: time-embedding MLP → FiLM ResBlocks → attention at 16/8 → skip-cat
decoder with `num_res_blocks+1` blocks per level → zero-init output conv.
- [ ] Self-test 1: out.shape == in.shape; untrained output exactly 0 (zero init).
- [ ] Self-test 2: skip bookkeeping — assert the skip list is empty after the
      decoder consumed it.
- [ ] Bug to remember: decoder in_channels = current + skip channels; track
      them in a list during construction, don't compute arithmetically.

## Level 5 — Overfit gate (30 min wall-clock)
- [ ] Train on ONE repeated batch: loss 1.0 → <0.05 within ~400 steps and
      samples reproduce the batch. Only after this gate, spend GPU-hours.
      (Loss stuck at 1.0 ⇒ t or ε plumbing broken; loss falls but samples are
      noise ⇒ sampler broken — the loss never tests the sampler!)

## Level 6 — DDIM + striding (45 min)
Build: subsequence ts = linspace(T−1, 0, S); per-jump α_eff = ᾱ_t/ᾱ_s (never
reuse per-step β!); DDIM step x_prev = √ᾱ_prev·x̂₀ + √(1−ᾱ_prev)·ε̂.
- [ ] Self-test 1: α_eff with adjacent steps == α_t exactly.
- [ ] Self-test 2: DDIM with the same seed twice → bit-identical samples;
      ancestral → different across runs (fresh z).
- [ ] Self-test 3: DDIM final step (ᾱ_prev=1) returns x̂₀ exactly.
- [ ] Re-derive: recompute ε̂ from the CLAMPED x̂₀ so (x̂₀, ε̂, x_t) stay
      consistent.

## Level 7 — Flow matching (45 min)
Build: x_t = (1−t)x₀ + t·x₁; loss MSE(v_θ(x_t, t·1000), x₁−x₀); Euler sampler
t: 0→1.
- [ ] Self-test 1: interpolant endpoints: t=0 → x₀, t=1 → x₁ exactly.
- [ ] Self-test 2 (the strongest single test in this whole file): build an
      ORACLE velocity net that returns the true x₁−x₀ given the state; your
      Euler sampler must reconstruct x₁ to ~1e-6 in ONE step. Catches
      direction, dt, and time-scaling bugs simultaneously.
- [ ] Bug to remember: t∈[0,1] fed raw to a sinusoidal embedding designed for
      0..1000 collapses all time embeddings — scale it, identically in train
      and sample.

## Level 8 — Reflow (30 min)
Build: sample N trajectories from the trained flow, store COUPLED (x₀ = the
exact starting noise, x₁̂ = where it landed); retrain the same architecture on
straight lines between couples; sample with ~10 steps.
- [ ] Self-test: few-step CD/quality of reflowed model ≫ plain FM at the same
      step count (ours: 25.7 vs 36.9 CD at 10 steps).
- [ ] Bug to remember: re-randomizing x₀ in __getitem__ silently turns Reflow
      back into plain FM — the pairing IS the method.

## Level 9 — DPM-Solver (1.5 h, needs Level 6 understood)
Re-derive first: PF-ODE semi-linearity → variation of constants → λ = log(α/σ)
change of variables → x_t = (α_t/α_s)x_s − σ_t(e^h−1)ε̂, h = λ_t−λ_s.
Build: DDPM→DPM notation (α=√ᾱ, σ=√(1−ᾱ), λ), order-1 step, midpoint order-2.
- [ ] Self-test 1: your order-1 step == your DDIM step numerically (they are
      the same formula in different clothes — prove it by expanding e^h).
- [ ] Self-test 2: λ strictly DECREASES with t (increases toward data); h > 0
      for a denoising step.
- [ ] Insight to remember: order-2 wins at LOW NFE (≤20); at 50+ NFE on easy
      data it can tie or lose (coarser grid per step) — ours: 15.4 (o1) vs
      19.6 (o2) at 50 NFE.

## Level 10 — Training hygiene (learned the hard way)
- [ ] EMA shadow + apply/restore bracketing for every sampling call, WITH
      decay warmup min(d,(1+step)/(10+step)): the shadow starts as a clone of
      the random init, which retains d^step weight -- at d=0.9999 that is
      13.5% after 20k steps = mush samples and noise-level KID from a
      perfectly healthy model (we measured 0.474). Self-test: early in
      training, EMA samples should look no worse than raw-weight samples.
- [ ] bf16 autocast (or fp16 + GradScaler + clipping) — and a non-finite-loss
      skip guard: one bad batch must never write NaN into the weights.
      (Our x₀ run died at step ~20k without it; restart cost 1.5 h.)
- [ ] Loss curves don't measure sample quality: evaluate checkpoints with
      KID/CD, log sample grids every few thousand steps.
- [ ] Never trust "the file exists" as "the code ran" — check for execution
      evidence (outputs, metrics, timestamps).
