# Phase 4 — First-Principles Mathematics of DDPM
*(with paper cross-references woven in — Phase 5)*

Notation. x₀ ∈ ℝᴰ: a clean data point (D = 3·64·64 here). x₁,…,x_T: progressively noisier versions (T = 1000). q(·): fixed forward process. p_θ(·): learned reverse process, θ = U-Net weights. N(x; μ, Σ): Gaussian density. I: identity. All Gaussians here are isotropic (Σ = σ²I), so every derivation reduces to scalar algebra applied coordinate-wise.

---

## 0. The Gaussian toolkit (prove once, use everywhere)

### 0.1 Density
N(x; μ, σ²I) = (2πσ²)^(−D/2) exp(−‖x−μ‖²/(2σ²)).

### 0.2 Linear transform
If ε ~ N(0, I) then a·ε + b ~ N(b, a²I).
*Proof:* E[aε+b] = aE[ε]+b = b. Cov = a²Cov(ε) = a²I. A linear map of a Gaussian is Gaussian (its characteristic function E[exp(is·(aε+b))] = exp(is·b − a²‖s‖²/2) is a Gaussian's). ∎

### 0.3 Sum of independent Gaussians
If u ~ N(0, σ₁²I), v ~ N(0, σ₂²I) independent, then u+v ~ N(0, (σ₁²+σ₂²)I).
*Proof:* characteristic functions multiply: exp(−σ₁²‖s‖²/2)·exp(−σ₂²‖s‖²/2) = exp(−(σ₁²+σ₂²)‖s‖²/2). ∎
**This single fact is why the forward process collapses to one shot (Section 2).**

### 0.4 Reparameterization trick
Sampling x ~ N(μ, σ²I) is identical to computing x = μ + σε with ε ~ N(0,I) (by 0.2). Written this way, x is a *differentiable function* of μ and σ — so gradients flow through sampling. Every place the code calls `x_t = sqrt_ab*x0 + sqrt_1mab*noise` is this trick.

### 0.5 Completing the square (the workhorse)
For scalars: a x² − 2bx = a(x − b/a)² − b²/a. Hence any density of the form exp(−½(a x² − 2bx + const)) is N(b/a, 1/a). We will use this to read off means/variances of products of Gaussians without ever integrating.

### 0.6 KL divergence between Gaussians
KL(N(μ₁,σ₁²I) ‖ N(μ₂,σ₂²I)) = D·[ log(σ₂/σ₁) + (σ₁² )/(2σ₂²) − ½ ] + ‖μ₁−μ₂‖²/(2σ₂²).
*Derivation:* KL = E₁[log N₁ − log N₂]
= E₁[ −D log σ₁ − ‖x−μ₁‖²/(2σ₁²) + D log σ₂ + ‖x−μ₂‖²/(2σ₂²) ]
E₁‖x−μ₁‖² = Dσ₁². For the last term write x−μ₂ = (x−μ₁)+(μ₁−μ₂):
E₁‖x−μ₂‖² = Dσ₁² + ‖μ₁−μ₂‖² (cross term vanishes since E₁[x−μ₁]=0).
Collect terms → formula. ∎
**Key takeaway:** when both variances are *fixed* (as in DDPM), KL = ‖μ₁−μ₂‖²/(2σ₂²) + const — a KL between Gaussians degenerates into an MSE between means. This is the moment the whole ELBO becomes regression.

---

## 1. Forward process — definition and why those coefficients

DDPM (Ho et al. 2020, **Eq. 2**) defines a Markov chain with a fixed schedule 0 < β₁ < … < β_T < 1:

  q(x_t | x_{t−1}) = N(x_t; √(1−β_t) · x_{t−1}, β_t I),  i.e. x_t = √(1−β_t)·x_{t−1} + √β_t·ε_t.

**Why the √(1−β_t) shrink factor?** Suppose x_{t−1} has zero mean and unit variance per coordinate. Then Var(x_t) = (1−β_t)·1 + β_t = 1. The chain is *variance-preserving* (Song et al., Score-SDE 2021, call it the VP-SDE): the signal shrinks exactly as fast as noise accumulates, so activations never blow up and the same network can handle all t. Drop the shrink factor and variance grows unboundedly (that's the VE / "variance exploding" family, NCSN/EDM territory).

Define α_t := 1 − β_t and ᾱ_t := ∏_{s=1}^t α_s. Schedule in this HW: linear, β from 1e-4 to 0.02, T=1000 (DDPM §4). Then ᾱ_T ≈ e^{Σlnα} ≈ 4·10⁻⁵ ≈ 0 — the endpoint is (numerically) pure N(0, I), which is what lets sampling *start* from `randn`.

## 2. Closed form q(x_t | x₀) — the training shortcut

Claim (DDPM **Eq. 4**): q(x_t | x₀) = N(x_t; √ᾱ_t·x₀, (1−ᾱ_t) I).

*Derivation by composing two steps, then induction.*
x₁ = √α₁ x₀ + √(1−α₁) ε₁
x₂ = √α₂ x₁ + √(1−α₂) ε₂ = √(α₂α₁) x₀ + √(α₂(1−α₁)) ε₁ + √(1−α₂) ε₂.
ε₁ ⊥ ε₂, so by toolkit 0.3 the two noise terms merge into one Gaussian with variance
α₂(1−α₁) + (1−α₂) = α₂ − α₂α₁ + 1 − α₂ = 1 − α₁α₂ = 1 − ᾱ₂. ✓
Induction step: assume x_{t−1} = √ᾱ_{t−1} x₀ + √(1−ᾱ_{t−1}) ε̄. Then
x_t = √α_t x_{t−1} + √(1−α_t) ε_t = √(α_t ᾱ_{t−1}) x₀ + √(α_t(1−ᾱ_{t−1})) ε̄ + √(1−α_t) ε_t,
merged noise variance = α_t(1−ᾱ_{t−1}) + (1−α_t) = 1 − α_t ᾱ_{t−1} = 1 − ᾱ_t. ∎

So training never simulates the chain: **x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, ε~N(0,I)** — one gather, one fused-multiply-add. This line IS `forward_process()`. Interpretation: √ᾱ_t = surviving signal fraction; signal-to-noise ratio SNR(t) = ᾱ_t/(1−ᾱ_t), monotonically decaying from ~10⁴ to ~10⁻⁴ — the schedule is really a choice of SNR decay curve (this reframing is EDM's starting point, Karras et al. 2022).

## 3. The true posterior q(x_{t−1} | x_t, x₀) — the answer key

By Bayes' rule, using Markovianity (q(x_t|x_{t−1},x₀)=q(x_t|x_{t−1})):

  q(x_{t−1}|x_t,x₀) = q(x_t|x_{t−1}) · q(x_{t−1}|x₀) / q(x_t|x₀).

All three factors are known Gaussians (Sections 1–2). Work in exponent space, per coordinate; drop everything not involving x_{t−1} into "C":

−2·log q(x_{t−1}|x_t,x₀) = (x_t − √α_t x_{t−1})²/β_t + (x_{t−1} − √ᾱ_{t−1} x₀)²/(1−ᾱ_{t−1}) + C

Expand and collect powers of x_{t−1}:
coefficient of x_{t−1}²: A = α_t/β_t + 1/(1−ᾱ_{t−1})
coefficient of −2·x_{t−1}: B = (√α_t/β_t)·x_t + (√ᾱ_{t−1}/(1−ᾱ_{t−1}))·x₀

By completing the square (toolkit 0.5) the posterior is N(B/A, 1/A).

Simplify A: A = [α_t(1−ᾱ_{t−1}) + β_t] / [β_t(1−ᾱ_{t−1})] = (1−ᾱ_t) / [β_t(1−ᾱ_{t−1})]
(numerator: α_t − ᾱ_t + β_t = (α_t + β_t) − ᾱ_t = 1 − ᾱ_t).

**Posterior variance** (DDPM **Eq. 7**):  β̃_t := 1/A = β_t · (1−ᾱ_{t−1})/(1−ᾱ_t).

**Posterior mean** (DDPM **Eq. 7**): μ̃_t(x_t,x₀) = B/A =
  [ √α_t (1−ᾱ_{t−1}) x_t + √ᾱ_{t−1} β_t x₀ ] / (1−ᾱ_t).

Sanity checks: coefficients of x_t and x₀ are positive and — check by substituting x_t = √ᾱ_t x₀ — the mean is a convex-like blend pulling x_t slightly toward the signal. β̃_t < β_t always (conditioning on x₀ removes uncertainty). β̃₁ ≈ 0 (last reverse step is nearly deterministic).

**ε-form of the mean** (needed for the algorithm). Invert the closed form: x₀ = (x_t − √(1−ᾱ_t) ε)/√ᾱ_t. Substitute into μ̃ and simplify:
μ̃_t = [√α_t(1−ᾱ_{t−1}) x_t + (√ᾱ_{t−1}β_t/√ᾱ_t)(x_t − √(1−ᾱ_t)ε)] / (1−ᾱ_t)
 = x_t·[α_t(1−ᾱ_{t−1}) + β_t] / [√α_t(1−ᾱ_t)] − ε·[β_t√(1−ᾱ_t)] / [√α_t(1−ᾱ_t)]
 = (1/√α_t) · ( x_t − (β_t/√(1−ᾱ_t)) · ε ).            (DDPM **Eq. 11** with ε in place of ε_θ)
(The x_t bracket used α_t(1−ᾱ_{t−1}) + β_t = 1−ᾱ_t again.)

## 4. Reverse model and the ELBO

Model (DDPM **Eq. 1**): p_θ(x_{0:T}) = p(x_T) ∏ p_θ(x_{t−1}|x_t), p(x_T)=N(0,I),
p_θ(x_{t−1}|x_t) = N(x_{t−1}; μ_θ(x_t,t), σ_t² I) with **fixed** σ_t² (HW/DDPM choice; Improved DDPM learns it). Both σ_t²=β_t and σ_t²=β̃_t work (DDPM §3.2 found them empirically equivalent; they are the two ends of the reverse-variance interval — β̃_t is optimal for a delta data distribution, β_t for a unit-Gaussian one).

**ELBO derivation.** For any latent-variable model, with the forward q as the inference distribution:
log p_θ(x₀) = log ∫ p_θ(x_{0:T}) dx_{1:T} = log E_{q(x_{1:T}|x₀)}[ p_θ(x_{0:T})/q(x_{1:T}|x₀) ]
≥ E_q[ log p_θ(x_{0:T}) − log q(x_{1:T}|x₀) ]   (Jensen: log is concave)  =: −L.  (DDPM **Eq. 3**)
The gap is exactly KL(q(x_{1:T}|x₀) ‖ p_θ(x_{1:T}|x₀)) ≥ 0, so the bound is tight iff q matches the model's true reverse.

**Decomposition into per-step KLs** (DDPM **Eq. 5**, full algebra):
L = E_q[ −log p(x_T) − Σ_{t=1}^T log p_θ(x_{t−1}|x_t) + Σ_{t=1}^T log q(x_t|x_{t−1}) ].
Rewrite each forward factor backwards via Bayes (for t ≥ 2):
q(x_t|x_{t−1}) = q(x_t|x_{t−1},x₀) = q(x_{t−1}|x_t,x₀) · q(x_t|x₀) / q(x_{t−1}|x₀).
Substituting, the ratio q(x_t|x₀)/q(x_{t−1}|x₀) **telescopes** across t=2..T leaving q(x_T|x₀)/q(x₁|x₀); the q(x₁|x₀) cancels against the t=1 term. Result:
L = E_q[ KL(q(x_T|x₀)‖p(x_T)) + Σ_{t=2}^T KL( q(x_{t−1}|x_t,x₀) ‖ p_θ(x_{t−1}|x_t) ) − log p_θ(x₀|x₁) ]
 =: L_T + Σ_{t≥2} L_{t−1} + L₀.
- L_T: no parameters (q fixed, p(x_T) fixed); ≈0 because ᾱ_T≈0. Ignored.
- L₀: decoder term; DDPM §3.3 uses a discretized Gaussian; in practice (and in this HW) it is absorbed into the same MSE.
- L_{t−1}: the heart. **Why this form is beautiful:** every KL compares two Gaussians whose *variances are fixed* — by toolkit 0.6:
  L_{t−1} = ‖μ̃_t(x_t,x₀) − μ_θ(x_t,t)‖² / (2σ_t²) + const.   (DDPM **Eq. 8**)
Variational inference has collapsed into mean-matching regression against the Section-3 answer key.

## 5. The ε-parametrization and L_simple

Choose μ_θ to have the same functional form as μ̃ (Section 3, ε-form), with a network ε_θ predicting the noise (DDPM **Eq. 11**):
  μ_θ(x_t, t) = (1/√α_t) ( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t, t) ).
Then, with x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε,
μ̃_t − μ_θ = (1/√α_t)(β_t/√(1−ᾱ_t)) (ε_θ − ε), so
  L_{t−1} = β_t² / (2σ_t² α_t (1−ᾱ_t)) · ‖ε − ε_θ(x_t,t)‖² .  (DDPM **Eq. 12**)

**L_simple** (DDPM **Eq. 14**): drop the t-dependent weight → L_simple = E_{t~U{1..T}, x₀, ε} ‖ε − ε_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t)‖².
Why dropping the weight helps (DDPM §3.4): the true weight is huge at small t (tiny β_t² but σ_t²α_t(1−ᾱ_t) tinier), over-emphasizing nearly-clean images; uniform weighting shifts effort toward high-noise steps where the *perceptually* hard work happens. This is a deliberate bias away from likelihood toward sample quality — the exact tension Improved DDPM (Nichol & Dhariwal 2021, hybrid loss) and EDM (explicit λ(σ) weighting) later revisit. **This L_simple, with mean over batch and pixels (MSE), is exactly what `compute_loss` implements.**

Training = DDPM **Algorithm 1**: sample x₀, t~Uniform, ε~N(0,I); one gradient step on ‖ε−ε_θ(√ᾱ_t x₀+√(1−ᾱ_t)ε, t)‖².

## 6. Sampling algorithm

DDPM **Algorithm 2**: x_T ~ N(0,I); for t = T,…,1:
  x_{t−1} = (1/√α_t) ( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t) ) + σ_t z,  z~N(0,I) if t>1 else z=0.
The z=0 at t=1 is because the output should be the mean of the final (near-deterministic, β̃₁≈0) step — adding noise there directly degrades the image (a classic Q5(a) bug). This is exactly `reverse_process` (one line of it) inside `sample`'s loop.

## 7. Strided sampling (the math HW Q7 needs)

The per-step β_t are *chain-specific*; you cannot reuse them on a subsequence. But Section 2's closed form only involves ᾱ. For any sub-schedule τ₁<…<τ_S (e.g. every 10th step), the marginals q(x_{τᵢ}|x₀)=N(√ᾱ_{τᵢ}x₀,(1−ᾱ_{τᵢ})I) still hold, and the effective one-step transition has
  α'ᵢ := ᾱ_{τᵢ}/ᾱ_{τᵢ₋₁},  β'ᵢ := 1 − α'ᵢ,  β̃'ᵢ := β'ᵢ(1−ᾱ_{τᵢ₋₁})/(1−ᾱ_{τᵢ}),
because ∏ over the sub-chain must reproduce the same ᾱ. Run Algorithm 2 with primed quantities on the S steps. (This is the "respaced DDPM" of Improved DDPM §4 / the σ=σ̂ special case of DDIM, Song et al. 2021.) Quality degrades gracefully as S shrinks — that's the Q7 finding.

## 8. Alternative parametrizations (the math HW Q6 needs)

From x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε, the pair (x_t known; predict one of ε, x₀) are linearly interchangeable:
  x̂₀ = (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t   ⟺   ε̂ = (x_t − √ᾱ_t x̂₀)/√(1−ᾱ_t).
**x₀-prediction:** train f_θ(x_t,t) ≈ x₀ with ‖x₀ − f_θ‖²; at sampling, plug x̂₀ into the *original* posterior mean μ̃_t(x_t, x̂₀) (Section 3 — no ε-form needed). Same algorithm, same posterior, different network target. Loss relation: ‖ε−ε̂‖² = (ᾱ_t/(1−ᾱ_t))·‖x₀−x̂₀‖² = SNR(t)·‖x₀−x̂₀‖² — so "unweighted x₀ loss" = "1/SNR-weighted ε loss": x₀-prediction *implicitly reweights* timesteps (down-weights low-noise steps, over-weights the ambiguous high-noise ones where x₀ is nearly unrecoverable) → typically slightly worse KID at T=1000. 
**v-prediction** (Salimans & Ho 2022, distillation paper, Eq. 11): v := √ᾱ_t ε − √(1−ᾱ_t) x₀, the "velocity" on the (x₀, ε) circle — well-conditioned at both SNR extremes; the modern default (used by Stable Diffusion 2, Imagen Video).
**Score connection:** ∇_{x_t} log q(x_t|x₀) = −(x_t−√ᾱ_t x₀)/(1−ᾱ_t) = −ε/√(1−ᾱ_t); by Tweedie/vincent denoising score matching the marginal score satisfies ∇ log q(x_t) = −E[ε|x_t]/√(1−ᾱ_t) ≈ −ε_θ/√(1−ᾱ_t). ε-prediction *is* score estimation up to scale — the bridge to Score-SDE, probability-flow ODEs, and flow matching (HW2+).

## 9. Equation → code map

| Math | Paper | Code |
|---|---|---|
| β schedule, α, ᾱ buffers | DDPM §4, Eq.4 | `DDPM.__init__` (register_buffer) |
| x_t = √ᾱ x₀ + √(1−ᾱ) ε | Eq. 4 | `forward_process` |
| L_simple | Eq. 14 / Alg. 1 | `compute_loss` |
| μ_θ, σ_t, one step | Eq. 11 / Alg. 2 | `reverse_process` |
| full loop, z=0 at t=1 | Alg. 2 | `sample` |
| ᾱ-ratio respacing | Improved DDPM §4 | `sample(num_steps=...)` (Q7) |
| x₀ ↔ ε swap | Eq. 4 inverted | Q6 variant of `compute_loss`/`reverse_process` |
