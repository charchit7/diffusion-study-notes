# DDIM and DPM-Solver From Zero
*The two fast samplers we implemented (CMU HW2, KAIST A2), derived from
scratch. Prerequisites: 05 Parts A–B (Gaussian toolkit, forward shortcut) and
06 Part A (velocity/Euler). New tools built here: a little more calculus —
each piece taught before use.*

Notation (DPM style, used throughout): α_t := √ᾱ_t (signal scale),
σ_t := √(1−ᾱ_t) (noise scale), so the forward shortcut reads
**x_t = α_t x₀ + σ_t ε**. λ_t := log(α_t/σ_t) is the log signal-to-noise
ratio: huge at t≈0 (clean), very negative at t=T (noise), strictly
decreasing in t.

---

## Part 1 — DDIM, with no calculus at all

### 1.1 The key observation: training never sees trajectories
Recall (09 Part C): the loss decomposes over timesteps, and each term only
involves x_t drawn from the MARGINAL q(x_t|x₀) = N(α_t x₀, σ_t²I). The
network is trained purely on "here is a marginal sample at level t". So ANY
sampling procedure whose states visit those same marginals is compatible
with the same trained network — the Markov ancestral chain is just one
choice. (This is the whole insight of Song et al. 2021.)

### 1.2 Build a sampler by "re-noising the estimate"
Suppose we stand at x_t and the network gives ε̂, hence the clean estimate
x̂₀ = (x_t − σ_t ε̂)/α_t (invert the shortcut — one line of algebra).
To move to a LOWER noise level s < t, manufacture a sample that looks like a
level-s marginal:

    x_s = α_s x̂₀ + (noise part with total variance σ_s²).

The noise part is our design freedom. Split it into a reused piece and a
fresh piece, sizes chosen so the variances add correctly (05 §A1 Rule 2):

    x_s = α_s x̂₀ + √(σ_s² − η²σ̃²) · ε̂ + η σ̃ · z,   z ~ N(0, I)   (1)

- Check the bookkeeping: if x̂₀ were the true x₀ and ε̂ a true unit noise,
  the total noise variance is (σ_s² − η²σ̃²) + η²σ̃² = σ_s². Marginal
  preserved. ✓ (σ̃ is the ancestral posterior std from 05 Part C.)
- **η = 0: DDIM.**  x_s = α_s x̂₀ + σ_s ε̂ — reuse ALL of the predicted
  noise, add none. Fully deterministic: the same x_T always maps to the same
  image. This is exactly HW2 Eq. 1 and our `ddim_step`.
- **η = 1: ancestral DDPM again.** With σ² = σ̃², expanding (1) in terms of
  (x_t, ε̂) reproduces the posterior mean of 05 Part C plus σ̃z. (Verify
  numerically in two lines — our test suite does exactly this comparison.)

So DDPM's stochastic sampler and DDIM sit at two ends of one dial, and the
dial never touches training. Why DDIM tolerates big jumps better: taking a
large stride with η=1 injects a large fresh noise the model must then remove
with fewer remaining steps; η=0 injects nothing, so the only error is the
model's own imperfection in ε̂.

### 1.3 What DDIM secretly is
Fix the ε̂ it reuses and look at the states it visits: x = α x̂₀ + σ ε̂ as
(α, σ) slide along the schedule — a smooth CURVE through the noise levels.
Deterministic path-following = an ODE solver in disguise. Part 2 makes that
precise and then improves on it.

---

## Part 2 — The calculus toolkit (built gently, like 06 Part A)

### 2.1 Derivative = velocity (recap of 06 A1)
x'(t) ≈ (x(t+Δ) − x(t))/Δ. Two rules, both checkable numerically:
- Product rule: (uv)' = u'v + uv'. Check with u=t, v=t²: (t³)' = 3t² and
  u'v+uv' = t² + 2t² = 3t². ✓
- Exponential: (e^{at})' = a·e^{at} ("growth rate proportional to size" —
  that's the definition of compounding). Special case a=1 at t=0 gives A2 of
  09. Also (log u)' = u'/u (run the exponential rule backwards through
  u = e^{log u}).

### 2.2 Integrals = adding up velocity
∫_s^t v(τ)dτ is "total displacement from velocity v" — the exact version of
Euler's sum of small steps. The only integral we will ever compute:
∫ e^{−λ} dλ = −e^{−λ} (check: derivative of −e^{−λ} is e^{−λ} by 2.1). ✓

### 2.3 The integrating-factor trick (the one real technique)
Problem: solve x'(t) = f(t)·x(t) + b(t) — "growth at rate f plus an external
input b". Divide out the growth: define A(t) with A'/A = f (i.e. A = e^{∫f}),
and look at the rescaled variable x/A:

    (x/A)' = x'/A − x·A'/A² = (x' − f·x)/A = b(t)/A(t).      (product rule)

The rescaled variable has NO self-growth — it just accumulates input. Add up:

    x(t)/A(t) − x(s)/A(s) = ∫_s^t b(τ)/A(τ) dτ
    ⇒  x(t) = (A(t)/A(s))·x(s) + A(t)·∫_s^t b(τ)/A(τ) dτ      (2)

**The linear part is EXACT — forever, for any step size.** Only the input
integral needs approximating. This one formula is all of DPM-Solver.

---

## Part 3 — The probability-flow ODE (derived the gentle way)

Take one data point x₀ and one FIXED noise ε, and follow the deterministic
curve x(t) = α_t x₀ + σ_t ε (this is precisely the path DDIM walks, §1.3).
Differentiate with the toolkit:

    x'(t) = α'_t x₀ + σ'_t ε.

Now eliminate (x₀, ε) in favor of the current state: from the shortcut,
x₀ = (x − σ_t ε)/α_t. Substitute and collect terms in x and ε:

    x' = (α'_t/α_t)(x − σ_t ε) + σ'_t ε
       = (α'_t/α_t)·x + (σ'_t − σ_t·α'_t/α_t)·ε                     (3)

A **semi-linear** equation: linear in x (rate f(t) = α'_t/α_t = (log α_t)')
plus an ε-input. For the population instead of one pair, the same
best-guess-is-the-average argument as flow matching (06 §A3/D2) replaces the
unknowable per-pair ε with its conditional mean — the network's ε̂(x, t).
Equation (3) with ε̂ is the probability-flow ODE. (The rigorous route goes
through the Fokker–Planck equation — the one result here we take on faith,
axiom #3 of these notes; (3) is its exact single-pair shadow.)

## Part 4 — Solve the linear part exactly, then change clocks

Apply formula (2) to (3): f = (log α)' so A(t) = α_t, and
b(t) = (σ'_t − σ_t (log α_t)')·ε̂. Then

    x_t = (α_t/α_s)·x_s + α_t ∫_s^t [σ'_τ/α_τ − σ_τ α'_τ/α_τ²]·ε̂(x_τ, τ) dτ.

Tidy the bracket: it is exactly (σ_τ/α_τ)' by the product/quotient rule —
and σ/α = e^{−λ} by the definition of λ. So the bracket is (e^{−λ_τ})' =
−λ'_τ e^{−λ_τ}, and **changing the clock from τ to λ** (each tick of the
integral becomes a tick of λ; the λ' cancels — that's what substitution IS):

    x_t = (α_t/α_s)·x_s − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂(x_λ, λ) dλ     (4)

Everything hard now lives in one exponentially-weighted integral of ε̂ over
log-SNR. λ is the natural clock of diffusion: equal steps in λ are equal
multiplicative steps in signal-to-noise.

## Part 5 — DPM-Solver-1: freeze ε̂, integrate exactly

Approximate ε̂ as CONSTANT over the step (its value at the start, λ_s). Pull
it out of (4) and use the one integral we know (2.2):

    ∫_{λ_s}^{λ_t} e^{−λ} dλ = e^{−λ_s} − e^{−λ_t}

    x_t = (α_t/α_s) x_s − α_t (e^{−λ_s} − e^{−λ_t}) ε̂
        = (α_t/α_s) x_s − α_t e^{−λ_t} (e^{h} − 1) ε̂,    h := λ_t − λ_s > 0
        = (α_t/α_s) x_s − σ_t (e^{h} − 1) ε̂                          (5)

(using α_t e^{−λ_t} = α_t·σ_t/α_t = σ_t). This is our `first_order_step`.

**Claim: (5) is exactly DDIM.** Expand e^h = e^{λ_t}/e^{λ_s} =
(α_t/σ_t)/(α_s/σ_s) = α_t σ_s/(α_s σ_t). Then

    σ_t(e^h − 1) = α_t σ_s/α_s − σ_t, so
    x_t = (α_t/α_s) x_s − (α_t/α_s)σ_s ε̂ + σ_t ε̂
        = α_t · (x_s − σ_s ε̂)/α_s + σ_t ε̂  =  α_t x̂₀ + σ_t ε̂.   ∎

The re-noising construction of Part 1 and the exact-linear-part ODE solution
are the same formula — derived twice, from opposite ends. (Our test suite
confirms it numerically; KAIST A2 grading calls it "DPM-Solver-1 ≡ DDIM".)

## Part 6 — DPM-Solver-2: evaluate ε̂ at the midpoint instead

Why freezing at the start is the error: over the step, ε̂ drifts; Euler-type
rules pay an error ∝ h²·(slope of ε̂). The classic fix (midpoint rule): use
the value at the CENTER of the interval — start-vs-mid drift and mid-vs-end
drift then cancel to first order, leaving error ∝ h³.

Algorithm (per step s → t, h = λ_t − λ_s):
1. Find the λ-midpoint time: s_mid = t(λ) at λ = (λ_s + λ_t)/2 (invert the
   λ schedule — table lookup / interpolation in code).
2. Half-step with the start estimate (formula (5) with h/2):
   u = (α_mid/α_s)·x_s − σ_mid(e^{h/2} − 1)·ε̂(x_s, s)
3. Re-evaluate the network AT the midpoint: ε̂_mid = ε̂(u, s_mid).
4. Full step from s using the midpoint estimate:
   x_t = (α_t/α_s)·x_s − σ_t(e^{h} − 1)·ε̂_mid.

Cost: 2 network calls per step ⇒ compare solvers at equal NETWORK CALLS
(NFE), not equal steps. Empirical shape (our KAIST A2 numbers): at 50 NFE on
easy 2D data, order-1 15.4 vs order-2 19.6 Chamfer (both converged; order-2's
halved grid slightly hurts); at ≤20 NFE order-2 wins clearly — its design
regime, matching the h³-vs-h² error analysis.

## Part 7 — Cheat sheet

| # | Result | Where |
|---|---|---|
| 1 | x_s = α_s x̂₀ + √(σ_s²−η²σ̃²) ε̂ + ησ̃ z — one dial: η=1 DDPM, η=0 DDIM | §1.2 |
| 2 | x' = f x + b solved exactly: x_t = (A_t/A_s)x_s + A_t∫b/A, A=e^{∫f} | §2.3 |
| 3 | Conditional-path ODE: x' = (log α)'x + (σ' − σ(log α)')ε | §3 |
| 4 | λ-form: x_t = (α_t/α_s)x_s − α_t ∫e^{−λ}ε̂ dλ | §4 |
| 5 | Solver-1: x_t = (α_t/α_s)x_s − σ_t(e^h−1)ε̂  ≡ DDIM (proved) | §5 |
| 6 | Solver-2: same with ε̂ evaluated at the λ-midpoint (2 NFE/step) | §6 |

Axioms so far across all notes (everything else is derived): (i) sum of
independent Gaussians is Gaussian (05 §A4), (ii) FM's average-arrow field
transports the mixture (06 §D1), (iii) the per-pair ODE averages to the
population PF-ODE (§3 here). All three are the same flavor: "per-pair
constructions average into population laws."
