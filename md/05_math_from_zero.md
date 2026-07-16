# DDPM Math From Zero
*Every equation derived assuming no calculus and shaky algebra. Longer but gentler
than `04_mathematics.md`. Read top to bottom; nothing is used before it is built.*

Throughout, we work with ONE pixel (a single number). Images are just many pixels,
and every formula applies to each pixel independently — so scalar math is all we need.

---

## Part A — The tools

### A1. Mean and variance (the only two statistics we need)

A random number X has:
- **mean** E[X]: the long-run average if you sampled X forever.
- **variance** Var(X) = E[(X − mean)²]: the average *squared* distance from the mean.
  Big variance = spread out. Zero variance = always the same number.
- **standard deviation** σ = √Var: same thing in the original units.

Two rules we will use constantly. Both follow just from "average of a sum = sum of
averages" (that's all E[·] is — a glorified average):

**Rule 1 (scaling):** Var(a·X) = a²·Var(X).
*Why:* a·X sits a times further from its mean, distance gets squared → a².
Check with numbers: X is ±1 coin flip (mean 0, Var = 1). Then 3X is ±3,
Var = average of 9 = 9 = 3²·1. ✓

**Rule 2 (adding independent randomness):** if X and Y are independent,
Var(X + Y) = Var(X) + Var(Y).
*Why:* Var(X+Y) = E[(X+Y)²] (assuming means are 0 to keep it short)
= E[X² + 2XY + Y²] = E[X²] + 2·E[XY] + E[Y²].
Independence makes E[XY] = E[X]·E[Y] = 0. Left with Var(X) + Var(Y). ∎
Check: two independent coin flips, sum has values −2, 0, 0, +2 → variance
= (4+0+0+4)/4 = 2 = 1 + 1. ✓

⚠ Note variances add — standard deviations do NOT (√2 ≠ 1+1).

### A2. The Gaussian (normal distribution)

"X ~ N(μ, σ²)" means: X is random, centered at μ, spread σ, with the bell-curve
shape. Its density (the curve's height at point x) is

    N(x; μ, σ²) = (1/√(2πσ²)) · exp( −(x−μ)² / (2σ²) )

You do not need to integrate this, ever, in DDPM. You only need to *read* it:
**everything about a Gaussian is stored in the exponent.** If someone hands you

    (something) · exp( −(x − STUFF₁)² / (2·STUFF₂) )

you instantly know: this is a Gaussian with mean STUFF₁ and variance STUFF₂.
The messy front factor is forced (curves must have total area 1), so we can ignore
it and track only exponents. This "read the exponent" trick replaces all calculus
in the posterior derivation below.

### A3. Building any Gaussian from the standard one (reparameterization)

Let ε ~ N(0, 1) ("standard normal": mean 0, variance 1 — `torch.randn`). Then

    X = μ + σ·ε   is exactly   N(μ, σ²).

*Why:* mean: E[μ + σε] = μ + σ·0 = μ. Variance: Rule 1 gives Var(σε) = σ²·1
(adding the constant μ shifts everything equally, changing no distances). Shape
stays bell-curved because stretching and shifting a bell is still a bell. ∎

This is THE most important line in the file. Every `mean + std * torch.randn(...)`
in the code is this identity.

### A4. Sum of two independent Gaussians is Gaussian

If u ~ N(0, a²) and v ~ N(0, b²) are independent, then u + v ~ N(0, a² + b²).
Gaussian-ness of the sum we take on faith (it's the one fact we don't prove; it is
why the Gaussian is called "stable"). The variance a² + b² is just Rule 2.
Check: N(0,9) + N(0,16) = N(0,25): σ's are 3 and 4, combined σ is 5, not 7.

### A5. Completing the square (algebra, not calculus)

Any expression  a·x² − 2b·x  can be rewritten as  a·(x − b/a)² − b²/a.
*Proof by expanding the right side:* a·(x² − 2(b/a)x + b²/a²) − b²/a
= a·x² − 2b·x + b²/a − b²/a = a·x² − 2b·x. ∎
Numeric check: 3x² − 12x (a=3, b=6) → 3(x−2)² − 12 → 3x²−12x+12−12 ✓.

Combined with A2: if a density's exponent is −½(a·x² − 2b·x + junk-without-x),
complete the square → −½·a·(x − b/a)² + const, and by "read the exponent":

    **mean = b/a,  variance = 1/a.**    (★ memorize this)

### A6. KL divergence: "how different are two distributions"

KL(P‖Q) ≥ 0, and = 0 only if P = Q. Think: a penalty for using Q when the truth
is P. The only fact we need (standard result, derivable by expanding definitions
with A1-style algebra — done in 04_mathematics.md §0.6):

    KL( N(μ₁, σ²) ‖ N(μ₂, σ²) ) = (μ₁ − μ₂)² / (2σ²)      [same variance!]

Read it: **for equal-variance Gaussians, "make the distributions match" literally
means "make the means match, squared-error style."** This single line is why DDPM
training ends up being plain MSE regression.

---

## Part B — The forward process (destroying the image)

### B1. One step

Fix T = 1000 numbers 0 < β₁ < β₂ < … < β_T < 1 (the *schedule*; in this HW,
evenly spaced from 0.0001 to 0.02). One noising step is:

    x_t = √(1−β_t) · x_{t−1} + √β_t · ε_t,      ε_t ~ N(0,1) fresh each step

In words: shrink the current pixel slightly (multiply by √(1−β_t) ≈ 0.9999),
then add a small amount of fresh noise (√β_t ≈ 0.01 worth).

**Why the strange √(1−β_t) instead of just adding noise?** Bookkeeping of variance.
Suppose Var(x_{t−1}) = 1. Then by Rules 1 and 2:

    Var(x_t) = (√(1−β_t))²·1 + (√β_t)²·1 = (1−β_t) + β_t = 1.   ✓

The shrink factor is chosen SO THAT variance stays exactly 1 forever ("variance
preserving"). Signal fades, noise grows, total "energy" constant — the network
always sees inputs of the same scale, at every t. Kill the shrink factor and
variance grows every step until activations explode.

### B2. Shortcut: jump to step t directly (the key training trick)

Define α_t = 1 − β_t and ᾱ_t = α₁·α₂·⋯·α_t (product; "alpha bar"). Claim:

    x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε,       ε ~ N(0,1)        (Eq. ★★)

meaning: to corrupt to level t you DON'T need t steps — one multiply, one add.

*Derivation for two steps, slowly.* Write out step 2 using step 1:

    x₂ = √α₂·x₁ + √β₂·ε₂
       = √α₂·(√α₁·x₀ + √β₁·ε₁) + √β₂·ε₂          [substitute x₁]
       = √(α₂α₁)·x₀ + √α₂√β₁·ε₁ + √β₂·ε₂         [multiply through]

The last two terms are independent Gaussians (fresh ε's!). By A4 they merge into
ONE Gaussian noise with variance = sum of variances:

    α₂·β₁ + β₂ = α₂(1−α₁) + (1−α₂) = α₂ − α₂α₁ + 1 − α₂ = 1 − α₁α₂

So x₂ = √(α₁α₂)·x₀ + √(1−α₁α₂)·(one standard noise) — exactly Eq. ★★ with
ᾱ₂ = α₁α₂. Repeat the same substitute-and-merge t times (nothing new happens,
the same algebra just recycles) and you get Eq. ★★ for every t. ∎

**Read Eq. ★★:** √ᾱ_t = "fraction of original signal still alive at time t";
√(1−ᾱ_t) = "how loud the noise is." Since each α < 1, the product ᾱ_t slides
from ≈1 down to ≈0.00004 at t=1000. So x₁₀₀₀ = 0.006·x₀ + 0.99998·noise ≈ pure
noise. That is the whole point: **the endpoint of destruction is a distribution
we can sample for free** (`torch.randn`).

Worked numbers (linear schedule): ᾱ₁ = 0.9999; ᾱ₂₅₀ ≈ 0.48 (about half signal,
half noise energy); ᾱ₅₀₀ ≈ 0.05; ᾱ₁₀₀₀ ≈ 0.00004.

Code: `forward_process()` is literally Eq. ★★ — gather √ᾱ_t and √(1−ᾱ_t) per
sample, fused multiply-add.

---

## Part C — The posterior (the "answer key" for one undo-step)

### C1. The question it answers

Suppose I know BOTH the noisy pixel x_t AND the original clean pixel x₀.
Question: what was x_{t−1}? Answer: still random (many micro-paths possible), but
its distribution is exactly computable, and it is Gaussian:

    q(x_{t−1} | x_t, x₀) = N( μ̃_t , β̃_t )

We now find μ̃_t and β̃_t with nothing but A5's complete-the-square.

### C2. Bayes' rule, in one paragraph

For events/densities: P(A|B) = P(B|A)·P(A)/P(B). Intuition: "probability of A
given that B happened" ∝ "how well A explains B" × "how likely A was to begin
with." Here A = the unknown x_{t−1}, B = the observed x_t:

    q(x_{t−1}|x_t, x₀) ∝ q(x_t|x_{t−1}) · q(x_{t−1}|x₀)
                         └── explains x_t ──┘  └── prior from x₀ ──┘

(The denominator doesn't contain x_{t−1}, so it's part of the ignorable front
factor — remember A2: only the exponent matters.)

### C3. Multiply the two Gaussians and read off the answer

Both factors are known:
- q(x_t|x_{t−1}) = N(x_t; √α_t·x_{t−1}, β_t)      [one forward step, B1]
- q(x_{t−1}|x₀)  = N(x_{t−1}; √ᾱ_{t−1}·x₀, 1−ᾱ_{t−1})   [shortcut ★★ at t−1]

Multiplying densities = adding exponents. Write the combined exponent (call the
unknown x := x_{t−1} to reduce clutter; times −2 to drop the −½'s):

    (x_t − √α_t·x)²/β_t + (x − √ᾱ_{t−1}·x₀)²/(1−ᾱ_{t−1})

Expand both squares, keep only terms containing x (everything else is "junk"
constant that the normalization eats):

    from the first:  (α_t/β_t)·x² − 2·(√α_t·x_t/β_t)·x   + junk
    from the second: (1/(1−ᾱ_{t−1}))·x² − 2·(√ᾱ_{t−1}·x₀/(1−ᾱ_{t−1}))·x + junk

Collect:  a·x² − 2b·x  with

    a = α_t/β_t + 1/(1−ᾱ_{t−1})
    b = √α_t·x_t/β_t + √ᾱ_{t−1}·x₀/(1−ᾱ_{t−1})

By the ★ rule (A5): **variance = 1/a, mean = b/a.** Now simplify a:

    a = [α_t(1−ᾱ_{t−1}) + β_t] / [β_t(1−ᾱ_{t−1})]

Numerator: α_t − α_t·ᾱ_{t−1} + β_t = (α_t + β_t) − ᾱ_t = 1 − ᾱ_t
(using α_t+β_t = 1 and α_t·ᾱ_{t−1} = ᾱ_t by definition of the product). So:

    **β̃_t = 1/a = β_t · (1−ᾱ_{t−1}) / (1−ᾱ_t)**            (posterior variance)

And mean = b/a = b·β̃_t. Multiply each of b's two terms by β̃_t:

    term 1: (√α_t·x_t/β_t) · β_t(1−ᾱ_{t−1})/(1−ᾱ_t) = √α_t(1−ᾱ_{t−1})/(1−ᾱ_t) · x_t
    term 2: (√ᾱ_{t−1}·x₀/(1−ᾱ_{t−1})) · β_t(1−ᾱ_{t−1})/(1−ᾱ_t) = √ᾱ_{t−1}·β_t/(1−ᾱ_t) · x₀

    **μ̃_t = [√α_t(1−ᾱ_{t−1})·x_t + √ᾱ_{t−1}·β_t·x₀] / (1−ᾱ_t)**   (posterior mean)

∎ No calculus happened — expand, collect, complete the square, read.

Sanity read: μ̃ is a weighted blend of "where we are" (x_t) and "where we came
from" (x₀). At t=1: β̃₁ ≈ 0 — the last undo step is almost deterministic. This is
why sampling adds NO noise at the final step.

These two boxed formulas are `posterior_variance`, `posterior_mean_coef_x0`,
`posterior_mean_coef_xt` in `ddpm.py::__init__`, and `reverse_process` computes
exactly  μ̃ = coef_x0·x̂₀ + coef_xt·x_t.

---

## Part D — Training: why the loss is just MSE on the noise

### D1. What we want vs. what we can compute

We want the model to maximize the probability of the data. Directly impossible
(we'd have to consider every noising path). The standard workaround (same as in
VAEs) is a lower bound — the ELBO — which decomposes into one term per timestep.
**The complete gentle derivation — every log expanded, the T=3 telescoping
written out line by line, the Gaussian KL fully expanded — is in
`09_elbo_from_zero.md`** (it needs only log rules and one inequality,
log x ≤ x−1, proved there). The punchline:

    total objective = Σ_t  KL( q(x_{t−1}|x_t, x₀)  ‖  p_θ(x_{t−1}|x_t) )  + edge terms
                            └── answer key (Part C) ──┘  └── our model ──┘

In words: **at every timestep, make the model's undo-step distribution match the
answer-key distribution.** The edge terms are ≈0 / absorbed.

### D2. KL of Gaussians → mean matching (where regression is born)

We CHOOSE the model to be Gaussian with the same variance as the answer key:
p_θ(x_{t−1}|x_t) = N(μ_θ(x_t, t), β̃_t). Then A6 collapses each KL:

    KL_t = ( μ̃_t − μ_θ(x_t, t) )² / (2β̃_t)

Matching distributions became matching means: predict the correct center of the
undo step, squared error. This is the moment probability theory hands the problem
to deep learning.

### D3. From mean-matching to noise-matching

The answer key mean μ̃_t needs x₀ — unknown at test time. But look at Eq. ★★:
x_t, x₀, ε are three numbers tied by ONE equation, so knowing (x_t, and any one
of x₀ or ε) gives the third:

    x₀ = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t                         (solve ★★ for x₀)

Substitute this x₀ into μ̃_t (Part C) and simplify — expand, put over the common
denominator (1−ᾱ_t)√α_t, and use 1−ᾱ_t = α_t(1−ᾱ_{t−1}) + β_t from C3 (full
expansion in 04_mathematics.md §3; it is 6 lines of the same collect-terms
algebra as C3):

    μ̃_t = (1/√α_t) · ( x_t − (β_t/√(1−ᾱ_t)) · ε )

Beautiful: **the correct undo-mean is "current pixel minus a known multiple of
the noise that was added."** The only unknown is ε. So let the network predict
ε — call the prediction ε_θ(x_t, t) — and define the model mean by the same
formula with ε_θ in place of ε. Then

    μ̃_t − μ_θ = (β_t / (√α_t·√(1−ᾱ_t))) · (ε_θ − ε)

and the per-step loss D2 becomes  (known positive number depending on t) · (ε − ε_θ)².

### D4. The final simplification

DDPM (Ho et al., §3.4) simply DELETES the t-dependent front weight:

    **L_simple = average over (image, random t, random ε) of  (ε − ε_θ(x_t, t))²**

Why deleting a "correct" weight is allowed and even good: the weight is enormous
for small t (nearly-clean images), so keeping it would spend all capacity
polishing almost-clean pixels; uniform weighting redistributes effort toward the
noisy regime where generation is actually hard. It's a deliberate trade of
likelihood for sample quality.

And that's the entire training loop (`compute_loss`):
pick image → pick random t → pick random ε → make x_t by Eq. ★★ → ask network
for ε̂ → MSE(ε̂, ε) → backprop.

---

## Part E — Sampling (using the trained network)

Start x_T = `torch.randn` (legal because ᾱ_T ≈ 0 made the endpoint standard
normal, Part B). Then for t = T down to 1, sample from the model's undo step
using A3 (reparameterization):

    ε̂  = ε_θ(x_t, t)                                  [network call]
    μ  = (1/√α_t)·( x_t − (β_t/√(1−ᾱ_t))·ε̂ )          [D3's formula]
    x_{t−1} = μ + √β̃_t · z,   z ~ N(0,1)  — except z = 0 when t = 1

Why fresh noise z at every step (aren't we REMOVING noise?): at high t the network
can only estimate the *average* over all images consistent with x_t; the injected
z is what lets the trajectory commit to one specific image instead of the blurry
average. And why z = 0 at the last step: β̃₁ ≈ 0 (C3 sanity read) — the final
answer is the mean; noising your finished image only hurts.

(Our code takes an equivalent route: convert ε̂ → x̂₀ by D3's inversion, clamp
x̂₀ to [−1,1] since real images live there, then use Part C's μ̃(x_t, x̂₀)
directly. Same math — D3 *is* that substitution — plus the clamp for robustness,
plus it works unchanged for the x₀-network and for skipping steps.)

### E1. Skipping steps (Q7) — why you can't reuse β, and what to use

The per-step β_t belongs to *adjacent* pairs (t−1, t). If you jump t → s with
s < t−1, the effective "one big step" must satisfy the shortcut ★★ at both ends.
Since ᾱ accumulates multiplicatively, the jump's alpha is the ratio:

    α_eff = ᾱ_t / ᾱ_s     (check: for s = t−1 this is exactly α_t ✓)

then β_eff = 1 − α_eff, and reuse Part C's formulas with (α_eff, β_eff, ᾱ_s, ᾱ_t).
Everything else is unchanged. Fewer steps = fewer network calls = faster but each
jump is bigger and relies harder on an imperfect ε̂ → quality degrades gradually.

### E2. Predicting x₀ instead of ε (Q6) — one line of algebra

Eq. ★★ ties x₀ and ε together given x_t. So a network can predict either:

    ε̂  = (x_t − √ᾱ_t·x̂₀) / √(1−ᾱ_t)      or      x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t

Train with MSE against x₀ instead of ε; sample by plugging x̂₀ straight into
Part C's μ̃. Substituting one loss into the other shows
(ε-loss) = [ᾱ_t/(1−ᾱ_t)] · (x₀-loss): the two trainings weight timesteps
DIFFERENTLY (x₀-prediction cares less about low-noise steps). Same model family,
different effective curriculum — that's the entire Q6(c) story you'll see in KID.

---

## Part F — Cheat sheet (all boxed results in one place)

| # | Formula | Meaning | Where derived |
|---|---------|---------|---------------|
| 1 | x_t = √(1−β_t)x_{t−1} + √β_t ε | one noising step, variance-preserving | B1 |
| 2 | x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε | jump to any noise level in one shot | B2 |
| 3 | β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t) | answer-key variance of the undo step | C3 |
| 4 | μ̃_t = [√α_t(1−ᾱ_{t−1})x_t + √ᾱ_{t−1}β_t x₀]/(1−ᾱ_t) | answer-key mean | C3 |
| 5 | μ̃_t = (x_t − β_t ε/√(1−ᾱ_t))/√α_t | same mean, written with ε | D3 |
| 6 | L_simple = E‖ε − ε_θ(x_t,t)‖² | the whole training loss | D4 |
| 7 | x_{t−1} = μ + √β̃_t z, z=0 at last step | one sampling step | E |
| 8 | α_eff = ᾱ_t/ᾱ_s | step-skipping (fast sampling) | E1 |
| 9 | x̂₀ = (x_t − √(1−ᾱ_t)ε̂)/√ᾱ_t | ε ↔ x₀ conversion | E2 |

Total math used: average-of-sum = sum-of-averages, two variance rules,
"read the exponent," completing the square, Bayes. No integrals were harmed.
