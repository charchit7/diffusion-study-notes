# The ELBO From Zero
*Fills the one gap `05_math_from_zero.md` deliberately skipped (its Part D1):
the complete chain from "maximize the probability of the data" to "MSE on the
noise", with no step hidden. Same rules: nothing used before it is built,
every expansion written out. Prerequisites: 05 Parts A–C (Gaussian toolkit,
forward process, posterior).*

Notation reminder: x₀ = data, x₁..x_T = the noising chain, q = the fixed
forward process, p_θ = our model of the reverse process,
p_θ(x_{0:T}) = p(x_T)·∏_{t=1}^T p_θ(x_{t−1}|x_t) with p(x_T) = N(0, I).

---

## Part A — Two tools

### A1. Log rules (that's almost all we need)
log(ab) = log a + log b, log(a/b) = log a − log b, log(∏ aᵢ) = Σ log aᵢ.
These convert the huge products in p_θ(x_{0:T}) into manageable sums.

### A2. The inequality log x ≤ x − 1 (our only inequality)
Claim: for every x > 0, log x ≤ x − 1, with equality only at x = 1.
*Gentle argument:* e^y ≥ 1 + y for all y (compound growth beats simple
growth: e^y is what you get compounding continuously at rate y, 1+y is
simple interest; check numerically: y=1 → 2.718 ≥ 2 ✓, y=−0.5 → 0.607 ≥ 0.5 ✓,
y=0 → equality). Substitute y = log x: x ≥ 1 + log x. Rearranged: done. ∎

### A3. KL divergence is never negative (proved with A2)
KL(q‖p) := E_q[log(q/p)]. Then, writing log(q/p) = −log(p/q):

    KL(q‖p) = −E_q[log(p/q)] ≥ −E_q[p/q − 1]          (A2 applied inside)
            = −( E_q[p/q] − 1 ) = −( ∫ q·(p/q) − 1 ) = −(1 − 1) = 0.  ∎

(∫q·(p/q) = ∫p = 1 because p is a probability density — "total area 1".)
This single fact replaces Jensen's inequality everywhere below.

---

## Part B — The master identity (three lines, only log rules)

We want to make log p_θ(x₀) big but can't compute it (it integrates over all
noising paths). Trick: bring in ANY helper distribution over the latents —
we'll use the forward process q(x_{1:T}|x₀) — and split:

Start from the definition of the ELBO (a quantity we CAN estimate):

    ELBO := E_q [ log ( p_θ(x₀, x_{1:T}) / q(x_{1:T}|x₀) ) ]

Rewrite the joint via the conditional: p_θ(x₀, x_{1:T}) =
p_θ(x_{1:T}|x₀) · p_θ(x₀). Substitute and use log rules:

    ELBO = E_q [ log p_θ(x₀) + log ( p_θ(x_{1:T}|x₀) / q(x_{1:T}|x₀) ) ]
         = log p_θ(x₀) − E_q [ log ( q(x_{1:T}|x₀) / p_θ(x_{1:T}|x₀) ) ]
         = log p_θ(x₀) − KL( q(x_{1:T}|x₀) ‖ p_θ(x_{1:T}|x₀) ).

(log p_θ(x₀) exits the expectation because it doesn't depend on x_{1:T}.)
Rearranged, the **master identity**:

    log p_θ(x₀) = ELBO + KL( q ‖ model's true reverse )        (★)

By A3 the KL ≥ 0, so **log p_θ(x₀) ≥ ELBO**: maximize the computable ELBO
and you push up the incomputable likelihood. The gap is exactly how far the
fixed forward q is from the model's own reverse — the bound tightens as the
model learns.

## Part C — Expand the ELBO for the diffusion chain

Insert the two factorizations and convert products to sums (log rules):

    ELBO = E_q [ log p(x_T) + Σ_{t=1}^T log p_θ(x_{t−1}|x_t)
                            − Σ_{t=1}^T log q(x_t|x_{t−1}) ]      (C1)

Every piece is a known Gaussian. But it's mis-aligned: p-terms condition on
x_t going DOWN, q-terms condition on x_{t−1} going UP. We re-point the
q-terms with Bayes.

### C2. The Bayes flip (for t ≥ 2)
The chain is Markov, so q(x_t|x_{t−1}) = q(x_t|x_{t−1}, x₀) — conditioning on
x₀ adds nothing new when x_{t−1} is known. Now apply Bayes' rule *inside the
world where x₀ is given*:

    q(x_t|x_{t−1}, x₀) = q(x_{t−1}|x_t, x₀) · q(x_t|x₀) / q(x_{t−1}|x₀).

Every factor on the right is something we've computed in 05: the posterior
(Part C there) and two shortcut marginals (Part B there).

### C3. Telescoping, written out fully for T = 3
Sum of the flipped q-terms for t = 2, 3 plus the t = 1 term:

    log q(x₁|x₀)
  + log q(x₁|x₂,x₀) + log q(x₂|x₀) − log q(x₁|x₀)      [t=2 flipped]
  + log q(x₂|x₃,x₀) + log q(x₃|x₀) − log q(x₂|x₀)      [t=3 flipped]

Now cancel line by line: +log q(x₁|x₀) (line 1) kills −log q(x₁|x₀) (line 2);
+log q(x₂|x₀) (line 2) kills −log q(x₂|x₀) (line 3). Left standing:

    log q(x₃|x₀) + Σ_{t=2}^{3} log q(x_{t−1}|x_t, x₀).

The same cancellation works for any T (each +log q(x_t|x₀) meets its
−log q(x_t|x₀) from the next line): the sum collapses to
**log q(x_T|x₀) + Σ_{t=2}^T log q(x_{t−1}|x_t, x₀)**.

### C4. Regroup into named terms
Substitute back into (C1), pair each q-term with its matching p-term, and
recall KL(a‖b) = E[log a − log b] under a:

    −ELBO = E_q [ KL( q(x_T|x₀) ‖ p(x_T) )                      =: L_T
          + Σ_{t=2}^T KL( q(x_{t−1}|x_t,x₀) ‖ p_θ(x_{t−1}|x_t) ) =: L_{t−1}
          − log p_θ(x₀|x₁) ]                                     =: L₀

(The outer E_q supplies the conditioning variables each KL needs.) Meaning:
- **L_T**: does the forward process actually end at the prior? No parameters
  inside (both distributions fixed); with ᾱ_T ≈ 4·10⁻⁵ it is ≈ 0. Ignore.
- **L₀**: the final decode step; in this HW it is absorbed into the same MSE
  (formally: a Gaussian log-likelihood is itself a scaled squared error —
  expand log N(x₀; μ_θ(x₁,1), σ²I) = const − ‖x₀−μ_θ‖²/2σ²).
- **L_{t−1}**: the heart — one term per step, each "make your reverse step
  match the answer key q(x_{t−1}|x_t, x₀) from 05 Part C".

## Part D — Each KL collapses to a squared error

Both arguments of L_{t−1} are Gaussians with the SAME fixed variance σ_t²
(we chose the model that way): q(x_{t−1}|x_t,x₀) = N(μ̃_t, σ_t²) and
p_θ = N(μ_θ, σ_t²). Expand the KL from its definition — per coordinate,
with a ~ N(μ₁, σ²):

    KL = E_a[ log N(a; μ₁,σ²) − log N(a; μ₂,σ²) ]
       = E_a[ −(a−μ₁)²/2σ² + (a−μ₂)²/2σ² ]        (the (2πσ²) fronts cancel)

Expand (a−μ₂)² by inserting ±μ₁:  (a−μ₁+μ₁−μ₂)²
       = (a−μ₁)² + 2(a−μ₁)(μ₁−μ₂) + (μ₁−μ₂)².
Take E_a term by term: E(a−μ₁)² = σ² ; E(a−μ₁) = 0 kills the middle; the last
is constant. So:

    KL = [ −σ² + σ² + (μ₁−μ₂)² ] / 2σ² = **(μ₁−μ₂)² / (2σ²)**.  ∎

Therefore  **L_{t−1} = ‖μ̃_t(x_t, x₀) − μ_θ(x_t, t)‖² / (2σ_t²)** (+ nothing).
Variational inference has become regression on the posterior mean.

## Part E — From mean-matching to the ε-loss (all algebra)

From 05 Part D3, the answer-key mean in ε-form is
μ̃_t = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε), and we parametrize the model
identically with ε_θ in place of ε. Subtract — the x_t parts cancel exactly:

    μ̃_t − μ_θ = (1/√α_t)(β_t/√(1−ᾱ_t)) (ε_θ − ε)

Square, divide by 2σ_t², and the per-step term becomes

    L_{t−1} = β_t² / ( 2 σ_t² α_t (1−ᾱ_t) ) · ‖ε − ε_θ(x_t, t)‖²    (E1)

— a **weighted** noise-matching MSE, weight w_t = β_t²/(2σ_t²α_t(1−ᾱ_t)).

## Part F — L_simple: dropping the weight, and why

Plug σ_t² = β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t) into w_t and simplify:
w_t = β_t / (2 α_t (1−ᾱ_{t−1})). Numbers (linear schedule): w₁ ≈ 0.5/(1−ᾱ₀)…
huge at small t (denominator → tiny since ᾱ₀ ≈ 1), modest at large t. Training
on (E1) as-is spends nearly all gradient signal polishing almost-clean images.
DDPM's empirical move (their Eq. 14): set every weight to 1:

    **L_simple = E_{x₀, t~U{1..T}, ε} ‖ε − ε_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t)‖²**

This is no longer exactly the ELBO — it's a reweighted bound that trades
likelihood for sample quality by up-weighting the high-noise steps where
generation is genuinely hard. (Making this trade principled is the story of
Improved DDPM's hybrid loss and EDM's explicit λ(σ) weighting.)

## Part G — Sanity checks you can run on the derivation itself
1. T = 1 collapse: the sums vanish and −ELBO = L_T − E log p_θ(x₀|x₁) — a
   plain autoencoder bound. ✓ (Diffusion = a very deep chain of these.)
2. Every L term is a KL or a log-likelihood ⇒ −ELBO ≥ 0 pieces behave.
3. If the model were the TRUE reverse, each KL = 0 and (★) says
   log p = ELBO — the bound is tight exactly when learning is done.
4. Dimensional check in (E1): all of β_t, σ_t², (1−ᾱ_t) are variances —
   the weight is 1/variance, matching KL's (μ₁−μ₂)²/2σ² shape. ✓

*Now 05_math_from_zero.md Part D can be read as the summary it was meant to
be — every "the punchline is" sentence there is derived line-by-line here.*
