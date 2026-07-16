# Flow Matching From Zero
*Same rules as `05_math_from_zero.md`: nothing used before it is built, every step
expanded, numeric sanity checks throughout. Read that file's Part A first (mean,
variance, reparameterization) — we reuse it. New tools needed here: just one —
"velocity," built below without assuming calculus.*

As before, we work with ONE pixel (a single number); images are many pixels and
every formula applies to each independently.

⚠ Convention flip vs DDPM: flow matching runs time **from 0 = noise to 1 = data**
(DDPM ran T = noise down to 0 = data). Also t is now a *continuous* number in
[0, 1], not an integer step count.

---

## Part A — The tools

### A1. Velocity (all the "calculus" we need)

If a particle's position at time t is x(t), its **velocity** at time t is:

    v(t) ≈ ( x(t + Δ) − x(t) ) / Δ        for a tiny time-step Δ

"Distance moved divided by time taken," measured over an instant. That is the
entire definition of a derivative that we need. Two facts, checkable by plugging
into the formula:

**Fact 1 (constant motion):** if x(t) = a + b·t, then
    v = ((a + b(t+Δ)) − (a + bt))/Δ = bΔ/Δ = **b**, at every t.
Straight-line motion has constant velocity. (Exact — no approximation needed;
the Δ's cancel perfectly because the path is straight.)

**Fact 2 (Euler's method — driving by GPS):** if you know the velocity everywhere,
you can reconstruct the path by taking small steps:

    x(t + Δ) ≈ x(t) + v(x(t), t) · Δ

"Where you'll be = where you are + how fast you're going × how long you go."
Smaller Δ = more accurate. If the true motion is a straight line, Euler is EXACT
even with one giant step (Fact 1: v never changes, so no error accumulates).
This one sentence is the entire flow-matching sampler.

### A2. Velocity field

A **velocity field** v(x, t) is an arrow attached to every location x at every
time t: "if you are standing here at this time, move this way at this speed."
Think weather-map wind arrows. Release a dust particle anywhere; the field
carries it along a path. Release a whole CLOUD of dust and the field carries the
entire cloud — reshaping the cloud's density as it goes.

That reshaping is the punchline: **a velocity field is a machine that transforms
one probability distribution into another over time.** Start the cloud shaped
like N(0, 1) at t=0; if the field is chosen well, the cloud at t=1 is shaped
like the data distribution. Generation = drop a random dust grain at t=0, follow
the arrows to t=1, read off where it landed.

So the whole game is: **learn the arrows.** A neural network v_θ(x, t) takes
(position, time) and outputs an arrow — exactly the same interface as DDPM's
ε_θ(x_t, t), just a different meaning for the output.

### A3. One tool from statistics: "the best guess is the average"

Claim: the number c that minimizes E[(Y − c)²] (average squared error against a
random Y) is c = E[Y], the mean.
*Proof by completing the square (05 file, A5):*
E[(Y − c)²] = E[Y²] − 2c·E[Y] + c² . As a function of c this is a parabola;
write it as (c − E[Y])² + (E[Y²] − E[Y]²). The first term is the only part
containing c, and it is smallest (zero) at c = E[Y]. ∎

Upgraded version we'll need: if the guess may DEPEND on some observed
information (here: your position x and time t), the best guesser outputs the
**conditional average** E[Y | what you observed]. Same proof, applied separately
at each observed value. Remember this — it is the entire "trick" of flow
matching (Part D).

---

## Part B — The big idea, before any formulas

DDPM learned to *undo noise* step by step, with fresh randomness injected at
every step (a drunkard stumbling from static toward a face). Flow matching asks
a cleaner question:

> Find a velocity field so that dust shaped like pure noise at t=0 flows into
> dust shaped like the data at t=1.

Differences from DDPM worth internalizing now:

1. **Deterministic ride.** After the ONE random draw at t=0 (your starting dust
   grain), the trip is pure arrow-following. No noise re-injection. All the
   diversity of generation lives in the starting point.
2. **We get to DESIGN the route.** DDPM's route was dictated by the noising
   process (curved, wiggly). Flow matching lets us declare "particles travel in
   straight lines" — and straight lines are exactly what Euler's method
   integrates with the fewest steps (Fact 1/2). This is why flow-matching models
   sample well with 10–50 steps while vanilla DDPM needs ~1000.
3. **Same training cost.** Despite the fancy "transport" language, training will
   collapse to: pick a noise, pick an image, pick a time, do one MSE. Just like
   DDPM.

The apparent obstacle: the arrows we need are a property of the *entire data
distribution* — how could we ever know the correct arrow at some random midpoint?
Part C+D dissolve this with one lovely observation.

---

## Part C — Conditional paths: pretend there is only ONE image

Fix one noise sample x₀ ~ N(0,1) and one dataset image x₁. If the world
contained only this pair, the natural route between them is the straight line:

    x_t = (1 − t)·x₀ + t·x₁ ,    t ∈ [0, 1]          (Eq. ▲, "the interpolant")

Check endpoints: t=0 → x₀ ✓ ; t=1 → x₁ ✓. At t=0.3 you're 30% of the way to
the image. (This straight-line choice is the "rectified flow" / "conditional
optimal transport" path — Liu et al. 2022, Lipman et al. 2023. Other curved
choices exist; straight is best for fast sampling and is what modern systems
like Stable Diffusion 3 / Flux / Meta MovieGen use.)

What velocity does this path have? It's of the form a + b·t with a = x₀ and
b = x₁ − x₀, so by Fact 1:

    v = x₁ − x₀ ,  constant along the whole trip        (Eq. ▲▲, "the target")

The arrow just points from your noise straight at your image, always. Given the
pair, the correct answer is trivial. This is called the **conditional** velocity
("conditional" = computed while knowing which pair you committed to).

Note it can be rewritten using Eq. ▲: since x₀ = (x_t − t·x₁)/(1−t), knowing
any two of (x_t, x₀, x₁, v) gives the others — the same three-numbers-one-
equation structure as DDPM's (x_t, x₀, ε).

---

## Part D — The marginalization problem, and the trick that solves it

### D1. The problem

At generation time the network stands at position x at time t and does NOT know
which (x₀, x₁) pair it belongs to — because it belongs to MANY. Different
noise-image pairs' straight lines cross the same point:

    pair A: noise −2 → image +4:  at t=0.5, x = 1, arrow = +6
    pair B: noise  0 → image +2:  at t=0.5, x = 1, arrow = +2

Standing at (x=1, t=0.5), which arrow is correct? For the CLOUD to flow
correctly, the field must use the average of all arrows passing through that
point, weighted by how likely each pair is to be there — the **marginal**
velocity:

    v*(x, t) = E[ x₁ − x₀ | x_t = x ]        ("average arrow through here")

(That this average is exactly what transports the whole cloud correctly is the
continuity-equation theorem of Lipman et al. 2023, Thm 1 — the one result we
cite without proof, as with Gaussian stability in the DDPM notes. Intuition:
each conditional line correctly transports its own two-point "mini-cloud"; a
density is a weighted mixture of mini-clouds; averaging the arrows transports
the mixture.)

This v* looks uncomputable — it averages over the whole dataset at every point.

### D2. The trick: MSE against the EASY target already learns the HARD one

Train the network by regression against the *conditional* arrow, which costs
nothing to compute:

    L = E over (x₀, x₁, t) of  ( v_θ(x_t, t) − (x₁ − x₀) )²      (Eq. ▲▲▲)

Now recall tool A3 (upgraded form): the function that minimizes a squared error
against a random target, given observed (x, t), is the conditional average of
that target:

    best v_θ(x, t) = E[ x₁ − x₀ | x_t = x ]  =  v*(x, t).

**The minimizer of the trivial loss IS the intractable marginal field.** We never
compute the average over the dataset; the optimizer performs it for us, because
that is simply what least-squares does when the target is noisy. (Same logic in
DDPM, where ε_θ silently learns E[ε | x_t] — flow matching just makes the trick
the headline.)

One more comfort: expanding the square shows the trivial loss and the true loss
E(v_θ − v*)² differ by a constant that does not contain θ (the variance of the
arrows around their mean), so even their gradients agree. Nothing is lost.

### D3. Why this is the whole paper

That's flow matching. No forward-process derivation, no Bayes posterior, no
ELBO, no KL chain — the three-page pipeline of the DDPM notes is replaced by:
(1) pick straight lines, (2) note their velocity is x₁ − x₀,
(3) least-squares learns conditional means. Everything else is engineering.

---

## Part E — The algorithms

### Training (compare DDPM Algorithm 1 line by line)

    repeat:
      x₁  ~ dataset                     # clean image        (DDPM: same)
      x₀  ~ N(0, I)                     # noise              (DDPM: ε)
      t   ~ Uniform[0, 1]               # continuous!        (DDPM: integer 0..T−1)
      x_t = (1−t)·x₀ + t·x₁             # interpolate        (DDPM: √ᾱ x₀ + √(1−ᾱ) ε)
      loss = ( v_θ(x_t, t) − (x₁−x₀) )² # regress the arrow  (DDPM: MSE on ε)

Same shape-bug traps as DDPM: t must broadcast as (B,1,1,1) against images, the
network takes t through the same sinusoidal embedding (it accepts floats — check
`blocks.py`: `t.float()[:, None] * freqs`), and the U-Net is reused unchanged.

### Sampling = Euler (Fact 2), driving from noise to data

    x = randn()                         # t = 0: drop a dust grain
    for i in 0 .. N−1:                  # N ≈ 10–100 steps
        t = i / N
        x = x + v_θ(x, t) · (1/N)       # one Euler step of size Δ = 1/N
    return x                            # t = 1: an image

No noise inside the loop, no special last step, no posterior variance — compare
with DDPM Algorithm 2 and enjoy. If the learned field were perfectly straight,
N = 1 would already work (Fact 1: constant velocity ⇒ Euler exact); in practice
the *marginal* field is curved even though each conditional line is straight
(averaging straight arrows with different directions bends the flow), so a few
dozen steps are used. Making the marginal flow actually straight by re-training
on the model's own (start, end) pairs is **Reflow / rectified flow distillation**
(Liu et al. 2022) — the road to 1-step generation, and cousin of consistency
models.

---

## Part F — The bridge back to DDPM (both are one family)

Put the DDPM forward shortcut and the FM interpolant side by side (time reversed
so both run noise → data):

    DDPM:  x = √(1−ᾱ)·noise + √ᾱ·image      coefficients on a CIRCLE  (a²+b²=1)
    FM:    x = (1−t)·noise + t·image        coefficients on a LINE    (a+b=1)

Both are "slide from noise to image along some schedule"; they differ only in
the path of the mixing coefficients. Consequences:

- The variance-preserving circle made DDPM's algebra heavy (ᾱ products, β̃'s);
  the straight line makes FM's algebra trivial. Same idea, better coordinates.
- ε-prediction, x₀-prediction (HW1 Q6!), v-prediction, and FM's velocity are all
  linear re-labelings of the same learned information: given x_t and any one of
  them, the others follow from the one mixing equation. FM's "velocity" target
  v = x₁ − x₀ is precisely the flow-matching analogue of DDPM's v-prediction.
- DDPM sampling ≈ stochastic path-following (noise re-injected each step);
  FM sampling = deterministic path-following. DDIM sits exactly in between:
  it is DDPM's *deterministic* sampler — historically the first hint that the
  arrows were all that mattered. Score-SDE makes this formal: every diffusion
  has a "probability flow ODE" whose velocity field generates the same
  distributions; FM just learns such a field directly, for a straighter path.
  **Both DDIM and that ODE (plus the DPM-Solver that solves it fast) are
  derived from scratch in `10_ddim_dpmsolver_from_zero.md`.**

## Part G — Cheat sheet

| # | Formula | Meaning | Where |
|---|---------|---------|-------|
| 1 | v ≈ (x(t+Δ)−x(t))/Δ | velocity = distance/time over an instant | A1 |
| 2 | x ← x + v·Δ | Euler step: the entire sampler | A1 |
| 3 | x_t = (1−t)x₀ + t·x₁ | straight path from noise to one image | C |
| 4 | v = x₁ − x₀ | its (constant) velocity: the training target | C |
| 5 | L = E(v_θ(x_t,t) − (x₁−x₀))² | the entire training loss | D2 |
| 6 | argmin = E[x₁−x₀ \| x_t] | least-squares learns the marginal field free | A3+D2 |

Total math used: distance/time, one substitution, and "the best guess is the
average." Even less than DDPM needed.
