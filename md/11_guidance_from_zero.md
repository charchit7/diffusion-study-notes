# Guidance From Zero (Classifier & Classifier-Free)
*Part 7 of the from-scratch series — the one production-critical idea the
earlier notes only used, never derived. Prerequisites: Bayes (05 §C2), log
rules (09 §A1), and one result from 10 §3: the network's ε̂ is a scaled
estimate of the score, ∇log q(x_t) ≈ −ε̂(x_t, t)/σ_t.*

## 1. The problem: steering generation

Everything so far generates unconditionally: start from noise, get *a* face,
*a* image — no control over which. Real systems generate from a condition c
(a text prompt, a class label, an attribute). Mathematically we want to
sample from p(x | c) instead of p(x).

The score is the natural handle. All our samplers (ancestral, DDIM, Euler,
DPM-Solver) consume the network's ε̂, and ε̂ is just the score in disguise
(10 §3). So: **if we can write down the score of p(x_t | c), every sampler
we built works unchanged — swap the ε̂ it eats.**

## 2. Bayes in score space (three lines)

Take logs of Bayes' rule p(x|c) = p(x) · p(c|x) / p(c):

    log p(x|c) = log p(x) + log p(c|x) − log p(c)

Now take the gradient with respect to x (the "which direction makes this
more likely" arrow — same derivative-as-velocity idea as 06 §A1, applied per
pixel). The last term doesn't contain x, so its gradient is zero:

    ∇log p(x|c) = ∇log p(x) + ∇log p(c|x)                    (★)

Read it: **the conditional score = the unconditional score + "the direction
that makes a classifier more confident the image shows c."** The first term
keeps images realistic; the second steers them toward the condition.

## 3. Classifier guidance (Dhariwal & Nichol 2021)

Implement (★) literally: train a separate classifier p_φ(c | x_t) that works
on NOISY images at every level t, and at each sampling step nudge with its
gradient. In ε-language (multiply (★) by −σ_t, using ε̂ = −σ_t·∇log p):

    ε̂_guided = ε̂(x_t, t) − s · σ_t · ∇log p_φ(c | x_t)

with a strength dial s. s = 1 samples p(x|c) exactly; s > 1 samples the
sharpened p(x)·p(c|x)^s — trading diversity for condition-fidelity (check by
putting s into the log-Bayes line: the classifier term gets multiplied).

Why this fell out of favor: you must train a second, noise-robust model; and
"ascend a classifier's gradient" is exactly how adversarial examples are
made — the guidance can satisfy the classifier without satisfying a human.

## 4. Classifier-free guidance: eliminate the classifier with Bayes again

The trick (Ho & Salimans 2022): solve (★) FOR the classifier term:

    ∇log p(c|x) = ∇log p(x|c) − ∇log p(x)

The classifier's direction is just conditional-score minus unconditional-
score — two things a diffusion model can supply itself. Substitute back into
the sharpened version ∇log p(x) + s·∇log p(c|x):

    guided score = ∇log p(x) + s·(∇log p(x|c) − ∇log p(x))
                 = (1−s)·∇log p(x) + s·∇log p(x|c)

In ε-form (multiply by −σ_t; both scores become their ε̂'s):

    **ε̂_cfg = ε̂_uncond + s · (ε̂_cond − ε̂_uncond)**           (CFG)

- s = 0: unconditional. s = 1: plain conditional. s > 1: **extrapolate
  PAST the conditional prediction, along the direction the condition
  changed** — amplify exactly the part of the denoising that is due to c.
- (Convention warning: some papers write w with s = 1+w; Stable Diffusion's
  `guidance_scale` is this s.)

## 5. One network, both predictions: condition dropout

ε̂_cond and ε̂_uncond come from the SAME network: during training, replace c
with a learned "null" condition ∅ (empty prompt) 10–20% of the time. The one
model then serves both experts; at sampling, run it twice per step — once
with c, once with ∅ — and combine with (CFG). That doubled forward pass is
exactly the `torch.cat([latents]*2)` + `chunk(2)` in our SDS code
(`guidance/sd.py::get_noise_preds`) and in every diffusers pipeline.

## 6. What the dial does (and where we saw it)

- s ≈ 1–3: faithful but sometimes lax on the prompt.
- s ≈ 7–8: the sweet spot for text-to-image (SD's 7.5 default; PDS used it).
- s ≈ 25+ (our SDS runs, per DreamFusion): prompts obeyed hard, but colors
  oversaturate and contrast blows out — extrapolating far past ε̂_cond
  pushes x̂₀ outside the data range each step (the clamp fights it, the
  palette shows it). Look at our SDS hotdog: technically perfect, slightly
  radioactive. That's s = 25 talking.
- Diversity shrinks as s grows: all samples crowd toward the mode the
  condition pins down. (CFG trades the distribution's breadth for its peak.)

## 7. Guidance beyond ε: flow matching

Nothing in §2–4 cared that the model predicts noise. For a velocity model,
scores turn into velocities the same way, and

    v_cfg = v_uncond + s · (v_cond − v_uncond)

is the working formula — this is exactly the `guidance_scale` hook we saw in
the KAIST flow notebook's sample(), and how SD3/Flux-class flow models are
steered in practice.

## 8. Where it would plug into OUR code

Our CelebA DDPM is unconditional, but the dataset ships 40 attributes and the
recipe is now three edits: (1) embed the attribute vector and add it to the
time embedding (same FiLM path), (2) drop the condition to a null embedding
10% of the time in `compute_loss`, (3) apply (CFG) inside `reverse_process`
/`ddim_step` by batching the two predictions. That upgrade is the natural
"Controllability track" project — and everything else (samplers, EMA,
training loop) stays untouched.

## 9. Cheat sheet

| # | Formula | Meaning |
|---|---|---|
| 1 | ∇log p(x\|c) = ∇log p(x) + ∇log p(c\|x) | Bayes in score space |
| 2 | ε̂_guided = ε̂ − s·σ_t·∇log p_φ(c\|x_t) | classifier guidance |
| 3 | ε̂_cfg = ε̂_∅ + s(ε̂_c − ε̂_∅) | classifier-free guidance |
| 4 | train with c → ∅ at ~10–20% | one net = both experts |
| 5 | v_cfg = v_∅ + s(v_c − v_∅) | same dial for flow models |

Axiom count: zero new ones — §2–4 is only Bayes, log rules, and the ε↔score
identity we already had. Guidance is a free consequence of the framework.
