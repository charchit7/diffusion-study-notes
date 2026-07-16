# Phase 3 — Mental Model: What Is Actually Happening?

*No equations in this file. Read this until the picture is vivid, then go to `04_mathematics.md`.*

## 1. The core problem

Generative modeling = "here are 100k face photos; build a machine that produces *new* faces from the same distribution." The hard part: the distribution of natural images is a fantastically thin, curved "sheet" (manifold) inside a million-dimensional space. Almost every point in pixel space is TV static; faces occupy a vanishingly small region. Sampling from it directly is hopeless — we can't even write down where it is.

## 2. The diffusion trick: destroy, then learn to rebuild

**Forward process** — the "destruction" direction. Take a real photo and add a tiny amount of TV static. Then a bit more. Do this 1000 times. After 1000 steps the photo is indistinguishable from pure static. Key properties:

- It's *dumb*: no learning, no network. Just a fixed recipe of "how much static at each step."
- It's *gentle*: each step changes the image only slightly, so consecutive steps are nearly identical images.
- It's a *bridge*: it connects the impossibly-complicated distribution (faces) to the simplest distribution in existence (pure Gaussian static), through 1000 intermediate distributions that morph smoothly from one to the other.

Analogy: drop of ink in a glass of water. At t=0 the ink is a perfect intricate swirl (a face). Molecules jiggle randomly (noise added). At t=1000 the water is uniformly grey (pure noise). Diffusion in physics runs one way only — that's the forward process.

**Reverse process** — the learned direction. If you watched a *movie* of the ink dispersing and played it backwards, you'd see grey water spontaneously organize into a swirl. That backwards movie is what we want to learn. Physics says you can't un-mix ink — *unless* you know, at every moment, the statistics of where the ink came from. A neural network can learn exactly that from data.

The gentleness of the forward steps is what makes learning possible: undoing 1000 tiny corruptions is 1000 easy problems ("this image is slightly staticky, clean it slightly") instead of one impossible problem ("here is static, produce a face").

## 3. Why predict the noise?

At every step the noisy image is literally `signal + static`. Ask the network: "here is a corrupted image and how far along the corruption is (t) — which part of what you see is the static?" If it can point at the static, we can subtract (some of) it. Three equivalent things the network could predict — the noise ε, the clean image x₀, or the small step backwards — are just algebraic rearrangements of each other (this equivalence is exactly HW Q6). DDPM chose ε because its target always has the same scale (unit-variance static) at every t, which makes one network with one loss work across all 1000 sub-problems.

Crucial subtlety: at high noise levels the task is *ambiguous* — many different faces could have produced the same very-noisy image. The network can't know which; the best it can do is predict the *average* static consistent with all of them. That's why one denoising jump from pure noise gives a blurry "average face," and why we instead take 1000 small steps, **re-injecting a little fresh randomness at each step**. The re-injected randomness is what lets the process commit to *one* face rather than the average of all faces — each small step makes a small decision, and small decisions compound into a specific, sharp image.

## 4. Training, in one sentence each

- Pick a random photo from the dataset.
- Pick a random corruption level t (1..1000) — uniformly, so the network gets practice at every level.
- Jump *directly* to that corruption level (a math shortcut lets us corrupt in one shot instead of t sequential steps — this is why training is cheap).
- Ask the network which part is static; penalize squared error against the static we actually added.
- Repeat millions of times. The network never sees a full forward or reverse trajectory during training — only random single snapshots.

## 5. Sampling (inference), in one sentence each

- Start from pure static (free to generate — it's just `randn`).
- For t = 1000 down to 1: ask the network for the static content, take a small step toward "cleaner," add a pinch of fresh static (except at the very last step — the final answer should be clean).
- After 1000 steps: a new face that never existed.
- This is why DDPM sampling is slow (1000 network calls per image) — the pain point behind HW Q7, and later DDIM / flow matching.

## 6. The scheduler / variance schedule

The "recipe card" for the destruction: a list of 1000 numbers (β₁…β₁₀₀₀) saying how much static to add at each step, increasing from tiny (1e-4) to moderate (0.02). Why increasing? Early corruption should be delicate — that's where the fine detail (skin texture, hair strands) lives and where the network learns the hardest lessons; late corruption can be coarse because the image is already mostly static. From these 1000 numbers everything else is precomputed bookkeeping: cumulative "how much of the original signal survives to step t" (ᾱ_t, a number sliding from ~1 to ~0), the shortcut coefficients, the reverse-step sizes. The schedule is *design*, not learning — changing it changes what the model spends its capacity on (Improved DDPM's cosine schedule is exactly this knob).

## 7. The posterior — the one non-obvious object

Question: "given the noisy image at step t, AND the original clean photo, what did step t−1 look like?" With both endpoints known this has an exact, known answer (a slightly-less-noisy blend of the two, plus a precisely known amount of remaining uncertainty). This is the **ground-truth answer key** for each reverse step. The network can't use it at test time (no clean photo then!), but during training we teach the network's reverse step to *match* this answer key. The entire DDPM loss is "match the answer key at every t" — and predicting ε turns out to be the cleanest way to parameterize that match.

## 8. Why the loss curve is a bad progress bar

The target (which static was added) is random every time, and at high t the task is irreducibly ambiguous — so the loss plateaus at a nonzero floor and jitters forever (HW Q5b), while sample quality keeps improving. Diffusion people watch *samples* and KID/FID, not the loss. Corollary: a beautiful loss curve proves only that training-time code works; sampling is a separate untested code path (HW Q5a).

## 9. The cast of characters (map to code)

| Concept | Code |
|---|---|
| destruction recipe (β, ᾱ) | buffers in `DDPM.__init__` |
| one-shot corruption | `DDPM.forward_process` |
| "which part is static?" network | `UNet(x_t, t)` |
| corruption-level dial fed to network | sinusoidal `TimestepEmbedding` → FiLM in every ResBlock |
| match-the-answer-key loss | `DDPM.compute_loss` (MSE on ε) |
| one small cleaning step | `DDPM.reverse_process` |
| static → face loop | `DDPM.sample` |
| smoothed network for nicer samples | `EMA` |
