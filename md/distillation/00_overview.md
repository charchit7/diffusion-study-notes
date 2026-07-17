# Text-to-Image Diffusion Distillation — The Field, Four Ways
*Master overview for the line-by-line tutorials in this folder. Read this
first; then each method's tutorial (methodology section + annotated code).
Prerequisites: our notes 06 (flow matching/Reflow), 10 (DDIM, x̂₀ identity,
score = −ε/σ), 11 (CFG), and the KAIST SDS writeup.*

## 0. The problem, in numbers

A production text-to-image model (SD1.5/SDXL-class) needs 25–50 U-Net calls
per image, ×2 for CFG. Distillation asks: **can a student reproduce the
teacher's output distribution in 1–4 calls, with CFG baked in?** The four
families below answer differently — and the differences are exactly about
WHAT is matched: trajectories, distributions, geometry, or realism.

## 1. The taxonomy (the mental map that matters most)

```
                       what does the student imitate?
        ┌──────────────────┬───────────────────┬──────────────────┬────────────────┐
        │ A. the teacher's │ B. the teacher's  │ C. a straightened │ D. "does it    │
        │ ODE TRAJECTORY   │ output            │ GEOMETRY          │ look real"     │
        │ (jump further    │ DISTRIBUTION      │ (make 1 step      │ (discriminator)│
        │ along it)        │ (match scores)    │ exact by design)  │                │
        ├──────────────────┼───────────────────┼──────────────────┼────────────────┤
        │ Progressive dist.│ SDS / VSD         │ Rectified Flow    │ ADD/SDXL-Turbo │
        │ Consistency, LCM │ DMD, DMD2 ★       │ Reflow            │ SDXL-Lightning │
        │ PCM ★, CTM       │ SwiftBrush, SiD   │ InstaFlow ★       │ (+GAN terms in │
        │ sCM, SANA-Sprint │ SenseFlow ('25)   │ PeRFlow           │ DMD2/PCM/Sprint)│
        └──────────────────┴───────────────────┴──────────────────┴────────────────┘
                 ★ = has a line-by-line tutorial in this folder
```

Modern SOTA systems are HYBRIDS: DMD2 = B + D; PCM = A + D; SANA-Sprint =
A(continuous) + D; FLUX-schnell reportedly = D on latents. Family C is the
odd one out — it changes the object being learned rather than the loss.
The newest axis (late 2025) is **B + RL**: DMDR ★ ("DMD meets Reinforcement
Learning", the recipe behind Z-Image-Turbo) makes the reward model and the
distillation loss train JOINTLY — see §3b.

---

## 2. Family A — Trajectory distillation → LCM → PCM

### The core idea (consistency)
Every noise x_T sits on ONE deterministic ODE trajectory ending at an image
x₀ (our notes 10: the PF-ODE / DDIM path). Define a **consistency function**
f(x_t, t) = "the endpoint of the trajectory through (x_t, t)" — then
f is constant along a trajectory (self-consistency), and f(x_T, T) IS
one-step generation. You cannot supervise f directly (you'd need the
endpoint), but you can enforce self-consistency locally:

    take x_t → run ONE teacher ODE step (DDIM/DPM-Solver) → x_{t−k}
    loss = d( f_θ(x_t, t), f_θ⁻(x_{t−k}, t−k) )        [θ⁻ = EMA target]

Local consistency chains into global: if every adjacent pair agrees, the
whole trajectory maps to one endpoint. **This is a bootstrapped fixed-point
argument, not regression to data** — the deepest conceptual jump in family A.

### LCM's three practical moves (Luo et al. 2023)
1. Do it in LATENT space on SD (cheap 64×64×4 tensors, frozen VAE).
2. Distill the **CFG-augmented** ODE: the teacher step uses
   ε_∅ + w(ε_c − ε_∅) with w SAMPLED per batch and fed to the student as an
   embedding — guidance becomes a conditioning input; inference needs ONE
   forward pass, no CFG doubling (our note 11's two-pass cost, deleted).
3. Skipping-step: teacher jumps k≈20 timesteps per consistency pair
   (a strided DDIM step — the exact ᾱ-ratio machinery from our HW1 Q7).
Plus **LCM-LoRA**: run the same distillation with LoRA adapters only —
"acceleration as a plug-in" that composes with style LoRAs.

### PCM: why LCM breaks, and the fix (Wang et al. 2024)
LCM's f maps everything to t=0, so multi-step sampling must RE-NOISE after
every step (stochastic, inconsistent across step counts, blurry at 2–4
steps); and the baked-in w range limits usable guidance. PCM splits the
trajectory into M phases and learns a consistency function PER PHASE
(mapping to the phase's start edge). Multi-step sampling becomes
deterministic edge-hopping — no re-noising — and 1..16-step results come
from ONE model by choosing how many phases to traverse. An adversarial loss
on the phase edges sharpens the 1-step regime. PCM ⊃ LCM (M=1 recovers it).

**Kinship to our work:** consistency targets are built from exactly our
DDIM/strided step; the boundary conditions c_skip/c_out that force
f(x₀, 0) = x₀ are the same zero-init/identity-at-init design philosophy as
ControlNet's zero-convs and our U-Net's zero head.

---

## 3. Family B — Distribution matching → DMD2

### The core idea
Don't follow trajectories at all. Let G_θ (the student) map noise → image in
one shot, and minimize the reverse KL between the student's output
distribution and the teacher's data distribution — using SCORES:

    ∇_θ KL(p_fake ‖ p_real) = E[ −( s_real(x_t) − s_fake(x_t) ) · ∂x_t/∂θ ]

- s_real: the frozen TEACHER's score (its ε-prediction, converted by our
  10 §3 identity) — "which way is more like real data".
- s_fake: a SECOND diffusion model, **trained online on the student's own
  outputs** — "which way is more like what the student currently makes".
The student ascends (real − fake): move toward data, away from your own
current habits. **This is our KAIST SDS with the crucial upgrade**: SDS's
baseline was the injected noise ε (making it mode-seeking and oversaturated
— our radioactive hotdog); DMD replaces it with a learned fake-score, which
is the Variational Score Distillation idea (ProlificDreamer) applied to
2D generation. Same detached-gradient implementation trick we wrote in
`guidance/sd.py`.

### DMD → DMD2 (Yin et al. 2024, NeurIPS oral)  {#dmd2}
DMD1 needed an expensive regression anchor (teacher-sampled noise→image
pairs + LPIPS) to stay stable. DMD2: (1) drop it; stabilize with
**two-time-scale updates** (fake score updated 5–10× per generator update —
a GAN-style critic schedule); (2) add a small **GAN discriminator head** on
the fake-score network's features, trained on REAL images — recovering
detail the teacher itself lacks (students can beat the teacher!); (3)
few-step generators via **backward simulation** — train the 4-step student
at its own intermediate states, not the forward-process states, killing the
train/inference mismatch. Distills SDXL to 1–4 steps at near-teacher (or
better) quality; the de-facto open SOTA of this family.

### 3b. DMDR: distillation meets RLHF (Jiang et al. 2025) — the sequel
The production pipeline after DMD2 was sequential: distill, THEN align with
human preferences (RL on a reward model). Both steps fight each other: RL on
a few-step student **reward-hacks** (drifts off the data manifold chasing the
reward), and plain DMD compresses the WHOLE distribution uniformly instead of
the region people actually prefer. DMDR optimizes both jointly: the reward
**tilts** the distribution being matched (preference-aware distillation)
while the DMD term anchors the student to the teacher — distillation as the
regularizer that prevents reward hacking, reward as the weighting that makes
distillation selective. Two stages: reward-tilted matching with dynamic
distillation strategies, then joint DMD+RL. This is the recipe behind
Z-Image-Turbo, and the open demo code (ImageNet SiT) builds directly on the
DMD2 codebase — our DMDR tutorial reads as a diff against the DMD2 one.

---

## 4. Family C — Straighten, then step → InstaFlow

### The core idea
One-step generation fails because the PF-ODE's trajectories are CURVED (the
marginal average of straight per-pair lines is bent — our note 06 Part E).
Family C doesn't fight the curvature with better losses; it REMOVES it:

    Reflow: sample (noise ε, image X̂) couples FROM THE TEACHER, retrain the
    flow on straight lines between these couples (never re-pair them!).

Each reflow provably straightens transport (couples stop crossing); after
one reflow, trajectories are nearly straight, and by our 06 Fact 1
(straight ⇒ Euler exact) a single Euler step almost lands on the answer.
Then a final **distillation** stage fits a true one-step map onto the
2-rectified-flow's endpoints with a perceptual (LPIPS) loss — legitimate
now, because the straightened map is nearly deterministic per noise.

**We have literally built this** at toy scale: KAIST A3's Reflow (CD 36.9 →
25.7 at 10 steps) — InstaFlow is the same three lines of math applied to
Stable Diffusion with 199 A100-days of compute. Order matters and is the
paper's headline: distill-without-straightening fails (the student must fit
an average of crossing trajectories); straighten-then-distill works.

---

## 5. Family D — Adversarial distillation (context; closed training code)

ADD/SDXL-Turbo (Sauer et al. 2023): student takes 1–4 steps; loss = a
discriminator on DINOv2 features (realism) + distillation to the teacher's
denoised prediction. SDXL-Lightning (ByteDance): progressive + adversarial
distillation with the discriminator built from the teacher's own UNet
encoder. Hyper-SD: trajectory-segmented consistency + human-feedback +
LoRA. These ship weights, not training code — which is exactly why the
tutorials in this folder cover DMD2/PCM (their GAN branches teach the same
techniques with code you can read). SANA-Sprint ('25) is the modern hybrid:
continuous-time consistency (sCM, TrigFlow parametrization — no discrete
schedule at all) + latent adversarial distillation, with code in NVlabs/Sana.

---

## 6. Choosing between them (the practical table)

| | LCM(-LoRA) | PCM | DMD2 | InstaFlow |
|---|---|---|---|---|
| Family | trajectory | trajectory+GAN | distribution+GAN | geometry |
| Sweet spot | 4–8 steps | 1–16 steps, one model | 1–4 steps | 1 step |
| 1-step quality | weak (blurry) | good (adv.) | **best-in-class** | good |
| Extra nets during training | none (EMA target) | +discriminator | +fake score, +disc. head | none |
| Training cost | low (LoRA: hours) | low-medium | high (two models live) | very high (reflow data gen) |
| CFG at inference | baked in (w-embed) | baked in | baked in (fixed w) | baked in |
| Plug-in LoRA form | ✅ the classic | ✅ | ✅ (community) | ✗ (full model) |
| Code | official + diffusers | official | official (full) | inference + recipe |

(DMDR extends the DMD2 column: same costs plus a reward model in the loop;
choose it when the goal is few-step AND preference-aligned — the current
frontier for production systems.)

Rules of thumb: need cheap acceleration of an existing fine-tuned model →
LCM-LoRA/PCM-LoRA. Need maximum 1–4-step quality and can afford training
two networks → DMD2. Doing research on fast sampling → understand all four;
family C composes with A/B/D (straighten first, then distill by any method).

## 7. How this folder maps to our curriculum

- Consistency/PCM ⇢ notes 10 (every target is a DDIM/strided step; x̂₀
  boundary identity), HW1 Q7 (ᾱ-ratio striding).
- DMD2 ⇢ KAIST SDS/PDS writeup (same gradient plumbing; fake score =
  upgraded baseline), notes 11 (CFG inside the teacher branch).
- InstaFlow ⇢ notes 06 Parts C–F + KAIST A3 Reflow (identical mechanism,
  industrial scale).
- All four ⇢ notes 08 Level 8's warning ("the pairing IS the method")
  recurs everywhere: LCM pairs (x_t, teacher-step(x_t)), DMD2 pairs nothing
  (that's the point), InstaFlow pairs (ε, X̂) and must never re-sample.

## Sources
- [DMD2 repo](https://github.com/tianweiy/DMD2) · [DMD2 paper](https://arxiv.org/abs/2405.14867) · [DMD paper](https://arxiv.org/html/2311.18828)
- [PCM repo](https://github.com/G-U-N/Phased-Consistency-Model) · [PCM paper](https://arxiv.org/abs/2405.18407)
- [InstaFlow repo](https://github.com/gnobitab/InstaFlow) · [Rectified Flow paper](https://arxiv.org/abs/2209.03003)
- [LCM repo](https://github.com/luosiallen/latent-consistency-model)
- [SANA-Sprint](https://arxiv.org/abs/2503.09641) · [Sana repo](https://github.com/NVlabs/Sana)
- [DMDR repo](https://github.com/vvvvvjdy/dmdr) · [DMDR paper](https://arxiv.org/abs/2511.13649) · [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- Survey context: [DMD topic page](https://www.emergentmind.com/topics/distribution-matching-distillation-dmd), [Inference-time distillation survey](https://arxiv.org/pdf/2412.08871)
