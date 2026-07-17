# DMDR, Line by Line — Distillation Meets RLHF (the DMD2 Sequel)

*A beginner's reading companion to the open DMDR training demo
(`distillation/dmdr/train_cc/sit/`, from "Distribution Matching Distillation
Meets Reinforcement Learning", arXiv 2511.13649 — the recipe behind
Z-Image-Turbo). This is a SEQUEL to our DMD2 tutorial
(`dmd2_line_by_line.md`): the DMD gradient, the fake-score chase, the
detached-target trick, TTUR, and backward simulation are all inherited and
cited there rather than re-derived — this document covers what DMDR ADDS.
Family context is `00_overview.md §3b` (family B + RL); CFG is notes
`11 §4` (and `11 §7` for velocity models, which is what runs here).*

**The codebase in one sentence:** take DMD2's two-score matching loop,
replace the second full teacher copy with tiny LoRA "guidance units" riding
on ONE frozen teacher body (fake score = teacher + LoRA, real score =
teacher), delete the GAN and the entire dataset, and add a differentiable
reward (DINOv2's log-probability of the conditioning class) to the
generator's loss — *jointly with* the DMD term, so the student matches a
**reward-tilted** teacher distribution instead of being RLHF'd after
distillation.

---

## Methodology

### M1 — The problem: distill, then align — and why the sequence fights itself

Production diffusion pipelines after DMD2 ran two independent post-training
stages in sequence: (1) distill the teacher to a few-step student, (2) align
the student with human preferences by fine-tuning against a reward model
(RLHF / reward-feedback learning). The DMDR paper's abstract states the two
failure modes of keeping them separate, and the demo's README quantifies one
of them:

- **RL alone reward-hacks.** A few-step student fine-tuned on a reward has
  nothing anchoring it to the data manifold; it drifts toward whatever
  off-distribution textures maximize the reward. (Abstract: "DMD serves as
  an effective regularizer to mitigate reward hacking during RL training.")
- **DMD alone is preference-blind.** Distillation compresses the *whole*
  teacher distribution uniformly — the student spends capacity on regions
  nobody prefers. (Abstract: "RL enables more preference-aware and
  controllable distillation rather than uniformly compressing the full data
  distribution.")

The demo's own results table (`train_cc/sit/README.md`) shows exactly the
quality↔diversity trade this buys on ImageNet 256 with a 1-step SiT-XL:
turning RL on moves Inception Score 232→416 and Precision 0.77→0.90 while
Recall drops 0.64→0.40 and FID rises 2.13→6.95 — the reward *concentrates*
the student on high-scoring regions, at diversity's expense. The README
frames this honestly as a trade-off dial, not a free win.

### M2 — DMDR's move: reward-tilted distribution matching (one identity)

DMD2's generator objective was `∇KL(p_fake ‖ p_real)` estimated by a
difference of two scores (our DMD2 tutorial §M2, Eq. M2.1/M2.2). DMDR's
generator objective in this demo is, verbatim from the training loop (§T4):

    loss_gen = dmd_loss + w · CE(DINOv2(decode(G(z,y))), y)

Cross-entropy against the conditioning class `y` is minus the log-probability
the reward model assigns to it, i.e. minus a reward `r(x, y) =
log p_DINO(y|x)`. So the objective is

    L(θ) = KL( p_fake ‖ p_real ) − w · E_{x∼p_fake}[ r(x) ].

Now one line of algebra (only log rules, notes `09 §A1`): for the *tilted*
distribution `p̃(x) ∝ p_real(x)·exp(w·r(x))`,

    KL( q ‖ p̃ ) = E_q[log q − log p_real − w·r] + log Z
                = KL( q ‖ p_real ) − w·E_q[r] + const.

**The additive "DMD loss + reward loss" of the code is exactly reverse-KL
against the reward-tilted teacher distribution** `p_real·e^{w·r}/Z` — up to
the usual DMD approximations (score estimated by networks, per-sample
normalizer, sampled t). That is the paper's "Reward-Tilted Distribution
Matching" in gradient form, and it makes both abstract claims mechanical:

- The reward **tilts the matching target**: the student is no longer asked
  to cover `p_real` uniformly but to concentrate where `e^{w·r}` is large —
  preference-aware distillation.
- The DMD term **anchors against reward hacking**: any drift off
  teacher-support costs KL immediately, because `s_fake − s_real` (the DMD
  gradient direction, DMD2 §M2) points back toward teacher-land. The reward
  can only re-weight *within* the teacher's support, not escape it. `w`
  (`--dino-loss-weight`, 0.05 in the shipped script) is the temperature of
  the tilt.

Contrast with sequential RLHF: there the KL anchor is at best an explicit
penalty to the *pre-RL student*, and with a few-step student the reward
gradient quickly finds adversarial directions. Here the anchor is the live,
continuously re-estimated distribution gap to the *teacher*.

### M3 — What this demo changes vs DMD2's machinery

Everything below is verified against the code; DMD2 section numbers refer to
`dmd2_line_by_line.md`.

1. **One score body instead of two.** DMD2 kept a frozen `real_unet` AND a
   fully trainable `fake_unet` copy (DMD2 §G1). DMDR's demo keeps ONE frozen
   SiT-XL teacher body and adds rank-32 LoRA matrices on every attention
   block's q/v projections (`guidance_units.py`, §L below). The *fake* score
   is the body with LoRA applied at scale `--lora-scale-f = 2.0`; the *real*
   score is the same body with LoRA at scale 0 (pure teacher) — with a
   twist: early in training the real branch leaks a cosine-decaying fraction
   of the LoRA (§T4). Only the LoRA parameters train on the guidance side
   (~millions instead of ~675M) — this is VSD/ProlificDreamer's LoRA
   fake-score idea (DMD2 §M3) returning after DMD2 had upgraded it to a full
   copy.
2. **No GAN, no dataset.** DMD2's second innovation was a GAN head on real
   LAION latents (DMD2 §M4.2, §G5). This demo has *neither a discriminator
   nor a single real image*: the only data loader is a `dummy_dataloader` of
   random tensors that exists purely because DeepSpeed demands one (§T1).
   Training is data-free — signal comes from the teacher's weights and the
   frozen reward model. In `00_overview §1` taxonomy terms: DMD2 = B + D;
   this demo = B + RL, with the reward model replacing the discriminator as
   the "signal beyond the teacher".
3. **Flow matching, not ε-prediction.** The backbone is SiT (velocity
   prediction on the linear path `x_t = (1−t)x₀ + t·ε`, notes `06`), so
   every ε↔x̂₀ conversion from DMD2 becomes the one-liner `x̂₀ = x_t − t·v̂`
   — no `ᾱ` buffers, no float64 rescue near t≈1 (the conversion has no
   near-zero division), and CFG combines *velocities* (notes `11 §7`).
4. **The RL term is ReFL-style, not policy-gradient.** The reward is
   *differentiable* (frozen DINOv2 ViT-L/14 + linear ImageNet head), so the
   code simply backprops CE through DINOv2 → a differentiable resize →
   the VAE decoder → the student's one gradient-carrying step. No REINFORCE,
   no log-prob ratios, no value network — "RL" here means reward-gradient
   fine-tuning, following ReFL (see M5).
5. **Same TTUR, same detached-target trick, same backward simulation** — 5
   guidance updates per generator update (`--ratio-update 5.0`, DMD2 §M4.3),
   the `½‖x − (x−grad).detach()‖²` gradient injection verbatim (DMD2 §G3),
   and student-generated training inputs via a no-grad self-sampler
   (DMD2 §M4.4/§U2; here `v2x0_sampler`, §S2).

### M4 — The two-stage recipe as shipped (and what is paper-only)

The demo trains in two launches (`scripts/`):

**Stage 1 — cold start (`cold_start_sitxl.sh`)**: 20 001 generator steps of
*pure DMD* — `--encoder-type` is unset (no reward model is even loaded) and
`--cold-start-iter 20001` equals the step budget, so the reward branch can
never fire. What makes it "dynamic" is `--dynamic-step 10000`, which drives
**exactly two cosine-annealed schedules**, both decaying to neutral over the
first 10k generator steps:

1. **Annealed timestep sampling** (§H1): the noise levels used both to train
   the fake score and to query the DMD gradient are drawn from
   Beta(4, 1.5) — mass near t≈0.86, the high-noise end — annealing to
   Beta(1, 1) = Uniform(0,1). Early training focuses on coarse structure
   (where a 1-step student is worst), then spreads to all noise levels.
2. **Annealed real-branch LoRA scale** (§T4): the "real" score starts as
   teacher + 0.75×LoRA and cosine-decays to the pure teacher. Since the
   LoRA *is* the fake delta, the real target initially sits partway toward
   the fake score — shrinking `s_real − s_fake` and gentling the DMD
   gradient while the student is still far from the teacher — then hardens
   into the true teacher.

These are the demo's only two `dynamic_step`-driven mechanisms, which lines
up with the abstract's "two dynamic distillation training strategies in the
initial stage" — but the abstract does not name them, and I cannot verify
the paper's exact terminology from the demo alone; treat the pairing as a
strong inference, not a citation.

**Stage 2 — joint DMD+RL (`dmdr_sitxl_refl.sh`)**: `--resume-ckpt` from step
20 000, ~2 000 more generator steps with `--cold-start-iter 20000` (reward
active from the first resumed step), `--encoder-type dinov2`,
`--dino-loss-weight 0.05`, batch shrunk 64→24 (the with-grad VAE decode +
DINOv2 forward is memory-hungry) and lr halved to 1e-5. Because
`global_step ≥ 20000 > dynamic_step = 10000`, both dynamic schedules are
already fully annealed in stage 2: the real branch is the pure teacher and
timesteps are ~uniform. So stage 2 is exactly the abstract's "joint DMD and
RL optimization in the second stage".

**What the demo does NOT contain — said explicitly:** the abstract places
"Reward-Tilted Distribution Matching optimization" *in the initial stage*,
alongside the dynamic strategies. In this open demo, stage 1 has **no reward
term at all** (no reward model is constructed); the only reward-tilting is
stage 2's additive loss (which, per M2, *is* tilted matching in gradient
form). Whatever additional stage-1 reward-tilting machinery the paper
describes for the T2I/Z-Image models is not present in the ImageNet demo,
and this tutorial makes no claims about its implementation. Likewise there
is no GAN branch, no human-preference reward (ImageReward/HPS-style), and no
text conditioning here — class labels stand in for prompts.

**CFG's role — a deliberate ImageNet difference.** DMD2 baked CFG 6–8 into
the real branch (DMD2 §G3); this demo ships `--cfg-r 0`, i.e. the real score
is *unguided*. The README explains: on ImageNet, DMD works without CFG, and
raising it trades diversity for fidelity (their table: cfg 1.5 → FID 5.20,
IS 387; cfg 4 → FID 15.70, IS 454) — whereas on text-to-image the real
estimator *needs* large CFG, an observation their sibling paper
**Decoupled-DMD** (arXiv 2511.22677) develops into a full story (CFG
augmentation as the primary engine of few-step conversion, distribution
matching as regularizer). Decoupled-DMD is mentioned here only for context —
it is the *other* half of the Z-Image-Turbo recipe and has its own paper.

### M5 — Pedigree: ReFL (and where the acknowledgements point)

The repo's acknowledgements name DMD2, SRA, and **ReFL** (the ImageReward
repo). ReFL ("Reward Feedback Learning", Xu et al. 2023) fine-tunes a
diffusion model by: rolling the sampler *without* gradient to a randomly
chosen late step, taking ONE gradient-carrying denoising step, one-step
predicting x̂₀, decoding, and backpropagating a differentiable reward
through that single step. Map it onto this code and the correspondence is
exact: the no-grad roll is `v2x0_sampler` (backward simulation), the single
gradient step is the student's `pred_v` call in the generator turn, and the
reward is DINOv2's class log-prob instead of ImageReward. For the 1-step
demo the "roll" is empty — the gradient step IS the whole generation — so
ReFL degenerates to direct differentiable reward fine-tuning. What DMDR adds
*on top of* ReFL is precisely the DMD anchor sharing the same backward pass
(M2); the training script's filename (`train_sitxl_refl_deepspeed.py`) is
honest about the lineage.

### M6 — Map of the code

| File | Role |
|---|---|
| `guidance_units.py` | `QVLoRAUnits` — the rank-r LoRA pair for q/v; the *parameters* that turn one teacher body into both score estimators |
| `models_sit.py` | SiT-XL backbone (DiT-style transformer, velocity output); LoRA injection point in `Attention.forward`; `lora_scale` threaded through every block |
| `samplers.py` | `pred_v` (one call + velocity-CFG), `v2x0_sampler` (student's few-step SDE sampler = backward simulation), `euler_sampler` (teacher reference only) |
| `utils.py` | annealed timestep samplers (dynamic strategy #1), `get_sample`, differentiable DINOv2 preprocessing, small helpers |
| `train_sitxl_refl_deepspeed.py` | everything else: dual DeepSpeed engines, the 5:1 loop, guidance update, generator update (reward + DMD, one backward) |
| `arguments.py`, `configs/`, `scripts/`, `convert_weight/` | flags, ZeRO-2 config, the two launch scripts, REPA checkpoint conversion |

Two models, two optimizers, two `Accelerator`s (a DeepSpeed necessity, §T1):
**generator** = `gen_model`, a full SiT-XL/2 initialized from a REPA-trained
SiT-XL checkpoint, fully trainable; **guidance** = `guidance_model`, the same
architecture with LoRA, main weights copied from the same checkpoint and
frozen, only LoRA trains.

Shape conventions (B = 64 stage 1 / 24 stage 2 per GPU; resolution 256):

| Tensor | Shape |
|---|---|
| latent / noise / x0 / v_pred / grad | (B, 4, 32, 32) |
| class labels `y` | (B,) long, values 0–999 (1000 = null class) |
| timesteps `ts` | (B, 1, 1, 1) float in [0, 1] (t=1 pure noise), `.flatten()`→(B,) for the model |
| SiT token stream | (B, 256, 1152)  (16×16 patches of a 32×32 latent, patch 2) |
| LoRA factors per block | A: (1152, 32), B: (32, 1152) |
| decoded image → DINOv2 input | (B, 3, 256, 256) → (B, 3, 224, 224) |
| DINOv2 logits | (B, 1000) |

---

## §L — `guidance_units.py` (all of it) + the LoRA plumbing in `models_sit.py`

An honest note first: despite the name, this file does not contain any loss
— the DMD and reward math live inline in the training script (§T4). What it
contains is the 30-line module whose *parameters* are the entire trainable
guidance side. The training log even prints them as "Guidance units
trainable parameters". Conceptually it IS the heart of DMDR's parameter
economy: these matrices are the only thing separating `s_fake` from
`s_real`.

```python
class QVLoRAUnits(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
```
One LoRA unit per attention block. `in_features = out_features = 1152`
(SiT-XL width); the training scripts pass `rank=32`.

```python
        self.lora_QA = nn.Parameter(torch.randn(in_features, rank))
        self.lora_QB = nn.Parameter(torch.randn(rank, out_features))
        self.lora_VA = nn.Parameter(torch.randn(in_features, rank))
        self.lora_VB = nn.Parameter(torch.randn(rank, out_features))
```
Two low-rank factor pairs — one for the query projection, one for the value
projection (the classic LoRA target choice). `A@B` is a rank-32
1152×1152 update. Note both factors are initialized `randn` here — standard
LoRA zero-init happens *elsewhere*: `SiT.initialize_weights` in
`models_sit.py` sets `lora_QB` and `lora_VB` to zero, so the delta starts at
exactly 0 and the guidance model at step 0 *is* the teacher (the same warm
start DMD2 got by copying the whole UNet, DMD2 §G1). The A factors stay
N(0, 1) — unusually large by LoRA conventions (no 1/√rank scaling), absorbed
by the learning rate and the zero B.

```python
    def forward(self, q, v):
        q_l = q @ (self.lora_QA @ self.lora_QB)
        v_l = v @ (self.lora_VA @ self.lora_VB)
        return q_l, v_l
```
Compute only the *delta*, not the sum — the caller decides how much of it to
add. `q, v` arrive as (B, 256, 1152); the parenthesization materializes the
full 1152×1152 product first (fine at this size). Returning raw deltas is
what makes the **scale dial** possible, and the scale dial is the method:

```python
# models_sit.py, Attention.forward
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads * self.head_dim).permute(2, 0, 1, 3)...
        q, k, v = qkv.unbind(0)
        if self.lora is not None:
            q_l, v_l = self.lora(q, v)
            q = q + lora_scale * q_l
            v = v + lora_scale * v_l
```
The injection point: after the fused qkv projection, before heads are split.
`lora_scale` is a *runtime argument*, threaded from `SiT.forward(x, t, y,
lora_scale=...)` through every `SiTBlock` into every `Attention`. So ONE
set of weights answers as a whole family of models:

- `lora_scale = args.lora_scale_f = 2.0` → the **fake score** (trained to
  denoise student samples, §T3);
- `lora_scale = 0` → the **real score**, the untouched frozen teacher;
- `lora_scale = lora_scale_r ∈ (0, 0.75]` → the annealed hybrid used as the
  real branch early in training (§T4, dynamic strategy #2).

Contrast DMD2 §G1: two full 2.6GB UNets and a monkey-patched forward. Here
the real/fake pair costs 56 rank-32 factor pairs and a float.

The rest of `models_sit.py`, grouped (read once, then trust): standard
DiT/SiT — `PatchEmbed` + fixed 2D sin-cos positional embeddings,
`TimestepEmbedder` (sinusoidal → MLP; note `t` here is a *float in [0,1]*,
not an integer index), `LabelEmbedder` with a 1001-row table (row 1000 = the
CFG null class, inherited from REPA pretraining — this trainer itself never
drops labels: `forward`'s `no_drop=True` default makes `token_drop`
unreachable, so `--cfg-prob` only toggles the table size), 28 adaLN-Zero
blocks conditioned on `t_embed + y_embed`, a `FinalLayer`, `unpatchify` back
to (B, 4, 32, 32). Output = **velocity**, one call.

---

## §S — `samplers.py`: how a velocity model generates

### S1 — `pred_v`: one call, with velocity-CFG

```python
def pred_v(model, latents, t, y, cfg_scale, lora_scale=0, num_of_calsses=1000):
    do_cfg = cfg_scale > 1.0
```
The single gateway every score/generator call in the training loop goes
through. Strictly-greater-than-1 gates CFG — so the demo scripts'
`--cfg-r 0` means the real branch runs *unguided* (M4).

```python
    if do_cfg:
        latents = torch.cat([latents, latents], dim=0)
        t = torch.cat([t, t], dim=0)
        y_null = torch.tensor([num_of_calsses] * y.size(0), device=y.device)
        y = torch.cat([y_null, y], dim=0)
```
The doubled-batch CFG trick (DMD2 §G0, notes `11 §4`), class-conditional
flavor: the "empty prompt" is class index 1000, the `LabelEmbedder`'s extra
row. Order is uncond-first (`[y_null, y]`), matched by the chunk below.
(`num_of_calsses` [sic] — the typo ships.)

```python
    time_input = t.flatten()
    v_pred = model(latents, time_input, y, lora_scale=lora_scale)
```
`t` arrives broadcast-shaped (B,1,1,1) from the timestep samplers (§H1);
the model wants (B,). `lora_scale` rides through to §L — **this argument is
which distribution you are asking about.**

```python
    if do_cfg:
        v_pred_uncond, v_pred_cond = v_pred.chunk(2)
        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
    return v_pred.to(latents.dtype)
```
CFG applied to *velocities* — exactly notes `11 §7`: guidance is linear in
the score, the velocity is affine in the score, so the same
`u_∅ + s·(u_c − u_∅)` combination is valid. Returns (B, 4, 32, 32).

### S2 — `v2x0_sampler`: the few-step generator's sampler = backward simulation ★

The in-file comment sets the frame: *"In dmd, we always hypo that the model
predicting x0, so we need to modify the sampling process accordingly. We
only implement sde sampling here."* — i.e. treat each model call as an
x̂₀-predictor and re-noise with fresh noise between steps, the same
consistency-style stochastic sampler as DMD2's `sample_backward`
(DMD2 §U2), now in flow-matching clothes. This function is BOTH the
student's inference sampler AND the training-input factory (called under
`no_grad` at the top of every training iteration, §T2).

```python
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=latents.dtype)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0.0
```
The student's grid, descending from t=1 (noise) to t=0 (data). The second
line is the SD3-style **timestep shift** — warp the grid toward high noise
for `shift > 1` (useful at high resolution); the scripts use `shift = 1`,
making it the identity, and the `t_steps[-1] = 0.0` guard is then redundant
but harmless. With `num_steps = 1` (the recommended demo): grid = [1.0, 0.0].

```python
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            ...
            d_cur = model(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)
```
March the grid pairwise (same off-by-one discipline as MIT lab 1, Cell 2);
one model call per step gives the velocity `d_cur` (B, 4, 32, 32). Note the
call goes straight to `model`, not `pred_v` — no CFG, no LoRA: this is the
plain student.

```python
            x_0 = x_cur + (0 - t_cur) * d_cur
            all_x0.append(x_0)
            x_next = (1 - t_next) * x_0 + t_next * torch.randn_like(x_0)
    return x_next, all_x0, t_steps[:-1]
```
Three lines that ARE the sampler. Line 1: ride the current velocity all the
way to t=0 — on the linear path `x_t = (1−t)x₀ + t·ε` with `v = ε − x₀`,
algebra gives `x₀ = x_t − t·v` exactly; this is the flow-matching sibling of
DMD2's `get_x0_from_noise` (DMD2 §0), with no ᾱ buffer and no float64
rescue needed. Line 2: keep *every* intermediate x̂₀ — the training loop
will sample from this list to build inputs at intermediate grid times
(DMD2 discarded all but the chosen prefix; here the whole trajectory of
estimates is returned). Line 3: re-noise the estimate down to the next grid
level with FRESH noise — the "SDE" of the comment, DMD2 §U2's η=1-flavored
jump. Returns the final sample, the list of estimates, and the grid times
they were made at.

For `num_steps = 1` the whole function collapses to: `x₀ = z − v(z, t=1)`,
`all_x0 = [x₀]`, grid = [1.0] — **one model call, noise to image.** (The
first return is named `xT` at the call site but holds the final *clean*
sample; quirk ledger.)

### S3 — `euler_sampler` (grouped — teacher reference only)

A standard deterministic Euler ODE sampler (`x ← x + (t_next − t_cur)·v`,
MIT lab 1 Cell 3 with the SiT velocity as drift), with optional Heun
second-order correction and a CFG window (`guidance_low/high`) — SiT-repo
heritage. In this codebase it is called exactly once, before training, with
250 steps and `cfg_scale = args.cfg_r`, to render a reference grid from the
freshly loaded `gen_model` — which at step 0 still *is* the teacher (same
weights), so the saved `mutistep_teacher_*.png` shows what the student is
being distilled from. Two details worth noting and moving on: its CFG
batch order is cond-first (`[y, y_null]`, opposite of `pred_v`, each
internally consistent), and the saved filename says `500nfe` assuming
CFG-doubled calls even when `cfg_r = 0` makes it 250.

---

## §H — `utils.py`: the annealed samplers and the reward preprocessing

### H1 — `sample_continue` / `sample_discrete`: dynamic strategy #1 ★

```python
def sample_continue(B, alpha=4.0, beta=1.5, s_type="logit_normal", step=0, dynamic_step=1000):
    ...
    elif s_type == "logit_normal":
        if dynamic_step > 0:
            progress = min(step / dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141593)))
            alpha = 1.0 + (alpha - 1.0) * cosine_decay
            beta = 1.0 + (beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(alpha, beta).sample((B,))
        return t.reshape(B,1,1,1)
```
Where training *timesteps* come from — and where the first "dynamic
distillation strategy" lives. Despite the name, the `"logit_normal"` branch
draws from a **Beta distribution** (naming quirk; the scripts' `--gui-a 4.0
--gui-b 1.5` are Beta parameters). Beta(4, 1.5) has mode ≈0.86 and mean
≈0.73 — in this codebase t=1 is pure noise, so early training concentrates
both the fake-score fitting (§T3) and the DMD queries (§T4) on the
high-noise regime, where a 1-step student's errors are biggest and coarsest.
The `cosine_decay` block interpolates (α, β) from (4, 1.5) at step 0 to
(1, 1) — i.e. **Uniform(0,1)** — over `dynamic_step = 10000` generator
steps, then stays uniform. Compare DMD2, which sampled the DM loss's t
uniformly on [2%, 98%] for the whole run (DMD2 §G1): DMDR replaces the fixed
window with a curriculum. Returned shape (B, 1, 1, 1), ready to broadcast
against latents. The `"uniform"` branch (used nowhere for continuous t in
the scripts) draws Uniform[0.001, 1).

`sample_discrete` is the same machinery restricted to a *grid*: given the
student's `t_steps`, it draws a Beta (or uniform) sample and snaps it to the
nearest grid value (`argmin |t − grid|`), with the identical cosine
annealing. The training loop uses it with `--s-type-gen uniform` to pick
*which grid step* the generator trains at — for `num_steps = 1` the grid is
[1.0] and every draw returns 1.0, so all of this is future-proofing for the
multi-step setting.

### H2 — `get_sample`: pick the matching x̂₀ off the simulated trajectory

```python
def get_sample(x0_all, sample_t, t_steps):
    stacked_x0 = torch.stack(x0_all, dim=0)
    s_to_index = {value.item(): idx for idx, value in enumerate(t_steps)}
    indices = torch.tensor([s_to_index[value.item()] for value in sample_t])
    sample = stacked_x0[indices, torch.arange(b)]
```
Stack the per-grid-step x̂₀ list into (num_steps, B, 4, 32, 32), build a
value→index map for the grid, and for each batch item select the x̂₀ that
was predicted at that item's sampled grid time — a gather across the
simulated trajectory. Shape out: (B, 4, 32, 32). One honest flag for the
multi-step case: the x̂₀ *predicted at* grid time tᵢ is then re-noised back
*to* level tᵢ (§T2), whereas at inference the input at tᵢ is built from the
estimate of the *previous, noisier* step tᵢ₋₁ — an off-by-one relative to
strict inference matching (DMD2 §U2 used the previous step's estimate). At
`num_steps = 1` this is moot: ts is always 1.0 and `(1−1)·x̂₀ + 1·noise`
erases the choice entirely.

### H3 — The rest (grouped)

`mean_flat(x)` — per-sample mean over (C, H, W): (B,4,32,32) → (B,).
`DINOv2ProcessorWithGrad` — the reward model's preprocessing as
*differentiable tensor ops*: ImageNet mean/std normalization + bicubic
resize to 224. The "WithGrad" in the name is the whole point — the stock
HF image processor works on PIL/NumPy and would sever the graph; this one
keeps the reward differentiable end-to-end (ReFL's requirement, M5).
`ckpt_path_to_accelerator_path` — string surgery mapping the tiny
`torch_units/step-N.pt` bookkeeping file to the two DeepSpeed state
directories (`accelerator_1/step-N`, `accelerator_2/step-N`) for resume.

---

## §T — `train_sitxl_refl_deepspeed.py`: the loop

### T0 — Module-level helpers (grouped)

`array2grid` — clamp/scale a batch to a uint8 image grid for saving.
`sample_posterior` — VAE-moments sampler, **defined and never called**
(there is no data to encode; a fossil from the data-loading trainer this
was pruned from). `update_ema` — a standard EMA step, **also never called**:
despite the tutorial-brief's expectations, this demo keeps NO EMA of the
student (same dead-EMA situation as DMD2's unused `EMA` class, DMD2 §0).
`copy_main_weights(source, target)` — copies every non-`lora` parameter by
name; this is how the guidance model's frozen body gets the teacher weights.
`create_logger`, `requires_grad` — cosmetics and a flag-flipper.

### T1 — Setup (grouped — five decisions that matter)

1. **Two Accelerators, two DeepSpeed engines.** DeepSpeed fuses
   model+optimizer+scheduler into one engine per `prepare()` call, so two
   independently-optimized models need two plugins (`z2_a`, `z2_b`, both
   ZeRO stage-2 from `configs/z2.json`) and two `Accelerator` objects
   sharing global state; `gen_model`+`optimizer_gen` prepare under plugin a,
   `guidance_model`+`optimizer_gui` under plugin b, and micro-batch size is
   copied across. This replaces DMD2's FSDP arrangement (DMD2 §T1) — same
   two-parameter-group structure, different sharding stack. `fp16` mixed
   precision throughout (`--mixed-precision fp16`; DMD2 used bf16 —
   fp16 works here partly because flow-matching x̂₀ conversion has no
   near-t=1 blow-up).
2. **The two models.** `gen_model` = SiT-XL/2, `load_state_dict(...,
   strict=False)` from the converted REPA checkpoint
   (`convert_weight/sit_conver.py` downloads the REPA SiT-XL and strips its
   projector heads), fully trainable. `guidance_model` = same class with
   `use_lora=True, lora_rank=32`; `copy_main_weights` clones the body; then
   the loop over parameters freezes everything without `lora` in its name
   and collects LoRA params into `params_model` — the ONLY trainable
   guidance parameters (with `--lora-rank 0` it would fall through to full
   finetuning; see the quirk ledger for why that mode only makes sense with
   CFG). `optimizer_gui` gets just this group, weight decay 0; the generator
   gets AdamW with wd 0.01 (same "yes, wd on a diffusion backbone" shrug as
   DMD2 §T1).
3. **Frozen VAE + frozen reward model.** `AutoencoderKL` (sd-vae-ft-ema,
   the SD latent space — SiT-on-ImageNet convention, `latents_scale =
   0.18215`). If `--encoder-type dinov2`: `torch.hub.load(...,
   'dinov2_vitl14_lc')` — DINOv2 ViT-L/14 **with its linear ImageNet
   classifier head** — cast to fp16, `requires_grad_(False)`. Frozen ≠ cut
   off: gradients still flow *through* it to the generator. Stage 1 passes
   no encoder type and never builds it.
4. **The dummy dataloader.** Random tensors, wrapped in a DataLoader, passed
   to `accelerator.prepare` and never iterated: DeepSpeed needs a loader to
   deduce batch sizing. This is the "there is no dataset" fact of M3 made
   concrete.
5. **Resume + reference grid.** Resume loads the two DeepSpeed states via
   `ckpt_path_to_accelerator_path` and restores the two step counters. Then
   the pre-training `euler_sampler` grid (§S3) is rendered — the teacher
   baseline every later `samples_step_*.png` gets compared against.

### T2 — The inner loop: skeleton and backward simulation

```python
    for inflop in range(inner_step, int(args.max_train_steps * args.ratio_update + 100)):
        with accelerator.accumulate([gen_model, guidance_model]):
            gen_model.eval()
            guidance_model.train()
```
The loop counts **inner steps** = guidance updates; generator updates happen
every `ratio_update = 5` inner steps, so the loop bound is
`max_train_steps × 5` (+100 slack so the final checkpoint lands). This is
DMD2's two-time-scale rule (DMD2 §M4.3/§T2) with inverted bookkeeping: DMD2
looped over generator steps and gated the generator's *loss*; DMDR loops
over guidance steps and gates the generator's *turn*. Net effect identical —
the fake score gets 5 looks at the student per student move.

```python
            latent = torch.randn(local_batch_size, in_channels, latent_size, latent_size)...
            ys = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
            with torch.no_grad():
                with accelerator.autocast():
                    xT, all_x0, t_steps = v2x0_sampler(model=gen_model, latents=latent,
                                                       y=ys, num_steps=args.num_steps, shift=args.shift)
```
Fresh noise (B, 4, 32, 32) and random classes (B,), then ONE no-grad run of
the student's own sampler (§S2). `xT` (misnamed — it is the final *clean*
sample) is unused below; what matters is `all_x0`, the trajectory of the
student's x̂₀ estimates. **This single simulation feeds both turns** — the
economy mirrors DMD2's one-forward-two-turns trick (DMD2 §T2), shifted one
level up: one *simulation* per iteration, consumed by the guidance loss now
and (on every 5th iteration) the generator loss too. No per-sample
`broadcast` of the step choice is needed (contrast DMD2 §U2): the full
trajectory is kept, so every rank runs the same number of model calls by
construction.

```python
            ts = sample_discrete(bsz, t_steps.flip(0), alpha=args.gen_a, beta=args.gen_b,
                                 s_type=args.s_type_gen, step=global_step, dynamic_step=args.dynamic_step)...
            input_latent_clean = get_sample(all_x0, ts, t_steps)...
            noise_i = torch.randn_like(input_latent_clean)
            input_latent_gen = (1 - ts) * input_latent_clean + ts * noise_i
            v_target = noise_i - input_latent_clean  # may not use
```
Build the generator's training input: pick a grid time per sample (§H1;
uniform over the grid per the scripts), fetch the student's own x̂₀ at that
time (§H2), and re-noise to that level via the linear path — DMD2 §U3's
`prepare_denoising_data`, flow-matching edition. For `num_steps = 1`:
`ts ≡ 1.0`, so `input_latent_gen = noise_i` exactly — pure noise in, no
mask needed (DMD2 needed an explicit `pure_noise_mask`; here the t=1
arithmetic does it for free). `v_target` is computed and never used — the
comment admits it (a hook for an optional regression term that isn't in
this demo).

### T3 — Guidance update: the fake score chases the student (every inner step)

```python
            ts_gui = sample_continue(bsz, alpha=args.gui_a, beta=args.gui_b,
                                     s_type=args.s_type_gui, step=global_step, dynamic_step=args.dynamic_step)...
            noise = torch.randn_like(input_latent_clean)
            input_latent_gui = (1 - ts_gui) * input_latent_clean + ts_gui * noise
            gt_diff = noise - input_latent_clean
```
A *continuous* annealed timestep (Beta(4,1.5)→Uniform, §H1 — dynamic
strategy #1 acting on the fake score's training distribution), a fresh
noising of the student's sample, and the flow-matching target
`v = ε − x₀`. `input_latent_clean` came out of a `no_grad` block, so it is
already detached — the generator cannot receive gradient from its critic's
training (DMD2 §G4's first-line `detach()`, achieved here by construction).

```python
            with accelerator2.autocast():
                v_pred_fake = pred_v(guidance_model, input_latent_gui, ts_gui, ys,
                                     cfg_scale=0, lora_scale=args.lora_scale_f)
                diffusion_loss = mean_flat((v_pred_fake - gt_diff) ** 2)
                loss_gui = diffusion_loss.mean()
            accelerator2.backward(loss_gui)
            ...
            optimizer_gui.step()
            optimizer_gui.zero_grad(set_to_none=True)
            optimizer_gen.zero_grad(set_to_none=True)  # important! ...
```
The **vanilla flow-matching loss on generator samples** — DMD2 §G4 with
`‖ε̂ − ε‖²` swapped for `‖v̂ − v‖²`. Two method-bearing details: `cfg_scale=0`
(no CFG on the fake branch — DMD2 asserted the same, §G1: `p_fake` is what
the student actually emits) and `lora_scale=args.lora_scale_f = 2.0` — the
LoRA delta is applied at scale 2 both when trained and when queried, so
"fake score" consistently means "teacher + 2×LoRA". Only the LoRA factors
receive gradient (everything else is frozen), so this loss *sculpts the
delta between the two distributions directly*. Backward runs through
`accelerator2` (the guidance engine); the trailing generator `zero_grad` is
the same belt-and-braces cross-contamination insurance DMD2 wore (§T2).

### T4 — Generator update ★ (every 5th inner step): reward + DMD, one backward

```python
            if (inner_step % args.ratio_update == 0) and inner_step > 0:
                gen_model.train()
                guidance_model.eval()
                with accelerator.autocast():
                    v_pred_gen = pred_v(gen_model, input_latent_gen, ts, ys, cfg_scale=0, lora_scale=0)
                x0 = input_latent_gen + (0.0 - ts) * v_pred_gen
```
The student's entire generative act: one graph-carrying call at the sampled
grid time (t=1 in the demo: `x0 = noise − v̂`), then the ride-to-zero
conversion. `x0` (B, 4, 32, 32) is the only tensor in this whole block
attached to the generator's autograd graph — everything below either feeds
on it with gradient (reward), or measures it and injects a hand-made
gradient (DMD).

**The reward (ReFL) term** — new in this sequel:

```python
                if global_step >= args.cold_start_iter:
                    if args.encoder_type is not None:
                        with accelerator.autocast():
                            latent_x0 = (x0 - latents_bias) / latents_scale
                            samples = vae.decode(latent_x0).sample
                            samples = ((samples + 1) / 2.).clamp(0, 1)
                            samples = transform_rep(samples)
                            logistics = rep_model(samples)
                            dino_loss = criterion(logistics, ys)
                            reward_loss_dino_mean = dino_loss.mean()
```
Gate first: the reward only exists past `cold_start_iter` AND when a reward
model was built — in stage 1 neither holds, in stage 2 both do from the
first resumed step (M4). Then the differentiable reward chain, every link
frozen but grad-transparent: un-scale the latent → **VAE decode WITH
gradient** (B,4,32,32)→(B,3,256,256) — this is the expensive line, and why
stage 2's batch drops to 24 → map [−1,1]→[0,1] → differentiable
normalize+resize to (B,3,224,224) (§H3) → DINOv2+linear head → (B,1000)
logits → cross-entropy against the *conditioning class* `ys`. Minimizing CE
= maximizing `log p_DINO(y|x)` = the README's "directly maximize the
classification accuracy score". This is ReFL's backprop-through-reward (M5)
with a classifier standing in for a human-preference model. Note what is
absent, honestly: no reward baseline, no KL-to-old-policy term, no sampling
of rewards — the *only* thing standing between this term and classic
classifier reward-hacking (adversarial DINOv2-fooling textures) is the DMD
loss sharing the same backward. `reward_loss_dino_mean` is initialized to a
zeros tensor before the loop so the joint sum below is well-defined when the
gate is closed.

**Dynamic strategy #2 — the annealed real branch:**

```python
                if global_step < args.dynamic_step:
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * global_step / args.dynamic_step))
                    lora_scale_r = args.lora_scale_r * cosine_factor
                else:
                    lora_scale_r = 0.0
```
The real-score estimator's LoRA scale: `0.75 → 0` along a cosine over the
first 10k generator steps, then exactly 0 forever. Early on, the "real"
branch is teacher + 0.75×(the fake delta trained at scale 2) — i.e. the
matching target starts partway toward the student's own distribution, which
shrinks `x̂₀_fake − x̂₀_real` and softens the DMD gradient while the student
is farthest from the teacher; as training proceeds the target hardens into
the pure frozen teacher. By stage 2 (`global_step ≥ 20000`) this is always
the pure teacher. (Interpretation is mine from the mechanics; the paper's
name for this schedule isn't verifiable from the demo — M4.)

**The DMD loss** — DMD2 §G3, velocity edition:

```python
                ts_dmd = sample_continue(bsz, alpha=args.gui_a, beta=args.gui_b, ...)
                noise = torch.randn_like(input_latent_gen)
                input_latent_dmd = (1 - ts_dmd) * x0 + ts_dmd * noise
                with accelerator.autocast():
                    with torch.no_grad():
                        v_pred_fake = pred_v(guidance_model, input_latent_dmd, ts_dmd, ys,
                                             cfg_scale=0, lora_scale=args.lora_scale_f)
                        v_pred_real = pred_v(guidance_model, input_latent_dmd, ts_dmd, ys,
                                             cfg_scale=args.cfg_r, lora_scale=lora_scale_r)
```
Diffuse the student's sample to a random annealed level (same Beta
curriculum as the fake score's training — the two sides of the KL are
matched at the same t-distribution), then query **the same network twice**:
once as the fake score (LoRA at 2.0, no CFG), once as the real score (LoRA
at the annealed `lora_scale_r`, CFG at `cfg_r` — which the ImageNet scripts
set to 0, i.e. *unguided*; contrast DMD2's CFG-6–8 real branch, §G3, and
see M4 for why). Both under `no_grad` — the scores are measured, never
backpropagated through.

```python
                    x0_r = input_latent_dmd + (0.0 - ts_dmd) * v_pred_real
                    x0_f = input_latent_dmd + (0.0 - ts_dmd) * v_pred_fake
                    p_real = x0 - x0_r
                    p_fake = x0 - x0_f
                    grad = (p_real - p_fake) / (torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                    grad = torch.nan_to_num(grad)
                    dmd_loss = 0.5 * torch.nn.functional.mse_loss(x0.float(),
                                          (x0 - grad).detach().float(), reduction='mean')
```
Line for line the DMD2 gradient (DMD2 §M2.2/§G3): convert both velocity
predictions to x̂₀, form `grad ∝ x̂₀_fake − x̂₀_real` (descending it moves the
student's sample toward teacher-land and away from where the student
over-produces), normalize by the per-sample mean `|x − x̂₀_real|`
((B,1,1,1), broadcast), and inject via the detached-target trick — fourth
sighting after our KAIST SDS, PDS, and DMD2. Two deltas vs DMD2: an explicit
`+1e-8` in the denominator (DMD2 relied on `nan_to_num` alone), and no
float64 anywhere (the flow conversion `x_t − t·v` is benign). One subtlety
for careful readers: `input_latent_dmd` is built from the *graph-connected*
`x0`, so `p_real`/`p_fake` technically carry graph — but `grad` is consumed
only inside `.detach()`, so the sole gradient path into the generator is the
first `mse_loss` argument, exactly as intended (the stray subgraph is wasted
memory, not wrong math).

**The joint update — the line the paper is named after:**

```python
                    loss_gen = dmd_loss_mean + args.dino_loss_weight * reward_loss_dino_mean
                accelerator.backward(loss_gen)
                ...
                optimizer_gen.step()
                optimizer_gen.zero_grad(set_to_none=True)
                optimizer_gui.zero_grad(set_to_none=True)
```
One backward carries both gradients into the same parameters at the same
step: the reward's "concentrate where DINOv2 scores my class highly" and the
DMD term's "stay on the teacher's distribution". Per M2's identity this sum
IS reverse-KL against the reward-tilted teacher `p_real·e^{0.05·r}/Z` — the
tilt is the alignment, the KL is the anti-hacking regularizer, and neither
exists as a separate training phase. Clip at norm 1.0, step, zero both
optimizers (mirror insurance).

### T5 — Bookkeeping (grouped)

The remaining ~90 lines: a `logs` dict (note: on the 4-of-5 guidance-only
iterations, the `dmd`/`dino` entries are stale values from the last
generator turn) appended as JSONL to `loss_gui_log` every inner step and
`loss_gen_log` every generator step; `global_step` incremented only on
generator turns (so `--max-train-steps` counts generator updates);
checkpointing = the two DeepSpeed `save_state` directories plus a tiny
`torch_units/step-N.pt` holding only the two step counters (the resume path
of §T1 reassembles all three); periodic sampling = one no-grad
`v2x0_sampler` run on fixed `(xz, yz)` noise/classes, VAE-decoded and saved
as a grid next to the teacher reference from §T1 — the visual A/B this
method lives and dies by.

---

## Quirk ledger (honesty section)

- **`guidance_units.py` is not where the guidance math lives.** It holds
  30 lines of LoRA parameters; the DMD gradient, the reward chain, and the
  fake-score loss are all inline in `train_sitxl_refl_deepspeed.py` §T3–T4.
  The name refers to the trainable *units* of the guidance model.
- **Stage-1 "Reward-Tilted Distribution Matching" is not in this demo.**
  The cold-start script trains pure DMD (no reward model is constructed);
  reward-tilting only appears as stage 2's additive loss. The abstract
  places tilted matching in stage 1 for the full method — whatever that
  entails beyond this demo is paper-only and not covered here.
- **"logit_normal" is a Beta distribution.** Both samplers' branch of that
  name draws `torch.distributions.Beta(alpha, beta)`; the docstrings talk
  about means and standard deviations of a logit-normal that is never used.
- **Invalid argparse defaults.** `--s-type-gen/--s-type-gui` default to
  `"normal"`, which is not in `choices` — argparse does not validate
  defaults (verified), so omitting the flags crashes later in
  `sample_continue`'s `ValueError`. The scripts always pass values.
- **README documents a flag that doesn't exist.** `--use-lora-gen` (LoRA
  for the *generator*) appears in `train_cc/sit/README.md` but not in
  `arguments.py` — the demo's generator is always fully finetuned.
  Similarly the README's "LoRA rank of the few-step diffusion model will be
  set to twice this value" describes code that isn't here.
- **`--lora-rank 0` would silently break DMD.** With LoRA disabled the
  guidance model fully finetunes, and real/fake become the SAME network —
  with `cfg_r = 0` they return identical predictions and `grad ≡ 0`. Full
  finetuning only makes sense with `cfg_r > 1`, where "real" becomes the
  CFG-boosted fake network — which is exactly the Decoupled-DMD reading
  (M4). The scripts always use rank 32.
- **Dead code:** `sample_posterior` and `update_ema` are defined and never
  called — **no EMA of the student is kept**, and no data is ever encoded.
  `v_target` is computed and unused ("# may not use"). `xT`, the sampler's
  final sample, is discarded by the training loop.
- **The multi-step input is off by one grid step** (§H2): x̂₀ predicted at
  tᵢ is re-noised to tᵢ, not tᵢ₊₁ — harmless at `num_steps = 1` (the demo's
  setting), a train/inference mismatch to check before using `num_steps > 1`.
- **Opposite CFG batch orders** in `euler_sampler` (cond-first) and `pred_v`
  (uncond-first); each pairs its own chunks correctly.
- **Stale log entries** on guidance-only steps; the same `logs` dict goes to
  both JSONL files.
- **`dmd_loss`'s `reduction='mean'`** divides the injected gradient by
  B·C·H·W, and `torch.abs(p_real).mean()` in the denominator carries a
  wasted autograd subgraph — both cosmetic (DMD2 had the first too).
- **Misnamed/typo'd**: `num_of_calsses`, `inflop` (the loop variable),
  `logistics` (the logits), `mutistep` in the saved filename.

---

## What to carry forward (DMDR → everything else)

1. **RLHF and distillation compose as ONE loss**: `KL(p_fake‖p_real) −
   w·E[r]` = reverse-KL to the reward-tilted teacher `p_real·e^{w·r}`. The
   reward tilts the target (preference-aware distillation); the DMD term is
   the live regularizer against reward hacking. One backward, no sequential
   pipeline — and `w` is an explicit quality↔diversity dial (README table:
   IS 232→416, Recall 0.64→0.40).
2. **A differentiable reward is the cheapest RL there is** (ReFL): freeze
   the reward model, keep every preprocessing op in tensor-land
   (`DINOv2ProcessorWithGrad`), decode with grad, backprop CE through the
   student's single generative step. No policy gradients needed — but also
   no protection except the anchor loss you pair it with.
3. **LoRA scale as a distribution dial**: one frozen teacher body + rank-32
   q/v deltas gives `s_real` (scale 0), `s_fake` (scale 2.0), and anything
   between (the annealed real branch) — DMD2's two full copies collapsed to
   a float argument. Fake-score-as-LoRA is VSD's idea rehabilitated at
   modern scale.
4. **Curricula over constants**: DMD2's fixed t-window and fixed real target
   become two cosine schedules — timestep sampling Beta(4,1.5)→Uniform, and
   real-branch LoRA leakage 0.75→0 — both keyed to one `dynamic_step` knob
   and both OFF by the time RL switches on. Cold-start pure-DMD first, then
   joint DMD+RL: the two-stage recipe.
5. **DMD can run data-free**: no dataset, no GAN — teacher weights + reward
   model are the only signal sources (the dummy dataloader exists for
   DeepSpeed, not for data). What DMD2 bought with a GAN-on-real-data
   (beating the teacher on realism), DMDR buys with a reward model — and on
   ImageNet, *without* CFG in the real branch (`cfg_r = 0`), whose
   quality↔diversity trade-off is instead exposed as an explicit knob
   (README; the T2I story, where CFG is load-bearing, is Decoupled-DMD's).
6. **Flow matching simplifies every DMD line**: `x̂₀ = x_t − t·v̂` replaces
   the ᾱ gymnastics, t=1 arithmetic replaces the pure-noise mask, and no
   float64 rescue is needed anywhere.
