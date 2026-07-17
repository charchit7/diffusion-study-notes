# DMD2, Line by Line — Distilling Stable Diffusion to 1–4 Steps

*A beginner's reading companion to the DMD2 training code
(`distillation/DMD2/main/`, from "Improved Distribution Matching
Distillation for Fast Image Synthesis", NeurIPS 2024, arXiv 2405.14867;
its parent DMD is arXiv 2311.18828). Every line that carries method: what it
does, why it's there, and what breaks without it. Concepts link to our
notes: score = −ε̂/σ is `10 §3` (restated at the top of `11`), CFG is
`11 §4`, the detached-target SDS trick is `writeups/kaist_reports.md`
(A-SDS/PDS).*

**The codebase in one sentence:** a one-shot (or 4-shot) generator network is
trained so that the *distribution* of its outputs matches the teacher
diffusion model's distribution — the matching signal is the difference
between two score networks (the frozen teacher = "real", a continuously
retrained copy = "fake") plus a GAN classifier grafted onto the fake
network's bottleneck, updated in an alternating generator/critic loop.

---

## Methodology

### M1 — The problem

A teacher (SD v1.5 or SDXL) generates great images but needs ~50 UNet calls.
We want a student `G` that maps pure noise `z` (and a prompt) to a finished
latent in ONE forward pass (or 4). Plain regression onto teacher outputs
(consistency-style, or DMD1's paired-ODE loss) needs an expensive dataset of
(noise → teacher-sample) pairs and caps the student at the teacher's
per-sample quality. DMD's move: don't match *samples*, match the
*distribution*.

### M2 — Distribution matching = a difference of two scores

Let `p_real` be the teacher's distribution and `p_fake` the current
student's output distribution. DMD minimizes the reverse KL, averaged over
diffused (noisy) versions of both, because scores of *noised* distributions
are exactly what diffusion models estimate everywhere (not just on the data
manifold):

    L(θ) = E_t  KL( p_fake,t ‖ p_real,t ),      x_t = α_t·G_θ(z) + σ_t·ε.

Differentiate the KL through the sample (this is the classic
score-function-of-the-integrand step; DMD §3.1):

    ∇_θ L ≈ E_{z,t,ε} [ w_t · ( s_fake(x_t, t) − s_real(x_t, t) ) · ∂x_t/∂θ ]     (M2.1)

Read it: at the noisy point `x_t`, `s_fake − s_real` is the direction "more
typical of the student than of the teacher"; descending it pushes the
student's sample *toward teacher-land and away from where the student
over-produces*. Two ingredients:

- **`s_real`** — the frozen teacher's score. Never trained. Via the ε↔score
  identity of notes `10 §3` / `11` intro, `s(x_t) = −ε̂(x_t,t)/σ_t`, so the
  code only ever calls the teacher UNet's ε-prediction. Crucially the code
  applies **CFG inside this branch** (`real_guidance_scale = 6–8`, notes
  `11 §4`): the "real" distribution being matched is the *guided, sharpened*
  teacher, which is what users actually sample from.
- **`s_fake`** — a score model of the *student's own current outputs*. There
  is no formula for it, so DMD trains a second full UNet copy, `fake_unet`,
  with the ordinary denoising loss **on generator samples**, continuously,
  interleaved with generator updates. It is a moving target tracking a
  moving distribution — the whole training is a two-network chase.

Both scores are converted to x̂₀-space before subtracting. With
`x̂₀ = (x_t − σ_t ε̂)/α_t` you get `s_fake − s_real = (α_t/σ_t²)(x̂₀_fake −
x̂₀_real)`, and DMD chooses the weight `w_t = σ_t²/α_t · CS/‖x − x̂₀_real‖₁`
(C·S = channels × pixels) precisely so the prefactor cancels and only a
per-sample normalizer survives:

    grad = ( x̂₀_fake − x̂₀_real ) / mean|x − x̂₀_real|                       (M2.2)

That is *verbatim* what `compute_distribution_matching_loss` computes (§G3
below) — the single most important code block in the repo.

### M3 — Pedigree: SDS → VSD → DMD (the same trick we implemented at KAIST)

Our A-SDS implementation (`writeups/kaist_reports.md`) optimized ONE image
with the gradient `w(t)(ε̂_cfg(x_t) − ε)`, injected as
`loss = ½‖x − (x − grad).detach()‖²` so that `dL/dx = grad` with zero
backprop through the UNet. Rewrite SDS's baseline `ε` with the shortcut
`ε = (x_t − α_t x)/σ_t`: it is exactly the ε-prediction of a *degenerate
fake score* whose x̂₀ is the current image itself — SDS is Eq. (M2.1) with
`p_fake` = a Dirac at the current particle. That crude baseline is why SDS
outputs are blurry/oversaturated. Variational Score Distillation
(ProlificDreamer) replaced the Dirac with a *trained* LoRA score of the
particle distribution; DMD is VSD with the particle set replaced by a
generator network and the LoRA by a full UNet copy. The DMD2 code even
reuses our exact detached-target trick, line for line:

    loss = 0.5 * F.mse_loss(latents, (latents - grad).detach())

Same "the target is detached, so d(loss)/d(latents) = grad" mechanics; only
the grad changed from `(ε̂_real − ε)` to `(x̂₀_fake − x̂₀_real)/normalizer`.

### M4 — What DMD2 changed vs DMD1

1. **Dropped the regression/LPIPS pretraining loss.** DMD1 needed millions
   of precomputed (noise, teacher-ODE-sample) pairs and an LPIPS regression
   term to stabilize training. DMD2 removes it entirely — in this repo the
   flagship launch scripts are literally named `..._noode...` and the DM
   loss + GAN carry everything (an optional ODE warm-start,
   `train_sd_ode.py`, exists but is not used by the main SDXL recipes).
2. **Added a GAN term on real data.** The fake score is only an *estimate*;
   where it is wrong, Eq. (M2.2) points the wrong way, and pure DMD
   inherits any teacher blurriness. DMD2 grafts a tiny conv classifier head
   onto `fake_unet`'s mid-block features and trains it to separate real
   dataset latents from generator latents (non-saturating logistic loss);
   the generator gets the corresponding `softplus(−D(G(z)))` term. Because
   the head rides on the fake UNet, the critic and the fake score share a
   body and one optimizer ("guidance model" = fake UNet + head).
3. **Two-time-scale updates.** The fake score must *track* the shifting
   student distribution or Eq. (M2.1) uses a stale `s_fake`. DMD2 updates
   the guidance model (fake score + GAN head) **5×** more often than the
   generator (`--dfake_gen_update_ratio 5` in every launch script; DMD2
   §"TTUR").
4. **Multi-step generator with simulated backward process.** For SDXL, one
   step is not enough; DMD2 trains a 4-step student on timesteps
   {999, 749, 499, 249}. Problem: at training time, what should the
   *input* at t=499 be? Noised *real* data would be a train/inference
   mismatch — at inference the input at t=499 is the student's own partial
   output, re-noised. Solution: **backward simulation** — run the student's
   own first k sampling steps (no grad) to manufacture the input, then take
   one supervised step. §U2 below.

### M5 — Map of the code

| File | Role |
|---|---|
| `main/sd_guidance.py` | `SDGuidance`: real+fake UNets, DM gradient (M2.2), GAN head, both loss sides |
| `main/sd_unified_model.py` | `SDUniModel`: the student generator, backward simulation, glue between turns |
| `main/train_sd.py` | `Trainer`: alternating generator/guidance updates, FSDP/accelerate/wandb boilerplate |
| `main/utils.py` | `get_x0_from_noise` and small helpers |
| `main/sd_unet_forward.py` | patched diffusers UNet forward that can early-return bottleneck features |

Two parameter groups, two optimizers:
**generator** = `feedforward_model` (a UNet used as a one-call denoiser);
**guidance** = `fake_unet` + `cls_pred_branch`. `real_unet` belongs to
neither — frozen, bf16, inference only.

Shape conventions used throughout (B = batch):

| Tensor | SD v1.5 | SDXL |
|---|---|---|
| latent / noise / grad | (B, 4, 64, 64) | (B, 4, 128, 128) |
| text_embedding | (B, 77, 768) | (B, 77, 2048) |
| pooled_text_embedding | — | (B, 1280) |
| timesteps | (B,) long | (B,) long |
| UNet mid-block feature | (B, 1280, 8, 8) | (B, 1280, 32, 32) |
| GAN logits | (B, 1) | (B, 1) |

---

## §0 — `main/utils.py`: the four helpers that carry method

```python
def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample
```
The single most-called function in the repo: invert the forward shortcut
`x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε` for `x₀` (our notes `10 §1.2`, one line of
algebra). `alphas_cumprod` is the (1000,) schedule buffer;
`alphas_cumprod[timestep]` fancy-indexes a (B,) vector of ᾱ values, reshaped
to (B,1,1,1) to broadcast over (B,4,H,W). Every ε-prediction in this
codebase is immediately converted to x̂₀ with this — which, per §M2, is the
score in yet another disguise. Callers pass `.double()` inputs: near t≈999,
`√ᾱ_t` is tiny and the division amplifies rounding error, so the conversion
runs in float64 even when the networks run bf16.

```python
def cycle(dl):
    while True:
        for data in dl: yield data
```
Turns a finite DataLoader into an infinite iterator — the training loop is
step-based, not epoch-based, so it just calls `next(...)` forever.

```python
class NoOpContext:  # __enter__/__exit__ that do nothing
class DummyNetwork(nn.Module):  # a single Linear(32, 1), never used
```
`NoOpContext` lets `with self.network_context_manager:` be a bf16 autocast
or nothing, chosen once at init. `DummyNetwork` is pure FSDP appeasement —
its comment in `sd_guidance.py` explains: diffusers models are lazily
initialized, and FSDP wants at least one module with real dense parameters
at wrap time.

The rest of `utils.py` — image-grid builders (`prepare_images_for_saving`),
matplotlib histogram/valued-array renderers, LMDB row readers, `SDTextDataset`
(a list of prompts → tokenized ids of shape (1, 77) per item), an unused
`EMA` class — is logging/data plumbing; skim once, then trust.

---

## §G — `main/sd_guidance.py`: the heart

### G0 — `predict_noise`: one UNet call, with optional CFG

```python
def predict_noise(unet, noisy_latents, text_embeddings, uncond_embedding, timesteps,
    guidance_scale=1.0, unet_added_conditions=None, uncond_unet_added_conditions=None):
    CFG_GUIDANCE = guidance_scale > 1
```
A thin wrapper so callers can say "give me ε̂, guided or not." The flag
turns strictly-greater-than-1 scales into the doubled-batch CFG path.

```python
    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2)
        embeddings = torch.cat([uncond_embedding, text_embeddings])
        timesteps = torch.cat([timesteps] * 2)
```
The standard CFG batching trick (notes `11 §4`): run conditional and
unconditional in ONE forward by stacking along batch — (B,4,H,W) → (2B,4,H,W).
Order is **uncond first, cond second** (the in-code comment says exactly
this); get it backwards and guidance points away from the prompt.

```python
        if unet_added_conditions is not None:
            ...condition_input[key] = torch.cat(
                [uncond_unet_added_conditions[key], unet_added_conditions[key]])
```
SDXL's extra conditioning (pooled text embedding + `time_ids`) must be
doubled in the same uncond-then-cond order, key by key.

```python
        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```
Split the (2B,...) output back into two (B,...) halves and apply the CFG
formula `ε̂_cfg = ε̂_∅ + s·(ε̂_c − ε̂_∅)` — Eq. (CFG) of notes `11 §4`,
verbatim. The `else` branch is a plain single forward with the conditional
embedding. Returns ε̂ of shape (B,4,H,W).

### G1 — `SDGuidance.__init__`: two UNets and a classifier head

```python
        self.real_unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()
        self.real_unet.requires_grad_(False)
```
The **teacher**. Loaded from the same HF checkpoint as everything else,
frozen forever. This is `s_real`'s body.

```python
        self.fake_unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()
        self.fake_unet.requires_grad_(True)
```
The **fake score network** — a full second copy of the teacher, *trainable*.
Initializing at the teacher is the smart warm start: at step 0 the student's
outputs still resemble teacher outputs, so the teacher's own score is a
decent first estimate of `s_fake`.

```python
        if args.use_fp16:
            self.real_unet = self.real_unet.to(torch.bfloat16)
```
Only the teacher is cast to bf16 storage — "we don't backpropagate through
it", so half-precision weights are free memory savings. The fake UNet keeps
fp32 master weights and runs under autocast instead.

```python
        if self.gan_alone:
            del self.real_unet
```
Ablation switch (`--gan_alone`): pure-GAN training deletes the teacher
entirely. Useful mental probe: everything DMD-specific dies with this flag.

```python
        self.scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
        alphas_cumprod = self.scheduler.alphas_cumprod
        self.register_buffer("alphas_cumprod", alphas_cumprod)
```
Pull the teacher's noise schedule and pin ᾱ (shape (1000,)) as a buffer —
travels with `.to(device)`, saved in checkpoints, no grad (same pattern and
reason as our DDPM schedule buffers, notes `02 §3.2`). The scheduler object
itself is reused for its `add_noise` (forward shortcut) method.

```python
        self.min_step = int(args.min_step_percent * self.scheduler.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.scheduler.num_train_timesteps)
```
The DM loss samples t uniformly in [20, 980] (2%–98% defaults) — identical
to the SDS range in our KAIST work and for the same reason: t≈0 gradients
are noise-dominated, t≈999 carries almost no signal about x₀.

```python
        assert self.fake_guidance_scale == 1, "no guidance for fake"
```
Honest and important: CFG is applied to the *real* branch only. `p_fake` is
what the student actually emits for this prompt; sharpening it with CFG
would change what distribution the KL measures.

```python
        if self.cls_on_clean_image:
            self.fake_unet.forward = types.MethodType(classify_forward, self.fake_unet)
```
Monkey-patch the fake UNet's forward with the copy in `sd_unet_forward.py`.
That copy is a verbatim diffusers forward plus ONE addition — after the
mid-block, before any up-blocks:

```python
    # in sd_unet_forward.classify_forward
    if classify_mode:
        output_list = list(down_block_res_samples) + [sample]
        return output_list
```
i.e. when asked, return the encoder-half activations and the bottleneck
feature map instead of finishing the U. The GAN discriminator is literally
"the fake score network's encoder + a small head" — feature sharing for
free, one design choice that makes DMD2's GAN nearly parameter-free.

```python
            if args.sdxl:
                self.cls_pred_branch = nn.Sequential(
                    nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 32x32 -> 16x16
                    ... # GroupNorm + SiLU between each
                    nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0),   # 1x1 -> 1x1
                )
```
The **GAN head**: a stack of stride-2/stride-4 convs (each followed by
GroupNorm+SiLU) that collapses the (B,1280,32,32) SDXL bottleneck (or
(B,1280,8,8) for SDv1.5, with a shorter stack) down to a (B,1,1,1) logit.
No sigmoid — raw logits feed softplus losses later. The printed warning
above this block is real operational advice: these convs are *randomly
initialized*, and under multi-node FSDP hybrid sharding each node would
draw different randoms — the fix (save checkpoint 0 and reload) is
implemented in `train_sd.py` §T1.

```python
        self.diffusion_gan = args.diffusion_gan
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep
        self.network_context_manager = torch.autocast(...bfloat16) if self.use_fp16 else NoOpContext()
```
Knobs for §G2's noisy discriminator, and the reusable autocast context.

### G2 — `compute_cls_logits`: the discriminator forward

```python
        if self.diffusion_gan:
            timesteps = torch.randint(0, self.diffusion_gan_max_timestep, [image.shape[0]], ...)
            image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
        else:
            timesteps = torch.zeros([image.shape[0]], ...)
```
"Diffusion-GAN" trick: optionally judge *noised* latents at a random level
t ∈ [0, 1000) (the launch scripts use the full range) instead of clean ones
at t=0. Noise smooths the real/fake supports into overlap — the classic cure
for a discriminator that wins too easily and starves the generator of
gradient. Note the input `image` is a VAE **latent**, not pixels (the
comment "we are operating on the VAE latent space" says so).

```python
        with self.network_context_manager:
            rep = self.fake_unet.forward(image, timesteps, text_embedding,
                added_cond_kwargs=unet_added_conditions, classify_mode=True)
        rep = rep[-1].float()
        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
```
Run the patched forward in `classify_mode` — it returns the list of encoder
activations; `rep[-1]` picks only the **bottleneck** (mid-block output,
(B,1280,8,8) or (B,1280,32,32)). The head maps it to (B,1,1,1); squeezing
dims 2 and 3 gives (B,1) logits. The discriminator is prompt-*conditional*
(text embedding goes in), so it judges "realistic *for this caption*."

### G3 — `compute_distribution_matching_loss`: THE code block ★

This function *is* Eq. (M2.2). Input `latents` = the generator's x̂₀ output,
(B,4,H,W), **attached to the generator's autograd graph** — the only tensor
here that ever will be.

```python
        original_latents = latents
        batch_size = latents.shape[0]
        with torch.no_grad():
```
Keep a graph-connected handle (`original_latents`), then enter `no_grad` for
*everything else*: both score networks run gradient-free, exactly like the
frozen-UNet forward in our SDS implementation. The generator's gradient will
be injected analytically at the end.

```python
            timesteps = torch.randint(self.min_step, min(self.max_step+1, self.num_train_timesteps),
                                      [batch_size], device=latents.device, dtype=torch.long)
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
```
Diffuse the student's sample: draw t ∈ [20, 981), ε ~ N(0,I), form
`x_t = √ᾱ_t·G(z) + √(1−ᾱ_t)·ε` (the forward shortcut, notes `05 B`). The KL
of §M2 is matched *at this noisy point* — one random t per sample per step,
a stochastic estimate of the E_t.

```python
            # run at full precision as autocast and no_grad doesn't work well together
            pred_fake_noise = predict_noise(self.fake_unet, noisy_latents, text_embedding,
                uncond_embedding, timesteps, guidance_scale=self.fake_guidance_scale, ...)
            pred_fake_image = get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(), self.alphas_cumprod.double(), timesteps)
```
**s_fake**, in two moves: the fake UNet's ε̂ (no CFG — scale is asserted 1),
then the ε→x̂₀ conversion in float64. Per notes `10 §3`, this x̂₀ *is* the
fake score up to the (α_t/σ_t²) factor that §M2's weighting cancels. Quirk
flagged honestly: despite `use_fp16`, this call runs full precision — the
in-code comment blames the autocast/no_grad interaction.

```python
                pred_real_noise = predict_noise(self.real_unet, noisy_latents.to(torch.bfloat16), ...,
                    timesteps, guidance_scale=self.real_guidance_scale, ...)
            pred_real_image = get_x0_from_noise(
                noisy_latents.double(), pred_real_noise.double(), self.alphas_cumprod.double(), timesteps)
```
**s_real**: same recipe with three differences — the frozen teacher, inputs
cast to bf16 to match its storage (SDXL's added-condition dicts get the same
cast in the surrounding boilerplate), and **CFG at scale 6–8** inside the
call (§G0). The student is distilled against the *guided* teacher: this is
where "one-step SDXL at guidance 8" quality comes from, and also why DMD
students inherit CFG's saturation tendencies (notes `11 §4`'s
diversity-for-fidelity trade, baked into the target distribution).

```python
            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)
            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = torch.nan_to_num(grad)
```
The DMD gradient, Eq. (M2.2). Expand the numerator:
`p_real − p_fake = pred_fake_image − pred_real_image = x̂₀_fake − x̂₀_real`
∝ `s_fake − s_real` — the reverse-KL direction of Eq. (M2.1). The
denominator is the per-sample mean |x − x̂₀_real| over (C,H,W) — DMD's
`w_t = σ²/α · CS/‖x − x̂₀_real‖₁` weight with the σ²/α part already cancelled
by working in x̂₀-space. It's a *relative* error normalizer: samples the
teacher wants to change a lot get proportionally tempered gradients, keeping
scale stable across t. `keepdim=True` keeps it (B,1,1,1) for broadcasting;
`nan_to_num` guards the division (and the float64 corner cases) — a silent
NaN here would poison the generator within one step.

```python
        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(),
                                reduction="mean")
```
The **detached-target trick**, identical to our KAIST SDS line
(`kaist_reports.md`): the target `(x − grad).detach()` is a constant, so
`d(loss)/d(x) = 0.5·2·(x − (x−grad))/N = grad/N` — backprop through this MSE
hands autograd exactly the hand-computed `grad` (scaled by 1/numel from
`reduction="mean"`; a constant absorbed by the learning rate), which then
flows into `∂G/∂θ` through `original_latents`. No backprop ever touches
either score UNet. This one line is why the whole method costs only *one*
generator backward per step.

The function returns `{"loss_dm": loss}` plus a log dict (detached copies of
`x_t`, both x̂₀ predictions, `grad`, and its norm — these become the wandb
"what does the teacher want changed" visualizations in §T2).

### G4 — `compute_loss_fake`: training the fake score to chase the student

```python
        latents = latents.detach()
```
First line, and the most important one: the generator must NOT receive
gradient from its critic's training loss. From here on, `latents` is data.

```python
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.num_train_timesteps, [batch_size], ...)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        with self.network_context_manager:
            fake_noise_pred = predict_noise(self.fake_unet, noisy_latents, ...,
                timesteps, guidance_scale=1, ...)
        loss_fake = torch.mean((fake_noise_pred.float() - noise.float())**2)
```
This is the **vanilla DDPM ε-prediction loss** — our HW1 training loop,
unchanged — except the "dataset" is *the generator's current outputs*. Full
t-range [0,1000) (unlike the DM loss's clipped range: the fake score must be
accurate everywhere it might be queried). Under autocast this time. Minimize
`‖ε̂ − ε‖²` on samples of `p_fake` and, by the standard argument, `fake_unet`
converges to the score of `p_fake` at every noise level — the dynamically
trained `s_fake` that §M2 requires. The in-between `fake_x0_pred` computed
via `get_x0_from_noise` is logging-only. The gradient-checkpointing
enable/disable wrapping the function trades compute for memory on this, the
most frequently executed backward (5× per generator update).

### G5 — The two GAN losses (generator side / critic side)

```python
    def compute_generator_clean_cls_loss(self, fake_image, ...):
        pred_realism_on_fake_with_grad = self.compute_cls_logits(fake_image, ...)
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
```
Generator's GAN term: `softplus(−D(G(z))) = log(1 + e^{−D})`, the
**non-saturating generator loss** ("make the critic call my images real").
Note `fake_image` is graph-connected — gradients flow through the fake
UNet's *encoder activations* back to the generator (the parameters are
`requires_grad_(False)`-frozen during this turn, §U4, so only the generator
learns from it).

```python
    def compute_guidance_clean_cls_loss(self, real_image, fake_image, ...):
        pred_realism_on_real = self.compute_cls_logits(real_image.detach(), ...)
        pred_realism_on_fake = self.compute_cls_logits(fake_image.detach(), ...)
        classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
```
Critic's term: push `D(fake)` down (`softplus(+·)`) and `D(real)` up
(`softplus(−·)`) — the standard logistic discriminator loss (StyleGAN
convention). Both inputs `.detach()`ed: the critic learns *about* images,
never *through* them. `real_image` is a **real dataset latent** (LAION
images pre-encoded by the VAE into the LMDB dataset, §T1) — this is DMD2's
"GAN on real data": the only place actual data enters training, letting the
student exceed the teacher on realism. The sigmoid'd logits are logged as
realism histograms.

### G6 — `generator_forward` / `guidance_forward` / `forward`: the dispatcher

```python
        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, ...)
        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, ...)
```
Generator turn = DM loss (§G3) + generator GAN loss (§G5), computed on the
same generated batch. (The commented-out `torch.autograd.grad` block below
it is a leftover debugging probe for comparing the two losses' gradient
magnitudes — how the authors tuned `gen_cls_loss_weight`.)

```python
        fake_dict, fake_log_dict = self.compute_loss_fake(image, ...)
        if self.cls_on_clean_image:
            clean_cls_loss_dict, ... = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['images'], fake_image=image, ...)
```
Guidance turn = denoising loss on fakes (§G4) + critic loss on real vs fake
(§G5). Finally `forward(generator_turn=..., guidance_turn=...)` routes a
data dict to one side or the other — an awkward but FSDP-friendly shape: the
whole guidance model is ONE wrapped module with one optimizer, and which
loss it produces is a flag.

---

## §U — `main/sd_unified_model.py`: the student and the glue

### U1 — `SDUniModel.__init__` (grouped — construction only)

One class owns everything: `self.guidance_model = SDGuidance(args, ...)`
(all of §G), plus:

```python
        self.denoising_step_list = torch.tensor(
            list(range(self.denoising_timestep-1, 0, -(self.denoising_timestep//self.num_denoising_step))), ...)
        self.timestep_interval = self.denoising_timestep//self.num_denoising_step
```
The student's sampling grid, descending. With `denoising_timestep=1000,
num_denoising_step=4`: **[999, 749, 499, 249]**, interval 250 — the 4-step
SDXL student's entire inference schedule, fixed at construction. With 1
step: [999].

- `self.feedforward_model` — the **generator**: yet another copy of the
  teacher UNet (`initialie_generator` [sic] must be set), fully trainable,
  or LoRA-wrapped (`--generator_lora`, rank-64 adapters on
  attention/conv/time-emb modules) for the cheap SDXL recipe. Same
  architecture as the teacher but *used differently*: one call = one full
  denoise.
- Text encoders (CLIP for SD1.5; dual-encoder `SDXLTextEncoder` for SDXL)
  and the VAE (or `AutoencoderTiny` — decode is logging-only during
  training, so a tiny VAE is fine), all frozen. `build_condition_input`
  makes SDXL's constant (1,6) `time_ids = [h, w, 0, 0, h, w]`.
- `self.alphas_cumprod`, autocast context — same as §G1.

### U2 — `sample_backward`: backward simulation ★ (`@torch.no_grad()`)

The fix for §M4 point 4: manufacture training inputs at intermediate
timesteps by running *the student itself*.

```python
        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, ...)
        selected_step = broadcast(selected_step, from_process=0)
```
Pick which of the 4 grid points this batch will train on — ONE shared choice
per step. `broadcast` forces every GPU to agree (rank 0's draw wins);
FSDP-sharded UNet forwards are collective operations, so ranks running
different numbers of loop iterations below would deadlock.

```python
        generated_image = noisy_image
        for constant in self.denoising_step_list[:selected_step]:
            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long) * constant
            generated_noise = self.feedforward_model(noisy_image, current_timesteps,
                real_text_embedding, added_cond_kwargs=unet_added_conditions).sample
            generated_image = get_x0_from_noise(noisy_image, generated_noise.double(),
                self.alphas_cumprod.double(), current_timesteps).float()
```
Run the student's own sampler for the first `selected_step` grid points:
at each, one UNet call gives ε̂, converted to a clean estimate x̂₀. The
default `generated_image = noisy_image` before the loop covers
`selected_step == 0` (loop never runs; the "image" is pure noise, but see
below — it gets masked away). Note `noisy_image` starts as the *function
argument* — pure fresh noise, and the prompt is the **real-data caption**
(this function is called with the denoising batch's text).

```python
            next_timestep = current_timesteps - self.timestep_interval
            noisy_image = self.noise_scheduler.add_noise(
                generated_image, torch.randn_like(generated_image), next_timestep).to(noisy_image.dtype)
        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, ...)
        return generated_image, return_timesteps
```
Re-noise the estimate down to the next grid level with FRESH noise — this is
exactly the consistency-style stochastic sampler: predict x̂₀, jump to a
lower level via the forward shortcut, repeat. (Compare notes `10 §1.2`'s
η dial: this is the η=1-flavored "add all-fresh noise" end, not DDIM's
ε̂-reuse.) Return the last clean estimate plus the timestep the *next*
(gradient-carrying) step should train at. Two honest quirks: the final
loop iteration's `noisy_image` re-noise is computed and discarded (the
caller re-noises again itself), and the whole simulation costs up to 3 extra
no-grad UNet calls per generator turn.

### U3 — `prepare_denoising_data` / `prepare_pure_generation_data` (`@torch.no_grad()`)

```python
        indices = torch.randint(0, self.num_denoising_step, (noise.shape[0],), ...)
        timesteps = self.denoising_step_list.to(noise.device)[indices]
```
Per-sample random grid timesteps — the *non*-backward-simulation fallback.

```python
        if self.backward_simulation:
            # we overwrite the denoising timesteps
            # note: we also use uncorrelated noise
            clean_images, timesteps = self.sample_backward(torch.randn_like(noise), text_embedding, pooled_text_embedding)
        else:
            clean_images = denoising_dict['images'].to(noise.device)
```
The fork of §M4.4. With `--backward_simulation` (the flagship recipe):
inputs come from the student's own partial trajectories (§U2) — note the
just-drawn per-sample `timesteps` are simply *discarded* and replaced by the
shared one, and the simulation starts from `randn_like(noise)`, deliberately
uncorrelated with the `noise` used next. Without it: `clean_images` are
**real data latents**, so the student trains as a text-conditioned denoiser
of noised real images (the DMD2 paper's intermediate variant, with the
train/inference mismatch).

```python
        noisy_image = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        pure_noise_mask = (timesteps == (self.num_train_timesteps-1))
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]
```
Noise the clean image to the training level — EXCEPT at the topmost grid
point t=999, where the input is forced to be *pure* noise rather than
"mostly noise plus a whisper of image": at inference step 1 the student sees
exactly N(0,I), so training must too. This mask is also what rescues the
`selected_step == 0` garbage default from §U2 (its `generated_image` is
never actually used at t=999... it *is* used inside `add_noise`, but the
result is overwritten by the mask).

`prepare_pure_generation_data` is the 1-step path: encode the tokenized
prompt (the "text_embedding" argument is really token ids — the code
comments admit it), prepare the real batch's embeddings for the GAN, and set
`noisy_image = noise` — pure noise in, one call, image out.

### U4 — `forward`, generator turn: one call, then hand off

```python
            if self.denoising:
                timesteps, text_embedding, ..., noisy_image = self.prepare_denoising_data(...)
            else:
                timesteps = torch.ones(noise.shape[0], ...) * self.conditioning_timestep
                text_embedding, ..., noisy_image = self.prepare_pure_generation_data(...)
```
Multi-step (SDXL 4-step) vs one-step (`conditioning_timestep=999`) input
preparation. Then SDXL's added conditions are assembled; the SDXL
"unconditional" is `torch.zeros_like` on both embeddings — zeroed vectors,
not an encoded empty string (an SDXL-community convention; flag it, don't
fight it).

```python
            if compute_generator_gradient:
                with self.network_context_manager:
                    generated_noise = self.feedforward_model(noisy_image, timesteps.long(),
                        text_embedding, added_cond_kwargs=unet_added_conditions).sample
            else:
                ...
                with torch.no_grad():
                    generated_noise = self.feedforward_model(...).sample
```
**The student's entire generative act is this single UNet call.** Grad mode
depends on whose turn it is: on guidance-only steps (4 of every 5, §T2) the
same images are made under `no_grad` purely as critic food (with gradient
checkpointing toggled off — pointless without backward). For the 4-step
student, note only THIS one step carries gradient; the backward-simulation
steps that produced `noisy_image` were no-grad.

```python
            generated_image = get_x0_from_noise(noisy_image.double(),
                generated_noise.double(), self.alphas_cumprod.double(), timesteps).float()
```
ε̂ → x̂₀ once more ("this assumes all teacher models use epsilon prediction
(which is true for SDv1.5 and SDXL)" — the in-code comment). At t=999 with
ᾱ≈0 this division is why everything is `.double()`. This x̂₀ IS the
generated image (latent).

```python
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(generator_turn=True, ...,
                    generator_data_dict=generator_data_dict)
                self.guidance_model.requires_grad_(True)
```
Hand the graph-connected image to §G6's generator turn, with the guidance
model's parameters frozen for the duration — so the generator's GAN loss
can flow *through* fake-UNet activations without depositing parameter
grads ("avoid any side effects of gradient accumulation").

```python
            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(), "text_embedding": text_embedding.detach(), ... }
```
Stash a fully **detached** copy of everything the guidance turn will need —
this dict is the only bridge between the two halves of §T2's training step
(the same images train the critic that just criticized them). The `visual`
block above it VAE-decodes a few latents for wandb; skim and trust. The
`guidance_turn` branch of `forward` just forwards this dict to §G6.

---

## §T — `main/train_sd.py`: the alternating loop

### T1 — `Trainer.__init__` (grouped — infrastructure)

Five groups, briefly:

1. **Accelerator**: `mixed_precision="no"` — precision is managed *manually*
   (bf16 storage for the teacher, autocast contexts elsewhere), not by
   accelerate. Seed is `args.seed + process_index` so each GPU draws
   different noise.
2. **Datasets, three-plus-one cycled loaders**: `SDTextDataset` (a text file
   of prompts → tokenized ids) feeds `dataloader` (generator turns) and a
   *separate* `guidance_dataloader` — two independent prompt streams
   "as the generator and guidance model are trained at different paces"
   (the TTUR of §M4.3). `SDImageDatasetLMDB` (pre-encoded VAE latents +
   captions of ~500k LAION images) feeds `real_dataloader` (GAN real
   branch) and, if `--denoising`, a fourth `denoising_dataloader` (captions
   and fallback clean images for §U3). For SDv1.5, the empty-prompt
   embedding is precomputed once as `self.uncond_embedding`.
3. **FSDP randomness fix**: exactly the problem §G1's warning predicted —
   before FSDP wrapping, rank 0 saves the freshly built model and every
   rank reloads it, so the randomly initialized GAN head agrees across
   nodes. Then only the two submodules (`feedforward_model`,
   `guidance_model`) are wrapped — never the umbrella `SDUniModel`.
4. **Two optimizers, two schedulers**: AdamW over each side's
   `requires_grad` params, `lr = generator_lr / guidance_lr` (5e-7 for
   full-model SDXL, 5e-5 for LoRA), betas/weight-decay left at PyTorch
   defaults (yes, wd=0.01 on a UNet — the comment owns it), plus a
   500-step warmup scheduler the comment itself calls "not very useful".
5. **Checkpoint resume paths** and `self.dfake_gen_update_ratio` — the
   number that implements TTUR (**5** in every real recipe).

`save()`/`load()`/`fsdp_state_dict()` are checkpoint plumbing: FSDP full
state dicts gathered to rank-0 CPU (`FullStateDictConfig(offload_to_cpu=True,
rank0_only=True)`), saved as two `.bin` files, older checkpoints pruned; a
frank comment admits optimizer state isn't saved under FSDP (OOM).

### T2 — `train_one_step`: the alternation ★

```python
        noise = torch.randn(self.batch_size, self.latent_channel,
                            self.latent_resolution, self.latent_resolution, device=...)
```
Fresh latent noise, (B,4,64,64)/(B,4,128,128). One draw serves the whole
step.

```python
        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
```
**The two-time-scale switch.** With ratio 5: steps 0,5,10,… are full
(generator + guidance) steps; steps 1–4,6–9,… train the guidance model
only. The critic gets 5 looks at the student per student move — keeping
`s_fake` and the GAN head fresh enough that Eq. (M2.2) points the right way.

```python
        if COMPUTE_GENERATOR_GRADIENT:
            text_embedding = next(self.dataloader)
        else:
            text_embedding = next(self.guidance_dataloader)
```
Different prompt streams per turn type (misleading variable name: for
SD1.5 this holds tokenized ids, unsqueezed two lines later). Then the
denoising batch (captions + fallback images) and the GAN's real batch are
drawn from their own cycles.

```python
        generator_loss_dict, generator_log_dict = self.model(
            noise, text_embedding, uncond_embedding, ...,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True, guidance_turn=False)
```
§U4's generator turn: generate images; if it's a generator step, also
compute `loss_dm` and `gen_cls_loss` against the (frozen-for-now) guidance
model. Either way, `generator_log_dict['guidance_data_dict']` now holds the
detached batch for the second half.

```python
        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight
            if self.cls_on_clean_image and self.gen_cls_loss:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight
            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()
            self.optimizer_guidance.zero_grad()
```
**Generator update.** The full DMD2 generator objective:
`L_G = 1.0·loss_dm + 5e-3·gen_cls_loss` (weights from the SDXL scripts —
the GAN term is a small seasoning on the DM main course). Backward drives
the detached-target trick of §G3, one backward through ONE UNet. Clip at
norm 10, step, then zero **both** optimizers — the comment explains the
guidance zeroing: the GAN loss touched fake-UNet activations, so this is
belt-and-braces against any stray accumulation (with the `requires_grad`
sandwich of §U4 there shouldn't be parameter grads, but FSDP wrapping makes
"shouldn't" worth insuring).

```python
        guidance_loss_dict, guidance_log_dict = self.model(
            noise, text_embedding, uncond_embedding, ...,
            generator_turn=False, guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict'])
        guidance_loss += guidance_loss_dict["loss_fake_mean"]
        if self.cls_on_clean_image:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight
        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad()
```
**Guidance update — every step, no condition.** The same generated images
(detached, via the bridge dict) now train the critic side:
`L_D = loss_fake_mean + 1e-2·guidance_cls_loss` — denoising-score-matching
on fakes (§G4) keeps `s_fake` current; the logistic loss on real-vs-fake
(§G5) trains the GAN head. Mirror-image clip/step/zero-both. Note the
elegant economy: **one generator forward feeds both turns** — criticized
with gradient in turn 1, learned-from without gradient in turn 2.

Everything after (~200 lines) is logging, in three groups: scalar means/stds
`accelerator.gather`ed across GPUs and sent to wandb every step
(`--log_loss`); every `wandb_iters` steps, image grids of generated
latents, the teacher's x̂₀_real vs the fake score's x̂₀_fake side by side and
their normalized *difference image* (a literal visualization of the DMD
gradient — the best debugging picture this method has), plus realism-
probability histograms from the GAN head; and a `wait_for_everyone` barrier.

### T3 — `train()` and `parse_args` (grouped)

`train()` is a bare for-loop: `train_one_step()`, checkpoint every
`log_iters`, log iteration wall time. `parse_args` holds the defaults quoted
throughout; three asserts encode real constraints — gradient accumulation
unsupported, FSDP and gradient checkpointing mutually exclusive, and
`wandb_iters % dfake_gen_update_ratio == 0` (so visual steps are always
full generator steps and the generator-side logs exist).

---

## Quirk ledger (honesty section)

- **Everything is latents.** "image" variables are VAE latents throughout;
  pixels only exist inside logging decodes.
- **Precision patchwork**: teacher stored bf16; fake UNet autocast-bf16 for
  its own training but full-precision inside the DM loss ("autocast and
  no_grad doesn't work well together"); all ε→x̂₀ conversions float64.
- **Misnamed variables**: `text_embedding` is often tokenized ids;
  `initialie_generator` [sic]; `real_guidance_scale` is a CFG scale, not a
  GAN weight.
- **Discarded computation**: `sample_backward`'s last re-noise; the randint
  timesteps overwritten under backward simulation; `fake_x0_pred` in §G4
  (logging only); the always-instantiated-but-unused `DummyNetwork` and
  `EMA` class (no EMA of the student is kept in this trainer).
- **Double zero_grad of both optimizers after each turn** — deliberate
  cross-contamination insurance, not confusion.
- **SDXL uncond = zeros**, not an encoded empty prompt.
- **`loss_dm`'s `reduction="mean"`** silently divides the injected gradient
  by B·C·H·W; the learning rate absorbs it, but if you port this code and
  change resolution, your effective DM step size changes with it.
- **The GAN head is randomly initialized under FSDP** → the save-and-reload
  dance at startup is load-bearing, not paranoia.

---

## What to carry forward (DMD2 → everything else)

1. **Distribution matching = a difference of two scores** at a shared noisy
   point: `grad ∝ x̂₀_fake − x̂₀_real` (Eq. M2.2) — reverse-KL descent with
   the frozen teacher as `s_real` and a *continuously retrained* copy as
   `s_fake`. SDS is the special case where `x̂₀_fake` is the current image
   itself; VSD/DMD upgrade that baseline to a learned score.
2. **The detached-target trick travels**: `½‖x − (x − grad).detach()‖²`
   turns any hand-computed gradient into a loss, with zero backprop through
   the networks that computed it. Third sighting after our SDS and PDS.
3. **A critic must outrun its generator**: 5 guidance updates per generator
   update, two prompt streams, and the fake score trained on the FULL
   t-range even though the DM loss only queries [2%, 98%].
4. **Recycle the score network as a discriminator**: `classify_mode=True`
   early-returns the bottleneck; a 3–5 conv head makes it a
   prompt-conditional GAN critic — near-free parameters, and the
   diffusion-GAN noise trick keeps it beatable.
5. **Backward simulation**: when a few-step student trains at intermediate
   t, manufacture its inputs by running the student itself (no-grad),
   re-noising x̂₀ between grid points; force pure noise at the top step.
   Train/inference distribution match is a *data* problem, solved with
   compute.
6. **CFG lives inside the target**: the student distills the guided teacher
   (scale 6–8) and needs no CFG at inference — one UNet call, no doubled
   batch. What our notes `11` called a sampling-time dial becomes a
   training-time definition of `p_real`.
