# LCM Distillation, Line by Line — `train_lcm_distill_sd_wds.py`

*A beginner's reading companion to
`distillation/latent-consistency-model/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py`
(the LCM team's original release; the diffusers repo maintains a cleaned-up
copy). Every load-bearing line: what it does, why it's there, and what breaks
without it. Concepts link to our notes: DDIM step and x̂₀-identity = `10 §1.2`,
CFG = `11 §4–5`, ODE/velocity-field picture = `06 §A2` and `10 Part 3`.
Paper: Luo et al. 2023, "Latent Consistency Models" (arXiv 2310.04378).*

**The method in one sentence:** train a student U-Net so that its *one-jump
guess of the clean image* is the same no matter where on a single (guidance-
augmented) denoising trajectory you ask it — using the frozen Stable Diffusion
teacher's DDIM steps to manufacture pairs of points on that trajectory — so
that at inference the student can leap from pure noise to an image in 1–4
steps, with classifier-free guidance already baked into its weights.

---

## Methodology (read this before the code)

### 1. The consistency function: f(x_t, t) → x₀

A diffusion sampler (notes `10 Part 3`) is a slow walk down the
**probability-flow ODE (PF-ODE)**: a deterministic trajectory
{x_T, …, x_t, …, x₀} connecting noise to one specific image. Every point on
one trajectory "belongs to" the same endpoint x₀.

A **consistency model** (Song et al. 2023, which LCM builds on) learns that
endpoint map directly: a function f(x_t, t) that outputs the trajectory's
*origin* x₀ from any point on it. If you have f, sampling is one function
call: draw x_T ~ N(0, I), output f(x_T, T). Done. (Few-step variants
alternate: predict x₀, re-noise to a smaller t, predict again — that is what
`LCMScheduler` does at 4 steps.)

### 2. Self-consistency, and why distillation needs a teacher

How do you train f without ever running a full trajectory? Exploit its
defining property, **self-consistency**: any two points on the *same*
trajectory must map to the *same* origin,

    f(x_t, t) = f(x_{t'}, t')   for all t, t' on one trajectory.

So the loss is: take two adjacent points on one trajectory and penalize the
distance between the student's two predictions. The catch: you need two points
that genuinely lie on one trajectory. Data alone gives you x_t = ᾱ√·x₀ + σ·ε
at any t (the forward marginal), but two independently-noised samples are NOT
on a shared ODE path. The fix is **distillation**: from x_t, ask the *frozen
teacher* (SD 1.5) for its noise prediction, and take one numerical **DDIM
step** (notes `10 §1.2`, η = 0 case) to get x_{t−k}. Teacher + ODE solver =
a trajectory oracle. Each training example is then a pair
(x_t, x_{t−k}) on one (approximate) trajectory, and the loss pulls
f_student(x_t, t) toward f_target(x_{t−k}, t−k). This is exactly Algorithm 1
of the LCM paper (the code's comments cite it: "z_{t_{n+k}} in Algorithm 1").

### 3. The augmented PF-ODE: bake the guidance scale w into the model

Stable Diffusion is never sampled bare; it is sampled with **classifier-free
guidance** (notes `11 §4`): ε̂_cfg = ε̂_∅ + s·(ε̂_c − ε̂_∅), which doubles
the cost of every step (two U-Net calls, notes `11 §5`). Different s values
define *different* vector fields, hence different PF-ODE trajectories — the
LCM paper calls the family over all w the **augmented PF-ODE**.

LCM distills the *guided* teacher: during training, a guidance scale w is
sampled at random from U[w_min, w_max] per example, the teacher's CFG-combined
prediction (at that w) drives the DDIM step, and w is fed to the *student* as
an extra conditioning input — a sinusoidal **w-embedding** injected next to
the timestep embedding. The student thus learns the whole family of guided
trajectories at once, indexed by w. Payoff at inference: you get CFG-quality
samples from **one** U-Net call per step — no unconditional pass, no
doubling. Guidance becomes free.

### 4. EMA target network (avoiding self-referential collapse)

The self-consistency loss compares the student to *itself* at a different
timestep. If both sides of the loss were the same live network, the trivial
solution "output a constant" satisfies it. Consistency training therefore
uses two copies:

- **online student** `unet` — updated by the optimizer;
- **target student** `target_unet` — updated only as an exponential moving
  average of the online one (θ_target ← μ·θ_target + (1−μ)·θ_online,
  μ = 0.95 here), never by gradients, and its prediction is wrapped in
  `no_grad`.

The target provides a slowly-moving, stable regression target — the same trick
as target networks in Q-learning and the EMA teacher in BYOL.

### 5. The skipping-step technique (k = 20)

Adjacent fine-grained timesteps (t and t−1 out of 1000) give consistency
targets that barely differ — tiny signal, slow convergence. LCM instead
enforces consistency between points **k steps apart** on a coarse 50-point
DDIM grid: with 1000 training timesteps and 50 DDIM timesteps, k = 1000/50 =
20. One teacher DDIM step jumps t_{n+k} → t_n in a single solver call. This
"skipping-step" schedule is the reason LCM distillation converges in a few
thousand iterations rather than the hundreds of thousands the original
consistency-models recipe needed.

### 6. Boundary conditions: c_skip, c_out, and f(x₀, 0) = x₀

Self-consistency alone is satisfied by any constant function. What pins f to
the *true* origin is the **boundary condition** f(x₀, 0) = x₀: at t = 0 the
model must be the identity. Rather than hoping the network learns this, it is
enforced *by parameterization*:

    f(x, t) = c_skip(t) · x + c_out(t) · F_θ(x, t)

where F_θ is "raw network output converted to an x₀-estimate", and the scalar
schedules satisfy c_skip(0) = 1, c_out(0) = 0 — so at t = 0, f is the
identity regardless of θ, and the consistency loss propagates that anchor up
the trajectory: f at t must match f at t−k, which matches f at t−2k, … which
at t = 0 is the ground truth x₀. In this script (with σ_data = 0.5 and a
timestep scaling of 10):

    c_skip(t) = 0.25 / ((10t)² + 0.25),      c_out(t) = 10t / √((10t)² + 0.25)

Check: t = 0 gives c_skip = 1, c_out = 0 ✓; t = 999 gives c_skip ≈ 2.5e-9,
c_out ≈ 1 — at high noise, f is essentially the network's x₀-prediction.

### 7. LCM-LoRA: distillation as an adapter

The sibling script `train_lcm_distill_lora_sd_wds.py` makes one swap: instead
of cloning the full teacher into a trainable student, it freezes the teacher
weights and trains only **LoRA adapters** on top (`peft`'s `LoraConfig` /
`get_peft_model`). Because so few parameters move, it also drops the EMA
target network — the target-side prediction at t−k is computed by the *same*
`unet` under `no_grad` (its line 1237 calls `unet(...)`, not
`target_unet(...)`), and there is no w-embedding input (the architecture must
stay identical to the base model for the adapter to be portable). The
resulting "acceleration vector" can be added to *any* fine-tune of the same
base model — distillation shipped as a plug-in.

### The methodology in one training step (pseudocode)

```
x₀        = VAE.encode(image)                        # clean latent
t_{n+k}   ~ random point on the 50-step DDIM grid    # "start" timestep
x_{t_{n+k}} = √ᾱ·x₀ + √(1−ᾱ)·ε                       # forward-noise it
w         ~ U[w_min, w_max]                          # random guidance scale

# online side (gradients flow)
pred      = c_skip(t_{n+k})·x_{t_{n+k}} + c_out(t_{n+k})·x̂₀_student(x, t_{n+k}, w, c)

# teacher side (no_grad): one guided DDIM step to t_n
x̂₀_cfg, ε̂_cfg = teacher CFG at scale w              # notes 11 §4
x_{t_n}   = √ᾱ_prev·x̂₀_cfg + √(1−ᾱ_prev)·ε̂_cfg      # DDIM, notes 10 §1.2

# target side (no_grad, EMA weights)
target    = c_skip(t_n)·x_{t_n} + c_out(t_n)·x̂₀_target(x_{t_n}, t_n, w, c)

loss      = huber(pred, target);  backprop into student;  EMA-update target
```

Every block of the walkthrough below is one line of this box.

---

## Section 0 — Header and imports (lines 1–75), grouped

```python
import webdataset as wds
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, StableDiffusionPipeline, UNet2DConditionModel
```
Three stacks: **webdataset** streams training images from tar shards (LAION-
style) instead of a folder; **accelerate** handles multi-GPU/mixed precision;
**diffusers** supplies the SD 1.5 parts — VAE (`AutoencoderKL`), noise
schedule (`DDPMScheduler`), U-Net, and `LCMScheduler` (used only for
validation sampling: it is the *inference-time* consistency sampler).
`MAX_SEQ_LENGTH = 77` is CLIP's token limit. `check_min_version("0.18.0.dev0")`
guards against diffusers versions lacking `LCMScheduler`.

## Section 1 — Webdataset plumbing (lines 78–214), grouped

Data-loading boilerplate — read once, then trust:

- `filter_keys` (78–82): returns a function keeping only wanted dict keys per
  sample.
- `group_by_keys_nothrow` / `tarfile_to_samples_nothrow` (85–118): re-implement
  webdataset's tar-grouping so that a malformed sample *warns* instead of
  killing a multi-day run — the FIXME comment explains the LAION-400M edge
  case (same key prefix straddling two tars).
- `WebdatasetFilter` (121–138): drops images smaller than `min_size` or with
  watermark probability > 0.5, judged from LAION's per-sample json. (Defined
  but never used in this script — a leftover.)
- `Text2ImageDataset` (141–214): the pipeline. Resample shards → untar →
  shuffle(1000) → decode PIL → rename (`jpg/png/… → image`, `txt/… → text`) →
  resize shorter side to 512, random-crop 512×512, `to_tensor`, normalize to
  [−1, 1] → `to_tuple("image", "text")` → batch. Each dataloader item is a
  **2-tuple** `(image, text)` with `image: (B, 3, 512, 512)` in [−1, 1] and
  `text` a list of B strings. Remember "2-tuple" — it matters at line 1102.
  `with_epoch(num_worker_batches)` makes the infinite resampled stream look
  like finite epochs of the requested size.

## Section 2 — `log_validation` (lines 217–290), grouped

Builds a `StableDiffusionPipeline` from the teacher checkpoint but swaps in
the student `unet` **and `LCMScheduler`**, then samples the 4 fixed prompts
with `num_inference_steps=4` — the whole point of the method made visible: 4
steps, no CFG argument needed. Images go to tensorboard/wandb.

One real quirk: the `del pipeline / gc.collect() / return image_logs` block
(lines 286–290) is indented *inside* the `for tracker in ...` loop, so the
function returns after the first tracker. Works with one tracker; silently
skips the rest with several.

## Section 3 — The w-embedding (lines 293–320)

```python
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    assert len(w.shape) == 1
    w = w * 1000.0
```
`w: (B,)` — one guidance scale per batch element. Multiplying by 1000 moves
w ∈ [5, 15] into the range the sinusoidal embedding below was designed for
(the same trick and code as timestep embeddings, borrowed from Google's VDM —
see the docstring's link; timesteps live in [0, 1000], so scale w to match).

```python
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
```
Standard transformer/DDPM sinusoidal embedding: geometric frequencies from 1
down to 1/10000, outer product with w — `(B, 1) * (1, half_dim) →
(B, half_dim)` — then sin/cos concatenated → `(B, embedding_dim)`. This
vector is later passed to the U-Net as `timestep_cond`; internally the U-Net
(constructed with `time_cond_proj_dim`) projects it and **adds it into the
timestep embedding**, which is how one scalar w conditions every layer.

```python
def append_dims(x, target_dims):
    return x[(...,) + (None,) * dims_to_append]
```
(Lines 323–328.) Reshape helper: turns `(B,)` into `(B, 1, 1, 1)` so
per-sample scalars broadcast against `(B, 4, 64, 64)` latents.

## Section 4 — Boundary-condition scalings (lines 331–335) ★ methodology §6

```python
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out
```
The enforced boundary condition, verbatim from `LCMScheduler`. `timestep` is
the raw integer grid value (0…999), shape `(B,)`; both outputs are `(B,)`.
`timestep / 0.1` = `timestep * 10` — the "scaled timestep". Plug in t = 0:
c_skip = σ_d²/σ_d² = 1, c_out = 0 → f(x, 0) = x exactly, for any network
weights. As t grows, c_skip → 0 and c_out → 1: at high noise the input x is
useless and f is essentially the network's x₀-prediction. σ_data = 0.5 is the
assumed data std (Karras/EDM convention, roughly right for SD latents).

Honest quirk: the `timestep_scaling=10.0` **parameter is never used** — the
10 is hard-coded as `/0.1`. Change the argument and nothing happens.

## Section 5 — x̂₀-prediction (lines 338–357) ★ the identity from notes 10 §1.2

```python
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
```
Converts a raw U-Net output into an estimate of the clean latent. For
ε-prediction this is exactly our one-line inversion of the forward shortcut
(notes `10 §1.2`): x_t = α_t x₀ + σ_t ε ⇒ **x̂₀ = (x_t − σ_t ε̂)/α_t**. Here
`alphas`/`sigmas` are the *global* schedules √ᾱ and √(1−ᾱ), each `(1000,)`;
`extract_into_tensor` (lines 354–357) is the familiar gather-then-reshape from
our DDPM code: pick each sample's own scalar by its timestep, `a.gather(-1, t)`
→ `(B,)`, reshape to `(B, 1, 1, 1)` for broadcasting. The v-prediction branch
is the same identity in v-parameterization (SD 2.x). Output: `(B, 4, 64, 64)`.

## Section 6 — `DDIMSolver` (lines 360–384) ★ the trajectory oracle

```python
class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
```
Precomputes the coarse grid for the skipping-step technique (methodology §5).
With defaults, `step_ratio = 20` and `ddim_timesteps = [19, 39, 59, …, 999]`,
shape `(50,)` — fifty evenly-spaced trainable timesteps, k = 20 apart.

```python
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
```
ᾱ at each grid point, and ᾱ at each grid point's *predecessor*: entry i of
`_prev` is ᾱ at `ddim_timesteps[i−1]`; entry 0 (whose predecessor would be
t = −1) falls back to ᾱ₀, i.e. t = 0. Both `(50,)`. `to(device)` (374–378)
just moves the three buffers.

```python
    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
```
One deterministic DDIM step, η = 0 — precisely equation (1) of notes
`10 §1.2` with the fresh-noise term deleted: **x_s = α_s·x̂₀ + σ_s·ε̂**, with
α_s = √ᾱ_prev and σ_s = √(1−ᾱ_prev). Note it indexes by `timestep_index`
(position 0–49 on the coarse grid, `(B,)`), *not* by the raw timestep — the
caller must pass the same `index` it used to pick `start_timesteps`. Inputs
and output all `(B, 4, 64, 64)`. This single call is the teacher's "advance
the ODE by k = 20 fine steps" jump.

## Section 7 — EMA update (lines 387–398) ★ methodology §4

```python
@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
```
Polyak averaging, in-place: θ_target ← rate·θ_target + (1−rate)·θ_online.
`no_grad` + `detach` keep autograd entirely out. The script calls it with
`args.ema_decay = 0.95` — much *faster*-moving than the 0.999–0.9999 typical
of generative-model EMA, because here the EMA copy is a *training target*
that must track the student closely (methodology §4), not a polish-for-
release average.

## Section 8 — `parse_args` (lines 421–741), grouped

Standard HF-script argparse; the LCD-specific block (lines 620–669) is the
part worth reading:

- `--w_min 5.0`, `--w_max 15.0`: the guidance-scale sampling range for the
  augmented PF-ODE (methodology §3). The help text claims the *Imagen* CFG
  formulation ("all guidance scales have 1 added"), but as we'll see at line
  1207 the code actually implements the *LCM-paper* formulation — an internal
  inconsistency; more at Section 12.
- `--num_ddim_timesteps 50`: the coarse grid size, hence k = 1000/50 = 20.
- `--loss_type {l2, huber}`, `--huber_c 0.001`: consistency-loss distance.
- `--ema_decay 0.95`: target-network rate.
- `--cast_teacher_unet`: run the frozen teacher in fp16/bf16 to save memory.

**Real bug to know about:** lines 859 and 1143 use `args.unet_time_cond_proj_dim`
(the w-embedding width), but **no `--unet_time_cond_proj_dim` argument is ever
defined** in `parse_args` — this script crashes with `AttributeError` as
shipped. The maintained diffusers copy of the script adds the flag (default
256). Patch it in if you run this version.

`encode_prompt` (744–767): tokenize captions to 77 tokens, run CLIP under
`no_grad` → `prompt_embeds: (B, 77, 768)`. It can randomly blank captions
with probability `proportion_empty_prompts` — but see the quirk at line 1011.

## Section 9 — `main`: setup (lines 770–1049), grouped with highlights

Lines 771–812: `Accelerator` (with `split_batches=True` — the comment explains
this keeps LR-schedule step counts right with webdataset), logging levels,
seeding, output dir, optional Hub repo. Boilerplate.

```python
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler", ...)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), timesteps=..., ddim_timesteps=args.num_ddim_timesteps)
```
(Lines 814–826.) The teacher's own noise schedule — ᾱ over 1000 steps — is
split into the (α_t, σ_t) = (√ᾱ, √(1−ᾱ)) pair our notes use, each `(1000,)`,
and handed to the `DDIMSolver`. The student must live on the *teacher's*
schedule; a mismatched schedule would put teacher and student on different
trajectories and the consistency pairs would be garbage.

Lines 828–854: load tokenizer, CLIP text encoder, VAE, and the **teacher
U-Net** from the SD checkpoint, then freeze all of them
(`requires_grad_(False)`). Nothing about the teacher ever trains. (Ignore the
comments saying "SD-XL checkpoint" — copy-paste from the SDXL sibling script;
this one is plain SD.)

```python
    if teacher_unet.config.time_cond_proj_dim is None:
        teacher_unet.config["time_cond_proj_dim"] = args.unet_time_cond_proj_dim
    unet = UNet2DConditionModel(**teacher_unet.config)
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()
```
(Lines 856–863.) Student creation, and the architectural delta of the whole
method: the teacher's config is mutated to include `time_cond_proj_dim`, so
the freshly built student U-Net has one extra small linear layer
(`time_embedding.cond_proj`) that ingests the w-embedding (Section 3).
`strict=False` is required because the teacher's state_dict has no weights
for that new layer — every shared weight is copied from the teacher (warm
start), and **the `cond_proj` layer alone starts at random init**. (Mutating
the already-built teacher's config dict is a dirty but harmless trick — the
teacher's modules are already constructed.)

```python
    target_unet = UNet2DConditionModel(**teacher_unet.config)
    target_unet.load_state_dict(unet.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)
```
(Lines 865–870.) The EMA target (methodology §4): an exact clone of the
student at t = 0, gradient-free forever; only `update_ema` will move it.
So three U-Nets coexist in memory: frozen teacher, trainable student, EMA
target — the main VRAM cost of LCM distillation (and what LCM-LoRA removes).

Lines 872–908: precision hygiene — trainable student stays fp32; frozen
VAE/text-encoder/teacher move to device and (optionally) to fp16/bf16;
`alpha_schedule`, `sigma_schedule`, `solver` buffers also move to device.
Lines 910–942: accelerate save/load hooks so checkpoints contain both `unet`
and `unet_target`. Lines 944–988: xformers, TF32, gradient checkpointing,
optional 8-bit Adam; optimizer over `unet.parameters()` **only** — the
optimizer never sees teacher or target.

Lines 990–1049: dataset (Section 1) instantiated; `compute_embeddings_fn`
built with `functools.partial`; LR scheduler; `accelerator.prepare(unet,
optimizer, lr_scheduler)` (note the dataloader is *not* prepared — webdataset
handles its own sharding). Quirk at line 1011: the partial hard-codes
`proportion_empty_prompts=0`, so the CLI flag `--proportion_empty_prompts` is
parsed, validated… and ignored. Finally:

```python
    uncond_input_ids = tokenizer([""] * args.train_batch_size, ..., max_length=77).input_ids.to(...)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]
```
(Lines 1046–1049.) The empty-prompt embedding `(B, 77, 768)`, computed **once**
— the "∅" of CFG (notes `11 §5`). The teacher will be queried with it every
step; precomputing saves a text-encoder call per iteration.

## Section 10 — The training loop, part 1: batch → noisy latent (lines 1099–1139)

```python
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                image, text, _, _ = batch
```
`accelerator.accumulate` handles gradient accumulation transparently.
**Quirk/bug:** the dataset yields 2-tuples `(image, text)` (Section 1), so
this 4-way unpack raises `ValueError` as shipped — the `_, _` (original size,
crop coords) belong to the SDXL sibling script. Delete the two underscores to
run this file.

```python
                image = image.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text)
                pixel_values = image.to(dtype=weight_dtype)
```
Move images `(B, 3, 512, 512)` to GPU; encode captions →
`encoded_text = {"prompt_embeds": (B, 77, 768)}`.

```python
                latents = []
                for i in range(0, pixel_values.shape[0], 32):
                    latents.append(vae.encode(pixel_values[i : i + 32]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
```
VAE-encode in chunks of ≤ 32 (the encoder is memory-hungry at 512²), *sample*
from the posterior (not the mean), and apply SD's magic 0.18215 scaling so
latents have ~unit variance. `latents: (B, 4, 64, 64)` — this is the x₀ of
the methodology box. Everything from here on happens in latent space; that
"latent" is the L in LCM.

```python
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
```
The skipping-step sampling (methodology §5). `topk` = k = 20. `index: (B,)`
picks a random position on the 50-point coarse grid per sample;
`start_timesteps` = t_{n+k} ∈ {19, 39, …, 999} is where the *pair starts*;
`timesteps` = t_n = t_{n+k} − 20 is where the teacher's DDIM step will land.
The `where` clamps the one negative case (index 0: 19 − 20 = −1 → 0), which
matches `DDIMSolver`'s ᾱ_prev[0] = ᾱ₀ fallback — for every other index,
`timesteps` lands exactly on `ddim_timesteps[index − 1]`. Consistent
bookkeeping between the two is essential: the *same* t_n must be used for
the target's c_skip/c_out and U-Net call as the solver actually stepped to.

```python
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
```
Boundary scalings (Section 4) at *both* ends of the pair — `_start` for the
online student at t_{n+k}, plain for the target at t_n — each reshaped
`(B,) → (B, 1, 1, 1)`.

```python
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
```
Forward-diffuse to t_{n+k}: x = √ᾱ·x₀ + √(1−ᾱ)·ε, `(B, 4, 64, 64)` — the
z_{t_{n+k}} of the paper's Algorithm 1 (the code comment says so). Note the
pair's starting point comes from the *forward marginal*, not from simulating
the ODE from noise — that is what makes each training step cheap (one solver
step, not a whole trajectory), and it works because the teacher's ε̂ defines
a valid trajectory through *any* marginal sample (notes `10 §1.1`).

## Section 11 — Training loop, part 2: sample w, run the student (lines 1141–1170)

```python
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w_embedding = guidance_scale_embedding(w, embedding_dim=args.unet_time_cond_proj_dim)
                w = w.reshape(bsz, 1, 1, 1)
```
The augmented-PF-ODE dial (methodology §3): per-sample guidance scale
w ~ U[5, 15], `(B,)`. Two forms are kept: the sinusoidal `w_embedding`
`(B, 256)` to *condition the student*, and the raw scalar reshaped to
`(B, 1, 1, 1)` to *combine the teacher's CFG* below. Both moved to the
latents' device/dtype (lines 1146–1147).

```python
                prompt_embeds = encoded_text.pop("prompt_embeds")

                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample
```
The **online student's** forward pass at the pair's start: inputs
`(B, 4, 64, 64)`, `(B,)` timesteps, `timestep_cond=w_embedding` (the LCM-
specific input — this is where w enters the network), text `(B, 77, 768)`.
Output `noise_pred: (B, 4, 64, 64)` — an ε̂-prediction (SD 1.5 is
ε-parameterized). After the `pop`, `encoded_text` is `{}`, so
`added_cond_kwargs={}` — a harmless SDXL leftover.

```python
                pred_x_0 = predicted_origin(noise_pred, start_timesteps, noisy_model_input, ...)
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
```
Convert ε̂ → x̂₀ via the identity (Section 5), then apply the boundary
parameterization: **model_pred = f_θ(x_{t_{n+k}}, t_{n+k}, w, c)** — the
left-hand side of the consistency loss, the only tensor in this whole step
that carries gradients. `(B, 4, 64, 64)`.

## Section 12 — Training loop, part 3: teacher CFG + DDIM step (lines 1172–1209) ★ the heart

```python
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        cond_pred_x0 = predicted_origin(cond_teacher_output, start_timesteps, noisy_model_input, ...)
```
Teacher pass #1: conditional ε̂_c at the *same* noisy input and timestep, no
gradients, autocast for speed. Converted to a conditional x̂₀ too. Note the
teacher takes **no** `timestep_cond` — it never heard of w; w exists only in
how we *combine* its two passes.

```python
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = predicted_origin(uncond_teacher_output, ...)
```
Teacher pass #2: unconditional ε̂_∅, using the precomputed empty-prompt
embedding. This is the classic two-pass CFG evaluation of notes `11 §5` —
the very cost LCM is about to distill away.

```python
                        # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)
```
**The single most important block in the file.** CFG applied in *both* spaces
— x̂₀ and ε̂ — then one DDIM step (Section 6):
x_prev = √ᾱ_prev · x̂₀_cfg + √(1−ᾱ_prev) · ε̂_cfg, landing at t_n.
`x_prev: (B, 4, 64, 64)` is the second point of the trajectory pair; w
broadcasts as `(B, 1, 1, 1)`.

Two honest notes. (1) *Convention*: `cond + w·(cond − uncond)` =
`uncond + (1+w)·(cond − uncond)`, i.e. our notes-`11 §4` formula with
s = 1 + w. So w ∈ [5, 15] here behaves like SD `guidance_scale` ∈ [6, 16] —
and the argparse help (which claims Imagen convention, s = w) contradicts the
code comment (which says LCM-paper convention). The code line is the ground
truth: LCM-paper formulation. (2) *Both spaces*: because CFG is a linear
combination and ε ↔ x̂₀ is an affine map at fixed t, combining in ε-space and
then converting would give the same x̂₀ — computing both explicitly just
feeds `ddim_step`'s two slots directly.

## Section 13 — Training loop, part 4: target prediction and loss (lines 1211–1244)

```python
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_noise_pred = target_unet(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                    pred_x_0 = predicted_origin(target_noise_pred, timesteps, x_prev, ...)
                    target = c_skip * x_prev + c_out * pred_x_0
```
The **EMA target network** evaluates the consistency function at the *other*
end of the pair: f_θ⁻(x_{t_n}, t_n, w, c), same w-embedding, same prompt, but
at `timesteps` = t_n and input `x_prev`. All under `no_grad` — the target is
a fixed regression label for this step (methodology §4). `target:
(B, 4, 64, 64)`. Self-consistency demands `model_pred ≈ target`.

```python
                if args.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_type == "huber":
                    loss = torch.mean(
                        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                    )
```
The consistency-distillation loss. The "huber" branch is the smooth
**pseudo-Huber** distance √(d² + c²) − c: quadratic for |d| ≪ c, linear for
|d| ≫ c — robust to the occasional wild teacher/solver estimate, which is why
the LCM paper prefers it for latent-space distillation (c = 0.001 default).
Note this is a plain regression in latent space — no ELBO, no noise-matching;
the entire diffusion math already happened upstream.

```python
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
```
Standard optimization, gradients flowing only through `model_pred` → student
`unet`. Grad-norm clipping at 1.0 only on real (synced) steps.

## Section 14 — Training loop, part 5: EMA + bookkeeping (lines 1246–1299)

```python
            if accelerator.sync_gradients:
                update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
```
After each real optimizer step, drag the target 5% of the way toward the
student (μ = 0.95, Section 7). Order matters: target update *after* the
student step, once per effective batch — inside the accumulate block it would
move during half-formed gradients.

The rest is housekeeping: rotate checkpoints under `checkpoints_total_limit`,
`accelerator.save_state` every 500 steps, run `log_validation` (Section 2)
every 200 steps for *both* the target and online U-Nets (the target usually
looks better — it is the one you ship), log loss/lr, and at the very end save
`unet/` and `unet_target/` with `save_pretrained`.

---

## Quirk ledger (all verified against the code)

1. `args.unet_time_cond_proj_dim` used (lines 859, 1143) but never added to
   `parse_args` → `AttributeError` on launch. Add the flag (diffusers'
   maintained copy defaults it to 256).
2. `image, text, _, _ = batch` (line 1102) vs. a 2-tuple dataset (line 178)
   → `ValueError`. SDXL leftover.
3. Help text for `--w_min/--w_max` claims Imagen CFG convention; line 1207
   implements the LCM-paper convention (effective SD scale = 1 + w).
4. `timestep_scaling` parameter of `scalings_for_boundary_conditions` is
   dead; the ×10 is hard-coded as `/0.1`.
5. `compute_embeddings_fn` hard-codes `proportion_empty_prompts=0`; the CLI
   flag is ignored.
6. `log_validation` returns from inside its tracker loop (line 290).
7. `WebdatasetFilter` is defined but never applied to the pipeline.
8. Comments mention "SD-XL checkpoint" and numbered steps "20.4.x" — artifacts
   of the file this was adapted from; the numbering is still a useful map of
   Algorithm 1's stages.

## What to carry forward

1. **Consistency = same origin from anywhere on one trajectory.** The loss
   never sees a real image as a target — only the model's own prediction one
   solver step downstream; ground truth enters solely through the t = 0
   boundary condition.
2. **c_skip/c_out parameterization**: f(x, t) = c_skip·x + c_out·x̂₀(x, t)
   with c_skip(0) = 1, c_out(0) = 0 — identities you enforce by construction
   beat identities you hope to learn.
3. **Teacher + DDIM step = trajectory-pair factory** (notes `10 §1.2` with
   η = 0). Any better ODE solver could slot into `DDIMSolver.ddim_step`
   unchanged — equation vs. simulator separation, again (MIT Lab 1's lesson).
4. **w as an input, not a sampler trick**: sample the guidance scale during
   training, embed it like a timestep, and CFG's 2× inference cost disappears
   (notes `11 §4–5` for what w does).
5. **Skipping-step k = 20** turns 1000 fine steps into 50 coarse consistency
   pairs — bigger jumps, stronger training signal, faster convergence.
6. **EMA target network** (μ = 0.95): the self-referential loss needs a
   slow-moving second copy, exactly like target networks in RL.
7. **LCM-LoRA**: same loss, student = frozen teacher + adapters, no EMA copy,
   no w-embedding — few-step sampling shipped as a portable plug-in.
