# PCM, Line by Line — Phased Consistency Distillation with an Adversarial Loss

*A beginner's reading companion to
`Phased-Consistency-Model/code/text_to_image_sd15/train_pcm_lora_sd15_adv.py`
(PCM, "Phased Consistency Models," Wang et al., NeurIPS 2024, arXiv 2405.18407).
Every line that carries the method: what it does, why it's there, and what
breaks without it. Boilerplate is grouped and dispatched quickly. Concepts
link to our notes: DDIM step identity = `10 §1.2`, CFG = `11 §4`,
ᾱ-ratio respacing = `04 §"ᾱ-ratio respacing"` / HW Q7.*

**The script in one sentence:** distill Stable Diffusion 1.5 into a LoRA
student that, instead of learning one consistency function mapping every
noisy latent to t=0 (LCM's move), learns M *phased* consistency functions —
each mapping any point inside its phase of the ODE trajectory to that phase's
low-noise edge — trained with a solver-bootstrapped consistency loss plus a
GAN loss in noisy-latent space, so that M deterministic jumps (plain DDIM,
no re-noising) generate an image.

---

## Methodology (read this first)

### Background you already have: LCM in five lines

Latent Consistency Models distill a teacher diffusion UNet into a student
that learns a **consistency function** f(x_t, t) ≈ x_0 — "from anywhere on
the probability-flow ODE trajectory, jump straight to the clean end."
Training never simulates whole trajectories; instead:

1. Noise a real latent to a random grid time t_{n+1} (forward process, `05`).
2. Ask the **teacher** (with CFG, `11 §4`) for ε̂ there, and take one
   **skipping-step DDIM solver step** (`10 §1.2`) down to t_n — that's a
   point *provably on (approximately) the same trajectory*.
3. **Consistency loss**: the student's jump from t_{n+1} must land where the
   student's own (frozen/EMA) jump from t_n lands. Self-consistency along the
   trajectory, anchored by the **boundary condition** f(x_0, 0) = x_0
   (enforced via a c_skip·x + c_out·F(x) parameterization).

### Three ways this breaks for few-step text-to-image (the PCM diagnosis)

**Flaw 1 — everything maps to t=0, so multi-step sampling must re-noise.**
A consistency model gives you *one* jump: x_T → x̂_0. To spend 4 steps
instead of 1, LCM does: jump to x̂_0, then **add fresh noise** to climb back
up to an intermediate t, jump again, re-noise again… Each re-noising is a
new random draw, so (a) the result is *stochastic* and changes character as
you change the step count — 1-step, 2-step, 4-step results of the same seed
disagree — and (b) since the consistency loss is an L2-type average, 1–2
step outputs regress to the mean: **blurry**. (You can see the re-noising in
any LCMScheduler `step()`; nothing in *this* file re-noises at inference —
that's the point.)

**Flaw 2 — the CFG-augmented ODE gets baked in.** LCM distills the teacher's
*guided* ε̂ = ε_c + w(ε_c − ε_u) with large w (order 5–15) into the student.
The student then *is* a large-CFG model: apply CFG again at inference and
images over-expose; usable guidance scales collapse to ≈1. Verified in this
repo: the launch script `train_pcm_lora_sd15.sh` comments say verbatim "You
will find from the validation that using larger CFGs generate overexpourse
results," the trainer exposes `--not_apply_cfg_solver` to distill the
**unguided** ODE instead, and `log_validation` is deliberately run twice per
eval, at `cfg=1` **and** `cfg=7.5`, to watch exactly this.

**Flaw 3 — the consistency loss is weak supervision.** The target (student's
jump from one solver step below) and the prediction (student's jump from
here) are nearly identical tensors; their L2/Huber difference is a tiny,
noisy learning signal, and it tolerates blur — an averaged-out x̂_0 is a
great L2 citizen. At 1–2 steps, where the jump is enormous, this shows up
as smooth, detail-free images.

### PCM's fix, part 1: split the trajectory into M phases

Choose M **edge timesteps** s_0 = 0 < s_1 < … < s_{M-1} < s_M = T that cut
the ODE trajectory into M sub-trajectories ("phases"). Learn a consistency
function **per phase**: for t inside phase m (s_{m-1} < t ≤ s_m),

    f_m(x_t, t) ≈ x_{s_{m-1}}     — jump to the phase's *low-noise edge*,

with the boundary condition f_m(x_{s_{m-1}}, s_{m-1}) = x_{s_{m-1}} (identity
at the edge). Sampling with M steps is then **deterministic chaining**:

    x_{s_M} → f_M → x_{s_{M-1}} → f_{M-1} → … → f_1 → x_0

— no re-noising ever, so results are consistent across step budgets, and
each learned jump is short enough to be easy. M=1 recovers LCM exactly;
M→number-of-grid-points recovers the teacher's DDIM sampler. In this code
the edges live on a 50-point DDIM grid: `inference_indices =
floor(linspace(0, 50, M, endpoint=False))` — for the shipped 2-phase
adversarial run, grid indices {0, 25}, i.e. edge timesteps {0, 499} of 1000.

One neat implementation choice makes this nearly free: the consistency
function is **parameterized as a DDIM step**. The student UNet still
predicts ε; the jump to edge s is just the DDIM identity from our note
`10 §1.2`:

    f(x_t, t) = √ᾱ_s · x̂_0(x_t, t) + √(1−ᾱ_s) · ε̂(x_t, t)

(`DDIMSolver.ddim_style_multiphase_pred` below — it picks s = the right
phase edge per sample, then applies exactly this formula). Because the
student is "an ε-predictor whose DDIM steps are trained to be exact even
when huge," the finished LoRA drops into a **stock `DDIMScheduler`** with
`num_inference_steps = M` — which is precisely what `log_validation` does.
This is also why the phase split is kin to our strided-schedule work: both
build a coarse sub-schedule of the same forward process and rely on ᾱ-ratio
identities to move along it (`04`, HW1/HW2 Q7 respacing).

**How the boundary condition is enforced here (different from LCM!):** not
by a smooth c_skip/c_out schedule, but discretely, on the *target side*.
When the sampled grid index sits exactly at a phase edge, the target is
defined to be the solver point x_prev itself (c_skip=1, c_out=0 — the
identity), while the online student always outputs its full DDIM-style jump
(c_skip=0, c_out=1). See `scalings_for_boundary_conditions_target/_online`.

### PCM's fix, part 2: the adversarial consistency loss

For flaw 3, PCM adds a GAN. After the student jumps to the phase edge, its
output is **re-noised forward** to a random timestep *inside the phase
above* the edge — using `noise_travel`, a marginal-preserving forward hop
whose coefficient is the ᾱ ratio √(ᾱ_target/ᾱ_current) (our respacing
identity again). The "real" sample is the *distillation target* (the
teacher-solver-bootstrapped prediction) re-noised the same way. A
discriminator — the **frozen teacher UNet as a feature extractor** with
tiny trainable conv heads on 9 intermediate feature maps — is trained with
a hinge loss to tell them apart, and the student receives
`adv_weight × g_loss` on top of the consistency loss. Distribution-level
supervision where L2 was point-level: this is what buys 1-step sharpness.
Generator and discriminator strictly alternate, one optimizer step each.

### The training step, as an algorithm box

    given latents x0, prompt c:
      index  ~ U{0..49}                          # position on 50-pt DDIM grid
      t_start = ddim_timesteps[index]            # ∈ {19, 39, …, 999}
      t_end   = t_start − 20 (clamped ≥ 0)       # one solver step lower
      x_t     = add_noise(x0, ε, t_start)                       # forward
      ONLINE:  ε̂_θ(x_t, t_start) → DDIM-jump to phase edge s  → model_pred
      TEACHER: ε̂_teacher with CFG(w) at t_start → one DDIM step → x_prev   [no grad]
      TARGET:  ε̂_θ(x_prev, t_end) → DDIM-jump to same edge s  → target    [no grad]
               (target := x_prev itself if index is exactly an edge)
      GAN:     re-noise model_pred and target from s up to a random
               within-phase t_adv; alternate hinge-D / (consistency + hinge-G)

Everything below is this box, line by line.

---

## Part 0 — File map and grouped boilerplate

Three files matter:

- `train_pcm_lora_sd15_adv.py` (1530 lines) — the trainer, annotated in full below.
- `scheduling_ddpm_modified.py` — a stock diffusers `DDPMScheduler` **plus
  one added method, `noise_travel`** (annotated in Part 5); `add_noise` is
  the standard √ᾱ·x + √(1−ᾱ)·ε.
- `discriminator_sd15.py` — the frozen-teacher-backbone discriminator
  (annotated in Part 6).

Boilerplate you can skim once (lines 1–67, 93–137, 395–751, 782–827,
946–1032, 1061–1169, 1433–1524):

- **Imports** — HuggingFace stack: `accelerate` (multi-GPU/mixed precision),
  `peft` (LoRA), `diffusers` (models/schedulers), `transformers` (CLIP text
  encoder). Note two imports that matter: `from scheduling_ddpm_modified
  import DDPMScheduler` (the local file shadows the diffusers class — that's
  how `noise_travel` gets in) and `from discriminator_sd15 import
  Discriminator`. `from torch.optim import RMSprop` is imported and never
  used (dead).
- **`get_module_kohya_state_dict`** (70–90) — renames PEFT LoRA keys
  (`lora_A/B` → `lora_down/up`, dots→underscores) into the Kohya format the
  `StableDiffusionPipeline.load_lora_weights` path expects, and attaches the
  `alpha` scale per layer. Pure format shim for validation/saving.
- **`CustomImageDataset`** (93–137) — folder of `img.jpg` + `img.txt`
  caption pairs; resize→center-crop→ToTensor→normalize to [−1,1]
  (mean/std 0.5). Quirk: if a caption file is missing, the `while True:` /
  `continue` retries the *same* idx forever — an infinite loop for any
  image lacking a `.txt`. Also the dataset path is **hardcoded** later at
  line 1051: `CustomImageDataset("/mnt/data/wangfuyun/cc3m", …)`; there is
  no `--train_data_dir` flag. Edit the source to point at your data.
- **`parse_args`** (395–751) — the usual wall. The PCM-specific knobs:
  `--num_ddim_timesteps` (default 50; the solver grid), `--multiphase`
  (default 8 in code; the shipped adversarial run uses **2**), `--w_min/
  --w_max` (teacher CFG range; script uses 4–5 and the help text warns these
  are "Imagen formulation," i.e. +1 versus the usual scale), `--not_apply_
  cfg_solver` (distill the unguided ODE), `--adv_weight` (0.1) and
  `--adv_lr` (1e-5), `--loss_type` l2/huber.
- **`encode_prompt`** (755–779) — tokenize captions (pad to 77), one frozen
  CLIP pass → `prompt_embeds` of shape **(B, 77, 768)**. With probability
  `proportion_empty_prompts` a caption is replaced by `""` (condition
  dropout, `11 §5` — the second launch-script recipe sets 0.1 exactly so the
  student itself supports CFG at inference).
- **Accelerator / logging / seed / repo setup** (782–827) — one detail:
  `set_seed(args.seed + accelerator.process_index)` gives each GPU a
  *different* seed, so noise draws differ across ranks (you want that).
- **Checkpoint hooks** (946–976) — save/load only the LoRA adapter, not the
  full UNet. **Optimizers** (1017–1032): AdamW on `unet.parameters()`
  (PEFT makes only LoRA params trainable) and a second AdamW on the
  discriminator heads with `betas=(0, 0.999)` — β1=0 is standard GAN
  practice (no momentum on D). **LR scheduler, `accelerator.prepare`,
  resume, progress bar** (1061–1169) — standard. **Checkpoint pruning and
  final save** (1433–1524) — standard.

---

## Part 1 — The small helpers that carry the method

```python
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
```
(229–256) Sinusoidal embedding of the CFG scale w — LCM feeds this to the
student via `timestep_cond` so the student is w-conditional. **In this file
it is dead code**: every student call passes `timestep_cond=None`. The PCM
student is *not* conditioned on w — one consequence of moving away from the
CFG-augmented-ODE design (Methodology, flaw 2).

```python
def append_dims(x, target_dims):
    return x[(...,) + (None,) * dims_to_append]
```
(259–266) Right-pad shape with singleton dims: turns a per-sample scalar
(B,) into (B,1,1,1) so it broadcasts against latents (B,4,64,64). Same job
as our `extract` helper in the DDPM homework.

```python
def scalings_for_boundary_conditions_target(index, selected_indices):
    c_skip = torch.isin(index, selected_indices).float()
    c_out = 1.0 - c_skip
    return c_skip, c_out
```
(269–272) **The phased boundary condition, in two lines.** `index` (B,) is
each sample's position on the 50-point DDIM grid; `selected_indices` (M,)
are the phase-edge grid indices. `torch.isin` → 1.0 exactly where the
sampled index *is* a phase edge. Used later as
`target = c_skip * x_prev + c_out * f_θ(x_prev)`: when the solver point
x_prev sits exactly on a phase edge s_{m-1}, the target is **x_prev itself**
— the identity — which is precisely f_m(x_{s_{m-1}}, s_{m-1}) = x_{s_{m-1}}.
Everywhere else the target is the student's own jump (consistency
bootstrapping). Boundary conditions enforced through the *training target*,
not through a c_skip/c_out reparameterization of the network.

```python
def scalings_for_boundary_conditions_online(index, selected_indices):
    c_skip = torch.zeros_like(index).float()
    c_out = torch.ones_like(index).float()
```
(275–278) The online (gradient-carrying) branch gets c_skip≡0, c_out≡1 — a
deliberate no-op kept so the code shape mirrors LCM's. The online student
*always* outputs its full DDIM-style jump; it is never handed the identity
shortcut (if it were, edge-index samples would produce zero gradient).

```python
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
```
(281–292) ε-prediction → x̂_0, the inversion of the forward identity
x_t = α_t·x_0 + σ_t·ε (here `alphas` = √ᾱ, `sigmas` = √(1−ᾱ) — beware the
naming: these are the *square-rooted* schedules). `extract_into_tensor`
(295–298) is the standard gather-then-reshape-to-(B,1,1,1). The
`v_prediction` branch is the α·x − σ·v identity; SD1.5 uses epsilon.

---

## Part 2 — `DDIMSolver`: the trajectory grid, the solver step, and the phase jump

This class (301–355) is the geometric heart. It holds a **coarse 50-point
DDIM grid** over the teacher's 1000 timesteps and knows two moves: one grid
step down (`ddim_step`, used to build the target's x_prev), and a jump all
the way to the current phase's edge (`ddim_style_multiphase_pred`, the
consistency function's output head).

```python
class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        self.step_ratio = timesteps // ddim_timesteps
```
1000 // 50 = **20** raw timesteps per grid step — LCM's "skipping-step"
solver stride k.

```python
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
```
Grid times = {1·20−1, 2·20−1, …, 50·20−1} = **{19, 39, 59, …, 999}** — a
"trailing"-style grid whose top point is exactly t=999, the noisiest
trainable timestep. Shape (50,).

```python
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
```
For each grid point, precompute ᾱ there *and* at the previous grid point,
with the bottom point's "previous" defined as raw timestep 0. Quirk worth
flagging: the prev-ᾱ for grid index 0 is ᾱ_0 = α_0 (≈0.99915 for SD1.5's
scaled-linear schedule), **not exactly 1** — so a "step to t=0" retains a
whisper of noise coefficient √(1−ᾱ_0) ≈ 0.029. Inherited from LCM's code;
harmless in practice, and consistent with `set_alpha_to_one=False` in the
validation scheduler. The rest of `__init__` and `.to(device)` (312–323)
just convert to torch tensors and move devices.

```python
    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
```
Our note `10 §1.2` verbatim — **re-noise the estimate**:
x_prev = √ᾱ_prev·x̂_0 + √(1−ᾱ_prev)·ε̂. One deterministic DDIM step from
grid index i down to grid index i−1 (a 20-raw-step skip). This is the
skipping-step ODE solver applied to the *teacher's* (CFG-combined)
prediction to manufacture the training target's input.

Now the star:

```python
    def ddim_style_multiphase_pred(self, pred_x0, pred_noise, timestep_index, multiphase):
        inference_indices = np.linspace(
            0, len(self.ddim_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
```
**Where the M phases are born.** Split the 50 grid indices into M equal
chunks and keep each chunk's first index. For `multiphase=2`:
linspace(0,50,2) without endpoint = [0, 25] → edges at grid indices
**{0, 25}**. For M=4: [0, 12.5, 25, 37.5] → **{0, 12, 25, 37}**. These are
the *lower edges* of the phases, on the grid.

```python
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index = inference_indices[last_valid_index]
```
Vectorized "which phase am I in?" — for each sample, find the **largest edge
≤ my grid index**. Line by line: broadcast each sample's index against all
M edges → mask (B, M) of edges at-or-below me; `flip` + `argmax` finds the
*last* True (argmax returns the first max, so flip makes last-first); un-flip
the position; gather the edge. Example, M=2 (edges {0,25}): index 3 → edge 0;
index 25 → edge 25; index 40 → edge 25. `timestep_index` is now (B,) of
**phase-edge grid indices**, overwriting the input's meaning (a readability
quirk — same variable, new semantics).

```python
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, self.ddim_timesteps_prev[timestep_index]
```
Same DDIM identity as `ddim_step`, but the destination is
`ddim_alpha_cumprods_prev[edge]` — i.e. the raw timestep **just below** the
edge grid point: `ddim_timesteps_prev[edge]`. For M=2 that's raw timesteps
**{0, 499}** (since `ddim_timesteps_prev[25] = ddim_timesteps[24] = 499`);
for M=4, **{0, 239, 499, 739}**. So this method *is* the phased consistency
function's output head: f_m(x_t, t) = √ᾱ_s·x̂_0 + √(1−ᾱ_s)·ε̂ with s = the
phase's low-noise edge. It returns both the jump result and the edge
timesteps (B,) — the latter feed the adversarial branch. One giant DDIM step
per phase; training will make that giant step *exact*, which vanilla DDIM
only achieves as the step shrinks (`10 §1.3`).

`update_ema` (358–369) — defined, **never called**. Unlike LCM there is no
EMA target network here: the target branch reuses the online student under
`no_grad` (see Part 4.7). Dead code, but a meaningful design fact.

---

## Part 3 — Setup inside `main()`: schedules, five networks, LoRA (grouped)

```python
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler", ...)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )
```
(828–842) The *local, modified* DDPMScheduler (this is where `noise_travel`
comes from), loaded with SD1.5's scaled-linear β schedule; `alpha_schedule`/
`sigma_schedule` are the √ᾱ_t, √(1−ᾱ_t) tables (shape (1000,)) that
`predicted_origin` consumes; and the 50-point solver grid from Part 2.

(844–872) Load tokenizer, CLIP text encoder, VAE, and **two copies of the
SD1.5 UNet**: `teacher_unet` (frozen ε-oracle) and, at line 872,
`discriminator = Discriminator(teacher_unet)` — note the discriminator
**wraps the very same frozen teacher UNet object** as its feature backbone.

```python
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    discriminator.unet.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    discriminator_params = []
    for param in discriminator.heads.parameters():
        param.requires_grad = True
        discriminator_params.append(param)
```
(874–883) The freeze map, and the whole GAN economy: only the tiny conv
`heads` of the discriminator train (collected for their own optimizer).
The backbone stays the pretrained teacher — features are already
"diffusion-aware," so D needs almost no capacity of its own.

```python
    unet = UNet2DConditionModel.from_pretrained(...)   # third UNet copy: the student
    unet.train()
    ...
    lora_config = LoraConfig(r=args.lora_rank, target_modules=[
        "to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2", "conv1", "conv2", "conv_shortcut",
        "downsamplers.0.conv", "upsamplers.0.conv", "time_emb_proj"])
    unet = get_peft_model(unet, lora_config)
```
(885–923) The student = teacher weights + rank-64 LoRA on attention
projections, feed-forwards, resnet convs, up/down-samplers, and the time
embedding projection — broader coverage than attention-only LoRA. Only these
adapters receive gradients; "PCM" ships as a LoRA you fuse into stock SD1.5.

(925–944) Mixed-precision dtype selection; frozen parts move to fp16/bf16,
trainable student stays fp32 (asserted at 898); schedules and solver move to
device. (978–1015) xformers / TF32 / gradient checkpointing / 8-bit Adam —
optional speed knobs.

```python
    uncond_input_ids = tokenizer([""] * args.train_batch_size, ...)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]
```
(1109–1115) The empty-prompt embedding **(B, 77, 768)**, computed once —
the "second head" of classifier-free guidance (`11 §5`), needed every step
for the teacher's CFG combination.

---

## Part 4 — The training step (lines 1171–1431): the heart, line by line

### 4.1 Batch → latents

```python
                image, text = batch
                image = image.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text)
                pixel_values = image.to(dtype=weight_dtype)
```
Images **(B, 3, 512, 512)** in [−1,1]; captions → `{"prompt_embeds":
(B, 77, 768)}` via the frozen CLIP (with the empty-prompt dropout inside).
B = 20 in the shipped script.

```python
                latents = []
                for i in range(0, pixel_values.shape[0], 32):
                    latents.append(vae.encode(pixel_values[i:i+32]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
```
VAE-encode in chunks of ≤32 (memory), *sample* from the posterior (not the
mean), scale by 0.18215. `latents`: **(B, 4, 64, 64)** — everything from
here on lives in latent space.

### 4.2 Sample a position on the solver grid

```python
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=...).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
```
`topk` = 20 (the solver stride k). `index` (B,) ~ U{0..49} picks a grid
point per sample; `start_timesteps` (B,) ∈ {19,…,999} is where the student
will stand; `timesteps` = one grid step lower (clamped at 0 for index 0 —
19−20 would be −1) is where the teacher's solver step will land. Exactly
LCM's (t_{n+k}, t_n) pair on the coarse grid.

### 4.3 Phase edges and the boundary scalings

```python
                inference_indices = np.linspace(
                    0, len(solver.ddim_timesteps), num=args.multiphase, endpoint=False)
                inference_indices = np.floor(inference_indices).astype(np.int64)
                inference_indices = torch.from_numpy(inference_indices).long().to(timesteps.device)
```
The **same** edge computation as inside `ddim_style_multiphase_pred`
(duplicated code — they must stay in sync, and do). For M=2: tensor([0, 25]).

```python
                c_skip_start, c_out_start = scalings_for_boundary_conditions_online(
                    index, inference_indices)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in ...]
                c_skip, c_out = scalings_for_boundary_conditions_target(
                    index, inference_indices)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
```
Online scalings ≡ (0, 1) — no-op. Target scalings: c_skip (B,1,1,1) is the
**is-this-index-a-phase-edge** indicator from Part 1. With M=2, samples with
index 0 or 25 get c_skip=1. (The commented-out debug prints at 1232–1238 —
the authors sanity-checked exactly these tensors.)

### 4.4 Forward-noise to the grid point, sample CFG scale

```python
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
```
x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε at t = start_timesteps. Shape (B,4,64,64).

```python
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1)
```
Per-sample CFG strength w ~ U[w_min, w_max] (script: U[4,5]), broadcastable.
**Note what does *not* happen next**: w is never embedded or fed to the
student (contrast LCM's `w_embedding`); it only shapes the teacher's target
below. The student therefore distills "the teacher at a moderate guidance
level" rather than becoming a w-conditional model.

### 4.5 Online student prediction — the phased consistency function, applied

```python
                noise_pred = unet(
                    noisy_model_input, start_timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample
```
The gradient-carrying forward pass: student's ε̂_θ(x_t, t, c), shape
(B,4,64,64). `timestep_cond=None` — no guidance embedding (see 4.4).
(`encoded_text` is `{}` by now since `prompt_embeds` was popped — an
empty-dict leftover from the SDXL version of this script.)

```python
                epsilon_reconstruction_pred = noise_pred
                x0_reconstruction_pred = predicted_origin(...)
                pred_x_0 = predicted_origin(
                    noise_pred, start_timesteps, noisy_model_input,
                    noise_scheduler.config.prediction_type, alpha_schedule, sigma_schedule)
```
x̂_0 = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t. Quirk: `epsilon_reconstruction_pred` and
`x0_reconstruction_pred` are assigned and never used, and `predicted_origin`
is computed **twice** with identical arguments (1265 and 1274) — leftovers;
only `pred_x_0` matters.

```python
                model_pred, end_timesteps = solver.ddim_style_multiphase_pred(
                    pred_x_0, noise_pred, index, args.multiphase)
                model_pred = c_skip_start * noisy_model_input + c_out_start * model_pred
```
**The student's consistency output**: jump from (x_t, t) to the phase edge
s via √ᾱ_s·x̂_0 + √(1−ᾱ_s)·ε̂. `end_timesteps` (B,) are the raw edge
timesteps ({0 or 499} for M=2). The second line is the (0,1) no-op kept for
symmetry with LCM. `model_pred` (B,4,64,64) *carries gradients into the
LoRA*.

### 4.6 Prepare the adversarial samples (fake side)

```python
                adv_timesteps = torch.empty_like(end_timesteps)
                for i in range(end_timesteps.size(0)):
                    adv_timesteps[i] = torch.randint(
                        end_timesteps[i].item(),
                        end_timesteps[i].item()
                        + noise_scheduler.config.num_train_timesteps // args.multiphase,
                        (1,), ...)
```
Per sample, draw a timestep uniformly in [edge, edge + 1000/M) — i.e. a
random level **inside the phase sitting on top of the edge** (M=2: [0,500)
or [499,999)). The GAN will compare distributions *at these noise levels*,
not at t=0 — matching noise levels between real and fake is what lets a
single discriminator work across the whole trajectory. (A Python loop over
the batch — works, just unvectorized.)

```python
                real_adv = noise_scheduler.add_noise(
                    latents, torch.randn_like(latents), adv_timesteps)  # not used.
                fake_adv = noise_scheduler.noise_travel(
                    model_pred, torch.randn_like(latents), end_timesteps, adv_timesteps)
```
`fake_adv`: take the student's edge prediction (a "sample at timestep
end_t") and **travel it forward** to adv_t with fresh noise — Part 5 shows
the ᾱ-ratio identity. Gradients flow: fake_adv is a function of the LoRA.
`real_adv` here is dataset latents noised to adv_t — and the authors'
own comment says **"not used"**: it is overwritten in the D-step below, where
"real" is redefined as the noised *distillation target*. So despite this
line, the discriminator in this script never sees actual data latents —
honest flag; the GAN pits *student output* against *teacher-solver target*,
making it an adversarial **consistency** loss rather than an adversarial
data-matching loss. (The paper describes both regimes; this code implements
the consistency-GAN one.)

### 4.7 Teacher + CFG → one solver step → x_prev (the trajectory anchor)

```python
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.float(), start_timesteps,
                            encoder_hidden_states=prompt_embeds.float()).sample
                        cond_pred_x0 = predicted_origin(...)
                        if args.not_apply_cfg_solver:
                            uncond_teacher_output = cond_teacher_output
                            uncond_pred_x0 = cond_pred_x0
                        else:
                            uncond_teacher_output = teacher_unet(
                                ..., encoder_hidden_states=uncond_prompt_embeds.float()).sample
                            uncond_pred_x0 = predicted_origin(...)
```
Two frozen-teacher passes at the same (x_t, t): conditional and
unconditional ε̂ (each (B,4,64,64)), both converted to x̂_0. **The CFG
switch**: with `--not_apply_cfg_solver`, uncond := cond, so the guidance
term below cancels to zero and the student distills the *unguided* PF-ODE —
the recipe that (together with 10% empty prompts) restores usable
inference-time CFG. This flag is PCM's answer to flaw 2, selectable per run.

```python
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_teacher_output + w * (
                            cond_teacher_output - uncond_teacher_output)
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)
```
Classifier-free guidance, `11 §4`, in **both** x_0-space and ε-space — and
since x̂_0 is affine in ε̂, these two extrapolations are the *same* CFG
applied consistently (combining them back through the DDIM formula is
self-consistent). Formulation note: `cond + w·(cond − uncond)` is the
"Imagen form" — equivalent to the usual `uncond + w'·(cond − uncond)` with
w' = w+1, so w∈[4,5] here ≈ guidance scale 5–6 in standard convention
(the arg help text says exactly this). Then **one skipping-step DDIM
solver step** down the grid: x_prev ≈ the true (guided) trajectory point at
`timesteps` = start − 20. This is the only place the teacher acts; the
whole target branch hangs off x_prev. All under `no_grad` — the teacher is
an oracle, never a gradient path.

### 4.8 Target branch — the student's own jump from one step below

```python
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_noise_pred = unet(
                            x_prev.float(), timesteps,
                            timestep_cond=None,
                            encoder_hidden_states=prompt_embeds.float()).sample
                    pred_x_0 = predicted_origin(target_noise_pred, timesteps, x_prev, ...)
                    target, end_timesteps = solver.ddim_style_multiphase_pred(
                        pred_x_0, target_noise_pred, index, args.multiphase)
                    target = c_skip * x_prev + c_out * target
```
The **same online `unet`** evaluated at (x_prev, t−20), no grad — there is
no EMA/teacher-student copy of the consistency net in this script
(`update_ema` is dead); the target is the stop-gradient of the current
student, the "no-EMA" simplification. Its jump to the phase edge is computed
with the *same* `index` (both t and t−20 lie in the same phase except
exactly at edges — where the next line takes over):
**`target = c_skip·x_prev + c_out·target`** is the phased boundary
condition firing. If `index` is an edge (c_skip=1), x_prev *is* the edge
point x_{s_{m-1}}, and the target becomes the identity — anchoring the
recursion. Everywhere else (c_skip=0) it's pure self-consistency: "your
jump from t must equal your jump from t−20 along the teacher's trajectory."
Chained over random indices, every point in a phase gets pinned, through
its neighbors, to the phase edge. Note `end_timesteps` is recomputed here
(same values as in 4.5 — the mapping depends only on `index`).

### 4.9 The alternating GAN: discriminator step…

```python
                if global_step % 2 == 0:
                    optimizer_discriminator.zero_grad(set_to_none=True)
                    real_adv = noise_scheduler.noise_travel(
                        target.float(), torch.randn_like(latents), end_timesteps, adv_timesteps)
                    loss = discriminator(
                        "d_loss", fake_adv.float(), real_adv.float(),
                        adv_timesteps, prompt_embeds.float(), 1.0)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad(set_to_none=True)
```
Even global steps train **only the discriminator**. Here `real_adv` is
(re)defined: the *target* (teacher-anchored prediction) noise-traveled to
the same adv_timesteps as the fake — so real and fake differ only in
"student jump vs teacher-solver-bootstrapped jump," at matched noise level,
matched prompt. `d_loss` (Part 6) detaches both inputs internally, so no
student gradient leaks. Note the student gets **no update at all** on even
steps — yet its full forward graph (4.5) was built anyway; wasted
compute/memory on half the steps, flagged honestly.

### 4.10 …and generator step: consistency loss + adversarial push

```python
                else:
                    if args.loss_type == "l2":
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    elif args.loss_type == "huber":
                        loss = torch.mean(
                            torch.sqrt((model_pred.float() - target.float()) ** 2
                                       + args.huber_c**2) - args.huber_c)
```
**The phased consistency loss.** Both tensors are jumps-to-the-phase-edge,
(B,4,64,64): `model_pred` from t (with grad), `target` from t−20 (stopped).
The Huber variant (used by the launch script, c=0.001) is the
pseudo-Huber √(d²+c²)−c — quadratic for |d|≪c, ~|d| for large errors, less
outlier-dominated than L2 in latent space.

```python
                    g_loss = args.adv_weight * discriminator(
                        "g_loss", fake_adv.float(), adv_timesteps, prompt_embeds.float(), 1.0)
                    loss += g_loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
```
Generator loss on the *non-detached* `fake_adv` — gradients flow back
through `noise_travel` (an affine map, so it just scales gradients by
√(ᾱ_adv/ᾱ_edge)) into `model_pred` and the LoRA. Total: L_consistency +
0.1·L_adv. Clip, step, done.

(1433–1509) End-of-step bookkeeping: checkpoint pruning/saving, and
`log_validation` twice — `cfg=1` and `cfg=7.5`, both with
`num_inference_step=args.multiphase` (sample with exactly M steps — the
whole point). The logging branch keys off `(global_step − 1) % 2` because
`global_step` was already incremented: even *finished* steps log `d_loss`,
odd ones log `loss_cm = loss − g_loss` and `g_loss` separately.

---

## Part 5 — Companion: `noise_travel` in `scheduling_ddpm_modified.py`

The local scheduler file is a stock diffusers `DDPMScheduler` with one
method added (lines 526–554) — the reason the file exists:

```python
    def noise_travel(self, current_samples, noise, current_timesteps, target_timesteps):
        alpha_prod_target = alphas_cumprod[target_timesteps].flatten()
        alpha_prod_current = alphas_cumprod[current_timesteps].flatten()
        alpha_prod = alpha_prod_target / alpha_prod_current
        sqrt_alpha_prod = alpha_prod ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5
        ...
        noisy_samples = sqrt_alpha_prod * current_samples + sqrt_one_minus_alpha_prod * noise
```
Forward-noise **from timestep s to timestep t** (s < t) rather than from 0:
x_t = √(ᾱ_t/ᾱ_s)·x_s + √(1 − ᾱ_t/ᾱ_s)·ε. Why the ratio is right: if
x_s = √ᾱ_s·x_0 + √(1−ᾱ_s)·ε′, substituting gives x_t with total noise
variance (ᾱ_t/ᾱ_s)(1−ᾱ_s) + (1−ᾱ_t/ᾱ_s) = 1−ᾱ_t — the correct marginal.
This is **exactly the ᾱ-ratio identity of our strided-schedule work**
(`04 §"ᾱ-ratio respacing"`, Improved-DDPM §4, our `DDPM.sample(num_steps=…)`
from HW Q7): a sub-chain of the forward process between two arbitrary
timesteps is again a Gaussian forward step with ᾱ' = ᾱ_t/ᾱ_s. There we used
it to *sample* on a coarse grid; here PCM uses it to relocate GAN
comparisons to matched within-phase noise levels. The `while … unsqueeze`
loops are shape-padding to (B,1,1,1); the commented-out
`assert current_timesteps < target_timesteps` hints at the intended
direction (forward only — the ratio would exceed 1 otherwise and the sqrt
would NaN).

---

## Part 6 — Companion: `discriminator_sd15.py`

```python
def modified_forward(self, sample, timestep, encoder_hidden_states, ...):
    ...
    output_features = []
    for downsample_block in self.down_blocks:
        sample, res_samples = downsample_block(...)
        output_features.append(sample)
    if self.mid_block is not None:
        sample = self.mid_block(...)
        output_features.append(sample)
    for i, upsample_block in enumerate(self.up_blocks):
        sample = upsample_block(...)
        output_features.append(sample)
    return output_features
```
(16–345) A copy of diffusers' `UNet2DConditionModel.forward` with two edits:
it **collects the hidden state after every down block, the mid block, and
every up block** (SD1.5: 4 + 1 + 4 = 9 feature maps, channels
[320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320], spatial sizes walking
the 64→32→16→8→…→64 ladder), and it **returns those features instead of the
final ε** — the `conv_out` post-processing is deleted. The 300-odd lines
above the loop are the untouched timestep-embedding / attention-mask /
added-condition plumbing from upstream; skim them. Because `timestep` and
`encoder_hidden_states` are consumed normally, the extracted features are
noise-level-aware and **prompt-aware** — the discriminator is conditional on
both, for free.

```python
class DiscriminatorHead(nn.Module):
    def __init__(self, input_channel, output_channel=1):
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(inplace=True),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = ...  # same block
        self.conv_out = nn.Conv2d(input_channel, output_channel, 1, 1, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
```
Per-feature critic: conv → residual conv → 1×1 to a **1-channel logit map**
(a patch-GAN: one real/fake score per spatial location, shape (B,1,h,w)).
The authors' own comment flags the paper-vs-code gap: LeakyReLU here, GELU
in the paper — an honest, memory-motivated substitution.

```python
class Discriminator(nn.Module):
    def __init__(self, unet, num_h_per_head=4,
                 adapter_channel_dims=[320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320]):
        self.unet = unet
        self.heads = nn.ModuleList([
            nn.ModuleList([DiscriminatorHead(c) for _ in range(self.num_h_per_head)])
            for c in adapter_channel_dims])
```
4 independent heads on each of the 9 features = **36 lightweight critics**
riding one frozen teacher forward pass. The channel list must match the
9 collected features (asserted in `_forward`).

```python
    def d_loss(self, sample_fake, sample_real, timestep, encoder_hidden_states, weight):
        fake_outputs = self._forward(sample_fake.detach(), ...)
        real_outputs = self._forward(sample_real.detach(), ...)
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (torch.mean(weight * torch.relu(fake_output.float() + 1))
                   + torch.mean(weight * torch.relu(1 - real_output.float()))
                   ) / (self.head_num * self.num_h_per_head)
```
**Hinge GAN loss** (the SAGAN/BigGAN standard): D wants real logits > +1
and fake logits < −1; gradients vanish once a sample is confidently
classified (the relu clamps). Both inputs `.detach()`-ed — the D step can
never move the student. Averaged over all 36 heads.

```python
    def g_loss(self, sample_fake, timestep, encoder_hidden_states, weight):
        for fake_output in fake_outputs:
            loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / ...
```
Generator side: push fake logits up. Quirk: the standard hinge-GAN generator
loss is `−mean(fake_output)` (unsaturated); `relu(1 − fake)` instead **stops
pushing once a patch's logit exceeds 1** — a conservative, self-limiting
variant that plays gently with the consistency loss it's added to.
`feature_loss` (an L2 feature-matching alternative) exists but is never
called from the trainer.

---

## Part 7 — Validation = the punchline: plain DDIM, M steps

```python
        scheduler=DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
            clip_sample=False,   # important. DDIM will apply True as default which causes inference degradation.
            set_alpha_to_one=False,
        ),  # DDIM should just work well. See our discussion on parameterization in the paper.
```
(139–226, `log_validation`) The trained LoRA is merged into a **stock
`StableDiffusionPipeline` with a stock `DDIMScheduler`** and run with
`num_inference_steps = args.multiphase`. No custom sampler ships with
training — because the consistency function *was parameterized as a DDIM
step* (Part 2), M-step sampling is literally DDIM on a trailing grid whose
giant steps the student has been trained to make exact. Two honest caveats:
(i) `clip_sample=False` and `set_alpha_to_one=False` are required to match
the training-time parameterization (the solver's ᾱ_prev[0] = ᾱ_0 quirk from
Part 2); (ii) the trailing M-step grid (M=2: t = {999, 499}) matches the
training phase tops closely but not to the digit (training's top grid point
is 999 and the M=2 mid-edge is 499 — here it aligns exactly; for M=4 the
trailing grid {999, 749, 499, 249} sits near but not on the trained edges
{739+20…, 499+20…, 239+20…} region). Multi-step sampling is deterministic —
no re-noising anywhere — which is Methodology flaw 1, fixed and visible.

---

## Quirks and dead code (collected, so you trust the rest)

1. `guidance_scale_embedding`, `update_ema`, `RMSprop`, `feature_loss`,
   `epsilon_reconstruction_pred`, `x0_reconstruction_pred` — all defined
   /imported and never used. `predicted_origin` runs twice identically
   (lines 1265, 1274).
2. `real_adv` from real data latents (line 1300) is dead ("# not used.");
   the GAN's "real" is the noised **distillation target**, so this script's
   adversarial loss compares student vs teacher-anchored prediction, not
   student vs data.
3. Strict every-other-step G/D alternation via `global_step % 2`: on D
   steps the student forward graph is built but unused; with
   `gradient_accumulation_steps > 1` the parity logic would misalign
   (global_step only advances on sync).
4. Dataset path hardcoded (`/mnt/data/wangfuyun/cc3m`); a missing caption
   `.txt` makes `__getitem__` loop forever.
5. `ddim_style_multiphase_pred` overwrites its `timestep_index` argument
   with edge indices mid-function; phase-edge computation is duplicated
   between solver and trainer.
6. Solver's ᾱ_prev at the bottom grid point is ᾱ_0, not 1 — matched by
   `set_alpha_to_one=False` at inference.
7. Boundary condition is enforced via the *target* (c_skip indicator), not
   via network parameterization; online c_skip/c_out are a (0,1) no-op kept
   for LCM code symmetry.

---

## What to carry forward (PCM → everything else)

1. **Phases = many small consistency models.** Cutting the trajectory at M
   edges and mapping each phase to its own low-noise edge turns "1 step or
   re-noise" into deterministic M-step sampling. M=1 is LCM; M=grid-size is
   the teacher's DDIM.
2. **Parameterize the consistency function as a DDIM step**
   (√ᾱ_s·x̂_0 + √(1−ᾱ_s)·ε̂, our `10 §1.2` identity) and your distilled
   student runs in stock samplers — training a *sampler-compatible* object,
   not a new architecture.
3. **Boundary conditions can live in the target**: an indicator c_skip that
   swaps the target for the identity exactly at edges anchors the whole
   bootstrapped recursion.
4. **The ᾱ-ratio identity keeps paying rent**: our strided-schedule
   respacing and PCM's `noise_travel` are the same equation — sub-chains of
   the forward process are forward processes with ᾱ' = ᾱ_t/ᾱ_s.
5. **CFG is a training-data decision in distillation** (`11 §4`): distill
   the guided ODE and w is baked in (inference CFG breaks); distill the
   unguided ODE + condition dropout and inference CFG comes back.
6. **When L2 supervision is too weak, match distributions**: re-noise
   prediction and target to a common level, reuse a frozen pretrained UNet
   as discriminator features, hinge loss, tiny heads, alternate steps.
