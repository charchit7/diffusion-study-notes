# InstaFlow, Line by Line — One-Step Stable Diffusion via Rectified Flow

*A beginner's reading companion to the InstaFlow repo
(`distillation/InstaFlow/`, ICLR 2024, arXiv 2309.06380). Every line of
`code/pipeline_rf.py` — the rectified-flow inference pipeline — plus a short
tour of `code/rf_lora.py`. The repo ships INFERENCE code only; the training
methodology below is reconstructed from the repo README and the paper.
Concepts link to our notes: straight-line flows = `06 Parts C–F`,
Euler = `06 §A1`, CFG = `11`, and we built the Reflow mechanism ourselves at
toy scale in KAIST Assignment 3.*

**The repo in one sentence:** take Stable Diffusion's UNet, re-train it (via
Reflow) to be a *velocity field* whose noise→image trajectories are nearly
straight lines, then distill — after which "sampling" is a single Euler step,
`image_latent = noise + v(noise)`, and the whole pipeline file is just SD
plumbing wrapped around a ten-line Euler loop.

---

## Methodology — the part that matters

### The problem: SD's probability flow is curved

Stable Diffusion generates by numerically solving an ODE/SDE from noise to
data (notes 10). Even with the best solvers (25-step DPM-Solver), you pay ~25
UNet forward passes per image. Why can't you take ONE giant step? Our notes
06 §A1 answered this exactly:

> **Fact 1/2 (06 §A1): if the trajectory is a straight line, velocity is
> constant, and Euler's method is EXACT even with one giant step.**
> Conversely, error comes only from *curvature*.

SD's flow is curved. Not because each training pair's path is curved — in
flow matching every *conditional* path is a straight line (06 Part C) — but
because the *marginal* field the network learns is the **average** of all
straight arrows passing through a point (06 Part D), and averaging straight
lines with different directions bends the flow (06 Part E, closing
paragraph). One Euler step on a curved flow lands far from the data manifold:
blurry, washed-out images.

### Why naive distillation fails (and what Reflow fixes)

The obvious shortcut — "just train a student network to map noise `X0`
directly to the teacher's output `X1` in one step" — has a hidden trap, and
understanding it is the paper's core insight.

Under an *independent* coupling (any noise may pair with any image — which is
what vanilla diffusion training uses), the same region of noise space is
associated with many different images. A one-step student trained with an L2
loss on such pairs learns the **conditional average** of its targets (our
notes 06 §A3: "the best guess is the average"). Averages of images are gray
mush. Even with a perceptual loss, the student is asked to fit a
fundamentally one-to-many map with a one-to-one function — it must fail
somewhere. The paper measures this directly: progressive-distillation SD gets
FID 37.2 in one step; InstaFlow gets 23.3. Straightening first is worth 14
FID points.

Rectified Flow's Reflow operation (Liu et al. 2022, arXiv 2209.03003) fixes
the coupling *before* distilling:

1. **Simulate the teacher's ODE** to obtain *coupled* pairs
   `(X0, X1 = ODE(X0))`. Because the ODE is deterministic, this coupling is a
   *function*: each noise maps to exactly ONE image. No averaging conflict
   remains.
2. **Retrain the velocity field on straight lines between these couples**:
   `Xt = t·X1 + (1−t)·X0`, target velocity `X1 − X0` (identical training loop
   to 06 Part E — only the *source of the pairs* changed). The result,
   "2-Rectified Flow", provably transports noise to (approximately) the same
   image distribution but along straighter trajectories with lower transport
   cost. Iterating gives 3-RF, 4-RF, … each straighter than the last;
   InstaFlow stops at 2.

We implemented exactly this mechanism at toy scale in the KAIST Reflow
assignment
(`kaist/Diffusion-2025-Assignment3-Flow/flow_matching_solved.ipynb`,
"Rectified Flow: Making Paths as Straight as Possible"): its
`RectFlowDataset` stores paired endpoints where `X0[i]` is the exact prior
sample whose trajectory under the trained flow ended at `X1[i]`, and the
reflow training loop regresses `v_θ(Xt, t)` onto `X1 − X0` over those
couples. InstaFlow is that same loop, scaled from 2D points to a 0.9B-param
text-conditioned UNet. The theory backdrop is notes
`06_flow_matching_from_zero.md` Parts C–F: Part C builds the straight
interpolant and its constant velocity, Part D shows least-squares learns the
marginal field, Part E names Reflow as "re-training on the model's own
(start, end) pairs … the road to 1-step generation", and Part F places it all
in the DDPM family.

### The InstaFlow pipeline: three stages (README "Method" + paper §3)

**Stage 1 — Generate coupled (text, noise, image) triplets from the
teacher.** Take pre-trained SD 1.5 and a subset of text prompts from
laion2B-en. For each prompt, sample `X0 ~ N(0, I)` in latent space
(shape `(4, 64, 64)`), run the teacher's ODE — 25-step DPM-Solver with CFG
scale 6.0 (later runs used 5.0 to avoid over-saturation) — to get `X1`. Store
the triplet `(T, X0, X1)`. ~1.6M pairs for the reflow stage and another 1.6M
for distillation. Note that CFG is applied *while generating the pairs*, so
guidance is partially baked into the coupling — this is why the 2-RF
checkpoint wants only `guidance_scale ≈ 1.5` at inference instead of SD's
7.5 (code README: "optimal range is [1.0, 2.0]").

**Stage 2 — Text-conditioned Reflow → 2-Rectified Flow.** Initialize `v_θ`
from SD 1.5's UNet and minimize

    L = E over (T, X0, X1), t~U[0,1] of  || (X1 − X0) − v_θ(Xt, t | T) ||²
        with  Xt = t·X1 + (1−t)·X0

— literally 06 Part E's training algorithm with coupled pairs and a text
condition. Time runs **0 = noise → 1 = image** (RF convention, same as our
notes; opposite to DDPM). This retraining straightens the marginal flow:
because each `X0` now has a unique partner `X1`, arrows through a point stop
disagreeing, the average stops bending, and (06 §A1 Fact 1) a nearly straight
flow is nearly exactly integrable by ONE Euler step. Already at this stage
the model generates decent images in 2–8 steps. Cost: ~75 A100 GPU days
(11.2 + 64, two sub-stages of 70k + 25k iterations).

**Stage 3 — Distill 2-RF into a true one-step model (InstaFlow).** Freeze the
straightened coupling `(X0, X1' = ODE_2RF(X0))` and train a copy of the 2-RF
UNet so that a *single* Euler step reproduces the endpoint:

    L = E over (T, X0) of  D( ODE_2RF(X0 | T),  X0 + v_θ(X0 | T) )

where `D` is a differentiable image similarity — first L2, then **LPIPS**
(perceptual distance computed on decoded images), which "significantly
improves" the one-step FID. Distillation is cheap conceptually because the
map being fit is now *nearly one-to-one and nearly linear*: the student only
has to absorb the small residual curvature, not resolve a many-to-many
assignment. Cost: ~108 A100 GPU days (54.4 L2 + 53.6 LPIPS). Total pipeline:
199 A100 GPU days — the README's headline "merely supervised training".

The README stresses that **reflow and distillation are orthogonal
techniques**: reflow changes *which* noise goes with which image
(the coupling); distillation compresses the solver. Doing distillation
*after* straightening is the key ordering — the paper's ablation
(direct SD distillation FID 37.2 vs 2-RF distillation 23.3 on MS COCO
2017-5k) is the quantitative form of "don't ask a one-step student to average
a curved, many-to-many flow." Final results: InstaFlow-0.9B, FID 23.3 in
~0.09s/image; a wider 1.7B variant reaches 22.4; on MS COCO 2014-30k,
FID 13.1 at 0.09s beats StyleGAN-T (13.9 at 0.1s).

**What the shipped checkpoints are:** `XCLiu/2_rectified_flow_from_sd_1_5` =
Stage-2 output (few-step, supports CFG and negative prompts);
`XCLiu/instaflow_0_9B_from_sd_1_5` = Stage-3 output (call with
`num_inference_steps=1, guidance_scale=0.0`; no negative-prompt support —
CFG was consumed during pair generation and distillation). Both are plain SD
1.5-shaped UNets — which is why the pipeline below is 95% stock
StableDiffusionPipeline and 5% new math, and why vanilla SD LoRAs/ControlNets
still bolt on.

---

## `code/pipeline_rf.py` — the walkthrough

The file is a copy of diffusers 0.21.4's `StableDiffusionPipeline` with the
denoising loop surgically replaced. We go top to bottom; stock-SD plumbing is
grouped and summarized (read once, then trust), the rectified-flow surgery is
line by line.

### Lines 1–39 — License and imports

```python
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
```
Standard-library helpers (`inspect` is used later to sniff a scheduler's
signature), type hints, PyTorch, and the three CLIP pieces from HuggingFace
`transformers`: the image processor (feeds the safety checker), the text
encoder (prompt → embeddings), and the tokenizer (string → token ids).

```python
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (deprecate, logging, replace_example_docstring)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
```
The diffusers toolbox: `AutoencoderKL` is the VAE (pixels ↔ latents),
`UNet2DConditionModel` is the network that will play the role of our velocity
field `v_θ`, the three loader mixins add `.load_lora_weights()` /
textual-inversion / single-`.ckpt` loading for free, `randn_tensor` is a
device- and generator-aware `torch.randn`, and `DiffusionPipeline` is the
base class that provides `from_pretrained`, `.to(device)`, model
registration, CPU offload, and the progress bar. Note
`KarrasDiffusionSchedulers` is imported only as a *type annotation* — spoiler
for the punchline below: **this pipeline never calls the scheduler**.

```python
logger = logging.get_logger(__name__)
```
Diffusers' logger, used for the truncation and safety-checker warnings.

### Lines 42–53 — `rescale_noise_cfg` (dead code — flagged honestly)

```python
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
```
A CFG-overexposure fix from arXiv 2305.08891: after guidance amplifies the
prediction, rescale its per-sample standard deviation back toward the
text-conditional one, then blend by `guidance_rescale`. **Quirk: in this file
the function is defined but never called.** `__call__` accepts a
`guidance_rescale` argument (line 527) and then ignores it — both are
leftovers from the stock SD pipeline this file was copied from. If you pass
`guidance_rescale=0.7` expecting an effect, you get nothing.

### Lines 56–92 — Class declaration

```python
class RectifiedFlowPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
```
The one new class. The docstring says it plainly: "Pipeline for text-to-image
generation using Rectified Flow and Euler discretization … based on
StableDiffusionPipeline from the official Diffusers library (0.21.4)."
Multiple inheritance = base pipeline machinery + three loading conveniences.

```python
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
```
Metadata read by `DiffusionPipeline`: in what order to shuttle submodels
between CPU and GPU when offloading is enabled (matches the order they're
used during a generation), which components may be `None`, and which never
get offloaded.

### Lines 93–181 — `__init__`: register the parts, patch old configs

```python
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler,
                 safety_checker, feature_extractor, requires_safety_checker=True):
        super().__init__()
```
The seven components of a Stable Diffusion checkpoint, injected by
`from_pretrained` (which reads the checkpoint's `model_index.json` and builds
each one). Nothing rectified-flow-specific yet.

Then four defensive blocks, all stock SD, all "fix ancient checkpoint
configs and warn" — grouped here, read once:

- **`steps_offset != 1` patch (106–118)** and **`clip_sample` patch
  (120–131)**: old scheduler configs stored on the Hub had wrong defaults;
  the code emits a deprecation warning and hot-patches
  `scheduler._internal_dict`. Ironic given the scheduler is never used, but
  harmless.
- **Safety-checker warnings (133–147)**: warn if you disabled it; error if
  you kept it but forgot the `feature_extractor` it needs.
- **`sample_size < 64` patch (149–168)**: pre-0.9.0 SD checkpoints
  mis-declared the UNet's latent resolution as 32; patch it to 64.

```python
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=safety_checker,
            feature_extractor=feature_extractor)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
```
`register_modules` is what makes `pipe.to("cuda")` and `save_pretrained` see
the submodels. `vae_scale_factor`: SD's VAE has 4 resolution blocks, so
2³ = **8** — a 512×512 image lives as a 64×64 latent. `VaeImageProcessor`
handles the final tensor→PIL conversion including the `[−1,1] → [0,1]`
denormalize.

### Lines 183–210 — VAE slicing/tiling toggles

Four two-line methods (`enable_vae_slicing`, `disable_vae_slicing`,
`enable_vae_tiling`, `disable_vae_tiling`) that just forward to the VAE:
decode in batch-slices or spatial tiles to save memory on big
batches/resolutions. Pure pass-throughs; no RF content.

### Lines 212–240 — `_encode_prompt` (deprecated shim)

Old API kept for backwards compatibility: calls the new `encode_prompt`
(next) and re-concatenates its `(positive, negative)` tuple into the single
tensor the old API returned — note the order `[negative, positive]`
(`torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])`), the same
uncond-first order used in the CFG batch later. Skim and move on.

### Lines 242–399 — `encode_prompt`: text → embeddings (stock SD, but shapes matter)

This is the standard SD prompt encoder, copied verbatim. The flow, with
shapes for batch size B and `num_images_per_prompt = n`:

```python
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
```
If the text encoder carries LoRA layers, set their global scale first.

```python
        text_inputs = self.tokenizer(prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
```
Tokenize to exactly 77 tokens (CLIP's max): pad short prompts, truncate long
ones. The block that follows re-tokenizes with `padding="longest"` purely to
*detect* truncation and print a warning naming the dropped words — a nice
UX touch, no effect on computation.

```python
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]
```
Run CLIP; take the last hidden state, **shape (B, 77, 768)** for SD 1.5.
(`attention_mask` is only passed for text encoders configured to want it —
CLIP in SD 1.5 is not.)

```python
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
```
The repeat-then-view dance duplicates each prompt's embedding for each image
requested: (B, 77, 768) → (B, 77·n, 768) → **(B·n, 77, 768)**. (Repeating
along dim 1 then viewing is an MPS-friendly equivalent of
`repeat_interleave` on dim 0.)

Lines 346–397: if CFG is on and no precomputed negatives were passed, build
the **unconditional** embeddings — from `negative_prompt` if given, else from
the empty string `""` — with type/batch-size validation, then the same
tokenize → encode → repeat dance, giving `negative_prompt_embeds` of the same
**(B·n, 77, 768)**. Returns the pair `(prompt_embeds,
negative_prompt_embeds)`.

### Lines 401–441 — Safety checker, deprecated decoder, scheduler-kwargs sniffer

```python
    def run_safety_checker(self, image, device, dtype):
```
If a safety checker exists: CLIP-embed the decoded images and let the checker
black out flagged ones; returns `(image, has_nsfw_concept)`. (The code README
warns that for reproducing paper FID you should disable it — false positives
become black images and inflate FID.)

```python
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
```
Deprecated but instructive: SD latents are stored multiplied by
`scaling_factor` (0.18215) so they're roughly unit-variance; decoding undoes
that, and the VAE's `[−1,1]` output maps to `[0,1]`. The live path does the
same thing inline at line 684.

```python
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        ...
```
Uses `inspect` to check whether `scheduler.step` accepts `eta`/`generator`.
**Quirk: pure vestige.** This pipeline never calls `scheduler.step`, and the
dict this returns is built at line 654 and then never read. The `eta`
argument of `__call__` is likewise decorative.

### Lines 443–488 — `check_inputs`: fail fast

Validation only, grouped: height/width divisible by 8 (the VAE downsamples
by 8, so odd sizes can't round-trip); `callback_steps` a positive int;
exactly one of `prompt` / `prompt_embeds`; not both `negative_prompt` and
`negative_prompt_embeds`; and if both embeds are passed directly their shapes
must match (they get concatenated for CFG).

### Lines 490–505 — `prepare_latents`: draw `X0`

```python
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
```
The starting noise shape: **(B·n, 4, H/8, W/8)** — (1, 4, 64, 64) for one
512×512 image. In RF language this is `X0`, the t=0 end of the straight line.

```python
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
```
Fresh `N(0, I)` noise (reproducible via `generator`), or user-supplied
latents (how the README's latent-interpolation video is made: interpolate two
noises, feed them in here). The `init_noise_sigma` multiply is stock-SD
scaling for sigma-based schedulers; for the scheduler config these
checkpoints ship, it's 1.0 — a no-op kept for structural fidelity.

### Lines 507–655 — `__call__` part 1: setup

```python
    @torch.no_grad()
    def __call__(self, prompt=None, height=None, width=None,
        num_inference_steps: int = 50, guidance_scale: float = 7.5, ...):
```
Inference only, so no autograd (same reason as lab1 Cell 2's
`@torch.no_grad()`). **Quirk: the defaults are inherited from SD and are
wrong for both shipped checkpoints.** The code README tells you what to
actually pass: 2-RF wants `num_inference_steps` 2–25 and `guidance_scale`
1.0–2.0; InstaFlow-0.9B wants exactly `num_inference_steps=1,
guidance_scale=0.0`. The long docstring (529–593) is stock SD.

```python
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
```
Default 64·8 = 512. Then `check_inputs` (599), and batch size inferred from
`prompt` (str → 1, list → len, else from `prompt_embeds.shape[0]`).

```python
        do_classifier_free_guidance = guidance_scale > 1.0
```
CFG switch, same convention as SD (w=1 ⇒ formula collapses to the
conditional prediction, so skip the double pass). Consequence worth knowing:
for one-step InstaFlow, `guidance_scale=0.0` and `1.0` behave *identically* —
CFG is off either way and the scale value is never multiplied in. The
README's `0.0` is a signal to the reader, not a different computation.

```python
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(...)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
```
Encode, then stack **uncond first, cond second** into one **(2·B·n, 77,
768)** tensor so both CFG branches ride one UNet forward pass (notes 11).

```python
        # 4. Prepare timesteps
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]
```
**The first rectified-flow line in the file, and the timestep-convention
gymnastics live here.** Unpack it. Let `N = num_inference_steps` and let
`s = i/N` be RF time (0 = noise → 1 = image, our notes-06 convention). The
code computes `t_unet = 1000·(1 − s)`:

- `N=1` → `timesteps = [1000.0]`
- `N=2` → `[1000.0, 500.0]`
- `N=25` → `[1000.0, 960.0, …, 40.0]`

Three things to notice, honestly:

1. **The UNet keeps SD's clock.** SD's UNet was pre-trained with integer
   timesteps 0..999 where *large = noisy*. Reflow fine-tuning kept that
   convention (queried at continuous values) so the network's time embedding
   starts from meaningful pre-trained features: RF time `s` is mapped to
   `1000(1−s)` before hitting the UNet. The Euler loop below marches `s`
   forward 0→1 while the number fed to the UNet counts *down* 1000→1000/N.
2. **`t_unet = 1000` is outside SD's original 0..999 grid**, and the values
   are floats (960.0), not integers. Both are fine mechanically — the
   sinusoidal timestep embedding accepts any float (same fact we noted in 06
   Part E for our own UNet) — and the reflow fine-tune trained the model at
   exactly these query points, so this is the convention the checkpoint
   *expects*. But do not feed this pipeline a vanilla SD checkpoint: it would
   see timesteps it half-recognizes and a velocity target it never learned.
3. **These are left endpoints; the grid never touches `t_unet = 0`** (RF
   time never reaches 1 inside the loop — the final Euler step *lands* there).
   For `N=1` the single query is at pure noise, `s=0`: one-step generation
   evaluates the velocity field once, at the noise itself.

No scheduler involved: compare stock SD's
`self.scheduler.set_timesteps(...)` — deleted, replaced by one list
comprehension. (The training scripts are not in this repo, so the exact
training-time mapping is inferred from this inference code plus the paper's
`Xt = t·X1 + (1−t)·X0` convention.)

```python
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, ...)
```
Draw `X0`: **(B·n, 4, 64, 64)**.

```python
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        dt = 1.0 / num_inference_steps
```
First line: the unused vestige (see above). Second line: **the RF step
size** — uniform Euler steps over the unit interval, `Δ = 1/N`. For `N=1`,
`dt = 1.0`: one step covers the whole trip. This is 06 §A1 Fact 2 with the
biggest possible Δ, justified only because reflow made the path straight
(Fact 1).

### Lines 657–680 — `__call__` part 2: **the Euler loop (the whole point)**

```python
        # 7. Denoising loop of Euler discretization from t = 0 to t = 1
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
```
The comment states the RF convention outright. Compare this loop to lab1's
`EulerSimulator.step` — it is the same three-line idea wearing SD clothes.

```python
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
```
CFG batching: duplicate the state to (2·B·n, 4, 64, 64) so the uncond and
cond halves (matching the `[negative, positive]` embed order) share one
forward pass. Without CFG, pass latents straight through — the one-step
InstaFlow path takes this branch, so one image really is ONE UNet call.

```python
                vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t
```
Broadcast the scalar timestep to a per-sample vector, shape **(2·B·n,)** (or
(B·n,)): every sample shares one clock, but the UNet API wants one timestep
per batch element. Note it's a *float* tensor — continuous time, as promised.

```python
                v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample
```
**The UNet as velocity field.** Architecturally this is exactly SD 1.5's
`UNet2DConditionModel`; only the *meaning* of the output changed. Stock SD:
`eps_pred`, "which noise was added." Post-reflow: `v_pred ≈ E[X1 − X0 | Xt]`,
"the arrow from your noise toward your image" (06 Part C/D — same interface,
different target, exactly as our notes said: "the same interface as DDPM's
ε_θ(x_t, t), just a different meaning for the output"). Shape out = shape in:
**(2·B·n, 4, 64, 64)**.

```python
                if do_classifier_free_guidance:
                    v_pred_neg, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
```
CFG applied **to velocities, not noise predictions** — the same affine
extrapolation as notes 11, `v = v_neg + w·(v_text − v_neg)`, which matches
the paper's `α·v(Zt,t|T) + (1−α)·v(Zt,t|NULL)` with `α = guidance_scale`.
Chunk order follows the embed concat: first half uncond/negative, second half
text. Because guidance was already baked into the teacher pairs (Stage 1),
`w ≈ 1.5` suffices for 2-RF where raw SD needed 7.5 — over-guiding a model
whose training data was *already* guided causes the over-saturation the paper
mentions.

```python
                latents = latents + dt * v_pred
```
**The most important line in the repository.** Euler's method, verbatim from
06 §A1 Fact 2: where you'll be = where you are + arrow × time. No scheduler
`step()`, no `alpha_prod` gymnastics, no posterior variance, no noise
re-injection — compare a DDPM/DDIM `step` and enjoy. With `N=1` this line
runs once and reads `image_latent = noise + v(noise, prompt)`: **the entire
one-step generation path.** Everything InstaFlow's 199 GPU days bought is the
right to trust this single addition — straight trajectory ⇒ constant velocity
⇒ Euler exact in one giant step (06 §A1 Fact 1). Also note the update uses
`latents` (batch B·n), not `latent_model_input` (2·B·n) — the CFG duplicate
was only ever a compute trick.

```python
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
```
Progress-bar and user-callback bookkeeping, copied from stock SD. The
`scheduler.order` references (order = 1 for first-order schedulers) are the
scheduler's only appearances inside the loop — cosmetic.

### Lines 683–703 — `__call__` part 3: decode and return

```python
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
```
Latents `X1` (B·n, 4, 64, 64) → undo the 0.18215 scaling → VAE decode →
pixels **(B·n, 3, 512, 512)** in `[−1, 1]` → safety check. Or hand back raw
latents if asked.

```python
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
```
Per-image `[−1,1] → [0,1]` denormalize — skipped for flagged images, which
the safety checker already replaced with black (already in `[0,1]`). Then
tensor → numpy/PIL per `output_type`.

```python
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
```
Release CPU-offload hooks, and return in SD's standard envelope —
`result.images[0]` is your picture.

**Scorecard vs stock StableDiffusionPipeline** — the entire diff is:
(a) the `timesteps` list comprehension replacing `scheduler.set_timesteps`,
(b) `dt = 1/N`, (c) `v_pred` naming + CFG on velocities,
(d) `latents = latents + dt * v_pred` replacing `scheduler.step(...)`.
Four edits. The scheduler object, `eta`, `extra_step_kwargs`,
`guidance_rescale`, and `rescale_noise_cfg` all remain as fossils of the
original — a nice reminder that "rectified flow inference" is a *simpler*
special case of what the SD machinery already does, not a more complex one.

---

## `code/rf_lora.py` — briefly: bolting community styles onto a one-step model

This script demonstrates the README's "compatible with pre-trained LoRAs"
claim. Why it works at all: InstaFlow's UNet is a *fine-tune* of SD 1.5's
UNet (same architecture, nearby weights), so weight *deltas* that restyle SD
approximately restyle InstaFlow too.

```python
def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
```
Task-vector arithmetic on raw state dicts: add a scaled weight-difference
dictionary into the UNet. `strict=False` tolerates keys the checkpoint
doesn't cover.

```python
def load_hf_hub_lora(pipe_rf, lora_path='Lykon/dreamshaper-7', ...):
    ...
    dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]
```
Despite the "lora" name, this path handles *full fine-tuned* checkpoints
(e.g., DreamShaper): download base SD 1.5 and the styled model, form
`dW = W_styled − W_SD15` per tensor, then `W_instaflow + dW`. It also swaps
in the styled model's VAE and text encoder wholesale. This is the "InstaFlow
+ dreamshaper-7" combo the HF Space demo uses.

```python
def load_civitai_lora(pipeline, checkpoint_path, multiplier, device, dtype):
    ...
    curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
```
The true low-rank path, for CivitAI `.safetensors` LoRAs: parse key names
like `lora_unet_..._proj.lora_down.weight`, walk the module tree attribute by
attribute to find the target layer (the odd `while len(layer_infos) > -1:`
is an always-true loop escaped by `break` — janky, works), then merge
`ΔW = α/rank · (up @ down)` directly into `weight.data` (with a
squeeze/unsqueeze dance for conv layers, whose weights are 4-D). Merging into
weights (rather than keeping LoRA as runtime adapters) keeps one-step
inference at exactly one unmodified UNet call.

```python
def main(args):
    if args.instaflow:
        pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", ...)
    else:
        pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/2_rectified_flow_from_sd_1_5", ...)
```
Same pipeline class for both checkpoints — only the call differs:

```python
    if args.instaflow:
        images = pipe(args.prompt, num_inference_steps=1, guidance_scale=1.0, ...).images
    else:
        images = pipe(prompt=args.prompt, negative_prompt="painting, unreal, twisted",
                      num_inference_steps=n_step, guidance_scale=1.5, ...).images
```
InstaFlow: 1 step, guidance 1.0 (= CFG off; identical to the README's 0.0 —
see the `> 1.0` switch above), no negative prompt. 2-RF: default 25 steps,
guidance 1.5, negative prompts fine. The rest of `main` is filename
plumbing for saving the PNG.

---

## What to carry forward (InstaFlow → everything else)

1. **Straighten, then distill.** Distilling a curved flow asks a one-step
   student to fit a many-to-many map — it learns a blur (06 §A3: least
   squares learns the average). Reflow first makes the noise→image coupling
   deterministic and the map nearly linear; distillation then only mops up
   residual curvature. 37.2 → 23.3 FID is the price of skipping this.
2. **Reflow = the FM training loop with the pairs swapped**: same
   `‖(X1−X0) − v(Xt,t)‖²` loss (06 Part E), but `(X0, X1)` are *coupled*
   endpoints of the teacher's own ODE, not independent draws. We built
   exactly this in KAIST Assignment 3's `RectFlowDataset`.
3. **Straight ⇒ Euler exact ⇒ `x1 = x0 + v(x0)`** (06 §A1 Facts 1–2). The
   entire sampler is `latents = latents + dt * v_pred`; one-step generation
   is that line with `dt = 1.0`.
4. **Conventions are gymnastics, contracts are forever.** RF time `s ∈ [0,1]`
   (0 = noise) is fed to SD's UNet as `1000·(1−s)` — continuous, hitting
   1000 (outside the original 0..999 grid), never hitting 0 — because the
   checkpoint was fine-tuned to expect exactly that. Interface unchanged,
   semantics retrained: `ε_θ` became `v_θ` without touching architecture.
5. **CFG extrapolates whatever the network predicts** — noise in SD,
   velocity here — and guidance baked into the *training pairs* means far
   less guidance needed at inference (1.5 vs 7.5), and none at all for the
   distilled one-step model.
6. Reading pipelines: diff against the stock one. Here the diff is four
   edits, and the leftover fossils (`scheduler`, `eta`, unused
   `rescale_noise_cfg`) tell you the true story — RF inference *removes*
   machinery.

**References:** InstaFlow: Liu, Zhang, Ma, Peng, Liu, "InstaFlow: One Step
is Enough for High-Quality Diffusion-Based Text-to-Image Generation," ICLR
2024, arXiv 2309.06380. Rectified Flow / Reflow: Liu, Gong, Liu, "Flow
Straight and Fast: Learning to Generate and Transfer Data with Rectified
Flow," arXiv 2209.03003. Our notes: `06_flow_matching_from_zero.md`
(Parts A, C–F), `10_ddim_dpmsolver_from_zero.md`, `11_guidance_from_zero.md`;
KAIST `Diffusion-2025-Assignment3-Flow/flow_matching_solved.ipynb`.
