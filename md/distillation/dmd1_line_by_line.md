# DMD1, Line by Line — The Original Distribution Matching Recipe (CIFAR-10, EDM)

*A beginner's reading companion to the **unofficial** DMD reimplementation
(`distillation/dmd1/`, devrimcavusoglu/dmd), which reproduces "One-step
Diffusion with Distribution Matching Distillation" (Yin et al., arXiv
2311.18828, CVPR 2024) on CIFAR-10 with an NVLabs EDM teacher. This is the
**prequel** to our DMD2 chapter (`dmd2_line_by_line.md`): the original
recipe, including everything DMD2 later removed — above all the
pre-generated paired dataset and the LPIPS regression loss on it. Because
this repo is a community reimplementation (not MIT/Adobe's code), we flag
every place it may deviate from the paper. Concepts link to our notes:
score = −ε̂/σ is `10 §3`, the reverse-KL derivation is `dmd2_line_by_line.md
§M2–M3`, deterministic noise→image couples are
`instaflow_line_by_line.md`.*

**The codebase in one sentence:** a copy of the EDM CIFAR-10 UNet is trained
to map scaled noise to a finished image in ONE call, pulled by two forces —
a distribution-matching gradient (frozen teacher score minus a continuously
retrained "fake" score, evaluated on re-noised generator samples) and an
LPIPS regression loss that pins the generator to a 100k-pair dataset of
(noise, teacher-ODE-output) couples manufactured offline by the teacher's
own 18-step sampler.

---

## Methodology

### M0 — Where this sits, and how honest we can be

Read order: this chapter and `dmd2_line_by_line.md` share one core (the
DMD gradient); this one adds what DMD1 had and DMD2 deleted. DMD1's
generator objective is

    L_G = L_KL + λ_reg · L_reg,        λ_reg = 0.25,

where `L_KL` is the distribution-matching (reverse-KL) loss and `L_reg` is
a **regression loss on paired teacher samples** — LPIPS between `G(z_ref)`
and the image `y_ref` that the *teacher's ODE sampler* produced from that
same `z_ref`, precomputed for a whole dataset before training starts.
Alongside, a second network `mu_fake` (the fake score) is trained with a
denoising loss on the generator's outputs, alternating 1:1 with the
generator. That's the whole method: three networks (frozen teacher
`mu_real`, trainable critic `mu_fake`, trainable generator `G`, all three
the *same* EDM architecture initialized from the *same* teacher
checkpoint), two optimizers, two-and-a-half losses.

Honesty up front: this repo is an unofficial reproduction by two students
(the README and `goals.txt` say so plainly). They trained on ONE RTX 4090
with batch 48 instead of the paper's 7 GPUs × batch 56, sqrt-scaled the
learning rate accordingly, and did **not** reach the paper's CIFAR-10 FID
of 2.66. The README's "Assumptions" section admits the paper doesn't say
which model each hyperparameter applies to, so the repo uses the same
optimizer settings (AdamW, lr 5e-5, wd 0.01) for *all three* networks.
Wherever the code makes a choice the paper doesn't pin down, we say so.

### M1 — The reverse-KL gradient (the shared core — see DMD2 §M2 for the derivation)

DMD minimizes `E_t KL(p_fake,t ‖ p_real,t)` over noised versions of the
generator's output distribution vs the teacher's. The derivation (done
fully in `dmd2_line_by_line.md §M2`, pedigree SDS→VSD→DMD in its §M3)
gives

    ∇_θ L ≈ E_{z,t,ε} [ w_t · ( s_fake(x_t) − s_real(x_t) ) · ∂x_t/∂θ ],

with `s_real` the frozen teacher's score and `s_fake` the score of the
*generator's own current outputs*, estimated by a second network trained
online with an ordinary denoising loss on generator samples. Both scores
are expressed through denoised predictions; DMD1's paper (its Eq. 8)
chooses the weight

    w_t = σ_t²/α_t · C·S / ‖x − x̂₀_real‖₁      (C·S = channels × pixels)

precisely so that the score-difference prefactor cancels and the surviving
gradient is a plain difference of denoised images with a per-sample
normalizer:

    grad = ( x̂₀_fake − x̂₀_real ) / mean|x − x̂₀_real|.                (M1.1)

In this EDM/VE codebase the cancellation is even cleaner than in the SD
world: EDM has α_t ≡ 1 and its network *natively outputs x̂₀* (§M3 below),
so `s(x) = (D(x;σ) − x)/σ²`, the `σ²` cancels against `w_t = σ²·(...)`,
and `dmd/loss.py` literally computes `(pred_fake_image − pred_real_image)
/ weighting_factor` and injects it with the detached-target trick
`0.5·mse(x, (x − grad).detach())` — the same three lines as DMD2, and the
single most important code block in this repo (§L2). One notable
*difference* from DMD2: **no CFG anywhere** — CIFAR-10 EDM is
class-conditional via one-hot labels, not text, so `s_real` is the plain
teacher score, not a guided one.

### M2 — The regression branch: paired couples + LPIPS (what DMD2 deleted) ★

The reverse KL is **mode-seeking**: a generator that collapses onto a few
teacher modes can score a low KL while dropping whole classes of images,
and early in training — when `mu_fake` is a poor estimate of a fast-moving
`p_fake` — the DM gradient can point anywhere. DMD1's fix is brute-force
anchoring:

1. **Offline, before training**: run the teacher's own deterministic ODE
   sampler (EDM's Heun sampler, 18 steps ≈ 35 UNet calls each) on a large
   set of seeds and save the couples `(z_ref, y_ref = ODE(z_ref))`. This
   repo generates 10,000 pairs per CIFAR class = **100,000 couples**
   (`dmd/dataset/dataset_generator.py`), each stored as a stacked
   `(2, 3, 32, 32)` npy and then packed into one HDF5 file (§D3–D4). The
   paper did the same at much larger scale for text-to-image; this is the
   expensive part — you pay full teacher sampling for an entire dataset
   before step 0.
2. **Online, every step**: draw a couple from the dataset, push the *same*
   `z_ref` through the student, and penalize
   `L_reg = LPIPS(G(z_ref), y_ref)` (§L3).

Why this works: the teacher ODE is deterministic, so the coupling
`z_ref → y_ref` is **one-to-one** — regression onto it does not suffer the
"average of many targets" gray-mush failure that dooms naive one-step
distillation under an independent coupling (our
`instaflow_line_by_line.md`, "Why naive distillation fails"; DMD1's
couples are exactly the same object as InstaFlow's, minus the Reflow
straightening). Because `z_ref` covers noise space uniformly and each has
a definite teacher answer, the regression term drags the generator toward
*every* teacher mode — mode coverage — and it is a fixed, stationary
target, so it stabilizes the early phase when the two-network chase of the
DM loss is still finding its feet.

Why DMD2 deleted it: it caps the student at the teacher's per-sample ODE
quality, it costs a dataset's worth of teacher sampling (prohibitive for
SDXL), and DMD2's GAN-on-real-data term provides the missing stabilizer
more cheaply (`dmd2_line_by_line.md §M4.1–M4.2`). Watching DMD1 first
makes DMD2's deletions legible: same skeleton, the crutch swapped for a
critic.

### M3 — EDM parametrization: σ-space, not ᾱ-space

Our SD-world tutorials (notes `10` notation header) use
`α_t := √ᾱ_t`, `σ_t := √(1−ᾱ_t)`, `x_t = α_t x₀ + σ_t ε` — a
*variance-preserving* frame where the clean image is shrunk as noise
grows. EDM works in the **variance-exploding** σ-frame:

    x_σ = x₀ + σ·ε,        σ ∈ [0.002, 80],   α ≡ 1,

i.e. "time" IS the noise level, nothing is rescaled, and at σ=80 the image
is an invisible perturbation on top of huge noise. Three consequences you
must hold in mind while reading this repo:

- **The network is a denoiser, not an ε-predictor.** `EDMPrecond` (§W1)
  wraps the raw UNet `F` as

      D(x; σ) = c_skip(σ)·x + c_out(σ)·F(c_in(σ)·x, c_noise(σ)),

  with `c_in = 1/√(σ²+σ_d²)` (whitens the input: `Var(x_σ) = σ_d² + σ²`),
  `c_skip = σ_d²/(σ²+σ_d²)` and `c_out = σ·σ_d/√(σ²+σ_d²)` (chosen so the
  effective training target has unit variance at every σ), `c_noise =
  ln(σ)/4`, `σ_d = 0.5` (CIFAR pixel std). `D(x;σ)` directly approximates
  `E[x₀|x_σ]` — so where the DMD2 chapter needed `get_x0_from_noise` after
  every UNet call, here **the forward pass already returns x̂₀** and
  `loss.py` uses the outputs raw.
- **Score conversion**: from `x_σ = x₀ + σε`, Tweedie gives
  `∇log p_σ(x) = (D(x;σ) − x)/σ²` — the VE analogue of `s = −ε̂/σ` from
  notes `10 §3`. This is the identity behind §M1's cancellation.
- **The one-step generator is the denoiser called once at (almost) the top
  noise level.** DMD1 defines `G(z) := D_θ(σ_max·z; σ_max-ish)` — feed
  scaled pure noise, condition on a *fixed* σ, read out the denoised
  image. At σ≈79.6, `c_skip ≈ 4e-5` and `c_out ≈ 0.5`: the input skip is
  numerically dead and the "denoiser" is really a direct generator network
  whose output is `0.5·F(0.0126·x)`. That's why initializing `G` from the
  teacher works: the teacher already knows how to make a plausible x̂₀
  from pure noise (a blurry class-average); training sharpens it into a
  sample.

  Timesteps in this repo are indices into a 1000-point **Karras grid**
  (`get_sigmas_karras`, notes `10 §4`'s ρ=7 spacing), indexed *from the
  end*: `t=0 → σ=0` (clean), `t=999 → σ≈79.56` (= grid point 1, "T−1").
  The generator's fixed sigma is exactly that `σ_{T−1} ≈ 79.56`.

### M4 — Map of the code

| File | Role |
|---|---|
| `dmd/loss.py` | THE heart: `DistributionMatchingLoss` (M1.1), `GeneratorLoss` (KL + λ·LPIPS), `DenoisingLoss` (fake-score training) |
| `dmd/training/training_loop.py` | `train_one_epoch`: per-batch G-update then `mu_fake`-update, paired-batch handling |
| `dmd/train.py` | `run()`: loads the three EDM copies, optimizers, dataloaders, epochs, FID |
| `dmd/modeling_utils.py` | Karras schedule, VE `forward_diffusion`, fixed generator sigma, EDM pickle loading |
| `dmd/training/networks.py` | verbatim NVLabs EDM networks; we read only `EDMPrecond` |
| `dmd/generate.py`, `dmd/sampler.py` | teacher sampling; `save_format="pairs"` manufactures the couples |
| `dmd/dataset/*` | couple-dataset generation and the HDF5 `CIFARPairs` Dataset |
| `scripts/dataset_to_h5.py` | npy couples + annotations.json → one `cifar.hdf5` |
| `dmd/dnnlib`, `dmd/torch_utils`, `dmd/utils`, `dmd/fid.py` | NVLabs plumbing, logging, homemade FID (grouped in §B) |

Shapes (B = batch, default 56; everything is **pixels**, not latents —
CIFAR needs no VAE):

| Tensor | Shape |
|---|---|
| `z`, `z_ref`, `x`, `x_ref`, `y_ref`, `noisy_x`, `grad` | (B, 3, 32, 32) |
| `generator_sigma` | (1, B) — yes, really; §W4 |
| `sigma_t` from `forward_diffusion` | (B,) |
| `class_idx` / one-hot `class_ids` | (B,) long / (B, 10) |
| `weighting_factor` | (B, 1, 1, 1) |

DMD1 (this repo) vs DMD2, at a glance:

| | DMD1 (here) | DMD2 |
|---|---|---|
| anchor | paired dataset + LPIPS regression | GAN on real data |
| real data used | never (only teacher *samples*) | yes (GAN real branch) |
| critic:generator update ratio | 1:1 | 5:1 (TTUR) |
| CFG in `s_real` | none (class-conditional EDM) | scale 6–8 baked in |
| student steps | 1 | 1 or 4 (backward simulation) |
| x̂₀ from network | native (EDM denoiser) | via `get_x0_from_noise` |

---

## §D — Manufacturing the paired dataset (methodologically load-bearing)

This is the part of DMD1 with no DMD2 counterpart: before any distillation
step runs, the teacher must answer "what image do YOU make from this exact
noise?" for 100k noises.

### D1 — `dmd/sampler.py`: the teacher's ODE sampler (verbatim EDM)

```python
def edm_sampler(net, latents, steps: int = 18, sigma_min: float = 0.002,
                sigma_max: float = 80, rho: float = 7.0, S_churn: float = 0.0, ...):
    t_steps = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=latents.device)
    x_next = latents.to(torch.float64) * t_steps[0]
```
NVLabs' Algorithm 2, copied. 18 Karras-spaced σ levels from 80 down to
0.002 (+ a final 0). The input `latents` are **unit** Gaussians; scaling by
`t_steps[0] = 80` happens here — remember this, it collides with a training
detail in §W4. Float64 throughout: EDM's reference sampler prioritizes
numerical fidelity over speed.

```python
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        ...
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
```
Heun's method on the VE probability-flow ODE `dx/dσ = (x − D(x;σ))/σ`
(the σ-space twin of the DDIM ODE we derived in notes `10 §2`): one Euler
step, then a second UNet call to average the slopes — 2nd order, hence
2·18−1 = **35 teacher UNet calls per training pair**. `S_churn=0` in the
default config makes it fully deterministic — crucial, because the whole
value of the couple is that `z_ref` has ONE definite answer (§M2). This is
the cost DMD2 refused to pay: 100k pairs × 35 calls = 3.5M teacher
forwards before training even starts.

### D2 — `dmd/generate.py`: `EDMGenerator`, and the `pairs` save format

`EDMGenerator` is a convenience wrapper around `edm_sampler`: it loads the
teacher pickle (`load_edm`, §W5), keeps a `GenerationConfig` dataclass
(steps/σ-range/churn defaults exactly as D1), and its `__call__` shards a
seed list into batches (with `torch.distributed` barriers so rank 0
downloads the model first). The method that matters:

```python
    def generate_batch(self, seeds, class_idx=None, **kwargs):
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn([batch_size, self.model.img_channels,
                             self.model.img_resolution, self.model.img_resolution], device=device)
        class_ids = torch.tensor([class_idx] * batch_size, device=device)
        class_labels = encode_labels(class_ids, self.model.label_dim)
        images = edm_sampler(self.model, latents, ..., class_labels=class_labels, randn_like=rnd.randn_like)
        return latents, images
```
`StackedRandomGenerator` gives every sample its own seeded
`torch.Generator`, so pair #001234 is reproducible forever regardless of
batch size — a dataset of couples must be re-generatable. It returns BOTH
the unit-Gaussian `latents` (B,3,32,32) and the finished `images` — the
couple, in one tuple.

```python
    @staticmethod
    def _save_array_as_pairs(output_dir, images, latents, save_start_idx):
        for iid, image_np, latent_np in zip(instance_ids, images_np, latents_np):
            pairs = np.stack([image_np, latent_np], axis=0)
            np.save(os.path.join(samples_output_dir, f"{iid:06d}.npy"), pairs)
```
One `(2, 3, 32, 32)` float array per couple: index 0 the image (in
[−1, 1]-ish pixel range, float64 from the sampler), index 1 the **unit**
latent. Note the images are saved *raw*, not quantized to uint8 — the
regression target keeps full precision.

`DMDGenerator` at the bottom of the file is the one-step *inference* class
for the finished student: load the trained checkpoint
(`load_dmd_model`), then

```python
        g_sigmas = get_fixed_generator_sigma(len(seeds), device=device)
        if scale_latents:
            latents = latents * g_sigmas[0, 0]
        ...
        return self.model(latents, g_sigmas, class_labels=class_labels).to(device)
```
— scale noise by σ_{T−1}, ONE network call, done. This is the entire
payoff of the method: compare 35 calls in D1. (Small bug flagged: the
guard `if not latents and not seeds:` would crash with "Boolean value of
Tensor is ambiguous" if you actually pass a `latents` tensor; seeds-only
usage works.)

### D3 — `dmd/dataset/dataset_generator.py`: the factory loop

```python
def generate_distillation_dataset(model_path, output_dir, device=None,
                                  size_per_class: int = 10000, batch_size: int = 64):
    edm_generator = EDMGenerator(network_path=model_path, device=device)
    seeds = list(range(size_per_class))
    for cls in range(10):
        edm_generator(output_dir.as_posix(), seeds=seeds, class_idx=cls,
                      batch_size=batch_size, save_format="pairs", save_start_idx=instance_id)
```
10 classes × 10k seeds = 100k couples, class-balanced by construction, plus
an `annotations.json` mapping instance id → (npy path, class, seed). The
docstring recommends batch 1024 on a 24 GB GPU — this is an overnight job,
and exactly the kind of precompute DMD2's authors deleted. **Deviation
watch**: the paper doesn't state a CIFAR pair count; 100k is this repo's
choice (their `goals.txt` mentions ~1M for ImageNet in the paper).

### D4 — `scripts/dataset_to_h5.py` + `dmd/dataset/cifar_pairs.py`: pack and serve

`convert_json_to_h5` walks `annotations.json` and writes every npy into
one HDF5 file, `hf["/data/{iid}"] = pairs` with `class_idx`/`seed` as HDF5
attributes — 100k tiny files become one seekable blob (the README's
`download_data.sh` fetches a prebuilt `cifar.hdf5` from HuggingFace so you
can skip D1–D3 entirely).

```python
class CIFARPairs(Dataset):
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_dataset_path, "r")["data"]
        sample = self.dataset[str(index)]
        image, latent = pairs
        return_dict = {"instance_id": index, "image": image, "latent": latent,
                       "class_id": attributes["class_idx"], "seed": attributes["seed"]}
```
A standard PyTorch Dataset over the HDF5 file. The lazy `h5py.File` open
inside `__getitem__` is the classic multiprocessing dance: h5py handles
can't cross the DataLoader fork, so each worker opens its own on first
access. Unpacking `image, latent = pairs` splits the (2,3,32,32) stack
back into the couple. **This dict is the training batch** — every DMD1
step consumes one paired batch (§T2).

---

## §W — `dmd/modeling_utils.py` (+ `EDMPrecond`): the EDM wrappers

### W1 — `EDMPrecond.forward` (in `dmd/training/networks.py`, verbatim NVLabs)

```python
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
```
The preconditioning of §M3, literally. `sigma` arrives as any shape and is
`reshape(-1, 1, 1, 1)`-ed first (this forgiving reshape is what makes §W4's
odd (1,B) sigma shape survive). All three networks in DMD1 — teacher,
fake score, generator — ARE this class with a `SongUNet` inside
(`model_channels=128`, `channel_mult=(2,2,2)`, `label_dim=10`, ~56M
params); they differ only in what's frozen and what loss trains them. Every
call returns x̂₀ ≈ `E[x₀|x,σ]`, shape (B,3,32,32) — keep repeating this
while reading `loss.py`: **no ε anywhere in this codebase**. The rest of
`networks.py` (928 lines: `SongUNet`, `DhariwalUNet`, attention blocks,
FIR resampling) is untouched NVLabs EDM — trust it like a library.

### W2 — `get_sigmas_karras`: the grid

```python
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)
```
Karras ρ=7 spacing (interpolate in σ^{1/7}, then raise to the 7th power —
dense near σ_min, sparse near σ_max), **descending** 80 → 0.002, with a 0
appended: length n+1. Taken from k-diffusion. Used with n=18 for the
teacher sampler (D1) and n=1000 as this repo's stand-in for "the 1000
training timesteps."

**Deviation watch:** the paper's teacher is EDM, and EDM *training* draws
σ from a lognormal (P_mean=−1.2, P_std=1.2), not from a uniform index into
a Karras *sampling* grid. This repo re-uses the sampling grid as a training
schedule everywhere (DM loss, denoising loss). It's self-consistent and
sane, but it is a reimplementation choice — the paper never specifies a
discrete grid for the CIFAR losses.

### W3 — `forward_diffusion`: VE noising, indexed from the end

```python
def forward_diffusion(x, t, n: int = 1000, noise=None):
    if noise is None:
        noise = torch.randn_like(x, device=x.device)
    sigma = get_sigmas_karras(n, sigma_min=0.002, sigma_max=80, rho=7.0, device=x.device)
    ns = noise * sigma[-(t + 1), None, None, None]
    noisy_x = x + ns
    return noisy_x, sigma[-(t + 1)]
```
The forward shortcut of this world: `x_t = x + σ_t·ε` — compare notes `10`'s
`x_t = α_t x₀ + σ_t ε` with α ≡ 1. The index gymnastics: the schedule array
has 1001 entries descending 80→0.002→0, and `sigma[-(t+1)]` counts from the
back, so **t=0 → σ=0 (clean), t=999 → σ=79.56 (grid point 1)** — large t =
more noise, matching DDPM intuition, but the array itself is stored
backwards. `t` is a (B,) LongTensor, so the fancy-index returns (B,) sigmas
broadcast to (B,1,1,1) for the product. Two small quirks: the full
1001-point schedule is rebuilt on every call (cheap, but wasteful), and
callers pass `t` created on CPU while `x` is on GPU — PyTorch tolerates
CPU index tensors on CUDA data, so it works.

### W4 — `get_fixed_generator_sigma`: the generator's frozen clock

```python
def get_fixed_generator_sigma(size, device):
    sigma = get_sigmas_karras(n=1000, sigma_min=0.002, sigma_max=80.0, device=device)[1]  # sigma_(T-1)
    return torch.tile(sigma, (1, size))
```
The paper conditions the one-step generator at the fixed timestep T−1; in
EDM terms that is grid point 1 of the 1000-point schedule, **σ ≈ 79.564**
(not 80!). Returned tiled to shape **(1, size)** — an odd choice that only
works because `EDMPrecond.forward` reshapes sigma to (−1,1,1,1) anyway;
callers grab the scalar as `generator_sigma[0, 0]`.

**Quirk with teeth:** the paired dataset (D1) built its images from
`latents * t_steps[0]` = `80·z`, but training and inference scale `z_ref`
by this **79.564**. So the generator sees `79.564·z_ref` while the teacher
answered for `80·z_ref` — a 0.5% mismatch in input scale between the
couple's question and its answer. Almost certainly harmless (c_in shrinks
both to ~0.0126·x), but it is a genuine internal inconsistency of this
reimplementation worth knowing about, since the *entire premise* of the
regression loss is "same z, same answer."

### W5 — Loaders and labels (brief)

- `load_edm(model_path, device)`: unpickles NVLabs' distributed `.pkl`
  and returns the `"ema"` network. The `sys.path.insert(0, SOURCES_ROOT)`
  line is a real trick: the pickle stores module paths
  (`training.networks.EDMPrecond`), so the repo fakes the original import
  layout to deserialize. All three networks start life through this
  function — **teacher, fake score, and generator are three copies of the
  same EMA teacher weights** (paper-faithful: DMD initializes both
  trainables from the teacher).
- `load_dmd_model(...)`: rebuilds `EDMPrecond` with the CIFAR
  hyperparameters **hardcoded** (resolution 32, `label_dim=10`,
  `channel_mult=(2,2,2)`, `dropout=0.13`…), loads `model_g`/`model_d`
  state dicts, and optionally the two AdamW states for resuming. Change
  dataset → edit this function by hand; the config is not saved in the
  checkpoint.
- `encode_labels(class_ids, label_dim)`: `one_hot(class_ids, 10)` → (B,10)
  float — EDM's class conditioning format.
- `sample_from_generator` / `generate_samples`: convenience one-step
  samplers duplicating §D2's `DMDGenerator.generate_batch` logic (including
  the same tensor-truthiness bug).

---

## §L — `dmd/loss.py` in FULL: the heart ★

Forty-odd lines that contain the whole method. Read them slowly.

### L1 — Imports

```python
import torch
import torch.nn.functional as F
from piq import LPIPS
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torchvision.transforms import Resize

from dmd.modeling_utils import forward_diffusion
```
`piq` supplies LPIPS (VGG16-feature perceptual distance, notes: "does it
*look* the same to a convnet," robust where pixel MSE is blind).
Subclassing `_Loss` (PyTorch's private loss base) buys them a `reduction`
attribute and nothing else. `forward_diffusion` is §W3 — the only bridge
to the noise machinery.

### L2 — `DistributionMatchingLoss`: Eq. (M1.1), verbatim ★★

```python
class DistributionMatchingLoss(_Loss):
    """Loss function for DMD (Algorithm 2) proposed in ..."""
    def __init__(self, timesteps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timesteps = timesteps
```
`timesteps=1000` — the grid size for §W3. Nothing else is state.

```python
    def forward(self, mu_real: Module, mu_fake: Module, x: torch.Tensor,
                class_ids: torch.Tensor = None) -> torch.Tensor:
        b, c, w, h = x.shape
```
Signature tells the story: the loss *receives both score networks as
arguments* every call — no ownership, no optimizer entanglement; the
training loop decides what's frozen. `x` is `G(z)`, (B,3,32,32),
**graph-attached to the generator** — the only tensor here that ever will
be.

```python
        # In practice T_min, T_max choices follows DreamFusion as follows
        T_min, T_max = int(0.02 * self.timesteps), int(0.98 * self.timesteps)
        timestep = torch.randint(T_min, T_max, [b])
        noisy_x, sigma_t = forward_diffusion(x, timestep)
```
Draw t ∈ [20, 979] per sample — the DreamFusion/SDS 2%–98% clip, same
range and same justification as DMD2 (§G1 there): at tiny σ the gradient
is noise-dominated, at σ_max it says nothing about x₀. Through §W3's
end-indexing this means σ ∈ [0.0032, 71.3]. Then diffuse the *student's
own sample*: `noisy_x = x + σ_t·ε`, fresh ε drawn inside
`forward_diffusion`. The KL is matched at this shared noisy point.
(`timestep` lives on CPU; works, see W3.)

```python
        with (torch.no_grad()):
            pred_fake_image = mu_fake(noisy_x, sigma_t, class_labels=class_ids)
            pred_real_image = mu_real(noisy_x, sigma_t, class_labels=class_ids)
```
Both denoisers answer "what clean image do you think is under this noise?"
— (B,3,32,32) each, gradient-free (the stray parentheses around
`torch.no_grad()` are cosmetic). `pred_real_image` = the frozen teacher's
x̂₀ = `s_real` in disguise; `pred_fake_image` = the online critic's x̂₀ =
`s_fake` in disguise (§M3's Tweedie identity). No CFG, no ε→x̂₀
conversion, no float64 dance — the EDM wrapper did the work. Compare the
~20 lines this took in DMD2's SD world.

```python
        weighting_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3], keepdim=True)  # Eqn. 8
        grad = (pred_fake_image - pred_real_image) / weighting_factor
```
**The DMD gradient.** Numerator: `x̂₀_fake − x̂₀_real` ∝ `s_fake − s_real`
— the reverse-KL direction, "more typical of the student than of the
teacher," to be descended. Denominator: per-sample `mean|x − x̂₀_real|`
over C,H,W, kept as (B,1,1,1) — the paper's Eq. 8 weight
`σ²/α·CS/‖x−x̂₀_real‖₁` with the σ²/α part already cancelled by working in
x̂₀-space (§M1; in this VE frame α=1 so the cancellation is exact). It
normalizes by "how much the teacher wants this sample changed," keeping
gradient scale uniform across noise levels and samples. Two nits vs DMD2's
version: no `nan_to_num` guard here, and `weighting_factor` is not
detached — both harmless in practice (the target is detached next line, so
no gradient flows through the factor), but the missing NaN guard means a
degenerate batch would kill training (the loop's `isfinite` check, §T2, is
the actual safety net).

```python
        diff = (x - grad).detach()  # stop-gradient
        return 0.5 * F.mse_loss(x, diff, reduction=self.reduction)
```
The **detached-target trick**, third sighting after our KAIST SDS and
DMD2 (`dmd2_line_by_line.md §M3`, `kaist_reports.md`): the target is a
constant, so `dL/dx = (x − (x − grad))/N = grad/N` — backprop hands
autograd exactly the hand-computed gradient, which flows into `∂G/∂θ`
through `x`, never touching either score network. `reduction` defaults to
`"mean"`, so N = B·3·32·32 = 3072·B silently scales the injected gradient
— absorbed by the learning rate, but change resolution and your effective
step size changes too (same quirk ledger entry as DMD2).

### L3 — `GeneratorLoss`: KL + λ·LPIPS — the DMD1 signature ★

```python
class GeneratorLoss(_Loss):
    """Combined loss for the generator model. See § 3.4 (Final Objective).
    D_KL + lambda_reg * L_reg"""
    def __init__(self, timesteps: int = 1000, lambda_reg: float = 0.25, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self.dmd_loss = DistributionMatchingLoss(timesteps)
        self.lpips = LPIPS()
        self.lambda_reg = lambda_reg
```
The paper's §3.4 final objective, one class. `lambda_reg=0.25` matches the
paper's stated regression weight. `LPIPS()` is piq's VGG16 version with
default `reduction='mean'` → returns a scalar. (Micro-bug: `super()
.__init__(self, ...)` passes `self` as the `size_average` positional —
PyTorch's deprecated arg path shrugs it off, but it's a typo.) **Deviation
watch:** the official DMD used LPIPS as implemented in its own stack; piq's
weights/normalization may differ slightly — fine for reproduction, not
bit-exact.

```python
    def forward(self, mu_real, mu_fake, x, x_ref, y_ref, class_ids=None):
        loss_kl = self.dmd_loss(mu_real, mu_fake, x, class_ids)
```
Distribution matching on `x = G(z)` — **fresh random z**, nothing to do
with the paired data. The two branches deliberately use different noises:
the KL term must sample the generator's whole distribution, not just the
reference set.

```python
        # Apply preprocessing
        x_ref = (x_ref + 1) / 2.0
        y_ref = (y_ref + 1) / 2.0
        transform = Resize(224)
        x_ref = transform(x_ref)
        y_ref = transform(y_ref)
        loss_reg = self.lpips(x_ref, y_ref)
        return loss_kl + self.lambda_reg * loss_reg
```
**The regression branch — the loss DMD2 deleted.** `x_ref = G(z_ref)` is
the student's answer for the paired latent, `y_ref` the teacher's
precomputed ODE answer for the *same* latent (§D). Map [−1,1] → [0,1]
(piq's expected range), bilinearly upsample 32 → 224 (VGG16 was trained at
224; a 32×32 input would shrivel to ~1×1 through its poolings — the resize
is the reimplementers' pragmatic choice; the paper doesn't discuss CIFAR
LPIPS resolution), then perceptual distance. Because the coupling is
one-to-one (§M2), this is honest regression, not mush-inducing averaging —
it anchors mode coverage and steadies the early chase, at the price of a
teacher-sampled dataset and a quality ceiling. `Resize(224)` is
re-instantiated every call — wasteful but harmless. Total:
`loss_kl + 0.25·loss_reg`, one scalar.

### L4 — `DenoisingLoss`: the fake score chases the student

```python
class DenoisingLoss(_Loss):
    """Loss function for DMD (Equation 6 / Algorithm 3) ..."""
    def forward(self, mu_fake, x, t, class_ids=None):
        x_t, sigma_t = forward_diffusion(x.detach(), t)  # stop grad
        pred_fake_image = mu_fake(x_t, sigma_t, class_labels=class_ids)
```
DMD2's `compute_loss_fake`, EDM edition. `x.detach()` first — the
generator must never learn from its critic's training signal. Re-noise the
generator's samples with *fresh* noise and a *fresh* t (drawn by the
caller, §T2), then ask the trainable `mu_fake` to denoise. Note
`pred_fake_image` here IS graph-attached — this loss trains `mu_fake`.

```python
        # Algorithm SNR + 1 / sigma_data^2 for EDM (sigma_data = 0.5)
        weight = 1 / sigma_t**2 + 1 / mu_fake.sigma_data**2
        return torch.mean(weight[:, None, None, None] * (pred_fake_image - x.detach()) ** 2)
```
Weighted x̂₀-space MSE. The weight `1/σ² + 1/σ_d²` = `(σ²+σ_d²)/(σ²σ_d²)`
is **exactly EDM's λ(σ) = 1/c_out(σ)²** — the weighting NVLabs trained the
teacher with, which makes the effective per-σ target unit-variance. So the
fake score is trained with *the teacher's own recipe* but on *the
student's outputs* — precisely the paper's Eq. 6 ("same objective as the
base model"). By the standard denoising-score-matching argument, `mu_fake`
→ the denoiser (hence score) of `p_fake` at every σ. The caller draws
`t ~ DU(1, 1000)` — t=0 is excluded because σ=0 makes the weight 1/0²
(the in-loop comment says exactly this); the full remaining range is used,
wider than the DM loss's 2–98% clip, for DMD2's same reason: the critic
must be accurate everywhere it might be queried.

---

## §T — The training step: `train.py` + `training_loop.py`

### T1 — `dmd/train.py`, `run()`: three copies of one teacher

Skipping the argparse-style docstring and Neptune plumbing, the method-
relevant lines:

```python
    training_dataset = CIFARPairs(data_path)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
```
THE dataloader is the paired dataset (§D4). There is no separate "noise
loader": fresh z for the KL term is drawn inside the loop, and — notice —
**real CIFAR-10 training images are never loaded at all**. DMD1 trains
purely against the teacher (its samples and its score); real data appears
only in the FID evaluator (test split). Contrast DMD2, where real latents
feed the GAN.

```python
    mu_real = load_edm(model_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", device=device)
    ...
    mu_fake = load_edm(model_path=model_path, device=device)
    generator = load_edm(model_path=model_path, device=device)
```
Three `EDMPrecond`s. Quirk with teeth: the teacher's URL is **hardcoded**,
ignoring the CLI's `--model-path` (which only seeds `mu_fake` and
`generator`) — pass a different teacher and you'll silently distill from
the CIFAR-10 one anyway. Both trainables start as exact teacher clones
(paper-faithful; DMD2 does the same).

```python
    generator_optimizer = AdamW(params=generator.parameters(), **optimizer_kwargs)
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), **optimizer_kwargs)
    generator_loss = GeneratorLoss(timesteps=dmd_loss_timesteps, lambda_reg=dmd_loss_lambda)
    diffusion_loss = DenoisingLoss()
```
Two optimizers, identical settings (lr 5e-5, wd 0.01, betas (0.9, 0.999))
— the README's "Assumptions" section flags this explicitly: the paper
gives hyperparameters without saying which network they belong to, so the
repo applies them to both. No LR scheduler (a commented-out
`lr_scheduler.step` haunts `train()`), no EMA of the generator (commented
out in the loop) — the paper's larger runs likely had both; another honest
deviation. `resume_from_checkpoint` swaps this block for
`load_dmd_model(..., for_training=True)` (§W5).

`train()` itself is a thin epoch loop: `train_one_epoch(...)` → compute
FID on the CIFAR-10 *test* split (§B) → `checkpoint_handler.save(...)`
keeping `last_checkpoint.pt` and a FID-best `best_checkpoint.pt`. The
`try: import apex / fvcore` blocks at the top of the file are dead
template code from a timm-style project — never used.

### T2 — `training_loop.py`, `train_one_epoch`: the alternation ★

```python
    generator.requires_grad_(True).train()
    mu_fake.requires_grad_(True).train()
    mu_real.requires_grad_(False).eval()
```
Roles pinned once per epoch: two trainables, one frozen teacher. Note
`mu_fake` stays `requires_grad_(True)` even during the generator's turn —
safe only because §L2 runs it under `no_grad` and §T2 zeroes grads before
each step (DMD2 needed a `requires_grad_` sandwich because its GAN loss
flows through critic activations; DMD1 has no such path).

```python
    for pairs in metric_logger.log_every(data_loader_train, print_freq, header):
        y_ref = pairs["image"].to(device, non_blocking=True).to(torch.float32).clip(-1, 1)
        z_ref = pairs["latent"].to(device, non_blocking=True).to(torch.float32)
        z = torch.randn_like(y_ref, device=device)
```
Every iteration consumes one **paired batch**: the teacher's answer
`y_ref` (B,3,32,32), clipped to [−1,1] (the float64 ODE output can
slightly overshoot; LPIPS wants a bounded range), its question `z_ref`,
and a *fresh* unit Gaussian `z` for the KL branch — two noise sources, two
loss branches, one batch (§L3's deliberate split).

```python
        generator_sigma = get_fixed_generator_sigma(z.shape[0], device=device)
        # Scale Z ~ N(0,1) (z and z_ref) w/ sigma(T-1) to match the sigma at T-1
        z = z * generator_sigma[0, 0]  # scalar product
        z_ref = z_ref * generator_sigma[0, 0]
        class_idx = pairs["class_id"].to(device, non_blocking=True)
        class_ids = encode_labels(class_idx, generator.label_dim)
```
Both latents scaled by σ_{T−1} ≈ 79.56 so the generator's input looks like
a legal EDM state at its fixed conditioning level, `x_σ ≈ σ·ε` (§M3; and
recall §W4's 79.56-vs-80 mismatch against how `y_ref` was made). Labels
one-hot (B,10); both `x` and `x_ref` use the *pair's* class — the fresh-z
branch inherits the batch's class distribution, which is uniform by
dataset construction (§D3).

```python
        with amp_autocast():
            # Update generator
            # tanh after small experiment between (no-postprocess, tanh, clipping)
            x = generator(z, generator_sigma, class_labels=class_ids)
            x_ref = generator(z_ref, generator_sigma, class_labels=class_ids)
            l_g = loss_g(mu_real, mu_fake, x, x_ref, y_ref, class_ids)
            if not math.isfinite(l_g.item()):
                print(f"Generator Loss is {l_g.item()}, stopping training")
                sys.exit(1)
        update_parameters(generator, l_g, optimizer_g, max_norm)
```
**Generator turn.** TWO generator forwards per step — one per branch:
`x = G(z)` feeds the DM loss, `x_ref = G(z_ref)` feeds LPIPS against
`y_ref`; then `l_g = loss_kl + 0.25·loss_reg` (§L3), hard-exit on
NaN/inf (the safety net §L2 lacks internally), and
`update_parameters` = zero_grad → backward → `clip_grad_norm_(10)` → step.
`amp_autocast` defaults to `contextlib.suppress` (i.e. **no** mixed
precision unless wired up — the CLI never wires it). The orphaned comment
about `tanh` records an experiment that lost: the shipped code applies
*no* output post-processing to `x` during training, though `y_ref` was
clipped — a small asymmetry to know about.

```python
        with amp_autocast():
            # Update mu_fake
            t = torch.randint(1, 1000, [x.shape[0]])  # t ~ DU(1,1000) as t=0 leads 1/0^2 -> inf
            l_d = loss_d(mu_fake, x, t, class_ids)
            ...
        update_parameters(mu_fake, l_d, optimizer_d, max_norm)
```
**Critic turn, same iteration, ratio 1:1.** `loss_d` = §L4's weighted
denoising loss on the SAME `x` batch (detached inside), fresh t ∈ [1,999]
→ σ ∈ [0.002, 79.56]. Two things to notice against DMD2: (i) no TTUR —
one critic update per generator update (DMD2 found 5:1 necessary at SD
scale; at CIFAR scale with the regression anchor, 1:1 sufficed); (ii) the
critic trains on samples from the generator *as it was before this
iteration's update* — a one-step-stale distribution, standard
GAN-alternation slack. Also note the paired branch (`x_ref`) never feeds
the critic; `mu_fake` only ever sees fresh-noise samples.

```python
        if i % im_save_freq == 0:
            with torch.no_grad():
                x_t, sigma_t = forward_diffusion(x, t)
                real_pred = mu_real(x_t, sigma_t, class_labels=class_ids)
                fake_pred = mu_fake(x_t, sigma_t, class_labels=class_ids)
            grid = _save_intermediate_images(images_epoch_dir, [x, real_pred, fake_pred, x_ref, y_ref], f"iter_{i}")
```
Every 300 iters, the five-row debug grid the README documents:
`(x, x_real, x_fake, x_ref, y_ref)` — the student's sample, what the
teacher vs the critic think hides under its re-noised version (their
*disagreement* is the DM gradient — the same "what does the teacher want
changed" picture DMD2 logs), and the regression pair side by side. It
reuses the critic's `t` for a fresh re-noising; visualization only.
Everything else in the loop is `MetricLogger` bookkeeping; the epoch
returns averaged meters.

---

## §B — Boilerplate, grouped (skim once, then trust)

- **`dmd/dnnlib/`, `dmd/torch_utils/`** — verbatim NVLabs plumbing:
  `dnnlib.util.open_url` (downloads + caches the teacher pickle),
  `EasyDict`, `persistence` (lets pickled classes deserialize),
  `distributed`/`training_stats`/`misc` (multi-GPU helpers; effectively
  single-GPU here). Imported, never modified — treat as vendored library.
- **`dmd/fid.py`** — a homemade FID: InceptionV3 with `fc = Identity()`,
  bilinear 299-upsampling, features → scipy `sqrtm` Fréchet distance,
  comparing generator samples against the **CIFAR-10 test split (10k)**
  each epoch. Deviation watch: standard reported CIFAR FIDs (including
  the paper's 2.66) use 50k samples vs the *training* set with the
  canonical FID pipeline — this repo's numbers are directionally useful
  for checkpointing, not comparable to the paper's table.
- **`dmd/utils/`** — `seed_everything`, Neptune experiment creation,
  `MetricLogger`/`SmoothedValue` (torchvision-reference style),
  `CheckpointHandler` (last + FID-best checkpoints, JSON log lines),
  `torch_to_pillow` (the `(x·127.5 + 128)` uint8 mapping), image grids.
  A `_load_checkpoint_for_ema` helper and fvcore FLOP-count imports are
  dead template code.
- **`dmd/__main__.py`** — `fire` CLI: `generate-edm` (D2),
  `generate-dataset` (D3), `train` (T1). `scripts/download_data.sh`
  fetches the prebuilt `cifar.hdf5` couples and the authors' trained
  checkpoint from HuggingFace; `main.ipynb` is a Colab walk-through of the
  same calls.

---

## Quirk ledger (honesty section)

- **Unofficial, under-resourced reproduction**: 1× RTX 4090, batch 48
  (paper: 7×56), lr sqrt-scaled to ~1.75e-5 in the authors' actual runs,
  FID admittedly not matched to the paper's 2.66 (`goals.txt`). The
  method is faithfully shaped; the numbers are not the paper's.
- **σ = 79.564 vs 80**: couples generated from `80·z`, generator trained
  and run on `79.564·z_ref` (§W4) — a small internal inconsistency in the
  very pairing the regression loss depends on.
- **Hardcoded teacher URL** in `run()` ignores `--model-path` for
  `mu_real` (§T1).
- **Uniform t over a Karras *sampling* grid** stands in for the teacher's
  actual (lognormal-σ) training distribution in both losses (§W2) — a
  reimplementation choice the paper doesn't sanction or forbid.
- **LPIPS details are the reimplementers'**: piq's VGG16 LPIPS, [0,1]
  inputs, bilinear 32→224 upsample; `Resize(224)` rebuilt per call;
  `super().__init__(self, ...)` typo in `GeneratorLoss` (§L3).
- **No `nan_to_num`, no EMA, no LR schedule**: the NaN guard is an
  `isfinite`-and-exit in the loop; EMA and scheduler exist only as
  comments (§T1–T2). DMD2 kept the nan guard; the paper's big runs almost
  surely kept EMA.
- **`reduction="mean"` in the DM loss** divides the injected gradient by
  B·3072; resolution changes silently rescale the DM step (shared with
  DMD2's ledger).
- **`weighting_factor` not detached** — harmless (target is detached), but
  it builds pointless graph.
- **(1, B) sigma tensors** (`get_fixed_generator_sigma`) survive only
  because `EDMPrecond` reshapes; don't imitate.
- **Tensor-truthiness bug** in `sample_from_generator` /
  `DMDGenerator.generate_batch`: passing an actual `latents` tensor hits
  `if not latents ...` → runtime error; seeds-path works (§D2).
- **`y_ref` clipped, `x` not**: training compares an unclipped student
  against a clipped teacher target; the orphan "tanh" comment records the
  experiment (§T2).
- **Dead template code**: apex/fvcore imports, EMA helpers, commented
  schedulers — inherited from a timm-style project skeleton; ignore.
- **FID protocol nonstandard** (test split, 10k, homemade pipeline) —
  fine for model selection, wrong for paper comparison (§B).

---

## What to carry forward

1. **DMD1 = DMD gradient + paired-regression anchor.** The generator loss
   is `L_KL + 0.25·LPIPS(G(z_ref), ODE_teacher(z_ref))`: reverse-KL
   descent via `grad = (x̂₀_fake − x̂₀_real)/mean|x − x̂₀_real|` (§L2, the
   same Eq. as DMD2), plus a fixed one-to-one regression target that buys
   mode coverage and early-training stability at the price of an offline
   teacher-sampled dataset and a teacher-quality ceiling. DMD2's whole
   pitch is deleting branch 2 and replacing its stabilizing role with a
   GAN on real data.
2. **Deterministic couples are a recurring tool.** `(z, ODE(z))` pairs —
   here as an anchor, in InstaFlow as the entire training signal (after
   Reflow straightening), in DMD2's optional ODE warm-start — always for
   the same reason: a deterministic teacher sampler turns hopeless
   one-to-many regression into honest one-to-one regression.
3. **Parametrization does bookkeeping for you.** In EDM the network *is*
   x̂₀ (`D = c_skip·x + c_out·F(c_in·x)`), α ≡ 1, and
   `score = (D − x)/σ²` — so DMD's weighted score difference collapses to
   two forward calls and one subtraction, with none of the ε→x̂₀ float64
   gymnastics the SD/ᾱ world needs (notes `10` header for the
   dictionary between frames).
4. **A one-step generator can be the teacher's own architecture at a
   frozen timestep**: `G(z) = D_θ(σ_{T−1}·z; σ_{T−1})`, initialized from
   teacher weights, where c_skip ≈ 0 makes it a de-facto direct generator
   — the cheapest possible student design, reused by DMD2.
5. **The detached-target trick, fourth sighting**: `0.5·mse(x,
   (x − grad).detach())` injects any hand-computed gradient with one
   backward pass and zero backprop through the score networks (KAIST SDS
   → PDS → DMD2 → here).
6. **Critic-freshness is scale-dependent**: 1:1 alternation plus a
   regression anchor suffices at CIFAR scale; at SD scale DMD2 needed
   5:1 TTUR once the anchor was gone. When you remove a stabilizer,
   budget update ratio to compensate.
