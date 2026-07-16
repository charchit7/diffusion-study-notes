# Phase 2 — Codebase Study: cmu-10799-diffusion

*Reference notes for HW1 (DDPM). Companion to `notes/01_assignment_overview.md` (Phase 1, delivered in chat).*

## 1. Repository map

```
cmu-10799-diffusion/
├── train.py                 (689 L)  training loop: AMP, EMA, DDP, checkpointing  [2 TODOs]
├── sample.py                (197 L)  inference script                             [2 TODOs]
├── download_dataset.py               HF Hub → local Arrow dataset
├── modal_app.py                      cloud GPU wrapper (Modal)
├── configs/
│   ├── ddpm_babel.yaml               slurm/local config (all hyperparams blank = TODO)
│   └── ddpm_modal.yaml               Modal variant
├── src/
│   ├── methods/
│   │   ├── base.py          (123 L)  BaseMethod ABC: compute_loss(), sample()
│   │   └── ddpm.py          (136 L)  DDPM skeleton — THE core deliverable [all math TODO]
│   ├── models/
│   │   ├── blocks.py        (290 L)  PROVIDED: embeddings, ResBlock, Attention, up/down
│   │   └── unet.py          (169 L)  UNet skeleton — you assemble it [TODO]
│   ├── data/
│   │   └── celeba.py        (418 L)  dataset + loader + [-1,1] helpers [transforms TODO]
│   └── utils/
│       ├── ema.py           (159 L)  PROVIDED: EMA shadow weights
│       └── logging_utils.py          PROVIDED: file+console logger
├── scripts/
│   ├── train.sh                      slurm launcher (torchrun if num_gpus>1)
│   └── evaluate_torch_fidelity.sh    sample.py 1k images → `fidelity --kid` vs dataset dir
└── notebooks/                        1D playground, dataset exploration, sampling viz
```

## 2. Dependency / data-flow graph

```
configs/ddpm_babel.yaml
        │ yaml.safe_load
        ▼
train.py::train()
        ├─► create_dataloader_from_config ─► CelebADataset.__getitem__ ─► transform (TODO) ─► (3,64,64) in [-1,1]
        ├─► create_model_from_config ─► UNet(blocks: TimestepEmbedding/ResBlock/Attention/Down/Up)
        ├─► DDPM.from_config(model, config, device)      ← wraps UNet, owns β/α/ᾱ buffers (TODO)
        ├─► AdamW(model.parameters())
        ├─► EMA(unwrap_model(model))
        └─► loop:
              batch (B,3,64,64) ──autocast──► DDPM.compute_loss (TODO)
                                                 │ forward_process: x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε
                                                 │ UNet(x_t, t) → ε̂
                                                 │ MSE(ε̂, ε)
                                                 ▼
              scaler.scale(loss).backward() → clip → scaler.step → ema.update()
              every sample_every: generate_samples (TODO) → DDPM.sample (TODO, EMA weights) → save_samples (TODO)
              every save_every:  save_checkpoint {model, optimizer, scaler, ema, step, config}

sample.py ─► load_checkpoint ─► DDPM.from_config ─► ema.apply_shadow ─► method.sample(num_steps) ─► save_samples (TODO)
scripts/evaluate_torch_fidelity.sh ─► sample.py (1k pngs) ─► fidelity --kid --input1 generated --input2 dataset
```

## 3. File-by-file analysis

### 3.1 `src/methods/base.py` — the interface (PROVIDED)
- `BaseMethod(nn.Module, ABC)` holds `self.model` (UNet) + `self.device`; abstract `compute_loss(x) -> (scalar loss, metrics dict)` and `sample(batch_size, image_shape) -> (B,C,H,W)`.
- `state_dict()` returns `{'model': ...}` — note **DDPM.state_dict adds `num_timesteps`**, so checkpoint format is method-aware.
- Trap: `parameters()` delegates to `self.model.parameters()`. In `train.py` the optimizer is built from `model` directly (possibly DDP-wrapped) — consistent, but if you register learnable params on DDPM itself they will NOT be optimized. Scheduler constants must be **buffers**, not parameters (they're fixed anyway).

### 3.2 `src/methods/ddpm.py` — your core deliverable (ALL TODO)
Stubbed: `__init__` (build schedule), `forward_process`, `compute_loss`, `reverse_process` (one step, `@torch.no_grad`), `sample` (loop), `state_dict`, `from_config` reads `config['ddpm']{num_timesteps, beta_start, beta_end}`.
- Design implied by the skeleton: linear β schedule from `beta_start` to `beta_end` over `num_timesteps` (DDPM paper: 1e-4 → 0.02, T=1000). Precompute and `register_buffer`: `betas, alphas, alphas_cumprod (ᾱ), sqrt(ᾱ), sqrt(1−ᾱ), posterior variance β̃` — buffers move with `.to(device)` and live in fp32 even under AMP (good).
- Broadcasting helper (their "Pro tip 2"): gather per-sample coefficient at integer t and reshape `(B,) → (B,1,1,1)`:
  `def extract(a, t, shape): return a.gather(0, t).reshape(-1, 1, 1, 1)`.
- `reverse_process(x_t, t)` is already decorated `@torch.no_grad()` and `sample` calls `self.eval_mode()` — dropout off at sampling, as it must be.
- `sample()` receives `num_steps` via `**kwargs` from both `train.py::generate_samples` and `sample.py` (`config['sampling']['num_steps']`). For Q7 this must support `num_steps < num_timesteps` via a strided sub-schedule using ᾱ ratios.

### 3.3 `src/models/blocks.py` — PROVIDED building blocks (read-only, know the shapes)
| Block | In → Out | Notes |
|---|---|---|
| `SinusoidalPositionalEmbedding(dim)` | `(B,)` → `(B, dim)` | `[sin(t·f), cos(t·f)]`, geometric freqs `exp(−log(10000)·i/half)`; accepts int or float t |
| `TimestepEmbedding(D)` | `(B,)` → `(B, D)` | sinusoidal(D) → Linear(D→4D) → SiLU → Linear(4D→D). **Output dim = D**, so pass this D as `time_embed_dim` to every ResBlock |
| `GroupNorm32(32, C)` | same shape | computes in fp32, casts back — the AMP-stability trick. **Requires C divisible by 32** ⇒ base_channels must be a multiple of 32 |
| `ResBlock(Cin, Cout, D)` | `(B,Cin,H,W)+(B,D)` → `(B,Cout,H,W)` | GN→SiLU→3×3conv; time MLP → `(B,2·Cout)` FiLM: `h = GN(h)·(1+scale)+shift` (default `use_scale_shift_norm=True`); GN→SiLU→dropout→3×3conv; skip via 1×1 conv if Cin≠Cout |
| `AttentionBlock(C, heads)` | `(B,C,H,W)` → same | 1×1 qkv conv, attention over the H·W tokens, O((HW)²) — only use at ≤16×16 |
| `Downsample(C)` | `(B,C,H,W)` → `(B,C,H/2,W/2)` | stride-2 3×3 conv (learned) |
| `Upsample(C)` | `(B,C,H,W)` → `(B,C,2H,2W)` | nearest ×2 + 3×3 conv (avoids checkerboard) |

### 3.4 `src/models/unet.py` — assemble the U-Net (TODO)
- Constructor stores config (defaults: base=128, mult=(1,2,2,4), 2 res blocks/level, attention at [16,8], dropout 0.1) but builds **nothing**; `forward` raises.
- Contract: `forward(x:(B,3,64,64), t:(B,)) → (B,3,64,64)`. The `__main__` self-test checks exactly this and prints the param count (that's your Q4(a) "model size" number).
- Canonical DDPM-style assembly you must write: `conv_in(3→128)`; **encoder** per level i: `num_res_blocks` × ResBlock (channels → base·mult[i], attention if current resolution ∈ attention_resolutions), push every output to a skip list, `Downsample` between levels; **middle**: ResBlock → Attention → ResBlock; **decoder** mirrors with `num_res_blocks+1` ResBlocks per level, each taking `cat([h, skip.pop()], dim=1)` (so in_channels = current + skip channels — the classic shape-bug site), `Upsample` between levels; **head**: GroupNorm32 → SiLU → 3×3 conv → 3 channels. Time: one `TimestepEmbedding(base_channels)` computed once per forward, passed to every ResBlock.
- Resolution/channel trace at 64px, mult (1,2,2,4): 64²/128 → 32²/256 → 16²/256 (attn) → 8²/512 (attn); ~35–40M params at base 128.

### 3.5 `src/data/celeba.py` — data pipeline (transforms TODO)
- `CelebADataset` loads from HF Hub (`electronickale/cmu-10799-celeba64-subset`) or local dir; items are dicts with a PIL image (hub/Arrow) or a file path (local). `__getitem__` returns **image only** (unconditional generation), documented contract: `(3, 64, 64) in [-1, 1]`.
- `_build_transforms()` currently returns an **empty** `transforms.Compose([])` ⇒ `__getitem__` returns a PIL image and the DataLoader's default collate **crashes**. Nothing runs until you implement: `RandomHorizontalFlip` (train+augment only) → `ToTensor()` ([0,1]) → `Normalize(0.5,0.5)` or `Lambda(normalize)` ([−1,1]). Resize unneeded (already 64×64).
- Helpers you'll reuse everywhere: `normalize` ([0,1]→[−1,1]), `unnormalize` ([−1,1]→[0,1]), `make_grid`, `save_image` (wraps torchvision). **`save_image` does NOT unnormalize for you** — pass `unnormalize(samples).clamp(0,1)` yourself (this is exactly Kale's Q5(c) saturation bug).
- `create_dataloader`: shuffle=True for train, `drop_last=True`, workers/pin from config.

### 3.6 `src/utils/ema.py` — PROVIDED
- Shadow dict of fp32 param clones; `update()`: `shadow = d·shadow + (1−d)·param` after each optimizer step; decay warmup `min(d, (1+step)/(10+step))`.
- `apply_shadow()` backs up live weights and copies shadow in; `restore()` swaps back. `generate_samples` in train.py already brackets sampling with these — **sampling always uses EMA weights** once `step ≥ ema_start`.
- Gotcha: `apply_shadow` without `restore` (as `sample.py` does until exit) permanently mutates live weights — fine for inference, fatal if you copied that pattern mid-training.

### 3.7 `train.py` — training infrastructure (2 TODOs)
Flow: parse args → load YAML → `train()`:
1. DDP context from torchrun env vars; single-GPU if `num_gpus: 1`.
2. Seeds: `seed` for model init (identical across ranks), then re-seed `seed+rank` for data/noise divergence.
3. Build dataloader → UNet → (DDP wrap) → `DDPM.from_config` → AdamW (`lr, betas, weight_decay` from config) → `EMA(unwrap_model(model), ema_decay)` → `GradScaler(enabled=mixed_precision)`.
4. `--overfit-single-batch`: caches one batch (replicated to batch_size) and reuses it every step — your primary debugging tool (Q3b): loss should → ~0.0x and samples should reproduce the batch.
5. Per step: `optimizer.zero_grad()` → `autocast: loss, metrics = method.compute_loss(batch)` → `scaler.scale(loss).backward()` → `scaler.unscale_` + `clip_grad_norm_(gradient_clip_norm)` → `scaler.step` → `scaler.update` → `ema.update()`.
   - Your `compute_loss` must return `metrics` containing key **`'loss'`** — the pbar does `avg_metrics['loss']` and KeyErrors otherwise.
6. Every `sample_every`: `generate_samples(...)` — **TODO**: inside, call `samples = method.sample(num_samples, image_shape, num_steps=config['sampling']['num_steps'])`; EMA bracketing and train/eval mode toggles are already written around your line. Then `save_samples(samples, path, num_samples)` — **TODO**: `save_image(unnormalize(samples).clamp(0,1), path, nrow=int(num_samples**0.5))`.
7. Checkpoints: `{model, optimizer, scaler, step, config, ema}` — config embedded, which is how `sample.py` rebuilds everything from the .pt alone.
- Misleading comment: `# EMA update - DISABLED` — the very next line **calls** `ema.update()`. EMA is ON.

### 3.8 `sample.py` — inference (TODO + starter bugs)
- Rebuilds model+DDPM from checkpoint's embedded config, applies EMA unless `--no_ema`, loops batches of `--batch_size` calling `method.sample(batch_size, image_shape, num_steps=args.num_steps or config['sampling']['num_steps'])`, saves individual PNGs (default; what the KID script consumes) or one grid (`--grid`).
- **Starter bug 1** (line ~169): individual-image loop calls `save_samples(samples, img_path, 1)` passing the *whole batch* for every index `i` — must be `samples[i:i+2]` or refactor. If unfixed, every "individual" PNG is the full batch grid → KID computed on grids → garbage score.
- **Starter bug 2** (line ~186): grid path calls `save_samples(all_samples, args.output, nrow=8)` but the signature is `(samples, save_path, num_samples)` — `nrow` is an unexpected kwarg → TypeError. Fix when you write `save_samples`.
- KID pipeline expects PNGs in a flat dir; `fidelity --kid` defaults to kid-subset-size 1000 = your 1k samples vs dataset dir.

### 3.9 Configs, scripts, Modal
- `configs/ddpm_babel.yaml`: every hyperparameter is a **blank TODO**. Sane fill: base_channels 128, mult [1,2,2,4], 2 res blocks, attention [16,8], heads 4, dropout 0.1, scale_shift true; batch 128, lr 2e-4, wd 0.0, betas [0.9,0.999], ema_decay 0.9999, ema_start ~5000, clip 1.0, ~100–200k iters (budget-dependent); ddpm: T=1000, β 1e-4→0.02; sampling num_steps 1000; mixed_precision true.
- `scripts/train.sh`: slurm; reads `num_gpus` from config and launches torchrun if >1. Note its default `--gres=gpu:L40S:4` vs config `num_gpus: 1` comment — set both consistently.
- `scripts/evaluate_torch_fidelity.sh`: the Q4(c) oracle — regenerates 1000 samples then `fidelity --kid --input1 generated --input2 $DATASET_PATH`. DATASET_PATH must be a directory of raw images (the same distribution you trained on).
- `modal_app.py`: same train/sample entrypoints inside a Modal container with a persistent volume; $500 credits across all HWs.

## 4. Complete TODO inventory (implementation order)

1. `celeba.py::_build_transforms` — flip + ToTensor + [−1,1]. *(nothing runs before this)*
2. `unet.py::UNet.__init__/forward` — verify with the built-in `__main__` test.
3. `ddpm.py::__init__` — schedule buffers.
4. `ddpm.py::forward_process(x0, t, noise)` — reparameterized q(x_t|x₀).
5. `ddpm.py::compute_loss` — sample t~U{0..T−1}, ε~N(0,I), MSE(ε̂, ε); return `(loss, {'loss': loss})`.
6. Overfit-single-batch sanity run (CPU/1 GPU, minutes).
7. `ddpm.py::reverse_process` + `sample` (incl. t=0 no-noise case, strided support for Q7).
8. `train.py::generate_samples` (call sample) + `save_samples` (unnormalize+clamp+grid).
9. `sample.py::save_samples` + fix the two starter bugs (§3.8).
10. Config values → full training → KID script.

## 5. Starter-code gotchas found (before writing any code)

1. `sample.py` individual-save passes whole batch per file (§3.8) — silently ruins KID.
2. `sample.py` grid path passes `nrow=` to a function whose signature doesn't accept it.
3. `train.py` "EMA update - DISABLED" comment is false — EMA is active.
4. Empty `Compose([])` means the dataloader crashes (PIL images not collatable) until transforms are written.
5. `GroupNorm32(32, C)` ⇒ all channel counts must be multiples of 32.
6. `compute_loss` metrics must include the key `'loss'` (pbar KeyError otherwise).
7. `celeba.py` error message points to `dataset_processing/download_dataset.py`; the actual script is repo-root `download_dataset.py`.
8. Two "num steps" knobs exist: `ddpm.num_timesteps` (training T) vs `sampling.num_steps` (inference) — they coincide for Q4 (1000) and deliberately diverge in Q7; don't conflate them inside DDPM.
9. AMP: `compute_loss` runs under autocast — keep schedule buffers fp32 and let autocast handle casts; GroupNorm32 already protects norms.
10. `datasets` loads the whole split into a Python list (`list(dataset)`) — RAM-hungry but fine for this subset; don't "fix" it into lazy indexing mid-homework unless memory forces you.

## 6. One training iteration, with shapes (B=128, 64×64, T=1000)

```
batch                      (128,3,64,64) fp32 [-1,1] cuda
t ~ randint(0,1000)        (128,)        int64
ε ~ randn_like(x0)         (128,3,64,64)
√ᾱ_t, √(1−ᾱ_t) gathered    (128,1,1,1)   fp32 (broadcast)
x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε (128,3,64,64) fp16 under autocast
UNet: conv_in              (128,128,64,64)
  t → TimestepEmbedding    (128,128)      [→ (128,2C) per-ResBlock FiLM]
  enc L0 64²/128 → L1 32²/256 → L2 16²/256+attn → L3 8²/512+attn
  mid  8²/512 (Res→Attn→Res)
  dec mirrors, cat(skip) doubles Cin, → 64²/128
  head → ε̂                 (128,3,64,64)
loss = MSE(ε̂, ε)           scalar fp32
backward → clip(1.0) → AdamW → EMA
```

Sampling reverses: `x_T ~ N(0,I) (B,3,64,64)` → for t = T−1 … 0: ε̂ = UNet(x_t, t); μ = (x_t − β_t/√(1−ᾱ_t)·ε̂)/√α_t; x_{t−1} = μ + σ_t·z (z=0 at t=0) → `unnormalize + clamp` → grid/PNG.
