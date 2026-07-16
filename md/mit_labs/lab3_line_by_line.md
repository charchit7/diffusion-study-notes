# MIT Lab 3, Line by Line — A Conditional Generative Model for Images

*A beginner's reading companion to `solutions/lab_three_complete.ipynb`. Every
line of code: what it does, why it's there, and what breaks without it.
Assumes Python/PyTorch basics, transformer basics, and the cited notes:
CFG = `11 §4`, velocity fields & Euler = `06 §A1–A2`, Gaussian toolkit =
`05 §A3–A6`. Where a class is recycled from Lab 1/2 I say so and point at the
Lab 1 study (`lab1_line_by_line.md`).*

**The lab in one sentence:** graduate from 2-D toy clouds to 32×32 MNIST
digits you can *ask for by name* — train one flow-matching network that serves
both a conditional and an unconditional vector field (classifier-free
guidance, notes 11 §5), give it eyes worthy of images (a Diffusion
Transformer), then shrink the whole game into the latent space of a VAE
(latent diffusion, à la Stable Diffusion).

**Cell numbering:** code cells only, 0–36, in notebook order. The lab's six
parts map to: Part 0 (recycling) = Cells 1–9, Part 1 (MNIST) = 10–11,
Part 2 (CFG) = 12–16, Part 3 (DiT) = 17–24, Part 4 (VAE) = 25–34,
Part 5 (latent diffusion) = 35–36.

---

## Cell 0 — Imports: the new tools of an image lab

```python
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import uuid
import random
```
Standard library. `os` for creating checkpoint directories (`runs/<name>` in
Cell 9); `ABC`/`abstractmethod` for the same enforceable contracts as Lab 1
(lab 1 study, Cell 0); typing helpers are documentation-only; `math` for
`math.pi`/`math.sqrt` on plain Python floats; `uuid` and `random` exist purely
to generate cute run names like `misty-durian-3f2a91bc` (Cell 9).

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
```
Same cast as Labs 1–2: plotting, PyTorch, the autodiff pair `vmap`/`jacrev`
(used again for the default `Alpha.dt`/`Beta.dt` — see lab 1 study, Cell 11
for how that pair computes derivatives without hand math), a progress bar.
`seaborn`, `make_moons`, `make_circles` are imported but never used in this
lab — leftovers from Lab 2's toy datasets.

```python
from torchvision import datasets, transforms
from torchvision.utils import make_grid
```
New, because we now have *images*: `datasets.MNIST` downloads the dataset
(Cell 10), `transforms` resizes/normalizes it, and `make_grid` tiles a batch
of image tensors `(b, c, h, w)` into one big image for plotting — you'll see
it in every visualization cell.

```python
from einops import rearrange
from einops.layers.torch import Rearrange
```
The other new star. `einops.rearrange(x, 'b c h w -> b (h w) c')` reshapes
tensors by *naming the axes* — self-documenting and shape-checked at runtime.
`Rearrange` (capital R) is the same thing as an `nn.Module` layer, so it can
sit inside `nn.Sequential`. The Patchifier/Depatchifier (Cells 18, 20) are
built almost entirely out of these; a wrong einops pattern throws an error
instead of silently scrambling your image, which is why the lab prefers it
over `.view()`/`.permute()` chains.

---

## Cell 1 — `Sampleable`: the unlabeled contract (recycled)

```python
class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
```
Identical in spirit to Lab 1's `Sampleable` (lab 1 study, Cell 11): "I can
hand you `num_samples` draws from myself," returning shape `(b, d)` — or, as
this lab generalizes later, `(b, ...)` for images. It stays *unlabeled*: this
is the contract the noise distribution p_simple will satisfy, because noise
has no class.

## Cell 2 — `LabeledSampleable`: joint samples of data *and* labels

```python
class LabeledSampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```
The one-line upgrade that makes conditional generation possible. `sample` now
returns a pair: `samples: (b, ...)` **and** `labels: (b,)`. Formally we are
now sampling from the *joint* distribution p_data(z, y) — an MNIST draw gives
you an image z and its digit identity y ∈ {0,…,9} together. Every piece of
CFG training (Cell 13) starts with exactly this joint draw; if your data
source can't hand back labels, there is nothing to condition on.

## Cell 3 — `IsotropicGaussian`: p_simple as an object

```python
class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
```
A pedantic but useful wrapper around `torch.randn`. `shape` is the *per-sample*
shape — `[2]` for the GMM sanity check, `[1, 32, 32]` for MNIST pixels,
`[128, 4, 4]` for VAE latents — so the same class serves all three worlds.
Inheriting `nn.Module` exists for one reason only:

```python
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
```
A one-element buffer whose only job is to *remember the device*. Buffers move
with `.to(device)` (same trick as DDPM schedule buffers — lab 1 study,
Cell 13), so the class can create new random tensors on the right device:

```python
    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
```
`torch.randn(b, *shape)` draws standard normal of shape `(b, *shape)`; scaling
by `std` gives N(0, std²·I) by reparameterization (notes 05 §A3). The
`.to(self.dummy.device)` is the load-bearing part: without the dummy-buffer
trick, sampling inside a CUDA training loop would produce CPU tensors and you
would hit the classic "expected all tensors on the same device" crash.

## Cell 4 — `GMM`: a labeled toy distribution

```python
class GMM(nn.Module, LabeledSampleable):
  def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.means = nn.Buffer(means)
    self.covariances = nn.Buffer(covariances)
    self.weights = nn.Buffer(weights)
```
A Gaussian mixture where the *mixture component index is the label* — the
perfect sanity-check distribution for conditional generation, because
"condition on y = 1" should visibly mean "generate only from bump 1." Shapes:
`means (K, d)`, `covariances (K,)`, `weights (K,)` for K components. All
stored as buffers so `.to(device)` carries them along.

```python
  def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.multinomial(self.weights.cpu(), num_samples=num_samples, replacement=True).to(self.means.device)
```
Step 1: pick which bump each sample comes from, proportional to `weights` —
that's what `multinomial` with `replacement=True` does, returning `(b,)`
integer labels. The `.cpu()` round-trip is a pragmatic dodge: `multinomial`
on CUDA can raise device-side asserts in some versions, so sample the *tiny*
label tensor on CPU and move it back.

```python
    samples = torch.zeros(num_samples, self.means.shape[1]).to(self.means.device)
    for idx in range(len(self.means)):
      samples[labels == idx] = torch.randn_like(samples[labels == idx]) * self.covariances[idx] + self.means[idx]
```
Step 2: for each component, fill the rows assigned to it with
`randn·scale + mean` — reparameterization again (notes 05 §A3). Boolean-mask
assignment (`samples[labels == idx] = ...`) writes only the matching rows.
One naming quibble worth noticing: the attribute is called `covariances` but
it multiplies the noise directly, so it is actually used as a *standard
deviation* (a covariance of 0.2 would need `sqrt(0.2)` here). Harmless for a
toy, but don't copy the name into real code.

```python
    return samples, labels
```
The `LabeledSampleable` contract fulfilled: `(b, 2)` points and `(b,)` labels.

## Cell 5 — `ConditionalProbabilityPath`: the noising recipe, now label-aware

Recycled from Lab 2 with two changes flagged in the markdown: (1) the
conditioning variable is now the *pair* (z, y) ~ p_data(z, y); (2) all shapes
are generalized from `b d` to `b ...` so the same class handles 2-D points
and `b c h w` images.

```python
class ConditionalProbabilityPath(nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
```
A probability path is defined by its two endpoints: the easy distribution
p_simple (noise, t = 0) and the data distribution p_data (t = 1). **Note the
time convention:** in this lab, as in flow matching generally (notes 06
Part E), t runs *noise → data* as 0 → 1 — the opposite direction from DDPM's
timestep index.

```python
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples) # (b ...)
        x = self.sample_conditional_path(z, t) # (b ...)
        return x
```
"Give me a draw from the marginal p_t(x)": first pick a data point z (the
label y is discarded with `_` — marginal sampling doesn't need it), then
noise it to level t. This is the two-step definition
p_t(x) = ∫ p_t(x|z) p(z) dz executed literally: sample the integrand instead
of integrating.

```python
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
```
The four abilities any concrete path must provide, all conditioned on a fixed
z ("pretend there is only one image" — notes 06 Part C): draw (z, y); draw
x ~ p_t(x|z); evaluate the *training target* u_t(x|z) (the arrow the network
regresses against, notes 06 §D2); and evaluate the conditional score
∇log p_t(x|z) (unused in this lab's training but kept for SDE sampling).
Shapes throughout: `z, x: (b, ...)`, `t: (b,)`, outputs `(b, ...)`.

## Cell 6 — Alpha, Beta, and the Gaussian path (177 recycled lines)

This is the longest recycled cell; it defines the noise schedule abstractions
and the concrete Gaussian path used for *everything* downstream.

```python
class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(self(torch.zeros(1,)), torch.zeros(1,))
        assert torch.allclose(self(torch.ones(1,)), torch.ones(1,))
```
α_t is the "how much data" dial. The constructor *asserts the boundary
conditions* α_0 = 0, α_1 = 1 the moment any subclass is instantiated — a
runtime contract check: get the schedule endpoints wrong and the class
refuses to exist, instead of silently training a path that doesn't start at
noise or end at data.

```python
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)
```
Subclasses supply α_t; the base class supplies its time-derivative α̇_t *for
free* via autodiff — `jacrev` differentiates, `vmap` maps it over the batch
(exactly the machinery dissected in the lab 1 study, Cell 11). The
`unsqueeze`/`view(-1)` dance feeds `jacrev` the `(b, 1)` shape it wants and
flattens the resulting `(b, 1, 1)` Jacobian back to `(b,)`. This means you
can invent any differentiable schedule and never write its derivative.

```python
class Beta(ABC):
    def __init__(self):
        assert torch.allclose(self(torch.zeros(1)), torch.ones(1))
        assert torch.allclose(self(torch.ones(1)), torch.zeros(1))
```
β_t is the "how much noise" dial, with mirrored endpoints β_0 = 1, β_1 = 0:
all noise at t = 0, none at t = 1. Same autodiff `dt` as `Alpha`.

```python
class LinearAlpha(Alpha):
    def __call__(self, t): return t
    def dt(self, t): return torch.ones_like(t)

class LinearBeta(Beta):
    def __call__(self, t): return 1-t
    def dt(self, t): return - torch.ones_like(t)
```
The concrete choice: α_t = t, β_t = 1−t — the straight-line / conditional-OT
schedule of notes 06 Part C (Eq. ▲). Both override `dt` with the exact
constant (±1) rather than paying for autodiff; `ones_like(t)` keeps shape and
device correct.

```python
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.rearrange_scalar = Rearrange(f'b -> b{" 1" * len(p_simple_shape)}')
```
The concrete path p_t(x|z) = N(α_t z, β_t² I). It builds its own unit
Gaussian p_simple from a shape spec. The last line is the shape-generalization
workhorse: for `p_simple_shape = [1, 32, 32]` the f-string evaluates to
`'b -> b 1 1 1'`, i.e. a layer that turns a `(b,)` scalar-per-sample tensor
into `(b, 1, 1, 1)` so it broadcasts against images. This single line is what
lets one class serve `(b, 2)` points and `(b, 1, 32, 32)` images; forgetting
the reshape is the classic bug where `t * z` either crashes or, worse,
broadcasts along the wrong axis.

```python
    def sample_conditioning_variable(self, num_samples: int):
        return self.p_data.sample(num_samples)
```
Delegates to the labeled dataset — returns the `(z, y)` pair straight through.

```python
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.rearrange_scalar(self.alpha(t)) # (b 1 1 1)
        beta_t = self.rearrange_scalar(self.beta(t)) # (b 1 1 1)
        return alpha_t * z + beta_t * torch.randn_like(z)
```
Draw x_t ~ N(α_t z, β_t² I) by reparameterization (notes 05 §A3):
`mean + std·ε`. With the linear schedule this is x_t = t·z + (1−t)·ε — the
straight-line interpolant between a fresh noise draw and the data point
(notes 06 Eq. ▲, with x₀ = ε, x₁ = z). `randn_like(z)` guarantees matching
shape, dtype, and device.

```python
    def conditional_vector_field(self, x, z, t):
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        dt_alpha_t = self.rearrange_scalar(self.alpha.dt(t))
        dt_beta_t = self.rearrange_scalar(self.beta.dt(t))
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
```
The training target u_t(x|z), in the general Gaussian-path form. Sanity-check
it against the notes: plug in α = t, β = 1−t, α̇ = 1, β̇ = −1 and simplify —
you get u = (z − x)/(1−t), and substituting x = t·z + (1−t)·ε gives
u = z − ε, exactly the constant "noise-to-data arrow" x₁ − x₀ of notes 06
Eq. ▲▲. **Note the division by β_t:** at t = 1 this is 0/0 — which is why the
trainers (Cells 13, 35) never sample t all the way to 1.

```python
    def conditional_score(self, x, z, t):
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        return (z * alpha_t - x) / beta_t ** 2
```
∇log N(x; α_t z, β_t²) = (mean − x)/variance — the standard Gaussian score
(differentiate the log-density exponent, notes 05 §A2). Equals −ε/β_t when
x = α_t z + β_t ε. Not used for training here, but kept so an SDE sampler
could be bolted on.

## Cell 7 — ODE and SDE contracts, now with `**kwargs`

```python
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor: ...

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t, **kwargs) -> torch.Tensor: ...
    @abstractmethod
    def diffusion_coefficient(self, xt, t, **kwargs) -> torch.Tensor: ...
```
Byte-for-byte the Lab 1 contracts (lab 1 study, Cell 1) with two upgrades the
markdown announces: shapes are `b ...` instead of `b d`, and every method
grows a `**kwargs`. That `**kwargs` is the *entire* mechanism by which the
class label y reaches the sampler: `simulate(..., y=labels)` threads it down
through `step` into `drift_coefficient` (see Cell 12) without any of the
generic plumbing knowing what "y" means. It's loose typing on purpose —
maximal reuse, at the cost that a typo like `simulate(..., labels=y)` fails
only deep inside the network call.

## Cell 8 — Simulators: per-sample time grids

```python
class Simulator(ABC):
    @abstractmethod
    def step(self, xt, t, dt, **kwargs): ...

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs):
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x
```
Same march-through-the-grid loop as Lab 1 (lab 1 study, Cell 2), with one
structural change: `ts` is now `(b, nt)` — **each sample carries its own time
grid** — so `t = ts[:, t_idx]` is a `(b,)` vector, not a scalar. In this lab
all rows are identical (`linspace(...).expand(b, -1)`), but the shape is
future-proof. `nts - 1` steps for `nts` timestamps, `h` derived from the grid
(non-uniform grids work for free), `@torch.no_grad()` because sampling is
inference. `use_tqdm` is new: checkpoint callbacks call `simulate` many times
during training and would otherwise flood the log with progress bars.

```python
    @torch.no_grad()
    def simulate_with_trajectory(self, x, ts, use_tqdm=True, **kwargs):
        x_traj = [x.clone()]
        ...
        return torch.stack(x_traj, dim=1)
```
Identical to the Lab 1 movie-recording version: clone every frame (defensive
copy), stack along `dim=1` to get `(b, nt, ...)`. Unused in this notebook's
driver cells but part of the recycled kit.

```python
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
    def step(self, xt, t, h, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h
```
Euler's method (notes 06 §A1 Fact 2): next = current + arrow × step. The new
line is the `h.view(...)`: `h` arrives as `(b,)` (or `(b,1,1,1)` when the
driver builds a 5-D `ts` — see Cell 22) and must broadcast against
`(b, c, h, w)` images, so it is reshaped to `(b, 1, 1, …)` with as many
trailing 1s as `xt` has non-batch dims. Skip this and `h * drift` either
crashes or broadcasts across the wrong axis. `**kwargs` (carrying `y=`)
passes straight through to the ODE.

```python
class EulerMaruyamaSimulator(Simulator):
    def step(self, xt, t, h, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h \
             + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)
```
The stochastic sibling, unchanged from Lab 1 including the all-important
`sqrt(h)` on the noise (variances add over time, so std scales as √h — lab 1
study, Cell 4; notes 05 §A1). Defined but not driven in this lab: all image
sampling here is deterministic Euler.

```python
def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat([torch.arange(0, num_timesteps - 1, record_every),
                      torch.tensor([num_timesteps - 1])])
```
Utility for subsampling trajectory frames: indices 0, k, 2k, … plus always
the final index (the endpoint is the sample!). Same as Lab 1's
`every_nth_index`. Unused downstream here.

## Cell 9 — `Trainer`: the reusable training loop with checkpoints

New in Lab 3: a beefier base trainer with run directories, LR warmup, and a
checkpoint hook. Worth reading once carefully because *three* subclasses hang
off it (Cells 13, 32, 35).

```python
MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size
```
Bytes = element count × bytes per element, summed over parameters *and*
buffers. Purely informational (printed at training start), but a good habit:
a DiT that silently grew to gigabytes is caught at a glance.

```python
class Trainer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.opt = None
        self.output_dir = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor: ...

    def checkpoint(self, step: int):
      pass
```
The contract: subclasses must define *what the loss is*
(`get_train_loss`) and may override *what happens at checkpoints*
(`checkpoint`, a no-op by default). Model/optimizer are attached later in
`train` — the trainer is constructed before it knows its model.

```python
    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
```
AdamW with mild weight decay — the boring, correct default for transformers.
Overridable per subclass, though nobody does.

```python
    def random_name(self) -> str:
        adjectives = [...]; foods = [...]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"
```
Human-readable run names ("quiet-jicama-1a2b3c4d"); the UUID suffix makes
collisions practically impossible.

```python
    def train(self, model, num_steps, lr=1e-3, warmup_steps=500,
              ckpt_every: Optional[int] = 500, run_name=None, **kwargs):
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join("runs", run_name)
        os.makedirs(self.output_dir, exist_ok=False)
```
Every run gets a fresh directory `runs/<name>` for checkpoints and preview
images. `exist_ok=False` means reusing a `run_name` crashes immediately —
deliberate, so you can never silently overwrite a previous run's checkpoints.

```python
        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")
        self.opt = self.get_optimizer(lr)
        self.model.train()
        for pg in self.opt.param_groups:
            pg["lr"] = 0.0
```
Attach model, report its size, build the optimizer, switch to train mode
(matters once we have models with mode-dependent behavior), and zero the LR
so warmup starts from 0 rather than from `lr` on step 0.

```python
        pbar = tqdm(range(num_steps))
        for step in pbar:
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr
```
Linear LR warmup: ramp 0 → lr over `warmup_steps`, then hold constant.
Warmup protects the early steps, when gradients from a randomly initialized
transformer are noisy and a full-size Adam step can wreck the initialization
(this pairs with the adaLN-Zero trick in Cell 19 — both are
"start gently" devices).

```python
            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            self.opt.step()
```
The canonical four-beat PyTorch step. `set_to_none=True` frees gradient
memory instead of zero-filling it (slightly faster). Note that `**kwargs`
passed to `train` (e.g. `batch_size=256`) flows into `get_train_loss` —
that's how subclasses receive their batch size.

```python
            losses.append(float(loss.detach().item()))
            steps.append(step)
            pbar.set_description(f"Step {step}, lr={cur_lr:.2e}, loss={loss.item():.4f}")
```
Bookkeeping: record the scalar loss (`.detach().item()` drops the graph and
converts to Python float — keep tensors here and you leak the entire autograd
graph every step) and paint live stats onto the progress bar.

```python
            if ckpt_every is not None and step % ckpt_every == 0 and step > 0:
              self.model.eval()
              self.checkpoint(step)
              self.model.train()
        self.model.eval()
        return losses, list(range(num_steps))
```
Every `ckpt_every` steps (skipping step 0), flip to eval mode, run the
subclass's checkpoint hook (save weights, render sample grids), flip back.
The eval/train toggling matters because checkpoints *sample from the model*.
Finally leave the model in eval mode and return the loss curve. (Returning
`list(range(num_steps))` instead of the accumulated `steps` list is
redundant-but-equal.)

---

## Cell 10 — `MNISTSampler`: the dataset as a `LabeledSampleable`

```python
class MNISTSampler(nn.Module, LabeledSampleable):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1305,), (0.2891,)),
            ])
        )
```
Running this cell **downloads MNIST** (~10 MB) into `./data` on first use —
60,000 training images of handwritten digits with labels 0–9. The transform
pipeline runs per image at access time: `Resize((32, 32))` upsamples the
native 28×28 to 32×32 (a power of two, so it patchifies evenly by 4 or 8 in
Cell 18 and halves cleanly three times in the VAE encoder, Cell 28);
`ToTensor()` converts PIL → float tensor `(1, 32, 32)` in [0, 1];
`Normalize` subtracts MNIST's mean 0.1305 and divides by std 0.2891, giving
roughly zero-mean unit-variance pixels — the same scale as the N(0, I) noise
the path mixes them with. Skip normalization and the flow has to bridge
mismatched scales, which trains noticeably worse.

```python
        self.dummy = nn.Buffer(torch.zeros(1))
```
The device-tracking dummy buffer again (Cell 3).

```python
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
```
Instead of a DataLoader, sampling is "shuffle all indices, take the first b"
— a random subset *without replacement* each call. `zip(*[...])` transposes
the list of (image, label) pairs into two tuples; `torch.stack` makes the
`(b, 1, 32, 32)` batch. Note `.to(self.dummy)` (tensor, not `.device`) moves
to the dummy's device *and dtype* in one call. Labels become int64 because
`nn.Embedding` (Cells 14, 21) demands integer indices. This design is slower
than a DataLoader (Python loop, full `randperm` of 60k every batch) but keeps
the `LabeledSampleable` abstraction perfectly uniform with the GMM.

## Cell 11 — Watching MNIST dissolve along the path

A driver cell: visualize p_t(x|z) at several times for a grid of digits.

```python
num_rows = 3
num_cols = 3
num_timesteps = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampler = MNISTSampler().to(device)
```
Knobs, device pick, and a sampler instance — which is then, amusingly, never
used: the next statement builds the path with a *second* `MNISTSampler()`.
Dead variable; harmless (both wrap the same downloaded dataset).

```python
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)
```
The pixel-space path: p_t(x|z) = N(t·z, (1−t)²·I) over `(1, 32, 32)` images.
Because the path is an `nn.Module` and MNISTSampler is a submodule,
`.to(device)` propagates to the dummy buffers inside — this is the payoff of
all that buffer plumbing.

```python
num_samples = num_rows * num_cols
z, _ = path.p_data.sample(num_samples)
z = z.view(-1, 1, 32, 32)
```
Draw 9 clean digits `(9, 1, 32, 32)`; the `view` is a no-op safety reshape.
Labels discarded — this cell is about noising, not conditioning.

```python
fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))
ts = torch.linspace(0, 1, num_timesteps).to(device)
for tidx, t in enumerate(ts):
    tt = t.expand(num_samples) # b
    xt = path.sample_conditional_path(z, tt) # b 1 32 32
    grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
    axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    axes[tidx].axis("off")
plt.show()
```
One subplot per time t ∈ {0, 0.25, 0.5, 0.75, 1}. `t.expand(num_samples)`
broadcasts the scalar to the `(b,)` shape the path expects. Each panel:
noise the *same* 9 digits to level t, tile with `make_grid` (3 per row,
normalized into the display range), `permute(1,2,0)` because matplotlib
wants `(h, w, c)` not `(c, h, w)`, `.cpu()` because matplotlib can't eat CUDA
tensors. **What you see confirms the time convention:** t = 0 is pure static,
t = 1 is clean digits — noise on the left, data on the right, the direction
the sampler will travel.

---

## Cell 12 — `ConditionalVectorField` + `CFGVectorFieldODE`: guidance at sampling time

The markdown cells before this one derive CFG in the lab's notation. Bridge
to our notes: starting from u_t(x|y) = a_t x + b_t ∇log p_t(x|y), apply Bayes
in score space, ∇log p_t(x|y) = ∇log p_t(x) + ∇log p_t(y|x) (notes 11 §2),
scale the classifier term by w, and substitute back to get the boxed formula

    ũ_t(x|y) = (1−w)·u_t(x|∅) + w·u_t(x|y).

Rearranged, this is exactly notes 11 §4's
`eps_cfg = eps_uncond + s·(eps_cond − eps_uncond)` with s = w, written for
velocities instead of ε (notes 11 §7: the formula is representation-agnostic).
Watch the convention: this lab's w is Stable Diffusion's `guidance_scale` s,
*not* the "s = 1+w" convention some papers use (notes 11 §4's convention
warning). w = 1 is plain conditional; w > 1 extrapolates past the conditional
prediction.

```python
class ConditionalVectorField(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
```
The network contract for u_t^θ(x|y): inputs `x: (b, ...)`, `t: (b,)`,
`y: (b,)` integer labels (including the null label), output the same shape as
`x`. Both the MLP (Cell 14) and the DiT (Cell 21) implement this — which is
why the trainer and sampler never care which architecture they're driving.

```python
class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, null_label: int, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label
```
An ODE (Cell 7 contract) *wrapping* a network: this is where guidance lives.
Note what this placement means: **guidance is a sampling-time construction
only** — the training loss (next cell) never sees `guidance_scale`. Baking w
into training is a classic conceptual bug; the whole point of CFG is that one
trained model supports every w afterwards (notes 11 §5).

```python
    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * self.null_label
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field
```
Line by line: (1) one forward pass with the real labels → u_t^θ(x|y);
(2) build a label tensor of all-∅ (`ones_like` keeps shape/device/dtype,
times the null index); (3) a second forward pass → u_t^θ(x|∅) — the *same*
network serving as its own unconditional model, the condition-dropout payoff
(notes 11 §5, where production code fuses the two passes as
`torch.cat([latents]*2)` + `chunk(2)`; here they are two separate calls for
clarity); (4) blend with the boxed formula. Every Euler step therefore costs
two network evaluations — CFG's standing tax. Note `y` is a *named* argument
here: the `**kwargs` chain `simulate(..., y=labels) → step → drift_coefficient`
lands on this signature, so calling `simulate` without `y=` is a TypeError.
With w = 1 the unguided term is multiplied by 0 but still computed — a small
inefficiency the lab doesn't bother optimizing away.

## Cell 13 — `CFGTrainer`: flow matching with label dropout

```python
class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label
```
The trainer owns the *path* (which owns the data) and two CFG-specific knobs:
`eta` = probability of replacing a real label with ∅ (the "condition dropout"
rate, notes 11 §5 — papers use 10–20%; this lab uses 25–35%, sensible for
only 10 classes where the unconditional task needs plenty of gradient
signal), and `eps` = how far short of t = 1 to stop (see below). The assert
catches the two degenerate settings: η = 0 means the model never learns
u_t(x|∅) and the unguided pass at sampling time is garbage; η = 1 means it
never learns conditioning at all.

```python
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size) # b ..., b
```
One joint draw from p_data(z, y): images `(b, 1, 32, 32)` (or GMM points
`(b, 2)`) and labels `(b,)`. This is line 1 of the boxed training recipe in
the markdown.

```python
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = self.null_label
```
Label dropout, done right: draw one uniform number **per sample** and null
out exactly those rows where it falls below η. Note `torch.rand` (uniform),
not `torch.randn` (Gaussian — the markdown's own hint 3 warns about this
mixup; with `randn`, "P(ξ < 0.35)" would be ≈ 0.64, silently more than
doubling your dropout rate). The classic alternative bug is per-*batch*
dropout — `if random.random() < eta: y[:] = null` — which forces every batch
to be all-conditional or all-unconditional, gives much noisier gradients, and
correlates the two tasks with batch statistics. Also note this mutates `y` in
place, which is fine only because `y` was freshly sampled two lines up.

```python
        # Step 3: Sample t and x
        t = torch.rand(batch_size).to(z) * (1 - self.eps) # b
        x = self.path.sample_conditional_path(z,t) # b ...
```
t ~ U[0, 1−ε) with ε = 0.001. The ε exists because the regression *target*
below divides by β_t = 1−t (Cell 6): at t = 1 that's a division by zero, and
just below it the target's magnitude explodes. Clipping the time range
sidesteps the singularity. `.to(z)` moves t to z's device *and* dtype in one
call. Then draw x_t = α_t z + β_t ε from the conditional path — shape
`(b, ...)`, with the `(b,)` time broadcast handled inside the path by
`rearrange_scalar`.

```python
        # Step 4: Regress and output loss
        ut_theta = self.model(x,t,y) # b ...
        ut_ref = self.path.conditional_vector_field(x,z,t) # b ...
        return torch.square(ut_theta - ut_ref).mean()
```
The CFM loss: network prediction u_t^θ(x|y) vs. the analytic conditional
target u_t(x|z), plain MSE. The magic that makes regressing the *easy*
per-pair target learn the *intractable* marginal field is notes 06 §D2
("the minimizer of the trivial loss IS the marginal field"); the only Lab 3
novelty is that y rides along as an input — including, η of the time, y = ∅,
which is what teaches the same network its unconditional half. `.mean()`
averages over batch *and* all data dimensions, so the loss scale is
resolution-independent.

## Cell 14 — `MLP` + `MLPConditionalVectorField`: the sanity-check network

```python
class MLP(nn.Module):
  def __init__(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU, final_init: bool = False):
    super().__init__()
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    self.net = torch.nn.Sequential(*mlp)
```
A generic MLP builder from a dims list, e.g. `[5, 256, 256, 2]` → Linear,
SiLU, Linear, SiLU, Linear. The `idx < len(dims) - 2` guard omits the
activation after the final layer — outputs are unconstrained real vectors
(velocities can be negative!). SiLU (x·sigmoid(x)) is the smooth default
activation of the diffusion literature.

```python
    if final_init:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)
```
Optional zero-init of the last layer, so the whole MLP outputs exactly 0 at
initialization. Unused for the vector-field MLP but reused with
`final_init=True` inside residual blocks later (Cell 26), where "the branch
contributes nothing at init" makes the residual network start as an identity
map — the same stabilization philosophy as adaLN-Zero (Cell 19).

```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```
`nn.Linear` acts on the last axis, so the same MLP happily processes
`(b, d)` or `(b, n, d)` token tensors — which is why the DiT reuses this
exact class for its feed-forward blocks.

```python
class MLPConditionalVectorField(ConditionalVectorField):
  def __init__(self, dim: int, hidden_dim: int, class_dim: int, num_classes: int):
    super().__init__()
    self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim])
    self.class_embedding = nn.Embedding(num_classes + 1, class_dim)
```
The Lab-2-style network, upgraded with a label input. Two pieces: an MLP
whose input dimension is data (`dim`) + label embedding (`class_dim`) + time
(1 scalar), and an embedding table with **`num_classes + 1` rows — the +1 is
the null label ∅.** Forget the +1 and training crashes (or worse, silently
gathers garbage) the first time dropout produces index `num_classes`.
`nn.Embedding` is just a learnable lookup table: integer in, `(class_dim,)`
vector out.

```python
  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
      xyt = torch.cat([
          x,
          self.class_embedding(y),
          t.unsqueeze(-1)
      ], dim=-1)
      return self.mlp(xyt)
```
Conditioning by concatenation, the bluntest instrument that works: glue
`x (b, 2)`, `emb(y) (b, class_dim)`, and `t (b,) → (b, 1)` into one
`(b, dim + class_dim + 1)` vector and let the MLP sort it out. Fine at 2-D;
hopeless at 1024-D images — which is precisely the motivation speech for
Part 3.

## Cell 15 — Sanity check: train the MLP on a 3-mode GMM

Driver cell; every line grouped by purpose.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
covs = torch.tensor([0.2, 0.2, 0.2])
weights = torch.tensor([1/3, 1/3, 1/3])
gmm = GMM(means, covs, weights).to(device)
```
Build a 3-component mixture with means equally spaced on a circle of radius 2
(angles 0°, 120°, 240°), tight equal spreads, equal weights. Three
well-separated bumps = three unambiguous classes, ideal for eyeballing
whether conditioning works.

```python
path = GaussianConditionalProbabilityPath(
    p_data = gmm, p_simple_shape = [2],
    alpha = LinearAlpha(), beta = LinearBeta()
).to(device)
vector_field = MLPConditionalVectorField(dim=2, hidden_dim=256, class_dim=2, num_classes=3).to(device)
```
The same path class as MNIST, with `p_simple_shape=[2]` (so
`rearrange_scalar` becomes `'b -> b 1'`). The MLP embeds 4 labels
(3 classes + ∅) into 2-D. Note both `.to(device)` calls — the path carries
buffers, the model carries parameters; forget either and the first training
step dies with a device mismatch.

```python
trainer = CFGTrainer(path=path, eta=0.25, null_label=3)
losses, steps = trainer.train(model=vector_field, num_steps=3000, lr=1e-3, batch_size=250)
plt.plot(steps, losses); plt.xlabel("Step"); plt.ylabel("Loss"); plt.show()
```
Null label = 3 (first unused integer after classes 0–2), 25% dropout, 3000
steps of batch-250 — seconds on a GPU. Running it prints the run directory
and model size, shows a tqdm bar with live loss, and the plot shows the loss
falling to a *noise floor that is not zero*: the CFM target is stochastic
(different (z, ε) pairs through the same (x, t) — notes 06 §D2's variance
term), so the residual variance never trains away. Flat-but-nonzero is
success.

## Cell 16 — Visualizing guided vs. unguided GMM samples

Three panels: target, per-class guided samples, unconditional samples.

```python
guidance_strength = 1.0 # try changing me!
fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6))

ax = axes[0]
x_data, _ = gmm.sample(250)
x_data = x_data.detach().cpu().numpy()
ax.scatter(x_data[:, 0], x_data[:, 1], s=5, marker="*")
ax.set_title("Target")
```
Panel 1: ground truth — 250 real GMM draws (`.cpu().numpy()` for matplotlib),
small star markers.

```python
ax = axes[1]
cfg_vector_field = CFGVectorFieldODE(vector_field, guidance_scale=guidance_strength, null_label=3)
simulator = EulerSimulator(cfg_vector_field)
```
Assemble the sampling stack: trained net → CFG ODE wrapper (Cell 12) → Euler
simulator (Cell 8). Composition, exactly the Lab 1 pattern — the sampler
neither knows nor cares that its drift internally runs two network passes.

```python
batch_size = 250
labels = torch.arange(3).repeat_interleave(batch_size).to(device)
x_init = path.p_simple.sample(3 * batch_size) # b 2
ts = torch.linspace(0, 1, 100).expand(3 * batch_size, -1).to(device) # b nt
xs = simulator.simulate(x_init, ts, y=labels) # b 2
```
`repeat_interleave` builds `[0]*250 + [1]*250 + [2]*250` — 250 requests for
each class in one batch. Start all 750 particles at N(0, I), give each the
same 100-point grid on [0, 1] (`expand` creates a `(750, 100)` view without
copying), and simulate with `y=labels` riding the kwargs chain into the CFG
drift. 99 Euler steps later, `xs` are model samples.

```python
for idx in range(3):
    xs_idx = xs[idx * batch_size: (idx + 1) * batch_size].detach().cpu().numpy()
    ax.scatter(xs_idx[:, 0], xs_idx[:, 1], s=5, label=f"Mode {idx}", marker="*")
ax.legend(); ax.set_title(f"CFG w/ Guidance Strength {guidance_strength:.2f}")
```
Slice the batch back into its three class-blocks and scatter each in its own
color. Success looks like: each color sits *only* on its own bump. This panel
is the whole point of Part 2 in one picture. Re-run with
`guidance_strength = 3` and watch each cluster *tighten toward its mode* —
the diversity-for-fidelity trade of notes 11 §6.

```python
ax = axes[2]
batch_size = 750
labels = torch.ones(batch_size).long().to(device) * 3
x_init = path.p_simple.sample(batch_size)
ts = torch.linspace(0, 1, 100).expand(batch_size, -1).to(device)
xs = simulator.simulate(x_init, ts, y=labels).detach().cpu().numpy()
ax.scatter(xs[:, 0], xs[:, 1], s=5, label=f"Mode {idx}", marker="*")
ax.set_title(f"Unguided Samples")
```
Panel 3: pass the *null* label (3) for everyone — the model's unconditional
half, learned purely from the η = 25% dropout examples. It should reproduce
all three bumps with equal weight, i.e. match Panel 1. (`.long()` because
embeddings need int64; the leftover `label=f"Mode {idx}"` reuses a stale loop
variable — cosmetic slop.) Note the two panels together *are* the two terms
of the CFG formula: guided uses both, this one is the u_t(x|∅) ingredient
alone.

---

## Cell 17 — `FourierEncoder`: time as a vector of waves

Part 3 begins: building the DiT. First component — turn the scalar t into a
`dim`-vector the transformer's conditioning pathway can use. The markdown's
recipe: t ↦ [cos(2πw_i t), …, sin(2πw_i t)] with w_i ~ N(0, 1).

```python
class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))
```
`dim` must be even because the output is half sines, half cosines. The
frequencies start as N(0, 1) draws per the formula — but note they are an
`nn.Parameter`, **so the frequencies are learned**, not fixed like the
classic sinusoidal positional encoding (the lab follows lucidrains' Karras
U-Net implementation here). Random frequencies at multiple scales let the
network resolve both slow trends and fine detail in t; a raw scalar t would
force the network to carve fine time resolution out of one number.

```python
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1) # b 1
        freqs = t * self.weights * 2 * math.pi # b hd
```
`view(-1, 1)` flattens whatever shape t arrives in — `(b,)` from the trainer
or `(b, 1, 1, 1)` from the 5-D sampling grid (Cell 22) — into `(b, 1)`;
broadcasting `(b,1) * (1,hd)` gives one phase per (sample, frequency).

```python
        sin_embed = torch.sin(freqs) # b hd
        cos_embed = torch.cos(freqs) # b hd
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # b d
```
Evaluate both wave families and concatenate → `(b, dim)`. The `sqrt(2)`:
sin²+cos² average to ½ each over random phases, so scaling by √2 restores
unit variance per component — keeping the embedding on the same scale as the
label embedding it will be *added to* in Cell 21. (The markdown lists cos
first, code puts sin first — irrelevant, the next layer is a linear map.)

## Cell 18 — `Patchifier`: image → tokens

```python
class Patchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
    super().__init__()
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
```
A transformer eats sequences of tokens, not pixel grids; the ViT/DiT answer
is to cut the image into non-overlapping p×p patches and embed each as one
token. The assert enforces clean tiling — a 32-image with patch 5 would
otherwise silently drop border pixels inside the conv below.

```python
    self.net = nn.Sequential(
        nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
        Rearrange("b d h w -> b (h w) d"),
    )
```
The two-step recipe from the markdown in two layers. The conv with
`kernel_size = stride = patch_size` is the standard trick: each output pixel
sees exactly one p×p patch (no overlap, thanks to the stride) and maps its
c_in·p² values through a learned linear map to `dim` channels — patch
extraction and linear embedding fused into one op. Shape:
`(b, c_in, 32, 32) → (b, dim, 32/p, 32/p)`. Then `Rearrange` flattens the
spatial grid into a token axis: `(b, dim, h', w') → (b, h'·w', dim)` —
`n = (32/p)²` tokens of width `dim` (with the training config p = 4: 64
tokens of 256). Get this rearrange pattern wrong (e.g. `b (d h) w`) and
einops errors out — the reason the lab uses named-axis reshapes.

```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```
(The docstring's claimed return shape `(bs, 1, img_size, img_size)` is a
copy-paste error — the true output is `(b, n, dim)` as above. Trust the code,
not the comment.)

## Cell 19 — `MHA`, `modulate`, `DiffusionTransformerLayer`, `DiffusionTransformer`

The heart of Part 3: attention from scratch, then the adaLN-Zero DiT block of
Peebles & Xie (reference [1] in the notebook), then the stack.

```python
class MHA(nn.Module):
  def __init__(self, dim: int, heads: int):
    super().__init__()
    assert dim % heads == 0
    self.scale = (dim // heads) ** -0.5
    self.qkv = nn.Linear(dim, dim * 3)
    self.fold_heads = Rearrange('b n (h d) -> (b h) n d', h=heads)
    self.unfold_heads = Rearrange('(b h) n d -> b n (h d)', h=heads)
    self.out = nn.Linear(dim, dim)
```
Multi-headed *self*-attention. `dim` must split evenly across heads (each
head gets `dim/heads` channels). `scale = 1/√(head_dim)` is the softmax
temperature from "Attention Is All You Need" — without it, dot products of
high-dimensional vectors grow like √d, saturating the softmax into a
near-one-hot and killing gradients. One fused `Linear(dim, 3·dim)` computes
Q, K, V in a single matmul. The two `Rearrange` layers implement the heads
trick: split the channel axis into (heads × head_dim) and *fold heads into
the batch axis*, so all subsequent ops are plain batched matmuls that treat
each head as an independent example.

```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    q, k, v = self.qkv(x).chunk(3, dim=-1) # b n (h d)
    q, k, v = map(self.fold_heads, (q, k, v)) # (b h) n d
```
Project `(b, n, dim) → (b, n, 3·dim)`, split into three `(b, n, dim)` chunks,
then fold each to `(b·h, n, head_dim)`.

```python
    qk = torch.einsum('bid,bjd->bij', q, k) * self.scale # (b h) n n
    attn = torch.softmax(qk, dim=-1) # (b h) n n
```
The attention matrix: `einsum('bid,bjd->bij')` is Q·Kᵀ — entry (i, j) is the
dot product between token i's query and token j's key, i.e. "how much should
patch i look at patch j." Softmax over `dim=-1` (the *key* axis j) makes each
row a probability distribution — each query distributes one unit of attention
over all tokens. Softmax over the wrong axis is the classic from-scratch
attention bug: it still runs, still trains, just worse. No causal mask —
image patches may all see each other.

```python
    x = torch.einsum('bij,bjd->bid', attn, v) # (b h) n d
    x = self.unfold_heads(x) # b n (h d)
    return self.out(x)
```
Weighted average of values per query (`attn @ V`), unfold heads back into the
channel axis, and mix heads with a final linear layer (without `out`, head
outputs would stay in disjoint channel blocks forever).

```python
def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias
```
FiLM-style conditioning: scale-and-shift the normalized activations using
vectors *computed from the conditioning signal*. The `1 +` matters: paired
with the zero-init below, scale = 0 at initialization means multiplication by
exactly 1 — the modulation starts as a no-op instead of zeroing the signal.

```python
class DiffusionTransformerLayer(nn.Module):
  def __init__(self, dim: int, heads: int):
    super().__init__()
    self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
    self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
```
Pre-norm placement (norm *before* each sublayer — the stable modern
arrangement). `elementwise_affine=False` strips the norm's own learnable
scale/shift because adaLN is about to supply *condition-dependent* scale and
shift instead; keeping both would be redundant parameters.

```python
    self.ada_ln = nn.Sequential(
        nn.RMSNorm(dim, elementwise_affine=False),
        nn.Linear(dim, dim * 6)
    )
    nn.init.zeros_(self.ada_ln[1].weight)
    nn.init.zeros_(self.ada_ln[1].bias)
```
The adaLN head: normalize the conditioning vector c = t-embedding +
y-embedding, then one linear layer emits **six** `(b, dim)` vectors — scale,
bias, and gate for each of the two sublayers. The zero-init is
**adaLN-Zero** (the DiT paper's key stability trick, flagged in the lab's
hint): at initialization all six outputs are exactly zero, so scales
modulate by 1, biases add 0, and — crucially — the *gates* multiply each
residual branch by 0. The entire transformer therefore starts as the
identity function, and training grows the layers smoothly from there.
Without it, deep DiTs are noticeably harder to train from scratch.

```python
  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    c = rearrange(self.ada_ln(c), 'b d -> b 1 d') # b 1 d
    attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)
```
Compute all six modulation vectors in one pass and insert a length-1 token
axis (`b d → b 1 d`) so each `(b, 1, dim)` chunk broadcasts across all n
tokens — the conditioning is *per-image*, identical for every patch. This is
the broadcasting care the lab's hint warns about: chunk *then* forget the
unsqueeze and you'd be modulating with shapes that don't align.

```python
    x = x + attn_gate * self.attn(
      modulate(self.norm1(x), attn_scale, attn_bias)
    )
    x = x + ff_gate * self.ff(
      modulate(self.norm2(x), ff_scale, ff_bias)
    )
    return x
```
The DiT block, matching the paper's diagram: norm → condition-dependent
scale/shift → sublayer (attention, then MLP) → condition-dependent *gate* →
residual add. Read the conditioning story: t and y never enter as tokens;
they steer the computation by modulating every layer's normalization and
gating — the reason a single network can represent u_t(x|y) for all t and y
at once. `self.ff = MLP([dim, 4 * dim, dim])` (declared just above) is the
standard 4× expansion feed-forward, reusing Cell 14's MLP class.

```python
class DiffusionTransformer(nn.Module):
  def __init__(self, depth: int, n_tokens: int, dim: int, **layer_kwargs):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(DiffusionTransformerLayer(dim=dim, **layer_kwargs))
    self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))
```
The stack: `depth` identical layers in an `nn.ModuleList` (a plain Python
list would hide the parameters from `.parameters()` and the optimizer —
a silent classic). Positional encodings are a *learned* `(n_tokens, dim)`
parameter, one vector per patch position, initialized N(0, 1) — possible
because the token count is fixed (per the lab's hint). Without them,
self-attention is permutation-invariant: the model literally could not tell
the top-left patch from the bottom-right one.

```python
  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x = x + self.pos_encodings.unsqueeze(0)
    for layer in self.layers:
      x = layer(x, c)
    return x
```
`unsqueeze(0)` turns `(n, dim)` into `(1, n, dim)` so it broadcasts over the
batch (the hint's second broadcasting warning). Then run the layers,
threading the same conditioning vector `c` into every block. Output shape
unchanged: `(b, n, dim)`.

## Cell 20 — `Depatchifier`: tokens → image

```python
class Depatchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
      super().__init__()
      self.patch_size = patch_size
      assert img_size % patch_size == 0, "Image size must be divisible by patch size"
      h = w = img_size // patch_size
```
The inverse of Cell 18: `(b, n, dim)` back to `(b, c_out, 32, 32)`, needed
because the *output* of a flow model is an image-shaped velocity field.
`h = w` is the patch-grid side length (8 for p = 4), stored so the rearrange
below knows how to lay tokens back out.

```python
      self.net = nn.Sequential(
          nn.RMSNorm(dim, elementwise_affine=False),
          MLP([dim, 4*dim, final_dim * patch_size ** 2]),
```
Steps 1–2 of the markdown's recipe. A final norm settles the scale of the
transformer's outputs, then an MLP maps each token from `dim` to
`final_dim · p²` numbers — exactly enough to paint a `final_dim`-channel p×p
pixel block. Per-token, because `nn.Linear` acts on the last axis:
`(b, n, dim) → (b, n, f·p²)`.

```python
          Rearrange("b (h w) (f ph pw) -> b f (h ph) (w pw)", h=h, w=w, f=final_dim, ph=patch_size, pw=patch_size),
```
The crucial line, and the mirror image of patchification: split the token
axis back into its (h, w) grid, split each token's feature vector into an
`f × p × p` block, and interleave — token (i, j)'s block lands at pixel rows
`i·p…(i+1)·p`, columns `j·p…(j+1)·p`. Result: `(b, f, 32, 32)`. This is where
depatch shape bugs live: swap `(f ph pw)` to `(ph pw f)` and you get a
correctly-*shaped* but pixel-scrambled image that trains terribly with no
error message — einops' keyword arguments (`h=h, ph=patch_size, …`) are the
only thing pinning down the intended factorization, since `8·8·... ` numbers
can factor many ways.

```python
          nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1)
      )
```
Step 4: a 3×3 conv from `final_dim` channels down to `c_out` (1 for pixels,
128 for latents). Besides fixing the channel count, its overlapping receptive
field lets neighboring patches blend — smoothing the block-boundary artifacts
a purely per-patch reconstruction would show. `padding=1` keeps 32×32.

```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```
`(b, n, dim) → (b, c_out, img_size, img_size)`.

## Cell 21 — `DiffusionTransformerFlowModel`: the assembled u_t^θ(x|y)

```python
class DiffusionTransformerFlowModel(ConditionalVectorField):
  def __init__(self, img_size=32, patch_size=8, num_layers=12, c=1,
               dim=256, heads=4, final_dim=10, n_classes=11):
      super().__init__()
      # 0. Construct time_embedder and y_embedder
      self.time_embedder = FourierEncoder(dim)
      self.y_embedder = nn.Embedding(num_embeddings = n_classes, embedding_dim = dim)
```
The four components wired together, implementing the `ConditionalVectorField`
contract of Cell 12 (so the CFG trainer and sampler drive it exactly like the
MLP). Note `n_classes=11`: **10 digits plus the null label ∅ = 10** — the
hint spells this out; an embedding of size 10 would crash at the first
dropped label. Both embedders output `(b, dim)` so they can be added.

```python
      # 1. Construct patchifier
      self.patchifier = Patchifier(img_size=img_size, patch_size=patch_size, c_in=c, dim=dim)

      # 2. Construct DiT
      n_tokens = (img_size // patch_size) ** 2
      self.dit = DiffusionTransformer(depth=num_layers, n_tokens=n_tokens, dim=dim, heads=heads)

      # 3. Construct de-patchifier
      self.depatchifier = Depatchifier(img_size=img_size, patch_size=patch_size, dim=dim,
                                       final_dim=final_dim, c_out=c)
```
Consistency by construction: patchifier and depatchifier receive the same
`img_size`/`patch_size`, and the token count fed to the transformer's
positional table is *derived* — `(32/4)² = 64` in the training config —
rather than hand-entered. Deriving it kills the classic mismatch where you
change `patch_size` and forget to update `n_tokens`, which would crash on the
positional-encoding add.

```python
  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 1. Embed time and y
    t_embed = self.time_embedder(t) # b d
    y_embed = self.y_embedder(y) # b d
```
t (any shape flattening to `(b,)` — see Cell 17's `view(-1, 1)`) becomes a
`(b, dim)` Fourier vector; y a `(b, dim)` looked-up vector.

```python
    # 2. Patchify
    x = self.patchifier(x) # b n d

    # 3. Pass through DiT
    x = self.dit(x, t_embed + y_embed) # b n d

    # 4. Depatchify
    x = self.depatchifier(x) # b 1 32 32
    return x
```
The DiT-paper data path: image → tokens; conditioning = *sum* of time and
label embeddings (addition works because both live in the same `dim`-space at
unit scale — the √2 in Cell 17 and the N(0,1) embedding init keep them
comparable; the null label is nothing special here, just an 11th learned
vector); tokens through `num_layers` adaLN-modulated blocks; tokens → image.
Output: the velocity field, same shape as the input image. (The docstring's
`t: b 1 1 1` reflects the sampling-time call; training passes `(b,)` — both
work because of the FourierEncoder's flatten.)

## Cell 22 — `visualize_output` + `MNISTCFGTrainer`: watching training

```python
@torch.no_grad()
def visualize_output(model, path, samples_per_class: int = 10, num_timesteps: int = 100,
                     guidance_scales: List[float] = [1.0, 3.0, 5.0], save_path=None, use_tqdm=True):
  fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
```
Renders one panel of generated digits per guidance scale — the lab's
qualitative eval. `@torch.no_grad()` because this runs *inside training*
every checkpoint; without it each visualization would build (and keep) a
100-step autograd graph.

```python
  for idx, w in enumerate(guidance_scales):
      ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=10)
      simulator = EulerSimulator(ode)
```
Same sampling stack as the GMM (Cell 16), rebuilt per panel with a different
w — demonstrating concretely that guidance strength is a *free sampling-time
knob* on one trained model (notes 11 §5).

```python
      y = torch.tensor([0, 1, 2, ..., 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
      num_samples = y.shape[0]
      x0 = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)
```
Eleven rows of requests: ten per digit 0–9 *plus ten for the null label 10* —
the last grid row shows unconditional samples, a built-in sanity check that
the dropout branch learned something. 110 noise seeds `(110, 1, 32, 32)`.

```python
      ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
      x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)
```
Two details. First, the grid stops at **0.999, not 1.0** — matching the
training range t ∈ [0, 0.999) from Cell 13's `eps`; the model has never seen
t ≥ 0.999, so don't query it there. Second, `ts` is built 5-D:
`(b, nt, 1, 1, 1)`. The simulator only assumes `ts.shape[1]` is time, so its
slices `ts[:, i]` come out `(b, 1, 1, 1)` — pre-broadcast against images, and
flattened back to `(b, 1)` inside the FourierEncoder. A `(b, nt)` grid would
also have worked (the `h.view` in `EulerSimulator.step` handles it); the 5-D
version just does the broadcasting up front.

```python
      v_min, v_max = x1.min(), x1.max()
      x1 = (x1 - v_min) / (v_max - v_min)
      grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(0,1))
      axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
      axes[idx].axis("off")
      axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
```
Min-max normalize the generated batch into [0, 1] (the model outputs
*normalized-pixel-scale* values, roughly [−0.45, 3]; without rescaling,
imshow's autoscaling per panel would be inconsistent), tile 10 per row →
an 11×10 digit grid, display. What you should see across the three panels:
w = 1 gives correct but sometimes scruffy digits; w = 3 noticeably cleaner
and more "canonical"; w = 5 cleaner still but visibly less diverse — each
row's ten samples start looking like clones. That is the diversity-vs-
fidelity dial of notes 11 §6, in grayscale.

```python
  if save_path is not None:
      plt.savefig(save_path); plt.close()
  else:
    plt.show()
```
File mode for checkpoint callbacks (close the figure or 20 checkpoints leak
20 figures), interactive mode for notebook use.

```python
class MNISTCFGTrainer(CFGTrainer):
  def checkpoint(self, step: int):
    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_model.pt'))
    torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_opt.pt'))
    visualize_output(self.model, self.path, save_path=os.path.join(self.output_dir, f'step_{step:6d}_output.png'), use_tqdm=False)
```
The training logic is 100% inherited from `CFGTrainer`; this subclass only
fills the checkpoint hook: model weights, optimizer state (so runs are
resumable), and a fresh sample grid — so you can literally watch digits
emerge in `runs/<name>/` as training progresses. (Nit: `:6d` space-pads
rather than zero-pads, so filenames sort oddly; the VAE trainer later uses
the correct `:06d`.)

## Cell 23 — Training the pixel-space DiT

```python
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)
```
The MNIST pixel path again (fresh instance, fresh MNIST wrapper).

```python
dit = DiffusionTransformerFlowModel(
    img_size = 32, patch_size = 4, num_layers = 8, c = 1,
    dim = 256, heads = 8, final_dim = 10, n_classes = 11,
).to(device)
```
The actual config: patch 4 ⇒ 64 tokens (finer than the default 8 ⇒ 16 —
more tokens, more spatial detail, more attention compute), 8 layers of
width 256 with 8 heads (head_dim 32), ~9 MB of parameters. Small by modern
standards, plenty for MNIST.

```python
trainer = MNISTCFGTrainer(path = path, eta=0.35, null_label=10)
losses, steps = trainer.train(model=dit, num_steps = 20000, lr=0.4e-3, batch_size=256, ckpt_every=1000)
plt.plot(steps, losses); plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss vs. Step"); plt.show()
```
η = 0.35 label dropout, null = 10, 20k steps at batch 256 and lr 4e-4 (with
the 500-step warmup from Cell 9). The comment says ~15 minutes on an A100.
What running it looks like: the run-directory printout, a model-size line, a
tqdm bar ticking through 20,000 steps with the loss falling fast then
plateauing around its irreducible noise floor, and every 1000 steps a saved
`step_*_output.png` (early ones are gray mush; digit identity typically
becomes recognizable within the first few thousand steps). The final plot is
the loss curve — expect a steep drop then a long flat tail; flat ≠ done
learning nothing, it's the stochastic-target floor again (Cell 15).

## Cell 24 — Final samples at three guidance scales

```python
samples_per_class = 10
num_timesteps = 100
guidance_scales = [1.0, 3.0, 5.0]

visualize_output(
    model=dit, path=path,
    samples_per_class=samples_per_class,
    num_timesteps=num_timesteps,
    guidance_scales=guidance_scales,
)
plt.show()
```
Pure driver: re-run Cell 22's visualization on the trained model with
default knobs exposed for play. Three 11×10 grids (rows = digits 0–9 plus the
∅ row). Things worth actually trying: drop `num_timesteps` to 10 and see how
well straight-ish flows survive coarse Euler grids (notes 06 Part E); crank
w to 10+ and watch strokes go thick and samples collapse toward one
prototype per class — the over-guidance failure mode of notes 11 §6.

---

## Cell 25 — `ResidualBlock`: the VAE's convolutional unit

Part 4 begins. The VAE (reference [2] — the Stable Diffusion paper's
autoencoder, miniaturized) is built from three small blocks; this is the
first.

```python
class ResidualBlock(nn.Module):
  def __init__(self, channels: int, act: nn.Module = nn.SiLU):
    super().__init__()
    self.norm = nn.GroupNorm(1, channels)
```
`GroupNorm(1, c)` = all channels in one group = LayerNorm over the whole
`(c, h, w)` feature map, as the markdown notes. GroupNorm rather than
BatchNorm because it's batch-size independent (no train/eval statistics
divergence — one less thing to break).

```python
    self.conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding=1, stride=1)
    self.act1 = act()
    self.conv2 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 1, padding=0, stride=1)
    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)
```
The residual branch: 3×3 conv (spatial mixing; `padding=1` preserves h, w —
essential, or the skip-add below would shape-mismatch), nonlinearity, then a
1×1 conv (per-pixel channel mixing) that is **zero-initialized** — so the
whole branch outputs 0 at init and the block starts as the identity. The
same "start as identity" trick as adaLN-Zero (Cell 19) and `final_init`
(Cell 14), now in conv form. Channel count never changes inside the block.

```python
  def forward(self, x: torch.Tensor):
    x_skip = x
    x = self.norm(x)
    x = self.conv1(x)
    x = self.act1(x)
    x = self.conv2(x)
    return x_skip + x
```
Save the input, run the pre-norm branch, add back. The residual connection
is what lets gradients flow through many stacked blocks; remove `x_skip +`
and this depth of plain convs trains far worse. Shapes: `(b, c, h, w)`
throughout.

## Cell 26 — `AttnBlock`: a transformer block for feature maps

```python
class AttnBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.reshape1 = Rearrange('b c h w -> b (h w) c')
    self.norm1 = nn.LayerNorm(channels)
    self.mha = MHA(channels, 1)
    self.norm2 = nn.LayerNorm(channels)
    self.ff = MLP([channels, 2 * channels, channels], final_init=True)
```
Convolutions see only local neighborhoods; one attention block per
encoder/decoder stage adds *global* mixing (any pixel can consult any other —
useful for digit-scale coherence). The trick: temporarily treat the feature
map as h·w tokens of width c, then reuse the DiT machinery — the *same*
`MHA` class from Cell 19 (single head here) and the same `MLP` (2× expansion,
`final_init=True` zero-initializing the last layer: identity-at-init again).

```python
  def forward(self, x: torch.Tensor):
    b, c, h, w = x.shape
    x = self.reshape1(x)

    x_skip = x
    x = self.norm1(x)
    x = self.mha(x)
    x = x + x_skip

    x_skip = x
    x = self.norm2(x)
    x = self.ff(x)
    x = x + x_skip

    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
```
Record the spatial dims (needed to fold back), flatten to `(b, h·w, c)`, then
a textbook pre-norm transformer block: norm → attention → residual, norm →
feed-forward → residual — no adaLN this time, because the VAE has no
conditioning signal. Finally rearrange back to `(b, c, h, w)`; passing `h=h`
is mandatory here since einops cannot know how to factor h·w on its own
(64 = 8·8 = 16·4 = …) — omit it and you get an error, or with the wrong value
a transposed image.

## Cell 27 — `EncoderBlock`: two res + attention + optional downsample

```python
class EncoderBlock(nn.Module):
  def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
    super().__init__()
    self.res1 = ResidualBlock(in_channels)
    self.res2 = ResidualBlock(in_channels)
    self.attn = AttnBlock(in_channels)
    if downsample_channels is not None:
      self.downsample = nn.Conv2d(in_channels = in_channels, out_channels = downsample_channels, padding=1, stride=2, kernel_size=3)
    else:
      self.downsample = None
```
One encoder *stage*: refine (two residual blocks), globally mix (attention),
then — unless this is the last stage — shrink. The downsampler is a strided
conv: `stride=2` halves both spatial dims while the channel count typically
grows (trading resolution for features, the standard encoder bargain). The
`Optional` channel argument doubles as the on/off switch: `None` means "final
stage, keep resolution."

```python
  def forward(self, x: torch.Tensor):
    x = self.res1(x)
    x = self.res2(x)
    x = self.attn(x)
    if self.downsample is not None:
      x = self.downsample(x)
    return x
```
Shape: `(b, c_in, h, w) → (b, c_down, h/2, w/2)` when downsampling, else
unchanged.

## Cell 28 — `Encoder`: image → latent Gaussian parameters

```python
class Encoder(nn.Module):
  def __init__(self, in_channels: int, hidden_channels: list[int]):
    super().__init__()
    self.init_conv = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels[0], kernel_size=3, padding=1, stride=1)
```
Step 0: lift the 1-channel image into `hidden_channels[0]` feature channels
without touching resolution.

```python
    ch_in = hidden_channels
    ch_out = hidden_channels[1:] + [None]
    blocks = []
    for in_c, out_c in zip(ch_in, ch_out):
      blocks.append(EncoderBlock(in_c, out_c))
    self.blocks = nn.ModuleList(blocks)
```
A neat pairing idiom: zip the channel list against itself shifted by one,
padding with `None`. For the training config `[16, 32, 64, 128]` this builds
blocks (16→32, ↓), (32→64, ↓), (64→128, ↓), (128, no downsample) — the
trailing `None` lands exactly on the last block, implementing "downsample
for all but the last" without a special case. Trace the shapes:
`(b,1,32,32) → init conv → (b,16,32,32) → (b,32,16,16) → (b,64,8,8) →
(b,128,4,4) → (b,128,4,4)`.

```python
    z_dim = hidden_channels[-1]
    self.z_mean = nn.Sequential(
      nn.GroupNorm(1, z_dim),
      nn.Conv2d(in_channels = z_dim, out_channels = z_dim, kernel_size=1, stride=1, padding=0),
    )
    self.logvar = nn.Parameter(torch.zeros(()))
```
Two output heads, matching the markdown's spec of q_φ(z|x). The mean head is
norm + 1×1 conv → `z_mean: (b, 128, 4, 4)`. The log-variance is a **single
scalar parameter shared across all inputs and all latent dimensions** — not
the usual per-element network output. Log-parameterization means σ² = e^logvar
is positive by construction (the stability precedent of [2] the markdown
cites); the scalar choice keeps the posterior spread a single global knob.
Initialized at 0 ⇒ σ = 1. Note what this latent is: `128·4·4 = 2048` numbers
per image — *more* than the 1024 input pixels. The compression here is
spatial/structural (32×32 → 4×4 grid), not dimensional; real LDMs compress
much harder.

```python
  def forward(self, x: torch.Tensor):
    x = self.init_conv(x)
    for block in self.blocks:
      x = block(x)
    return self.z_mean(x), self.logvar
```
Returns the pair (mean tensor, scalar logvar) that defines
q_φ(z|x) = N(z_mean, e^logvar·I).

## Cell 29 — `DecoderBlock`: the mirrored stage

```python
class DecoderBlock(nn.Module):
  def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
    super().__init__()
    self.res1 = ResidualBlock(in_channels)
    self.res2 = ResidualBlock(in_channels)
    self.attn = AttnBlock(in_channels)
    if upsample_channels is not None:
      self.upsample = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=in_channels, out_channels=upsample_channels, kernel_size=3, padding=1, stride=1),
      )
    else:
      self.upsample = None
```
Same res-res-attn body as the encoder block; only the resampler flips
direction. Upsampling is nearest-neighbor 2× (each pixel becomes a 2×2 block)
*followed by* a 3×3 conv to smooth and change channels — deliberately chosen
over `ConvTranspose2d`, which is notorious for checkerboard artifacts.
Shape: `(b, c_in, h, w) → (b, c_up, 2h, 2w)` when upsampling.

```python
  def forward(self, x: torch.Tensor):
    x = self.res1(x); x = self.res2(x); x = self.attn(x)
    if self.upsample is not None:
      x = self.upsample(x)
    return x
```
Identical control flow to Cell 27.

## Cell 30 — `Decoder`: latent → reconstructed-image Gaussian parameters

```python
class Decoder(nn.Module):
  def __init__(self, out_channels: int, hidden_channels: list[int]):
    super().__init__()
    ch_in = hidden_channels
    ch_out = hidden_channels[1:] + [None]
    blocks = []
    for in_c, out_c in zip(ch_in, ch_out):
      blocks.append(DecoderBlock(in_c, out_c))
    self.blocks = nn.ModuleList(blocks)
```
The same zip-shift idiom as the encoder. The VAE (Cell 31) passes
`reversed(hidden_channels)` = `[128, 64, 32, 16]`, giving (128→64, ↑),
(64→32, ↑), (32→16, ↑), (16, no upsample): `(b,128,4,4) → (b,64,8,8) →
(b,32,16,16) → (b,16,32,32) → (b,16,32,32)`. Note there is no initial conv —
the latent already *is* a feature map.

```python
    x_dim = hidden_channels[-1]
    self.x_mean = nn.Sequential(
      nn.GroupNorm(1, x_dim),
      nn.Conv2d(in_channels = x_dim, out_channels = out_channels, kernel_size=1, stride=1, padding=0),
    )
    self.logvar = nn.Parameter(torch.zeros(()))

  def forward(self, x: torch.Tensor):
    for block in self.blocks:
      x = block(x)
    return self.x_mean(x), self.logvar
```
Mirror-image output heads: `x_mean: (b, 1, 32, 32)` via norm + 1×1 conv
(16 → 1 channels), plus another *scalar* learned log-variance — this one
parameterizing the decoder's observation noise p_θ(x|z) = N(x_mean,
e^logvar·I), whose role in the loss is explained next.

## Cell 31 — `VAE`: reparameterization and the ELBO loss

```python
class VAE(nn.Module):
  def __init__(self, data_channels: int, hidden_channels: list[int], beta: float = 0.1):
    super().__init__()
    self.beta = beta
    self._encoder = Encoder(data_channels, hidden_channels)
    self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))
```
Assembly, with the decoder's channel list reversed so the architectures
mirror. `beta` weights the KL term (a β-VAE knob): larger β presses the
posterior harder toward N(0, I), costing reconstruction quality but making
the latent space smoother — which is exactly what a *diffusion model in
latent space* wants, since it must learn to generate these latents from
Gaussian noise.

```python
  def encode(self, x): return self._encoder(x)
  def decode(self, z): return self._decoder(z)
```
Thin pass-throughs giving the outside world (interpolation in Cell 32, the
latent trainer in Cell 35) a clean API.

```python
  def forward(self, x: torch.Tensor):
    z_mean, z_logvar = self.encode(x)
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
    x_mean, x_logvar = self.decode(z)
    return z_mean, z_logvar, x_mean, x_logvar
```
The full pass. The middle line is **the reparameterization trick** — notes
05 §A3's `μ + σ·ε` identity, verbatim: `exp(0.5·logvar)` = e^{logvar/2} = σ
(half the logvar because variance → std is a square root). Writing the sample
as mean + std·noise, instead of calling a black-box sampler, is what lets
gradients flow *through the sampling step* back into the encoder — the entire
reason VAEs are trainable. `randn_like(z_mean)` handles shape/device; the
scalar `z_logvar` broadcasts over the whole `(b, 128, 4, 4)` mean.

```python
  def compute_loss(self, z_mean, z_logvar, x_mean, x_logvar, x_true):
    """ See display 85 from the text """
    eps = 1e-6
    # KL loss
    kl_loss = self.beta * (z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1).mean()
```
The negative ELBO, term one: KL(q_φ(z|x) ‖ N(0, I)). For diagonal Gaussians
the closed form per dimension is ½(μ² + σ² − log σ² − 1) — recognize each
piece in the code: `z_mean.pow(2)` = μ², `exp(z_logvar)` = σ²,
`− z_logvar` = −log σ², `− 1`. (KL between Gaussians is the same object as
notes 05 §A6, in its general unequal-variance form.) Two conventions to
notice: the ½ is dropped (absorbed into β — only the *ratio* of loss weights
matters), and `.mean()` averages over dimensions rather than summing (again a
constant rescale folded into β). **Signs are the classic bug surface here**:
flip `− z_logvar` to `+ z_logvar` and the "KL" can go negative and the
encoder is rewarded for infinite variance; the correct expression is ≥ 0 with
minimum exactly at μ = 0, σ = 1.

```python
    # Reconstruction loss
    mse_term = (x_true - x_mean).pow(2) / (torch.exp(x_logvar) + eps)
    confidence_term = x_logvar
    recon_loss = (mse_term + confidence_term).mean()
    return kl_loss + recon_loss
```
Term two: −E[log p_θ(x|z)] for a Gaussian decoder, i.e. the Gaussian negative
log-likelihood (x−μ)²/σ² + log σ² (constants and the ½ dropped as above).
Read the two pieces as a dialogue: the MSE term is reconstruction error
*discounted by the decoder's claimed noise level* σ² = e^x_logvar; the
`confidence_term` charges the decoder for claiming a large σ². Together they
make the observation noise self-calibrating — if the decoder inflates σ² to
excuse bad reconstructions, the log σ² penalty bites; if it claims σ² ≈ 0,
any residual error is amplified enormously. With a *fixed* σ this whole term
would collapse to plain MSE. `eps = 1e-6` guards the division if `x_logvar`
plunges very negative. Total loss = β·KL + NLL, minimized jointly over
encoder and decoder.

## Cell 32 — `MNISTVAETrainer` + latent interpolation

```python
class MNISTVAETrainer(Trainer):
  def __init__(self, mnist_sampleable: LabeledSampleable, batch_size: int = 64, **kwargs):
    super().__init__(**kwargs)
    self.mnist = mnist_sampleable
    self.batch_size = batch_size

  def get_train_loss(self):
    x, y = self.mnist.sample(self.batch_size)
    z_mean, z_std, x_mean, x_std = self.model(x)
    return self.model.compute_loss(z_mean, z_std, x_mean, x_std, x)
```
The simplest trainer subclass: sample a batch (labels drawn and discarded —
this VAE is unconditional), forward, ELBO loss. Note `batch_size` is stored
at construction here rather than threaded through `train(**kwargs)` as in
`CFGTrainer` — two styles of the same plumbing. (Naming nit: the variables
are called `z_std`/`x_std` but hold *logvars*; they're only ever passed to
`compute_loss`, which treats them correctly.)

```python
  @torch.no_grad()
  def checkpoint(self, step: int):
    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_model.pt'))
    torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_opt.pt'))
    b = 10
    x, _ = self.mnist.sample(b)
    _, _, x_mean, _ = self.model(x)
    x_all = torch.cat([x, x_mean], dim=0)
    grid = make_grid(x_all, nrow=b, normalize=True, value_range=(0,1))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    plt.axis("off"); plt.title("VAE Reconstruction")
    plt.savefig(os.path.join(self.output_dir, f'step_{step:06d}_output.png')); plt.close()
```
Checkpoint hook: save weights, then a two-row diagnostic image — 10 originals
concatenated *along the batch axis* with their 10 reconstructions (`x_mean`,
the decoder's mean, not a noisy sample); `nrow=b` folds the 20-image batch
into two rows of 10, top = real, bottom = reconstruction. Blurry-but-right
bottoms early, sharpening over training. `@torch.no_grad()` because this
forwards the model outside the training step.

```python
@torch.no_grad()
def visualize_latent_interpolation(x1, x2, vae: VAE, n_steps: int, save_path=None):
   z1_mean, z1_logvar = vae.encode(x1)
   z1 = z1_mean + torch.exp(0.5 * z1_logvar) * torch.randn_like(z1_mean) # 1 c h w
   z2_mean, z2_logvar = vae.encode(x2)
   z2 = z2_mean + torch.exp(0.5 * z2_logvar) * torch.randn_like(z2_mean) # 1 c h w
```
Encode two single images `(1, 1, 32, 32)` to latents `(1, 128, 4, 4)`, using
the same reparameterized sampling as training.

```python
   lambdas = torch.linspace(0, 1, n_steps).to(z1.device)
   zs = (1 - lambdas) * z1.unsqueeze(-1) + lambdas * z2.unsqueeze(-1) # 1 c h w n_steps
   zs = rearrange(zs, '1 c h w n -> n c h w')
   samples, _ = vae.decode(zs) # n_steps 1 h w
```
Linear interpolation in latent space, vectorized by a broadcasting trick:
`unsqueeze(-1)` makes the latents `(1, c, h, w, 1)`, which broadcast against
the `(n,)` lambdas to produce all n blends at once; the rearrange then
promotes the interpolation axis to the batch axis so one `decode` call
renders every intermediate. The *point* of the figure: if the latent space
is smooth (the KL term's doing), the decoded morph between two digits stays
digit-like at every step, instead of dissolving into non-digit mush — this
smoothness is what makes the latent space a hospitable home for a diffusion
model.

- Remaining lines: `make_grid` + `imshow` + optional `savefig` — the same
  display boilerplate as every other visualization cell.

## Cell 33 — Training the VAE

```python
device = torch.device('cuda')
mnist = MNISTSampler().to(device)
vae = VAE(
   data_channels = 1,
   hidden_channels = [16, 32, 64, 128],
   beta = 10.0,
).to(device)
```
Note the hard `torch.device('cuda')` — unlike earlier cells there's no CPU
fallback; on a CPU-only machine this cell dies here. The channel schedule
gives the 32→4 spatial reduction traced in Cell 28. β = 10 is *strong*
regularization — chosen to keep latents very close to N(0, I) for the sake
of Part 5, at some cost in reconstruction sharpness. (Heads-up: a comment in
Cell 36 claims this VAE had `beta = 1.0`; the code you actually ran says
10.0 — trust the code.)

```python
trainer = MNISTVAETrainer(mnist_sampleable = mnist, batch_size = 64)
losses, steps = trainer.train(model = vae, num_steps = 5000, lr = 1e-3, warmup_steps = 500, ckpt_every = 250)
plt.plot(steps, losses); plt.xlabel("Step"); plt.ylabel("Loss"); plt.show()
```
5000 steps at batch 64 — a few minutes on a GPU. Output: run directory, tqdm
bar, reconstruction PNGs every 250 steps in `runs/<name>/`, and a loss curve
that drops sharply and flattens. The loss can hover at seemingly "large"
values — remember it's β·KL + NLL with learned observation noise, not a pure
MSE, so its absolute scale isn't directly interpretable.

## Cell 34 — Interpolation demo

```python
vae.eval()
samples, _ = mnist.sample(2)
interpolated_samples = visualize_latent_interpolation(
   x1 = samples[:1],
   x2 = samples[1:2],
   vae = vae,
   n_steps = 10,
) # n_steps 1 h w
```
Grab two random digits (slicing with `[:1]`/`[1:2]` keeps the batch dim —
`samples[0]` would drop it and break `encode`), morph in 10 steps, display.
`vae.eval()` is belt-and-braces (Trainer already left it in eval mode; this
VAE has no dropout/batchnorm anyway). Expect one digit melting smoothly into
the other through plausible in-between shapes.

---

## Cell 35 — `LatentCFGTrainer`: diffusion moves into latent space

Part 5: the punchline. Everything from Part 2–3 is reused; the only change is
*where the flow lives* — in the VAE's `(128, 4, 4)` latent space instead of
`(1, 32, 32)` pixel space. This is the Stable Diffusion recipe [2] in
miniature: train an autoencoder once, then do generation in its latent space.

```python
class LatentCFGTrainer(Trainer):
    def __init__(self, mnist: MNISTSampler, vae: VAE, path: GaussianConditionalProbabilityPath,
                 eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.mnist = mnist
        self.vae = vae
        self.path = path
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label
```
Same fields as `CFGTrainer` (Cell 13) plus the two new actors: the raw MNIST
sampler and the *frozen, pre-trained* VAE. (`self.path = path` is assigned
twice — harmless copy-paste residue.) The path passed in will have
`p_data = None`, because this trainer builds its own "data" by encoding — see
`get_train_loss`.

```python
    def visualize_samples(self, save_path, samples_per_class=10, num_timesteps=100,
                          guidance_scales=[1.0, 3.0, 5.0], use_tqdm=False):
      ...
          z0 = self.path.p_simple.sample(num_samples)
          ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
          z1 = simulator.simulate(z0, ts, y=y, use_tqdm=use_tqdm)
          # Decode
          x1, _ = self.vae.decode(z1)
      ...
```
A near-verbatim copy of Cell 22's `visualize_output` with exactly one new
step, and it's the conceptual heart of latent diffusion: the sampler starts
from `(b, 128, 4, 4)` Gaussian noise, Euler-integrates the CFG velocity field
**in latent space**, and then a single `vae.decode` call turns the final
latent into pixels (taking `x_mean`, discarding the logvar). Everything
upstream of decode has never seen a pixel. The rest of the method — the
CFG ODE per guidance scale, the 11-class label tensor, min-max normalization,
`make_grid`, save-or-show — is line-for-line the Cell 22 walkthrough.

One genuine oddity to flag: right after this method's body, at class-body
indentation, sits a stray

```python
    plt.show()
```
— a leftover line that executes once *at class-definition time* (showing
nothing, since no figure is open) and never again. Delete-on-sight material;
it's a nice reminder that Python class bodies are just executed code.

```python
    def checkpoint(self, step: int):
      torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_model.pt'))
      torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_opt.pt'))
      self.visualize_samples(save_path=os.path.join(self.output_dir, f'step_{step:6d}_output.png'))
```
Same checkpoint pattern as Cell 22: weights + optimizer + a decoded sample
grid every `ckpt_every` steps.

```python
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z, y from MNIST + encode
        with torch.no_grad():
          xx, y = self.mnist.sample(batch_size) # b 1 h w, b
          z_mean, z_logvar = self.vae.encode(xx) # b c h w, 1
          zz = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
```
The one substantive edit versus `CFGTrainer.get_train_loss`, exactly as the
lab's hint prescribes: instead of `path.p_data.sample`, draw *pixels* + label
from MNIST, push them through the frozen encoder, and reparameterize
(notes 05 §A3 once more) to get a latent `zz: (b, 128, 4, 4)`. From here on,
`zz` plays the role z played in pixel space — the "data" the flow learns to
reach. The `torch.no_grad()` guard (hint 2) matters twice over: it stops
autograd from building a graph through the encoder every step (wasted memory
and compute — the optimizer only holds DiT parameters, so VAE grads would be
computed and thrown away), and it documents the design: **the VAE is frozen;
the diffusion model adapts to it, never the reverse.**

```python
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        yi = torch.rand(y.shape[0]).to(y.device)
        y[yi < self.eta] = self.null_label

        # Step 3: Sample t and x
        t = torch.rand(batch_size).to(zz) * (1 - self.eps) # b
        zx = self.path.sample_conditional_path(zz,t) # b ...

        # Step 4: Regress and output loss
        ut_theta = self.model(zx,t,y) # b ...
        ut_ref = self.path.conditional_vector_field(zx,zz,t) # b ...
        return torch.square(ut_theta - ut_ref).mean()
```
Steps 2–4 are *character-for-character* the CFG trainer of Cell 13 — same
per-sample label dropout (same `rand`-not-`randn` trap), same t ∈ [0, 0.999)
guard against the β_t = 0 singularity, same noising via
`sample_conditional_path` (now producing noisy *latents* `(b, 128, 4, 4)`),
same MSE against the analytic conditional velocity with the latent `zz` as
the conditioning point. That this required zero changes is the payoff of
Part 0's `b ...` shape generalization: the path, the loss, and the model
contract never assumed pixels.

## Cell 36 — Training the latent DiT

```python
vae = vae.to(device) # VAE(data_channels = 1, hidden_channels = [16, 32, 64, 128], beta = 1.0)
```
Reuse the VAE trained in Cell 33 (already on device; the `.to` is a no-op
safety). The trailing comment misremembers β as 1.0 — Cell 33 trained with
β = 10.0.

```python
c = 128
img_size = 4

path = GaussianConditionalProbabilityPath(
    p_data = None,
    p_simple_shape = [c, img_size, img_size],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)
```
The *latent* probability path: noise shape `[128, 4, 4]` matching the encoder
output, same linear schedule. `p_data = None` looks alarming but is safe by
construction: `LatentCFGTrainer.get_train_loss` never calls
`sample_conditioning_variable` (it builds its own conditioning latents via
the encoder), so the `None` would only crash if someone called
`path.sample_marginal_path` — nobody does.

```python
dit = DiffusionTransformerFlowModel(
    img_size = img_size, patch_size = 1, num_layers = 8, c = c,
    dim = 256, heads = 8, final_dim = 10, n_classes = 11,
).to(device)
```
The *same DiT class* re-instantiated for latents — the payoff of writing
Part 3 generically. Read the config: "images" are now 4×4 with 128 channels;
`patch_size = 1` makes every latent pixel its own token, so
n_tokens = (4/1)² = **16 tokens** (versus 64 in pixel space — a 4× shorter
sequence, hence ~16× cheaper attention; this compute saving is the entire
business case for latent diffusion at scale). The Patchifier's conv is now a
1×1 conv from 128 → 256 channels; the Depatchifier ends in a 3×3 conv back
to 128.

```python
mnist = MNISTSampler().to(device)
trainer = LatentCFGTrainer(mnist = mnist, vae = vae, path = path, eta=0.35, null_label=10)
losses, steps = trainer.train(model=dit, num_steps = 10000, lr=0.4e-3, batch_size=256, ckpt_every=500)
plt.plot(steps, losses); plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss vs. Step"); plt.show()
```
Same hyperparameters as the pixel run but only 10,000 steps — the smaller
token grid trains faster per step *and* needs fewer steps. Output: the usual
run directory, progress bar, checkpoint PNGs every 500 steps (each an 11×10
digit grid at w ∈ {1, 3, 5}, now produced by latent sampling + VAE decode),
and the loss curve. Expect latent samples to look slightly softer than the
pixel-space DiT's — they inherit the β = 10 VAE's blur — while the guidance-
scale behavior across the three panels mirrors Part 3 exactly, because CFG
never cared what space it operates in (notes 11 §7).

---

## What to carry forward (Lab 3 → real systems)

1. **Conditioning is a data question first**: `LabeledSampleable` returning
   (z, y) jointly is the whole prerequisite for conditional generation; the
   loss stays the same CFM MSE (notes 06 §D2) with y along for the ride.
2. **CFG = one network, two roles, blended at sampling time** (notes 11
   §4–5): per-sample label dropout to ∅ during training (`rand`, not
   `randn`; per-sample, not per-batch); at inference two forward passes per
   step combined as (1−w)·u(x|∅) + w·u(x|y). Guidance scale w is a free
   post-training dial: fidelity up, diversity down (notes 11 §6).
3. **The t ∈ [0, 1−ε) guard**: the Gaussian-path target divides by β_t —
   never train or sample at the singular endpoint.
4. **DiT = patchify → adaLN-modulated transformer → depatchify**; t and y
   steer via learned scale/shift/gate, not via tokens; adaLN-Zero (and
   zero-initialized residual branches generally) makes deep stacks start as
   the identity and train stably. Named-axis `Rearrange` is the antidote to
   patch/depatch scrambles.
5. **VAE in three formulas**: reparameterize z = μ + e^{logvar/2}·ε (notes
   05 §A3), KL(q‖N(0,I)) = ½(μ² + σ² − log σ² − 1) per dim (watch the
   signs; notes 05 §A6), Gaussian NLL with *learned* observation variance =
   MSE/σ² + log σ². β trades reconstruction sharpness for latent smoothness.
6. **Latent diffusion is literally the same trainer with an encoder bolted
   on**: freeze the VAE, encode under `torch.no_grad()`, run the identical
   CFM/CFG machinery on `(c, h', w')` latents, decode once at the very end.
   Fewer tokens = the compute win that scales this recipe to Stable
   Diffusion.
7. **Everything samples with the same two Lab-1 loops** — the CFG model is
   just another ODE plugged into the same Euler `step` (notes 06 §A1
   Fact 2), with `**kwargs` quietly carrying the condition through code
   written before conditions existed.
