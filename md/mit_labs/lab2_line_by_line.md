# MIT Lab 2, Line by Line — Flow Matching and Score Matching

*A beginner's reading companion to `solutions/lab_two_complete.ipynb`. Every
line of code: what it does, why it's there, and what breaks without it.
Concepts link to our notes: interpolant/target velocity = `06 §C`,
least-squares-learns-the-mean = `06 §A3/D2`, reparameterization = `05 §A3`,
score = −ε/σ = `10` (notation header + §3). Builds directly on the Lab 1
companion (`lab1_line_by_line.md`) — the ODE/SDE/Simulator machinery returns
almost verbatim.*

**The lab in one sentence:** design a *conditional* probability path p_t(x|z)
from noise (t=0) to one data point z (t=1), write down its exact conditional
vector field and score, regress two MLPs onto those per-sample targets — and
discover that the trained networks are the *marginal* vector field and score,
good enough to carry an entire noise cloud onto the data distribution with the
Lab 1 simulators.

⚠ Time convention, once and for all: in this lab **t = 0 is noise
(p_simple), t = 1 is data (p_data)** — the flow-matching convention of our
notes 06, and the *opposite* of DDPM's T=noise → 0=data. Every formula below
silently assumes it.

---

## Part 0 — Recycled machinery (Cells 0–4)

## Cell 0 — Imports and device

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
```
Abstract-base-class tools (same enforceable-contract role as Lab 1 Cell 0),
a wider set of type-hint helpers (`List[int]` for MLP layer sizes, `Type[...]`
to pass a *class* like `torch.nn.SiLU` as an argument), and `math` (stdlib —
imported but essentially unused; harmless).

```python
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
```
NumPy (this lab actually uses it: `np.histogram2d`, `np.percentile` in Cell 2),
matplotlib, matplotlib's colormap module `cm` (needed for
`cm.colors.Normalize`, the color-scaling object in Cell 2), and the `Axes`
type imported purely for signatures.

```python
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
```
PyTorch, its distributions toolbox (ready-made Gaussians/mixtures with correct
`sample()`/`log_prob()`), and the function-transform pair from Lab 1:
`jacrev(f)` = "function that returns f's derivatives," `vmap` = "apply it per
batch item without a Python loop." Lab 1 used them for scores; this lab uses
them for **time derivatives** α̇_t, β̇_t (Cell 8).

```python
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
```
Progress bars, statistical plotting, and — new — two scikit-learn toy-data
generators. `make_moons`/`make_circles` return 2D point clouds shaped like
interlocking crescents / concentric rings; they become non-Gaussian
`p_data` (and even `p_simple`!) targets in Part 4.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Pick GPU if available. Everything downstream is `.to(device)`-ed; a missed one
gives the classic "expected cuda but got cpu" error.

## Cell 1 — Sampleable, Density, Gaussian, GaussianMixture (Lab 1's Cells 11+13, consolidated)

```python
class Sampleable(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor: ...
```
Contract #1: "I can hand you points drawn from myself." `sample(n)` returns
`(n, dim)`. New versus Lab 1: the abstract **`dim` property** — later code
(e.g. `GaussianConditionalProbabilityPath.__init__`) must build a matching
Gaussian source without being told the dimensionality, so every distribution
now has to announce its own. `@property` stacked over `@abstractmethod` means
children expose it as an attribute (`p.dim`), not a method call.

```python
class Density(ABC):
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor: ...
```
Contract #2: "I can tell you how likely any point is," `(batch, dim)` in,
`(batch, 1)` out. Log, not raw density — raw densities underflow float
precision. Note what was *dropped* relative to Lab 1: the autodiff `score`
method. This lab computes scores analytically (Cell 10) or learns them
(Cell 22); it never differentiates a log-density again.

The split matters as much as in Lab 1: **real datasets are Sampleable but not
Density** — and Part 4's moons/checkerboard classes are exactly that, which is
why the linear path there will have no usable `conditional_score`.

```python
class Gaussian(torch.nn.Module, Sampleable, Density):
    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
```
Triple inheritance, each with a job: `nn.Module` supplies `register_buffer`
and `.to(device)`; the other two are our contracts. Buffers = constants that
travel with the module (move on `.to(device)`, saved in `state_dict`, no
gradients) — the same pattern as our DDPM schedule buffers.

```python
    @property
    def dim(self) -> int:
        return self.mean.shape[0]
```
Dimensionality read off the mean vector `(dim,)` — the new contract, fulfilled
for free.

```python
    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
```
Build the torch distribution *on demand* — a `@property` so it always uses the
buffers' **current** device (build it once in `__init__` and a later
`.to(device)` would strand it on CPU). `validate_args=False` skips input
checks for speed.

```python
    def sample(self, num_samples):
        return self.distribution.sample((num_samples,))
    def log_density(self, x):
        return self.distribution.log_prob(x).view(-1, 1)
```
Delegate both contracts to torch. `.view(-1, 1)` re-imposes the `(batch, 1)`
shape (torch returns `(batch,)`).

```python
    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)
```
Convenience constructor for N(0, std²·I) — note `std ** 2`: the constructor
takes a *covariance*, so the standard deviation must be squared (mixing up σ
and σ² is the classic Gaussian API bug). This is the `p_simple` factory used
everywhere.

```python
class GaussianMixture(torch.nn.Module, Sampleable, Density):
```
Same wrapper pattern around `D.MixtureSameFamily`: `means (nmodes, 2)`,
`covs (nmodes, 2, 2)`, `weights (nmodes,)` registered as buffers;
`D.Categorical(probs=weights)` picks WHICH bump each sample comes from,
`D.MultivariateNormal(means, covs)` is the batch of bumps, and torch composes
them into one distribution whose `log_prob` is the correct log-sum-exp over
components. `dim` reads `means.shape[1]`.

The two constructors are scenery generators:
- `random_2D` — `torch.manual_seed(seed)` for reproducible figures; means at
  uniform-random positions in a `scale`-sized box (`(rand−0.5)*scale`), plus an
  optional `x_offset` shift along x; equal spherical covariances via
  `diag_embed(ones)·std²`; equal (unnormalized — torch normalizes) weights.
- `symmetric_2D` — the one this lab actually uses: `nmodes` bumps equally
  spaced on a circle of radius `scale`. `linspace(0, 2π, nmodes+1)[:nmodes]`
  drops the duplicate 2π endpoint; means = `stack([cos, sin], dim=1)·scale`;
  weights `ones/nmodes`.

## Cell 2 — Plotting utilities (six small functions)

```python
def hist2d_samples(samples, ax=None, bins=200, scale=5.0, percentile=99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins,
                                       range=[[-scale, scale], [-scale, scale]])
```
Bin a `(n, 2)` cloud of points into a `bins × bins` count grid over the square
[−scale, scale]². The fixed `range` matters: without it numpy fits the bins to
the data, and panels at different times would use different coordinate systems.

```python
    cmax = np.percentile(H, percentile)
    norm = cm.colors.Normalize(vmax=cmax, vmin=0.0)
```
The subtlety of this function. Color-scale ceiling = the 99th-percentile *bin
count*, not the max. Near t=1 a conditional path concentrates thousands of
points into a couple of bins; scaling colors by the true max would render
everything else invisibly dark. Clipping at a percentile keeps the bulk of the
structure visible. (Part 4's Cell 31 tweaks `percentile` per frame for exactly
this reason.)

```python
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)
```
Draw the counts as an image. The `.T` + `origin='lower'` pair reconciles
matplotlib's image convention (row = y, origin top-left) with math convention
(x right, y up) — drop either and the picture is flipped/transposed.

`hist2d_sampleable`, `scatter_sampleable`, `kdeplot_sampleable`: draw
`num_samples` fresh points from a Sampleable and hand them to
histogram / scatter / seaborn-KDE respectively. Shared boilerplate per
function: `assert sampleable.dim == 2` (these are strictly 2D visualizers),
`if ax is None: ax = plt.gca()` (composable into subplot grids), and
`.detach().cpu()` on samples — matplotlib cannot eat GPU tensors; forgetting
`.cpu()` is the #1 plotting crash.

`imshow_density` / `contour_density`: evaluate a `Density` on a lattice —
`linspace` each axis (moved `.to(device)`, since `log_density` runs on the
model's device), `meshgrid` to a grid, `stack(...).reshape/-flatten` into a
`(bins², 2)` batch of query points, ONE `log_density` call, reshape back to
`(bins, bins)`, transpose, and `imshow`/`contour`. Same `.T`/`origin='lower'`
story as above. These paint the faint red (source) and blue (target) heatmaps
under every figure in the lab.

## Cell 3 — The ODE and SDE contracts (Lab 1 Cell 1, one shape changed)

```python
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t) -> torch.Tensor: ...
```
An ODE *is* a drift function u(x,t) — the arrow field of dx = u dt (notes 06
§A2). `SDE` adds `diffusion_coefficient` — the loudness of the random kicks in
dx = u dt + σ dW. No `step`, no `simulate`: equations describe dynamics,
simulators advance them — Lab 1's separation lesson, unchanged.

The one **real change is in the docstring shape contract: `t: (batch_size, 1)`**
— in Lab 1 `t` was a scalar `()`. Why: everything downstream
(α_t, β_t, the MLPs) consumes a per-sample time column, because training draws
a *different* t for every batch element and `torch.cat([x, t], dim=-1)`
(Cell 16) needs t as a `(bs, 1)` column. Simulation uses one shared clock, but
it is stored per-sample anyway so the same networks work in both modes.

## Cell 4 — Simulator, Euler, Euler–Maruyama, record_every

```python
class Simulator(ABC):
    @abstractmethod
    def step(self, xt, t, dt): ...
```
Same contract as Lab 1: advance the world one tick, `(bs, dim)` state, `(bs,1)`
time, `(bs,1)` dt.

```python
    @torch.no_grad()
    def simulate(self, x, ts):
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
        return x
```
`@torch.no_grad()`: simulation is inference; without it autograd would record
hundreds of steps and memory balloons. `len(ts) - 1` because each iteration
consumes a pair of timestamps — N stamps define N−1 steps. **The indexing is
the change from Lab 1**: `ts` is now `(bs, num_timesteps, 1)` — a clock *per
sample* — so the current time is the column `ts[:, t_idx]`, shape `(bs, 1)`,
and h likewise. (Note the small wart: `len(ts)` is `ts.shape[0]` = batch size,
not the number of timesteps — the code only works because every caller happens
to pass `num_samples ≥ num_timesteps`... actually no: `len(ts)-1` *is* wrong
reading, but callers pass `ts` expanded to `(num_samples, nts, 1)` where
`len(ts) = num_samples`. Compare `simulate_with_trajectory` below, which
correctly uses `nts = ts.shape[1]`. `simulate` is never called in this
notebook — only `simulate_with_trajectory` is — so the latent bug never fires.
Good catch-the-bug exercise.)

```python
    @torch.no_grad()
    def simulate_with_trajectory(self, x, ts):
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            ...
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
```
Keep the whole movie: clone each frame (defends against in-place `step`
implementations mutating every stored frame), stack along `dim=1` →
`(bs, num_timesteps, dim)`, index order "particle, time, coordinate."
`nts = ts.shape[1]` — the correct timestep count under the new convention.

```python
class EulerSimulator(Simulator):
    def step(self, xt, t, h):
        return xt + self.ode.drift_coefficient(xt,t) * h
```
Notes 06 §A1 Fact 2 verbatim: where you'll be = where you are + arrow × time.
`h` is `(bs,1)`, drift is `(bs,dim)`; broadcasting scales each particle's
arrow.

```python
class EulerMaruyamaSimulator(Simulator):
    def step(self, xt, t, h):
        return xt + self.sde.drift_coefficient(xt,t) * h \
                  + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)
```
Euler plus a fresh Gaussian kick per particle, scaled by σ and **√h, not h** —
over a step of length h Brownian noise accumulates *variance* h (variances
add, 05 §A1 Rule 2), so its standard deviation is √h. The single
most-tested fact about SDE simulation (see Lab 1 Cell 4 for the full story).

```python
def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, record_every),
        torch.tensor([num_timesteps - 1]),
    ])
```
Which trajectory frames to plot: indices 0, n, 2n, … plus ALWAYS the last one
(the endpoint is the sample; missing it would be silly). The `== 1` early
return avoids appending a duplicate last index. Two nits worth noticing:
the parameter `record_every` shadows the function's own name (legal, ugly);
and calling it with `record_every = num_timesteps // num_marginals` yields
`num_marginals + 1` frames unless the division is exact — e.g. 1000 steps,
`num_marginals=3` → indices {0, 333, 666, 999}, i.e. **four** scatter colors
in the "3-marginal" plots. Part 4's Cell 31 adds an `assert` to force exact
divisibility; the Part 2–3 plot cells just live with the extra frame.

---

## Part 1 — The central contract (Cell 5)

The markdown before this cell states the lab's premise (our notes 06 Parts
C–D, in the authors' notation): pick a conditional path p_t(x|z) with
p_0(·|z) = p_simple and p_1(·|z) = δ_z; its known conditional field
u_t^ref(x|z) generates it; the *marginal* field u_t^ref(x) — the average of
conditional fields over z — carries p_simple to p_data but is unknowable;
regressing against the conditional field secretly learns the marginal one
(least-squares learns conditional means, notes 06 §A3/D2).

## Cell 5 — `ConditionalProbabilityPath`, the lab's main interface

```python
class ConditionalProbabilityPath(torch.nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
```
A path object owns its two endpoints. `torch.nn.Module` as a base means that
if the endpoints are Modules (Gaussian, GaussianMixture), assigning them as
attributes auto-registers them as submodules — so ONE `.to(device)` on the
path moves the endpoint distributions too. That's why every construction later
reads `GaussianConditionalProbabilityPath(...).to(device)`.

```python
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        num_samples = t.shape[0]
        z = self.sample_conditioning_variable(num_samples) # (num_samples, dim)
        x = self.sample_conditional_path(z, t)             # (num_samples, dim)
        return x
```
The only *concrete* method, and it is one line of probability theory:
to sample the marginal p_t(x) = ∫ p_t(x|z) p(z) dz, sample z ~ p(z), then
x ~ p_t(x|z), then **throw z away**. Marginalization = hierarchical sampling.
Note the shape convention: `t` is `(num_samples, 1)` — a (possibly different)
time per sample — and the batch size is *read off t*. This method paints all
the "Ground-Truth Marginal" panels later.

```python
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor: ...
    @abstractmethod
    def sample_conditional_path(self, z, t) -> torch.Tensor: ...
    @abstractmethod
    def conditional_vector_field(self, x, z, t) -> torch.Tensor: ...
    @abstractmethod
    def conditional_score(self, x, z, t) -> torch.Tensor: ...
```
The four things a concrete path must know how to do, with uniform shapes —
z, x: `(num_samples, dim)`, t: `(num_samples, 1)`, all returns
`(num_samples, dim)` (except z's sampler). Read them as the four faces of one
object:
1. *where do conditioning points come from* (in practice: the training set —
   the notebook's later markdown note says exactly this),
2. *simulate the corruption analytically* (training data generator),
3. *the regression target for flow matching* (Cell 18),
4. *the regression target for score matching* (Cell 22).
Everything else in the lab is either an implementation of this interface or a
consumer of it.

---

## Part 2 — Gaussian conditional probability paths (Cells 6–15)

The markdown defines the object of study:
**p_t(x|z) = N(x; α_t z, β_t² I_d)** with p_simple = N(0, I_d), where
α, β: [0,1] → ℝ are monotonic, continuously differentiable,
α_1 = β_0 = 1 and α_0 = β_1 = 0. So t=0 gives N(0, I) (pure noise), t=1 gives
δ_z (the data point). This is DDPM's forward shortcut
x_t = √ᾱ·x₀ + √(1−ᾱ)·ε (05 §B2) with time reversed and the schedule freed
from the circle constraint — compare notes 06 Part F's circle-vs-line table.

## Cell 6 — PARAMS

```python
PARAMS = {
    "scale": 15.0,        # half-width of every plot window
    "target_scale": 10.0, # radius of the circle the mixture modes sit on
    "target_std": 1.0,    # std of each mixture bump
}
```
Plot/experiment constants in one dict "to avoid polluting the namespace" (the
authors' comment). Not config machinery — just three numbers reused by ~10
cells, so the figures all agree.

## Cell 7 — Meet p_simple and p_data

```python
p_simple = Gaussian.isotropic(dim=2, std = 1.0).to(device)
p_data = GaussianMixture.symmetric_2D(nmodes=5, std=PARAMS["target_std"],
                                      scale=PARAMS["target_scale"]).to(device)
```
The two endpoints for all of Parts 2–3: a unit Gaussian at the origin, and
five unit-std bumps on a circle of radius 10. These two module-level globals
are quietly reused by later plotting cells (e.g. Cell 21) — a notebook-ism to
be aware of if you run cells out of order.

The rest is a 1×3 figure: heatmap of p_simple (red), of p_data (blue), and
both overlaid. Per panel: a title, `set_xticks([])/set_yticks([])` to remove
ticks, and an `imshow_density(..., vmin=-10, alpha=0.25, cmap='Reds'/'Blues')`
call — `vmin=-10` floors the log-density colormap (log p plunges to −∞ in
empty space; without a floor all structure compresses into one color band),
`alpha=0.25` keeps the maps faint so later scatters read on top. Note the
third panel's two calls omit `ax=` and land on the current axes — it works
because it is the most recently created axes, another notebook-ism.

## Cell 8 — `Alpha` and `Beta`: schedules as verified contracts

```python
class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(self(torch.zeros(1,1)), torch.zeros(1,1))
        assert torch.allclose(self(torch.ones(1,1)), torch.ones(1,1))
```
The base constructor *tests its own subclass*: any concrete Alpha must map
t=0 → 0 and t=1 → 1, and violating the boundary contract crashes at
construction time rather than producing silently wrong paths. (The probe
tensors are `(1,1)` — the standard time shape with batch 1 — and live on CPU;
fine, because construction happens before any `.to(device)`.)
`torch.allclose` rather than `==`: float schedules may only hit the endpoints
to within rounding.

```python
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...
```
Subclasses supply the actual α_t, `(num_samples, 1)` in and out. Implementing
`__call__` makes schedule objects usable like functions: `self.alpha(t)`.

```python
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)              # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)      # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
```
α̇_t **for free, via autodiff** — the same `vmap(jacrev(...))` recipe Lab 1
used for spatial scores, now aimed at time. `jacrev(self)` differentiates the
`(1,1) → (1,1)` map, producing a `(1,1,1,1)` Jacobian per sample; `vmap`
runs it per batch item; the `unsqueeze`/`view(-1,1)` dance adds the dummy
per-sample axis first and flattens the Jacobian clutter after. Bookkeeping,
not math. Because this default exists, a subclass only *must* provide α_t
itself — its derivative is optional. `Beta` is the mirror image with the
flipped boundary contract β_0 = 1, β_1 = 0.

Why derivatives at all: the conditional vector field formula (Cell 10) is
built from α̇ and β̇ — the *velocity* of the schedule is what moves particles.

## Cell 9 — `LinearAlpha` and `SquareRootBeta`

The section fixes **α_t = t, β_t = √(1−t)**. Check the contracts: α_0=0, α_1=1
✓; β_0=1, β_1=0 ✓. (Aside for DDPM eyes: then x = t·z + √(1−t)·ε has
α² + β² = t² + 1 − t ≠ 1 — this is *not* the variance-preserving circle;
the lab's point is that any legal (α, β) pair works.)

```python
class LinearAlpha(Alpha):
    def __call__(self, t): return t
    def dt(self, t): return torch.ones_like(t)
```
α_t = t and its hand-written derivative α̇_t = 1 (overriding the autodiff
default — cheaper and exact). `ones_like(t)` keeps shape `(num_samples,1)`,
device, and dtype; returning the scalar `1.0` would break the shape contract.

```python
class SquareRootBeta(Beta):
    def __call__(self, t): return torch.sqrt(1-t)
    def dt(self, t): return - 0.5 / (torch.sqrt(1 - t) + 1e-4)
```
β_t = √(1−t), and β̇_t = −1/(2√(1−t)) by the chain rule — negative
(noise shrinks over time) and **diverging as t → 1**. The `+ 1e-4` in the
denominator is a numerical guard: at exactly t=1 the true derivative is −∞ and
the division would return `-inf`, which would poison every tensor it touches
downstream (`inf * 0 = nan`). The epsilon caps the blow-up at about −5000.
This divergence is not an implementation flaw — it is the honest geometry of a
path that must crush a Gaussian into a point, and it resurfaces twice below
(the SDE explosion note before Cell 15, and the t=0.9999 dodge in Cell 27).

## Cell 10 — `GaussianConditionalProbabilityPath`: the answer key

```python
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta):
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
```
For Gaussian paths the source is not a choice — the construction *requires*
p_simple = N(0, I_d) (that's what β_0 = 1, α_0 = 0 delivers at t=0) — so the
constructor builds it itself, using the `dim` property contract from Cell 1.
The path is fully specified by (p_data, α, β).

```python
    def sample_conditioning_variable(self, num_samples: int):
        return self.p_data.sample(num_samples)
```
z ~ p_data: the conditioning variable IS a data point. (The notebook's
markdown addresses the apparent circularity — "aren't we trying to learn to
sample p_data?" — with: in practice this line returns items from a finite
*training set*; only here, with a synthetic p_data, can it sample the true
distribution. The learned model must still turn *noise* into data at test
time; using data during training is exactly what "training" means.)

```python
    def sample_conditional_path(self, z, t):
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
```
**Problem 2.2's answer, and it is one reparameterization** (notes 05 §A3:
X = μ + σ·Z is exactly N(μ, σ²)): mean α_t z, std β_t, fresh unit noise with
z's shape/device via `randn_like`. Shapes: `alpha(t)` is `(ns,1)`, z is
`(ns,dim)` — broadcasting stretches the per-sample scalar across dims. This
line is this lab's Eq. ★★ (05 §B2): corrupt to ANY time in one shot, no
step-by-step simulation. It is also the training-data generator for both
trainers later.

```python
    def conditional_vector_field(self, x, z, t):
        alpha_t = self.alpha(t)        # (num_samples, 1)
        beta_t = self.beta(t)          # (num_samples, 1)
        dt_alpha_t = self.alpha.dt(t)  # (num_samples, 1)
        dt_beta_t = self.beta.dt(t)    # (num_samples, 1)
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
```
**Problem 2.3's answer**: the lecture formula
u_t(x|z) = (α̇_t − (β̇_t/β_t)·α_t)·z + (β̇_t/β_t)·x, transcribed with the four
schedule values pulled once each (readability + one autodiff call each, not
two). Where the formula comes from, in our notes' no-calculus style: a
particle riding the path stays at x_t = α_t z + β_t ε for its FIXED ε, so its
velocity is ẋ_t = α̇_t z + β̇_t ε; eliminate the unobservable ε using
ε = (x − α_t z)/β_t and collect terms in z and x — you get exactly the two
brackets above. (Same eliminate-and-collect move as notes 10 §3 for the
probability-flow ODE.) Sanity check with this lab's schedule (α=t, β=√(1−t)):
at t=0, α=0, α̇=1, β̇/β = −1/2, so u = z − x/2 — head toward your data point
while gently relaxing the noise; as t→1, β̇/β → −∞ and the field pulls x
violently onto z — that is δ_z forming (and the numerical trouble brewing).
All shapes: `(ns,1)` schedule columns broadcast against `(ns,dim)` states →
`(ns,dim)` out.

```python
    def conditional_score(self, x, z, t):
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2
```
**Problem 2.4's answer**: ∇_x log N(x; α_t z, β_t² I) = (α_t z − x)/β_t².
Derivation you can do in your head: log N = −‖x−μ‖²/(2σ²) + const, gradient
= −(x−μ)/σ², here μ = α_t z, σ = β_t. Two readings worth internalizing:
(1) the score is an arrow pointing from x back toward the mean α_t z, with
strength 1/β_t² — "uphill in probability" (Lab 1 Cell 15);
(2) substituting x = α_t z + β_t ε gives score = **−ε/β_t** — the
score/noise identity from our notes 10 (there written −ε/σ_t in the notation
header, powering §3's PF-ODE): learning the score and learning the noise are
the same job up to a known scale. Note also `beta_t ** 2` → this divides by
(1−t), which → 0 as t→1: the score, like β̇/β, explodes near the data end.

## Cell 11 — Visualizing one conditional path

```python
path = GaussianConditionalProbabilityPath(
    p_data = GaussianMixture.symmetric_2D(...).to(device),
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
).to(device)
```
Assemble the path (the constructors run the Cell 8 boundary asserts right
here). The outer `.to(device)` moves p_simple, p_data — everything registered
through the `nn.Module` machinery of Cell 5.

```python
z = path.sample_conditioning_variable(1) # (1,2)
ts = torch.linspace(0.0, 1.0, 7).to(device)
```
ONE conditioning point (this whole figure is about a single z — that is what
"conditional" means), and 7 snapshot times including both endpoints.

```python
plt.scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=75, label='z')
```
Mark z as a red star. `.cpu()` — matplotlib again.

```python
num_samples = 1000
for t in ts:
    zz = z.expand(num_samples, 2)
    tt = t.unsqueeze(0).expand(num_samples, 1) # (samples, 1)
    samples = path.sample_conditional_path(zz, tt) # (samples, 2)
    plt.scatter(samples[:,0].cpu(), samples[:,1].cpu(), alpha=0.25, s=8, label=f't={t.item():.1f}')
```
Per snapshot time: broadcast the single z to a `(1000, 2)` batch
(`expand` creates a view, no copy — safe because nothing writes to it),
broadcast the scalar t to the `(1000, 1)` time-column contract
(`unsqueeze(0)` makes the 0-dim scalar a `(1,)` vector so `expand` has an axis
to stretch), draw 1000 conditional samples, scatter them translucently. What
you should see: a unit blob at the origin at t=0 marching toward z while
shrinking — mean α_t z slides 0 → z, std β_t shrinks 1 → 0 — collapsing onto
the star at t=1. Remaining lines (figure size, axis limits, title, tick
removal, the two `imshow_density` underlays, legend sizing) are the Cell 7
styling pattern.

## Cell 12 — Wrapping the conditional field as an `ODE`

```python
class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        super().__init__()
        self.path = path
        self.z = z
```
Adapter: the Lab 1 simulators speak `ODE`; the path speaks
`conditional_vector_field(x, z, t)`. This class freezes one z and exposes the
rest as a drift. Composition again — nothing is re-derived.

```python
    def drift_coefficient(self, x, t):
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)
```
Per step: read the batch size off the state, broadcast the stored `(1, dim)` z
to `(bs, dim)` (`*self.z.shape[1:]` keeps it shape-agnostic — the docstring's
`(1, ...)` hints this adapter would survive image-shaped z), and delegate.
Without the expand, the path's `(ns,1)`-vs-`(ns,dim)` broadcasting would
still *happen* to work for z of batch 1 — but only by accident; being explicit
is the point.

## Cell 13 — Problem 2.3's three-panel check (samples vs trajectories vs ground truth)

The conceptual skeleton (everything else is the Cell 7/11 styling pattern —
axis limits, tick removal, red star for z, red/blue density underlays, legend
props — grouped and skipped from here on):

```python
num_samples = 1000; num_timesteps = 1000; num_marginals = 3
```
Knobs: how many particles ride the ODE, how many Euler steps, how many
snapshot times get scatter-plotted.

```python
torch.cuda.manual_seed(1)
z = path.sample_conditioning_variable(1) # (1,2)
```
Seed the **GPU** RNG so the figure picks the same mixture mode every run
(note: on a CPU-only machine this seeds nothing and z varies — a small
portability gotcha), then draw the single conditioning point.

```python
ode = ConditionalVectorFieldODE(path, z)
simulator = EulerSimulator(ode)
x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device)
xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
```
The lab's simulation idiom, worth reading carefully once:
- build equation, plug into simulator (Lab 1's composition pattern);
- start the whole population at p_simple = N(0, I) — the corruption process
  run *forward in the lab's clock* is the generation direction;
- the time grid: `linspace → (nts,)`, `view(1,-1,1) → (1, nts, 1)`,
  `expand(num_samples,-1,1) → (num_samples, nts, 1)` — the per-sample-clock
  contract of Cell 4. `expand` again: a view, zero memory for 1000 copies;
- simulate keeping every frame: `(1000, 1000, 2)`.

```python
every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
xts_every_n = xts[:,every_n,:]  # (bs, nts // n, dim)
ts_every_n = ts[0,every_n]      # (nts // n, 1)
for plot_idx in range(xts_every_n.shape[1]):
    tt = ts_every_n[plot_idx].item()
    ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), ...)
```
Middle panel ("Samples from Conditional ODE"): pick snapshot columns
(1000//3 = every 333rd step → frames {0, 333, 666, 999}; four colors despite
`num_marginals = 3`, the Cell 4 off-by-one), slice time index 0 of `ts` for
the labels (all rows share the clock, row 0 is as good as any), scatter each
snapshot. `.detach().cpu()` before plotting — `detach` is redundant under
`no_grad` but harmless and habitual.

```python
for traj_idx in range(15):
    ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')
```
Right panel ("Trajectories"): the first 15 particles' full paths as thin black
curves — individual behavior, complementing the middle panel's population
snapshots. The lab's recurring double lens (Lab 1 Cell 16).

```python
for plot_idx in range(xts_every_n.shape[1]):
    tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
    zz = z.expand(num_samples, 2)
    marginal_samples = path.sample_conditional_path(zz, tt)
    ax.scatter(...)
```
Left panel ("Ground-Truth Conditional Probability Path"): at the SAME snapshot
times, draw fresh analytic samples via Cell 10's one-liner. **This is the
correctness check for Problems 2.2+2.3 in one picture**: the middle panel got
its points by *integrating the vector field*; the left panel by *sampling the
Gaussian directly*; if your `conditional_vector_field` is right, the two
clouds match at every time. (Different random points — compare shapes, not
individual dots.)

## Cell 14 — Adding Langevin: the conditional SDE

The markdown premise: the SDE
dX_t = [u_t(X_t|z) + ½σ² ∇log p_t(X_t|z)] dt + σ dW_t has the SAME
time-marginals as the ODE — inject noise, then add ½σ²·score of drift to
clean it back up, and the distribution never notices. This is Lab 1's
Langevin fact (Cell 15: drift ½σ²·score + noise σ preserves p) *added on top
of* a moving reference field, and it's the template for every
diffusion-model sampler (notes 10 §1's η-dial lives on exactly this axis).

```python
class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path, z, sigma):
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma
```
Same adapter as Cell 12 plus the noise dial σ. (Copy-paste artifact: the
docstring still describes only `path` and `z`.)

```python
    def drift_coefficient(self, x, t):
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t) + 0.5 * self.sigma**2 * self.path.conditional_score(x,z,t)
```
The formula verbatim: reference arrows + ½σ² × uphill-in-probability arrows.
Both terms `(bs, dim)`. This is where Problem 2.4's `conditional_score` gets
consumed.

```python
    def diffusion_coefficient(self, x, t):
        return self.sigma
```
Constant noise level — but note it returns the *Python float*, not a
`(bs, dim)` tensor as the SDE contract's docstring demands. It works anyway:
in Euler–Maruyama the float multiplies `torch.sqrt(h) * randn_like(xt)` and
scalar-times-tensor broadcasts. A tensor-shaped
`self.sigma * torch.ones_like(x)` (Lab 1's style) would honor the contract;
keep the discrepancy in mind if you ever write a state-dependent σ.

The markdown after poses the numerical warning: substituting the score makes
the drift contain ½σ²(α_t z − X_t)/β_t², and **β_t → 0 as t → 1 makes the
drift explode, quadratically in σ** — a finite-step Euler–Maruyama cannot
follow it, so big σ produces visible artifacts. The stated practical fix:
let the noise decay too (σ_t = β_t), so the explosion cancels. File this
away — real diffusion samplers all shape their noise schedule for exactly
this reason.

## Cell 15 — Problem 2.4's three-panel check (SDE edition)

Structurally identical to Cell 13 — same knobs plus `sigma = 2.5`, same
seeding, same three panels — with exactly three substantive differences:

```python
sde = ConditionalVectorFieldSDE(path, z, sigma)
simulator = EulerMaruyamaSimulator(sde)
```
The equation is the Cell 14 SDE and the simulator is stochastic (√h·noise per
step, Cell 4).

```python
for traj_idx in range(5):
```
Only 5 trajectories in the right panel instead of 15 — SDE paths are jagged
scribbles; more would be unreadable.

And the check is the same but stronger: the middle panel (Euler–Maruyama
samples) must STILL match the left panel (analytic conditional samples) at
every snapshot — visual confirmation that Langevin augmentation preserves the
marginals, with a correctly implemented score. If your `conditional_score`
had the sign flipped, the noise would win and the cloud would visibly
over-spread. (Nit: the header comment still says "Run me for Problem 2.3!" —
copy-paste; this is 2.4.)

---

## Part 3 — Training on the Gaussian path (Cells 16–27)

The markdown restates the two-step trick (notes 06 §D): the marginal field
u_t^ref(x) = E_{z ~ p_t(z|x)}[u_t^ref(x|z)] is unknowable, but since the
minimizer of a squared error against a noisy target is the conditional mean
(06 §A3), regressing u_t^θ(x) against the *conditional* field under
(z ~ p(z), x ~ p_t(x|z), t ~ U[0,1)) — the **conditional flow matching loss
L_CFM** — has the marginal field as its exact minimizer.

## Cell 16 — `build_mlp` and `MLPVectorField`

```python
def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)
```
A `dims` list like `[3, 64, 64, 2]` becomes Linear(3→64), SiLU, Linear(64→64),
SiLU, Linear(64→2). The `if idx < len(dims) - 2` skips the activation after
the LAST layer — essential: the output is a velocity in ℝ², which can be any
real vector; squashing it through SiLU would forbid, e.g., large negative
components. `activation` is passed as a *class* (`Type[nn.Module]`) and
instantiated fresh each time (`activation()`) — sharing one instance would be
fine for stateless SiLU but the pattern is correct in general. SiLU
(x·sigmoid(x)) is the smooth default in diffusion codebases.

```python
class MLPVectorField(torch.nn.Module):
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])
```
The learned field u_t^θ: input size `dim + 1` — position ⊕ time — output size
`dim` — one arrow. The `+ 1` IS the time conditioning.

```python
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)
```
Concatenate `(bs, dim)` and `(bs, 1)` along the last axis → `(bs, dim+1)`,
one MLP pass → `(bs, dim)`. This is why the whole lab carries t as a
`(bs, 1)` column: `cat` demands matching batch shapes — feed a scalar t here
and you get an immediate dimension error. (Real image models replace this raw
concat with sinusoidal time embeddings — notes 06 Part E's training box — but
for 2D toys, raw t works.)

## Cell 17 — The `Trainer` scaffold

```python
class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass
```
The pattern: a Trainer owns a model and the ONLY thing a subclass must define
is how to compute one batch's loss. Flow matching vs score matching will
differ in `get_train_loss` alone — a strong hint that they are the same
algorithm with a different regression target.

```python
    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)
```
Adam, overridable. Note it is created fresh inside `train` — calling `train`
twice restarts optimizer state (momentum etc.), though the model weights
persist.

```python
    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs):
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
```
Move the model to the compute device (the models are constructed on CPU in
Cells 19/23/32/34 — this line is what makes that OK), build the optimizer
AFTER the move (optimizer state tensors follow parameter placement), switch to
train mode (a no-op for these plain MLPs — no dropout/batchnorm — but correct
hygiene).

```python
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')
        self.model.eval()
```
The canonical PyTorch five-step: clear stale gradients (forget this and
gradients *accumulate* across iterations — the classic silent bug), compute
loss, backprop, apply the update, report. `loss.item()` pulls the scalar to
CPU for display without keeping the graph alive. Two quirks to notice:
`tqdm(enumerate(range(n)))` is a roundabout `tqdm(range(n))` and, lacking a
`total`, shows a counter without an ETA bar; and **`train` returns `None`** —
yet every call site writes `losses = trainer.train(...)`. `losses` is always
`None`; nothing plots it, so nothing breaks — but if you ever want a loss
curve, you must collect and return `loss.item()` per step yourself.
"Epoch" here also really means "one freshly sampled batch": there is no
dataset and no dataloader, because the path can synthesize unlimited training
pairs — one luxury of knowing p_data.

## Cell 18 — Problem 3.1: the conditional flow matching loss

```python
class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
```
A trainer owns the model (via the base) *and* the path — the path is the
training-data factory and the answer key.

```python
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size,1).to(z)      # (bs, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, dim)
```
The Monte-Carlo sampling of the loss's expectation, one line per random
variable, exactly matching the blue subscripts in the markdown's L_CFM:
z ~ p_data (data point), t ~ U[0,1) (`torch.rand` is uniform on [0,1) — the
half-open interval conveniently dodges the β_1 = 0 singularity), and
x ~ p_t(x|z) via Cell 10's reparameterization line. The small gem is
**`.to(z)`**: tensor-to-tensor `.to` copies BOTH device and dtype from `z` —
`t` was born on CPU (`torch.rand` ignores the model's device) and this is the
one-token fix; forget it and the `cat([x,t])` inside the model throws a
device mismatch. Compare notes 06 Part E's training box: these three lines
are its first three lines with x₁ renamed z.

```python
        ut_theta = self.model(x,t)                        # (bs, dim)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, dim)
        error = torch.sum(torch.square(ut_theta - ut_ref), dim=-1) # (bs,)
        return torch.mean(error)
```
Prediction, target, squared error. Read the reduction order: `square` is
elementwise `(bs, dim)`; `sum(dim=-1)` collapses the coordinates into the
squared *norm* ‖·‖² per sample `(bs,)` — the norm in the loss formula; `mean`
averages over the batch — the 1/N in the Monte-Carlo estimate. (A blanket
`mean` of everything would differ only by the constant factor dim — harmless
here, but matching the formula keeps loss values comparable to theory.)
Why the target is computable at all: WE know which z generated this x — we
just made it — so the "conditional" field is evaluated with its condition in
hand. Why the loss will NOT go to zero: at a fixed (x, t), different z's
disagree, and the best any function of (x, t) can do is their conditional mean
(notes 06 §D2) — the leftover is the variance of the arrows around that mean,
an irreducible floor. The notebook bolds exactly this: "the loss should
converge, but not to zero!"

## Cell 19 — Train the flow model

```python
path = GaussianConditionalProbabilityPath(
    p_data = GaussianMixture.symmetric_2D(nmodes=5, ...).to(device),
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
).to(device)

flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64])

trainer = ConditionalFlowMatchingTrainer(path, flow_model)
losses = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=1000)
```
Rebuild the same path (each big cell reconstructs it — makes cells
self-contained at the cost of repetition), a 4×64 MLP (≈13k parameters —
2D toy scale), and 5000 Adam steps of batch 1000 at lr 1e-3. `flow_model` is
built on CPU; `train` moves it. `losses` is `None` (Cell 17's quirk). About a
minute on GPU. The global `flow_model` is reused by Cells 21, 25, 27.

## Cell 20 — Wrapping the learned field as an `ODE`

```python
class LearnedVectorFieldODE(ODE):
    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x, t):
        return self.net(x, t)
```
The three most satisfying lines in the lab: the *neural network* now sits
exactly where the *analytic* field sat in Cell 12 — same `ODE` socket, same
simulators, nothing downstream changes. (No `super().__init__()` and no z:
the learned field is the MARGINAL field; conditioning has been averaged out
by training. Docstring nit: `t` is `(bs, 1)`, not `(bs, dim)`.) This is the
payoff of Lab 1's equation/simulator split.

## Cell 21 — Does the learned ODE reproduce the marginal path?

Cell 13's triptych with two swaps. Swap one:

```python
ode = LearnedVectorFieldODE(flow_model)
simulator = EulerSimulator(ode)
x0 = path.p_simple.sample(num_samples)
ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device)
xts = simulator.simulate_with_trajectory(x0, ts)
```
Simulate the *learned marginal* ODE from noise — there is no z anywhere; this
is unconditional generation, the real thing.

Swap two, in the left ("Ground-Truth Marginal Probability Path") panel:

```python
    marginal_samples = path.sample_marginal_path(tt)
```
The comparison target is now `sample_marginal_path` (Cell 5's
sample-z-then-x-then-forget-z line) instead of the conditional sampler — fresh
z per point per snapshot, i.e. true marginal samples. What success looks like:
the middle panel's cloud spreads from the origin and splits into the five
mixture bumps in lockstep with the left panel; the right panel's trajectories
*curve* — each conditional field pushed straight at its own z, but averaging
over z bends the marginal flow (notes 06 Part E: straight conditionals,
curved marginal — the observation behind Reflow/rectification). Middle vs
left agreeing at ALL intermediate times, not just t=1, is the strong check:
the network learned the whole path, not just the endpoint. Everything else in
the cell (snapshot extraction via `record_every`, `num_samples // 10 = 100`
black trajectories, styling) is Cell 13 verbatim.

## Cell 22 — Problem 3.2: `MLPScore` and the score matching loss

The markdown sets it up: to add Langevin stochasticity at *sampling* time —
dX = [u_t^θ + ½σ² ∇log p_t(x)]dt + σ dW — we need the marginal score, and
the SAME trick applies: ∇log p_t(x) = E_{z ~ p_t(z|x)}[∇log p_t(x|z)], so
regressing against the *conditional* score (Cell 10's closed form) learns the
marginal one. Loss L_CSM = E‖s_t^θ(x) − ∇log p_t(x|z)‖².

```python
class MLPScore(torch.nn.Module):
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x, t):
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)
```
Byte-for-byte `MLPVectorField` with a different name. That is the lesson, not
laziness: a score field and a velocity field are both "ℝ^dim × time → ℝ^dim";
only the *training target* distinguishes them (and Cell 26 will show one is a
linear function of the other).

```python
class ConditionalScoreMatchingTrainer(Trainer):
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size,1).to(z)      # (bs, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, dim)

        s_theta = self.model(x,t)                    # (bs, dim)
        s_ref = self.path.conditional_score(x,z,t)   # (bs, dim)
        mse = torch.sum(torch.square(s_theta - s_ref), dim=-1) # (bs,)
        return torch.mean(mse)
```
Diff against Cell 18: `conditional_vector_field` → `conditional_score`. The
sampling lines, the `.to(z)` trick, the sum-then-mean reduction — identical.
Since the conditional score is −ε/β_t (Cell 10), this is *noise prediction
with a per-t scale* — DDPM's L_simple (05 §D4) is this same loss reweighted;
the 1/β_t factor means the target grows huge for t near 1, which is why this
loss converges to a (much larger) non-zero floor and why real systems prefer
predicting ε and rescaling.

## Cell 23 — Train the score model

Same shape as Cell 19: rebuild the path, `score_model = MLPScore(dim=2,
hiddens=[64,64,64,64])`, train — but only `num_epochs=1000` (score targets
are informative enough, and this is a demo). Global `score_model` feeds
Cells 25 and 27.

## Cell 24 — `LangevinFlowSDE`: two networks, one sampler

```python
class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma
```
The learned twin of Cell 14: both networks plus the noise dial. (Docstring is
again a stale copy mentioning `path` and `z` — neither exists here.)

```python
    def drift_coefficient(self, x, t):
        return self.flow_model(x,t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x, t):
        return self.sigma
```
dX = [u^θ + ½σ² s^θ]dt + σ dW, with the same return-a-float contract wobble
as Cell 14 (works via broadcasting). Conceptually this is the biggest cell in
the lab despite its size: **a whole family of samplers, indexed by σ, from
ONE pair of trained networks** — σ=0 is the deterministic flow (≈ DDIM),
σ>0 trades determinism for Langevin self-correction (≈ ancestral
diffusion) — the η-dial of notes 10 §1.2 rebuilt from parts.

## Cell 25 — The learned SDE triptych

Cell 21's figure with the SDE swapped in:

```python
num_timesteps = 300
sigma = 2.0 # Don't set sigma too large or you'll get numerical issues!
...
sde = LangevinFlowSDE(flow_model, score_model, sigma)
simulator = EulerMaruyamaSimulator(sde)
```
300 steps instead of 1000 (SDE snapshots are self-correcting; also cheaper —
each step is now TWO network calls, flow + score), and the σ warning is
Cell 14's β_t → 0 explosion, now with a *learned* score that also becomes
huge and less reliable near t=1. One more cosmetic difference: the density
underlays reference `path.p_simple`/`path.p_data` (attributes of the path)
rather than the Cell 7 globals — same objects, tidier habit. Success: the
middle panel's marginals STILL match the left panel's ground truth despite
per-step noise injection — Langevin preserving marginals, now end-to-end
learned. Trajectories (right panel) are jagged where Cell 21's were smooth,
yet land in the same five bumps.

## Cell 26 — Question 3.3: the score hiding inside the flow

The markdown derivation to hold onto: for Gaussian paths the marginal field
and marginal score obey the LINEAR identity
u_t^ref(x) = a_t·x + b_t·∇log p_t(x), with
(a_t, b_t) = (α̇_t/α_t, β_t²·α̇_t/α_t − β̇_t·β_t). Rearranged:
score = (u − a_t x)/b_t. So a trained flow model *contains* a score model:

```python
class ScoreFromVectorField(torch.nn.Module):
    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta
```
Not a new network — zero new parameters. A wrapper holding the trained field
and the schedules that define the conversion coefficients.

```python
    def forward(self, x, t):
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)

        num = alpha_t * self.vector_field(x,t) - dt_alpha_t * x
        den = beta_t ** 2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t

        return num / den
```
The markdown's s̃_t^θ(x) = (α_t u^θ − α̇_t x)/(β_t² α̇_t − α_t β̇_t β_t) —
the same rearrangement with numerator and denominator multiplied through by
α_t. All four schedule values are `(bs,1)` columns broadcasting against
`(bs,dim)`. Sanity-check the denominator with this lab's schedule (α=t,
β=√(1−t), α̇=1, β̇=−1/(2√(1−t))): (1−t)·1 − t·(−1/(2√(1−t)))·√(1−t)
= 1 − t + t/2 = **1 − t/2** — exactly the markdown's claim: nonzero on
[0,1), zero only at t=1. So the conversion is safe everywhere except the
final instant (dodged in the next cell by t ≤ 0.9999). Big picture: this is
the flow-matching face of the ε ↔ x₀ ↔ score ↔ velocity interconvertibility
family (notes 06 Part F "linear re-labelings"; notes 10 uses the same
identity to turn ε̂ into the PF-ODE drift). Diffusion models learn ONE
function wearing four costumes.

## Cell 27 — Quiver comparison: learned score vs converted score

```python
num_bins = 30
num_marginals = 4
```
A 30×30 grid of probe points, at 4 times.

```python
learned_score_model = score_model
flow_score_model = ScoreFromVectorField(flow_model, path.alpha, path.beta)
```
The two contenders: s_t^θ trained by score matching (Cell 23) vs s̃_t^θ
computed from the flow model (Cell 26). If both trainings succeeded, these
two functions — produced by *entirely different losses* — must agree.

```python
ts = torch.linspace(0.0, 0.9999, num_marginals).to(device)
xs = torch.linspace(-scale, scale, num_bins).to(device)
ys = torch.linspace(-scale, scale, num_bins).to(device)
xx, yy = torch.meshgrid(xs, ys)
xx = xx.reshape(-1,1)
yy = yy.reshape(-1,1)
xy = torch.cat([xx,yy], dim=-1)
```
Times stop at 0.9999, not 1.0 — the markdown's ε-dodge. At exactly t=1 the
code computes `beta_t = 0` and (thanks to Cell 9's epsilon)
`dt_beta_t ≈ −5000`, making Cell 26's denominator
β²α̇ − αβ̇β = 0 − 1·(−5000)·0 = 0 → division by zero; at t = 0.9999 the
denominator is safely ≈ 1 − t/2 ≈ 0.5. The rest builds the probe batch:
meshgrid the axes, flatten each to `(900, 1)`, concat → `(900, 2)` query
points — the same lattice-to-batch move as Cell 2's density plotting.

```python
for idx in range(num_marginals):
    t = ts[idx]
    bs = num_bins ** 2
    tt = t.view(1,1).expand(bs, 1)

    learned_scores = learned_score_model(xy, tt)
    learned_scores_x = learned_scores[:,0]
    learned_scores_y = learned_scores[:,1]

    ax = axes[0, idx]
    ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_scores_x.detach().cpu(), learned_scores_y.detach().cpu(), scale=125, alpha=0.5)
```
Per column: broadcast the scalar time to the `(900, 1)` contract, evaluate the
score at all 900 grid points in one call, split into x/y components, and draw
arrows with `quiver`. `scale=125` fixes quiver's arrow-length normalization to
the same constant in both rows — let matplotlib auto-scale each panel and the
visual comparison would be meaningless. `.detach()` genuinely matters here:
unlike the simulation cells there is no `no_grad()` context, so the model
outputs carry grad history and matplotlib/numpy conversion would raise without
it. The second half of the loop body repeats everything with
`flow_score_model` into row 1 (one dead line — `ax = axes` — is immediately
overwritten by `ax = axes[1, idx]`; harmless leftover). Row labels
("Learned with Score Matching" / "Computed from u_t^θ(x)") plus per-panel
density underlays and tick removal — styling. What agreement means: two
independently trained objects satisfying a theoretical identity is strong
evidence BOTH are close to the truth — you're checking theory with theory.
Expect the fields to point from empty space toward the five bumps, growing in
magnitude at later t (the 1/β² sharpening).

---

## Part 4 — Linear paths between arbitrary distributions (Cells 28–35)

The markdown defines the second path: fix z, take X_0 ~ p_simple, and set
**X_t = (1−t)·X_0 + t·z** — our notes 06 §C interpolant with x₁ = z, defining
p_t(x|z) as the law of X_t. Endpoints: p_0(·|z) = p_simple, p_1(·|z) = δ_z.
Conditional field: u_t(x|z) = (z − x)/(1 − t). Two observations the code will
exploit: (1) no closed-form conditional *score* — p_simple may have no density
we can write (Sampleable-not-Density strikes); (2) **p_simple need not be
Gaussian** — flow matching between arbitrary clouds.

## Cell 28 — Three non-Gaussian Sampleables

```python
class MoonsSampleable(Sampleable):
    def __init__(self, device, noise=0.05, scale=5.0, offset=None):
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)
```
A thin adapter over sklearn. Note it is NOT an `nn.Module` — no buffers, no
`.to()` — so it stores its `device` explicitly and moves each batch there
itself. The mutable-default dodge (`offset=None` then create zeros) is the
standard Python idiom. `dim` is hardcoded 2.

```python
    def sample(self, num_samples: int) -> torch.Tensor:
        samples, _ = make_moons(n_samples=num_samples, noise=self.noise, random_state=None)
        return self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device) + self.offset
```
sklearn returns numpy float64 points (labels discarded via `_`);
`random_state=None` keeps every call fresh (a fixed seed would make "sampling"
return the same cloud forever — deadly for training). The cast to float32
before `from_numpy` avoids a silent dtype mismatch with the float32 models;
then scale, move, shift. (Docstring typo: says shape `(num_samples, 3)`; it is
`(num_samples, 2)`.) `CirclesSampleable` is identical with
`make_circles(..., factor=0.5)` — two concentric rings, inner at half radius.

```python
class CheckerboardSampleable(Sampleable):
    def sample(self, num_samples: int) -> torch.Tensor:
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0,2).to(device)
        while samples.shape[0] < num_samples:
            new_samples = (torch.rand(num_samples,2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:,0] + self.scale) / grid_length) % 2 == 0 # (bs,)
            y_mask = torch.floor((new_samples[:,1] + self.scale) / grid_length) % 2 == 0 # (bs,)
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]
```
Hand-rolled **rejection sampling** — worth reading line by line because it's a
technique, not a library call:
- `grid_length`: the board spans [−scale, scale]² split into `grid_size`
  cells per side.
- Start with an EMPTY `(0, 2)` tensor (note it uses the *global* `device`,
  not `self.device` — works here because they're the same object; still an
  inconsistency to spot) and loop until enough survivors accumulate.
- Propose `num_samples` uniform points on the whole square:
  `(rand − 0.5) · 2 · scale`.
- Column index: shift x by +scale into [0, 2·scale], divide by cell size,
  `floor` → integer cell column; `% 2 == 0` → is it an even column? Same for
  rows.
- `logical_xor(~x_mask, y_mask)`: true when `~x_mask ≠ y_mask`, i.e. when
  `x_mask == y_mask` — keep points whose row and column parities MATCH (both
  even or both odd). That alternating keep/discard pattern IS the
  checkerboard. (An XNOR written as xor-of-a-complement — squint-worthy but
  correct.)
- Boolean-mask the survivors, append, and finally trim the overshoot with
  `samples[:num_samples]`. Each round keeps ≈half its proposals, so the loop
  runs ~2 iterations.

Why the checkerboard is the boss-level target: hard edges and disconnected
components — a density a Gaussian mixture can't fake, and a good stress test
for how sharply a learned flow can fold mass.

## Cell 29 — Visualize the new targets

```python
targets = {
    "circles": CirclesSampleable(device),
    "moons": MoonsSampleable(device, scale=3.5),
    "checkerboard": CheckerboardSampleable(device, grid_size=4)
}
```
Dict of name → distribution, then one loop: `hist2d_sampleable(target,
20000, bins=100, scale=7.5, ax=ax)` per panel plus `set_aspect('equal')`
(circles should look circular) and the usual tick/title styling. Histograms,
not scatter — 20k points reveal *density* (the two moons have uniform
intensity; the rings are thin and sharp). These have no `log_density`, so the
heatmap-of-density plots of Parts 2–3 are impossible — samples are all we
have, which is the entire point of Part 4.

## Cell 30 — Problem 4.1: `LinearConditionalProbabilityPath`

```python
class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)
```
Contrast with the Gaussian path's constructor: p_simple is now a *parameter*.
Any Sampleable pair works — this single signature change is observation (2).

```python
    def sample_conditioning_variable(self, num_samples: int):
        return self.p_data.sample(num_samples)
```
Same as before: z is a data point.

```python
    def sample_conditional_path(self, z, t):
        x0 = self.p_simple.sample(z.shape[0])
        return (1 - t) * x0 + t * z
```
**The interpolant, literally** (notes 06 §C, Eq. ▲): draw a FRESH source point
per call and slide. Batch size read off `z.shape[0]`; `t` is `(ns,1)`
broadcasting against `(ns,dim)`. Two things to notice: unlike the Gaussian
path there is no `randn_like` — the only randomness is x₀ itself (at fixed z
and t, X_t is a deterministic squash of the source distribution); and pairing
is *independent* — z and x₀ are drawn with no relation to each other, which is
exactly the independent-coupling assumption of vanilla flow matching.

```python
    def conditional_vector_field(self, x, z, t):
        return (z - x) / (1 - t)
```
u_t(x|z) = (z − x)/(1−t). Where it comes from: on the interpolant the velocity
is constant, ẋ = z − x₀ (06 §C Eq. ▲▲); eliminating x₀ via
x₀ = (x − tz)/(1−t) gives z − x₀ = (z − x)/(1−t) — "the remaining
displacement divided by the remaining time." Cross-check against Part 2's
machinery: with a Gaussian source this path is p_t(x|z) = N(tz, (1−t)²I),
i.e. a Gaussian path with α_t = t, β_t = 1 − t; plug α̇=1, β̇=−1,
β=1−t into Cell 10's formula: (1 + t/(1−t))·z − x/(1−t) = (z − x)/(1−t) ✓.
Same singularity at t=1 as ever — defined on [0,1) only; training's
`torch.rand` ∈ [0,1) never touches it.

```python
    def conditional_score(self, x, z, t):
        raise Exception("You should not be calling this function!")
```
Honest API design: the conditional density is the law of (1−t)X₀ + tz, and
for a samples-only p_simple (moons, checkerboard...) there is no formula to
differentiate — so the method fails LOUDLY instead of returning something
plausible and wrong. Consequence: no score matching and no Langevin/SDE
sampling on linear paths here — Part 4 is ODE-only. (This is why diffusion
*models* insist on Gaussian corruption: it buys the score.)

## Cell 31 — Problem 4.1's consistency check: three rows must agree

```python
num_samples = 100000
num_timesteps = 500
num_marginals = 5
assert num_timesteps % (num_marginals - 1) == 0
```
100k samples — histograms need mass. The `assert` fixes Cell 4's
`record_every` off-by-one: 500 % 4 == 0 makes the recorded frames land
exactly on 5 evenly spaced times.

```python
path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0),
    p_data = CheckerboardSampleable(device, grid_size=4)
).to(device)
z = path.p_data.sample(1) # (1,2)
```
Gaussian → checkerboard, and one conditioning point for the first two rows.

Row 1 — conditional path via `sample_conditional_path`:
```python
ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
for idx, t in enumerate(ts):
    zz = z.expand(num_samples, -1)
    tt = t.view(1,1).expand(num_samples,1)
    xts = path.sample_conditional_path(zz, tt)
    percentile = min(99 + 2 * torch.sin(t).item(), 100)
    hist2d_samples(samples=xts.cpu(), ax=axes[0, idx], bins=300, scale=scale, percentile=percentile, alpha=1.0)
```
Same broadcast idiom as Cell 11 (`expand(num_samples, -1)`: −1 = "keep that
dim"), but plotted as histograms. The `percentile` line is a purely cosmetic
hack: as t → 1 the cloud collapses toward z and nearly all mass enters a few
bins; sliding the color ceiling from the 99th percentile (t=0) toward 100
(t→1; `2·sin(t)` grows 0 → ~1.7, clipped at 100) keeps the collapsing blob
from saturating. Don't look for math in it.

Row 2 — the SAME path via the ODE:
```python
ode = ConditionalVectorFieldODE(path, z)
simulator = EulerSimulator(ode)
ts = torch.linspace(0,1,num_timesteps).to(device)
record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
```
Cell 12's adapter reused unchanged on the NEW path class — the interface
paying rent. Integrate 100k particles for 500 Euler steps, keep 5 frames,
histogram each (same percentile hack, labels from
`ts[record_every_idxs[idx]]`). Rows 1 and 2 agreeing is the
Problems-4.1 check: your sampler and your vector field describe the same
p_t(x|z).

Row 3 — the marginal path:
```python
    xts = path.sample_marginal_path(tt)
```
Fresh z per sample (Cell 5). Expect: unit blob at t=0 morphing into the FULL
checkerboard at t=1 — and at mid-times a smeared hybrid. Per-row styling
(limits, ticks, titles, y-labels, the red star on the last column of rows 1–2
marking z) grouped as usual.

## Cell 32 — Train flow matching on the linear path

```python
path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0),
    p_data = CheckerboardSampleable(device, grid_size=4)
).to(device)

linear_flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64])

trainer = ConditionalFlowMatchingTrainer(path, linear_flow_model)
losses = trainer.train(num_epochs=10000, device=device, lr=1e-3, batch_size=2000)
```
The punchline cell for software design: **`ConditionalFlowMatchingTrainer` is
reused verbatim** — it only ever spoke to the abstract path interface, so
swapping Gaussian → linear path costs zero new training code. Budget doubles
(10000 epochs, batch 2000): the checkerboard's hard edges are harder to
regress than five smooth bumps. Note the trainer calls only
`sample_conditional_path` and `conditional_vector_field` — the forbidden
`conditional_score` is never touched, so the `raise` never fires.

## Cell 33 — Ground truth vs learned, side by side

A 2×5 grid. Row 0: `path.sample_marginal_path(tt)` histogrammed at 5 times
(the analytic answer — same loop as Cell 31 row 3, batch 50000). Row 1:

```python
ode = LearnedVectorFieldODE(linear_flow_model)
simulator = EulerSimulator(ode)
ts = torch.linspace(0,1,100).to(device)
record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
for idx in range(xts.shape[1]):
    xx = xts[:,idx,:]
    hist2d_samples(samples=xx.cpu(), ax=axes[1, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
```
Simulate the learned field — with only **100 Euler steps** now (vs 1000 in
Part 2/3): the near-straight linear path is exactly what Euler integrates
accurately with few steps (notes 06 Fact 1/2 and Part B's "why FM samples in
tens of steps"). Snapshot, histogram, label; usual styling and row labels
("Ground Truth" / "Learned"). Success: a recognizable 4×4 checkerboard grown
out of a Gaussian blob, with matching *intermediate* marginals — the learned
field reproduces the whole path. Expect slightly soft edges: an MLP regressing
a discontinuous-density limit will blur corners; more capacity/epochs
sharpens them.

## Cell 34 — Problem 4.3: drop the Gaussian entirely

```python
path = LinearConditionalProbabilityPath(
    p_simple = CirclesSampleable(device),
    p_data = CheckerboardSampleable(device, grid_size=4)
).to(device)

bridging_flow_model = MLPVectorField(dim=2, hiddens=[100,100,100,100])

trainer = ConditionalFlowMatchingTrainer(path, bridging_flow_model)
losses = trainer.train(num_epochs=20000, device=device, lr=1e-3, batch_size=2000)
```
The source is now concentric circles — **no Gaussian anywhere in the
problem**. Nothing in the interpolant, the field, the trainer, or the sampler
ever used Gaussianity of p_simple; the code diff from Cell 32 is literally
the `p_simple=` line. Wider network (4×100) and 20000 epochs: bridging two
structured distributions gives a rougher marginal field (arrows from
independently paired circle-points and checker-points crisscross heavily), so
the conditional targets are noisier and the fit is harder. This
generality — flows between arbitrary marginals — is where flow matching
outgrows diffusion (whose forward process is welded to Gaussians); it's the
door to bridge/interpolant methods.

## Cell 35 — The bridge, visualized

Byte-level a copy of Cell 33 with `bridging_flow_model`, `num_samples =
30000`, and `ts = torch.linspace(0,1,200)` — 200 Euler steps this time (the
circle→checkerboard marginal field is curvier than Gaussian→checkerboard, so
Euler needs a finer grid; try 50 and watch the corners smear). Row 0 the
analytic marginal path (rings dissolving into checkerboard), row 1 the
learned one. The final markdown cell hands you the keys: swap `p_simple` /
`p_data` among the Cell 28–29 zoo and observe — moons→circles,
checkerboard→moons all train with the SAME three-line recipe.

---

## What to carry forward (Lab 2 → Lab 3 and beyond)

1. **One interface, whole lab**: `ConditionalProbabilityPath` = sample z,
   sample p_t(x|z), conditional field, conditional score. Every trainer,
   sampler, and figure consumed only this contract — that's why Gaussian and
   linear paths swap freely.
2. **The Gaussian path is DDPM's forward shortcut with designable (α, β)**:
   p_t(x|z) = N(α_t z, β_t² I); sampling it is one reparameterization
   (05 §A3); t runs 0=noise → 1=data here.
3. **Two closed forms to memorize**:
   u_t(x|z) = (α̇ − (β̇/β)α)z + (β̇/β)x, and
   ∇log p_t(x|z) = (α_t z − x)/β_t² = −ε/β_t (the score/noise identity,
   notes 10). Both explode as β_t → 0 — every t→1 epsilon, warning, and
   σ_t = β_t remedy in the lab traces back to this.
4. **Training = regress the conditional target; the marginal is what you
   get**: L_CFM and L_CSM differ only in the target line; least-squares
   learns conditional means (06 §A3/D2); the loss floor is the target's
   conditional variance — converges, never to zero.
5. **Flow and score are the same information**: identical MLPs, and
   s̃ = (α u^θ − α̇ x)/(β²α̇ − αβ̇β) converts one into the other with zero
   new parameters — the lab's member of the ε/x₀/score/velocity re-labeling
   family (06 Part F, 10 §3).
6. **σ is a sampler dial, not a training choice**: u^θ alone → deterministic
   ODE; u^θ + ½σ²s^θ + σdW → stochastic samplers with the same marginals
   (the η-dial of notes 10 §1.2).
7. **Linear paths free the source**: X_t = (1−t)X₀ + tz needs only samples
   from both ends — but forfeits the score (no density, no Langevin). Choose
   Gaussian paths when you want SDEs; linear paths when you want straightness
   and arbitrary endpoints.
8. Recurring PyTorch craft: `(bs,1)` time columns feeding `cat([x,t])`;
   `.to(z)` for device+dtype in one token; `expand` for free broadcasts;
   boundary asserts in ABC constructors; analytic `.dt` overrides of an
   autodiff default; rejection sampling with boolean masks; and the small
   honest bugs — `len(ts)` vs `ts.shape[1]`, `record_every`'s extra frame,
   `train()` returning `None` into `losses`.
