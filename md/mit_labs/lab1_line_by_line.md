# MIT Lab 1, Line by Line — Simulating ODEs and SDEs

*A beginner's reading companion to `solutions/lab_one_complete.ipynb`. Every
line of code: what it does, why it's there, and what breaks without it.
Concepts link to our notes: Euler = `06 §A1`, score = `10 §3`.*

**The lab in one sentence:** before any neural networks, learn to *simulate* —
follow deterministic arrows (ODEs), follow arrows plus random kicks (SDEs),
and watch how doing so moves entire *distributions* of points around. Every
diffusion sampler you'll ever write is one of these two loops.

---

## Cell 0 — Imports and device

```python
from abc import ABC, abstractmethod
```
Python's tools for **abstract base classes**. `ABC` marks a class as "a
template, not a thing you can build"; `@abstractmethod` marks a method as
"every child class MUST provide this." The lab uses them as enforceable
contracts: an `SDE` *is* anything with a drift and a diffusion — nothing more.

```python
from typing import Optional
```
Type-hint helper: `Optional[Axes]` means "an Axes object, or None." Purely
documentation — Python doesn't enforce it — but it tells readers which
arguments may be omitted.

```python
import numpy as np
```
NumPy. Used sparingly here (e.g., `np.pi`); PyTorch does the heavy lifting.

```python
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
```
Plotting. `plt` is the canvas-level interface; `Axes` is imported only so
function signatures can *name* the type of the `ax` argument.

```python
import torch
import torch.distributions as D
```
PyTorch, and its probability-distributions toolbox `D` — ready-made Gaussian
and mixture objects with correct `sample()` and `log_prob()` methods, so the
lab doesn't hand-roll densities.

```python
from torch.func import vmap, jacrev
```
The two pieces of PyTorch's *function-transform* API used later to get scores
for free: `jacrev(f)` returns a new function computing the **Jacobian**
(all derivatives) of `f`; `vmap` makes any function operate **per-batch-item**
without a Python loop. Together: exact ∇log p for every sample, no math done
by hand. (More at Cell 11.)

```python
from tqdm import tqdm
import seaborn as sns
```
`tqdm` wraps any loop in a progress bar (you'll see it around simulation
steps). `seaborn` = prettier statistical plots (line plots, histograms, KDE).

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Pick the GPU if one exists, else CPU. Every tensor created later is moved
`.to(device)`; forgetting one produces the classic
"expected device cuda but got cpu" error.

---

## Cell 1 — The ODE and SDE contracts

```python
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t) -> torch.Tensor:
```
An ODE, to this lab, is JUST a drift function: the arrow field
u(x, t) in dx/dt = u(x_t, t) (our notes 06 §A2 — "arrow attached to every
location and time"). Declaring it abstract means "any concrete ODE must say
what its arrows are"; there is no other requirement.

The docstring pins the **shape contract**, which is the real content here:
- `xt: (batch_size, dim)` — a whole *population* of points is simulated at
  once; batch = many independent particles.
- `t: ()` — a scalar (0-dimensional tensor): every particle shares one clock.
- returns `(batch_size, dim)` — one arrow per particle.

```python
class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t) -> torch.Tensor: ...
    @abstractmethod
    def diffusion_coefficient(self, xt, t) -> torch.Tensor: ...
```
An SDE is an ODE **plus a randomness dial**:
dx = u(x_t, t)dt + σ(x_t, t)dW. `drift_coefficient` is the deterministic
part (same as ODE); `diffusion_coefficient` says *how loud the random kicks
are* at each place and time. Note what is absent: no `step`, no `simulate` —
the equation object only *describes* dynamics; HOW to simulate them is a
separate concern (next cell). This separation is the lab's main software
lesson: swap equations and simulators independently.

---

## Cell 2 — The Simulator contract and the two loops

```python
class Simulator(ABC):
    @abstractmethod
    def step(self, xt, t, dt):
```
A simulator is anything that knows how to advance the world by one tick:
given the state at time t and a step size dt, produce the state at t+dt.
Abstract, because Euler is only one choice (midpoint, Heun, DPM-Solver…
are all just different `step`s — see our notes 10).

```python
    @torch.no_grad()
    def simulate(self, x, ts):
```
`@torch.no_grad()`: simulation is *inference*, not training — no gradients
are needed through hundreds of steps. Without this decorator PyTorch would
record every step in the autograd graph and memory would balloon.

```python
        for t_idx in range(len(ts) - 1):
```
March through the time grid. `len(ts) - 1` because each iteration consumes a
*pair* (ts[i], ts[i+1]) — N timestamps define N−1 steps. Off-by-one here is
the classic sampler bug.

```python
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
```
Current time, and step size *derived from the grid* (h = next − current)
rather than assumed constant — so non-uniform time grids work for free
(exactly what fancy samplers exploit).

```python
            x = self.step(x, t, h)
        return x
```
Overwrite the state each tick; return only the endpoint. Memory: O(1) in
time — nothing but the current state is kept.

```python
    @torch.no_grad()
    def simulate_with_trajectory(self, x, ts):
        xs = [x.clone()]
```
Same loop, but keep the whole movie. `x.clone()` matters: `step` returns new
tensors here, but cloning defends against any future in-place `step` —
without it every list entry could point at the same mutated memory and the
"trajectory" would be N copies of the final frame.

```python
        for t_idx in tqdm(range(len(ts) - 1)):
```
Identical loop wrapped in a progress bar (this is the version humans watch).

```python
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
```
Collect each frame; `torch.stack(xs, dim=1)` turns a list of
(batch, dim) frames into one (batch, num_timesteps, dim) tensor — dim=1 so
that index order reads "particle, time, coordinate."

---

## Cell 3 — Q1.1a: Euler simulator (three lines that matter)

```python
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
```
Composition: the simulator *holds* an equation. Any ODE object plugs in.

```python
    def step(self, xt, t, h):
        return xt + self.ode.drift_coefficient(xt,t) * h
```
Euler's method, verbatim from our notes (06 §A1 Fact 2): **where you'll be =
where you are + arrow × time**. `drift_coefficient` returns (batch, dim),
`h` is scalar; broadcasting scales every particle's arrow at once. That's
the entire deterministic sampler family in one line.

## Cell 4 — Q1.1b: Euler–Maruyama (the stochastic sibling)

```python
        return xt + self.sde.drift_coefficient(xt,t) * h \
                  + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)
```
Euler plus one new term: a fresh Gaussian kick per particle
(`randn_like(xt)` = right shape, right device), scaled by the local noise
level σ **and by √h — not h.** Why the square root: over a step of length h,
Brownian noise accumulates *variance* h (variances add over time — our notes
05 §A1 Rule 2), so its *standard deviation* is √h. Scale by h instead and
noise vanishes too fast as you refine the grid; your "SDE" silently becomes
an ODE. This √h is the single most-tested fact about SDE simulation.
Compare with our DDPM ancestral step (05 Part E): mean move + σ·z — same
skeleton, different clothes.

---

## Cell 5 — Q2.1: Brownian motion, the hydrogen atom of SDEs

```python
class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma
```
One knob: the noise volume σ.

```python
    def drift_coefficient(self, xt, t):
        return torch.zeros_like(xt)
```
NO arrows — pure randomness. `zeros_like` (not the scalar 0) keeps the
shape/device contract: downstream code can always add drift to noise
blindly. Also note the object ignores `t`: time-independence expressed by
simply not using the argument.

```python
    def diffusion_coefficient(self, xt, t):
        return self.sigma * torch.ones_like(xt)
```
Constant noise everywhere: σ·1, again shaped like the state. dx = σ·dW is
the textbook Wiener process: expect trajectories that wander with spread
growing like √t (variance σ²t) — which is exactly what the plot next cell
confirms.

## Cell 6 — Trajectory plotting utility (read once, then trust)

The one *conceptual* line:
```python
        trajectories = simulator.simulate_with_trajectory(x0, timesteps)
```
runs the actual simulation → (num_trajectories, num_timesteps, 1).

Everything else is presentation, in five groups:
- `if ax is None: ax = plt.gca()` — draw on the caller's axes if given, else
  grab the current one; makes the function composable into subplot grids.
- The `for trajectory_idx ...` loop slices one particle's path
  `trajectories[i, :, 0]`, moves it `.detach().cpu().numpy()` (matplotlib
  can't eat GPU tensors), and draws a thin translucent line — 500
  overlapping ghost-lines is what makes the *distribution* visible.
- Labels/ticks/grid lines: font sizes and a faint grid; cosmetics.
- `if show_hist:` block — takes the FINAL time-slice
  `trajectories[:, -1, 0]` and draws a sideways histogram on a small child
  axes to the right (`make_axes_locatable ... append_axes`), sharing the
  y-axis so trajectory endpoints and histogram bars line up. `binwidth` is
  chosen from the data range (~25 bins, floor 0.05) so the histogram isn't
  jagged for tight distributions. `decouple_hist_axis` exists for later
  plots where the endpoint spread differs wildly from the trajectory range.
- The final `fig.text(...)` block re-draws the title centered over BOTH the
  main axes and the histogram (a plain `ax.set_title` would center it over
  the trajectories only). Pure typography.

## Cell 7 — Driving Brownian motion

```python
sigma = 1.0
n_traj = 500
brownian_motion = BrownianMotion(sigma)
simulator = EulerMaruyamaSimulator(sde=brownian_motion)
```
Build equation, plug into simulator — the composition pattern in action.

```python
x0 = torch.zeros(n_traj,1).to(device)
```
All 500 particles start at 0 — so any spread you see was *created by noise*.

```python
ts = torch.linspace(0.0,5.0,500).to(device)
```
500 evenly spaced timestamps on [0, 5] ⇒ 499 steps of h≈0.01.

The rest sets up a figure and calls the Cell 6 utility with
`show_hist=True`. What you should see and why: a fan of paths opening like
√t, ending in a bell-shaped histogram of spread √5·σ ≈ 2.24 — Gaussian
noise added over time stays Gaussian (05 §A4), variance = σ²t.

---

## Cell 8 — Q2.2: Ornstein–Uhlenbeck — noise vs a restoring spring

```python
    def drift_coefficient(self, xt, t):
        return - self.theta * xt
```
The new ingredient vs Brownian motion: an arrow pointing BACK TOWARD ZERO,
stronger the further out you are (magnitude θ|x|). A spring. Diffusion is
constant σ as before. The whole process: dx = −θx dt + σ dW — tug-of-war
between a spring pulling in and noise kicking out.

## Cell 9 — OU side-by-side experiments

Three (θ, σ) pairs at fixed θ=0.25: σ=0 (pure spring: all trajectories decay
smoothly to 0 — an ODE in disguise), σ=0.5 (small jitter around the decay),
σ=2.0 (noise dominates; wide stationary band).

Line notes beyond Cell 7's pattern:
```python
x0 = torch.linspace(-10.0,10.0,n_traj).view(-1,1).to(device)
```
Start the particles SPREAD OUT (−10…10) rather than at a point — so you can
watch initial conditions being *forgotten*: whatever x0 was, trajectories
converge into the same statistical band. `view(-1,1)` reshapes the flat
(n,) linspace into the (n, 1) = (batch, dim) contract.

Top row plots 10 trajectories (individual behavior), bottom row 500 with
histograms (distributional behavior) — same system, two lenses; the
`decouple_hist_axis=True` on the bottom row lets the histogram zoom to the
narrow stationary band.

## Cell 10 — The punchline grid: what does OU converge TO?

```python
theta = sigma**2 / 2 / d
```
The key line. It *inverts* the stationary-variance formula: an OU process
settles into N(0, σ²/2θ) — a permanent tug-of-war standoff. Fixing
d := σ²/2θ and choosing θ from it builds a grid where **every cell in a row
has the same destination distribution** but different noise levels.

```python
time_scale = sigma**2
ts = torch.linspace(0.0, simulation_time / time_scale, 1000).to(device)
```
Bigger σ ⇒ faster convergence ⇒ shorten the simulated window by σ² so every
panel shows a comparable stage of its journey.

What the grid teaches (and why it's in a diffusion course): totally
different dynamics, same endpoint distribution — a first taste of "many
processes share marginals," the idea behind DDIM (notes 10 §1). And an OU
process run long enough is exactly DDPM's forward process: it forgets its
initial condition into a fixed Gaussian.

---

## Cell 11 — Density and Sampleable: the two faces of a distribution

```python
class Density(ABC):
    @abstractmethod
    def log_density(self, x): ...
```
Contract #1: "I can tell you how likely any point is" (log p(x), returned as
(batch, 1)). Log, not raw p: densities underflow float precision; sums of
logs are stable (same reason as 09 §A1).

```python
    def score(self, x):
        x = x.unsqueeze(1)
        score = vmap(jacrev(self.log_density))(x)
        return score.squeeze((1, 2, 3))
```
The score ∇log p — the "which way is uphill in probability" arrow (notes 10
§3) — obtained WITHOUT any hand math:
- `jacrev(self.log_density)` builds a function returning derivatives of
  log-density w.r.t. its input;
- `vmap(...)` applies it per sample without a loop;
- the `unsqueeze`/`squeeze` dance: `jacrev` computes the Jacobian of a
  (1, dim)→(1, 1) map, producing shape (batch, 1, 1, 1, dim)-ish; adding the
  dummy axis first and squeezing axes (1,2,3) after collapses it back to a
  clean (batch, dim). Bookkeeping, not math.
This is autodiff used as a *calculus engine*: define log p once, get scores
exactly, for any density.

```python
class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples): ...
```
Contract #2: "I can hand you points drawn from myself." Kept SEPARATE from
Density on purpose: real datasets are Sampleable but not Density (you can
draw photos; you cannot evaluate p(photo)) — the entire reason generative
modeling is hard, encoded as a class design.

## Cell 12 — 2D plotting helpers (four small functions)

`hist2d_sampleable` / `scatter_sampleable`: draw `num_samples` points and
show them as a 2D histogram / scatter. One subtlety: `.cpu()` on each
coordinate — matplotlib again.

`imshow_density` / `contour_density`: evaluate log-density on a grid —
`linspace` each axis, `meshgrid` to a lattice, `stack(...).reshape(-1, 2)`
to a (bins², 2) batch of query points, one `log_density` call, reshape back
to (bins, bins). The `.T` and `origin='lower'`: matplotlib's image
convention (row = y, origin top-left) vs math convention (x right, y up) —
transpose + origin flip aligns them. Same grid feeds `imshow` (heatmap) or
`contour` (level lines).

## Cell 13 — Concrete distributions: Gaussian and GaussianMixture

```python
class Gaussian(torch.nn.Module, Sampleable, Density):
```
Triple inheritance, each with a job: `nn.Module` gives `register_buffer` and
`.to(device)`; the other two are our contracts.

```python
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
```
Buffers = "constants that travel with the module" — same pattern (and same
reason) as our DDPM schedule buffers (02 §3.2): move with `.to(device)`,
saved in state_dict, no gradients.

```python
    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
```
Build the torch distribution *on demand* — a `@property` so it always uses
the buffers' current device. `validate_args=False` skips input checking for
speed. `sample`/`log_density` then delegate to it (`.view(-1, 1)` enforcing
the (batch, 1) contract).

`GaussianMixture` is the same wrapper around `D.MixtureSameFamily`:
- `D.Categorical(probs=weights)` picks WHICH bump each sample comes from;
- `D.MultivariateNormal(means, covs)` is the batch of bumps;
- torch composes them into one distribution with correct log_prob (a
  log-sum-exp over components) and sampling.

The two constructors are just scenery generators:
`random_2D` — nmodes bumps at uniform-random locations in a
`scale`-sized box (`(rand−0.5)*scale`), equal spherical covariances via
`diag_embed(ones)·std²`, fixed `seed` so everyone's figures match.
`symmetric_2D` — bumps equally spaced on a circle: angles from linspace
(dropping the duplicate 2π endpoint with `[:nmodes]`), means =
(cos, sin)·scale.

## Cell 14 — Visualizing the three targets

Builds one Gaussian and the two mixtures, then per density: heatmap +
faint contour lines overlaid. `vmin=-15` clips the colormap: log-densities
plunge to −∞ in empty regions; without a floor the interesting structure
compresses into one color band.

---

## Cell 15 — Q3.1: Langevin dynamics — the star of the lab

```python
class LangevinSDE(SDE):
    def __init__(self, sigma: float, density: Density):
```
An SDE built FROM a density — the object that turns "I know the target's
shape" into "I can generate from it."

```python
    def drift_coefficient(self, xt, t):
        return 0.5 * self.sigma ** 2 * self.density.score(xt)
```
Drift = climb the probability landscape (score = uphill arrow), with
strength σ²/2. The ½σ² is not style — it is the unique coefficient
for which the climb and the noise below *balance* so the stationary
distribution is exactly p (Fokker–Planck fact; the lab verifies it
empirically instead of proving it — our axiom-flavored move).

```python
    def diffusion_coefficient(self, xt, t):
        return self.sigma * torch.ones_like(xt)
```
Constant exploration noise. Intuition for the pair: pure climbing collapses
all points onto the peaks (mode-seeking); pure noise wanders anywhere;
climb + noise in the ½σ²-ratio makes points settle into p and *stay*
distributed as p. Connection: OU (Cell 8) is Langevin for a Gaussian target
— check: score of N(0, σ²/2θ) is −(2θ/σ²)x, times ½σ² gives −θx, the OU
drift. That is Q3.2's whole answer, and it's why diffusion models
(learned score + Langevin-style samplers, notes 10 §3) work at all.

## Cell 16 — Snapshot utilities

`every_nth_index(N, n)`: indices 0, n, 2n, … plus ALWAYS the last index
(`cat` with `[N−1]`) — endpoint matters most; the `n == 1` early return
avoids duplicating it.

`graph_dynamics(...)`: simulate once with trajectory, pick snapshot columns
(`xts[:, indices_to_plot]`), then per snapshot draw two rows: scatter of
particles over a faint target heatmap (are the points where the mass is?)
and a seaborn KDE contour of the SAME points (does their *density* match
the target's shape?). Scatter shows individuals, KDE shows the
distribution — the lab's recurring double-lens.

## Cell 17 — The experiment

Wide Gaussian start (cov 20·I — deliberately too spread out), 5-mode target,
σ=0.6, 1000 steps over t∈[0,5], plotted every 334 steps → 3 snapshots:
blob → blob draped over the modes → points sitting in the bumps with KDE
matching the target contours. **This picture is generative modeling**: a
known distribution transported onto a complicated one by simulating a
process — Labs 2–3 merely replace the exact score with a learned network.

## Cells 18–19 (optional) — Animation

`Camera(fig)` from `celluloid`: after drawing each frame, `camera.snap()`
stores it; `camera.animate()` compiles frames to an animation, saved as mp4
and returned as an inline HTML5 video. The plotting body is identical to
Cell 16's, minus subplots-per-time (one pair of axes reused per frame).
Nothing conceptually new — the same dynamics as a movie.

---

## What to carry forward (Lab 1 → everything else)

1. Equation objects (drift/diffusion) vs simulator objects (step/loop) —
   diffusion samplers are just `step` functions.
2. **√h on the noise** — variances add; std does not.
3. Sampleable vs Density — the split that defines the generative problem.
4. Score = ∇log p = uphill arrow; autodiff can compute it when the density
   is known; Lab 2 learns it when the density is a dataset.
5. Langevin: drift ½σ²·score + noise σ ⇒ stationary distribution p.
6. Different processes, same destination (OU grid) — foreshadows DDIM.
