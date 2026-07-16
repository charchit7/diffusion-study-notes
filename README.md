# Diffusion & Flow Matching — From-Zero Study Notes

A self-contained curriculum for learning diffusion models and flow matching
from first principles, written while solving the CMU 10-799 homeworks, four
KAIST CS492 assignments, and the MIT 6.S184 IAP labs end to end.

**Read online:** open `index.html` (or enable GitHub Pages on this repo and
browse it as a website).

## Contents
- **Theory (7 parts, read in order):** mental model → DDPM math → ELBO →
  flow matching → DDIM & DPM-Solver → classifier-free guidance → the
  rebuild-from-scratch checklist with no-training self-tests.
  Every derivation is complete: no calculus assumed until it is built, no
  "it can be shown", three explicitly-labeled axioms total.
- **MIT lab companions:** line-by-line explanations of every code cell in
  the three `iap-diffusion-labs` solution notebooks (what each line does,
  why it's there, what breaks without it) — including honest notes on the
  solution code's own quirks.
- `md/` — markdown sources of everything.

## How to study with this
Theory part N before the code that implements it; then try to rewrite each
lab cell from memory after reading its annotation. The checklist (theory
part 7) gives level-by-level self-tests for rebuilding DDPM → DDIM → FM →
Reflow → DPM-Solver from a blank file.
