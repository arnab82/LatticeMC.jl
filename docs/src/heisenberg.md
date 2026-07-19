# Heisenberg model: Stochastic Series Expansion (SSE) QMC

`LatticeMC.Heisenberg` computes the ground state of the S=1/2 antiferromagnetic
Heisenberg model,

$$H = J\sum_{\langle i,j\rangle} \mathbf{S}_i\cdot\mathbf{S}_j,\qquad J>0,$$

on a bipartite lattice, using **Stochastic Series Expansion (SSE)** QMC with
the operator-loop update (Sandvik). This is an algorithmically different Monte
Carlo from the rest of the package — no determinants, no auxiliary fields; it
samples terms of the power-series expansion of $e^{-\beta H}$ — and it is
*exact* (no sign problem) on a bipartite lattice, so an 8×8 = 64-site system
(a $2^{64}$ Hilbert space, hopeless for exact diagonalization) is a few-second
calculation.

## Quick start

```julia
using LatticeMC

lat = LatticeMC.build_heisenberg_square(8, 8; pbc=true)   # 64 sites, bipartite
result = LatticeMC.run_sse(lat; beta=32.0, num_thermal=5000, num_measure=50000)
result.energy_per_site      # -0.6735(2) J  (matches the known 8x8 value)
result.energy               # ~ -43.1 J for the whole lattice
```

Take `beta` (inverse temperature) a few times the linear size `L` so the
finite-temperature result has converged to the ground state; `run_sse`
increases the internal operator-string length automatically during
thermalization. Measured against the literature, `run_sse` reproduces the
square-lattice ground-state energy per site essentially exactly:

| lattice | SSE (this code) | reference |
|---|---|---|
| 4×4 | -0.7015(1) | -0.701780 |
| 6×6 | -0.6790(1) | -0.678872 |
| 8×8 | -0.6735(1) | -0.673487 |
| ∞ (TD limit) | — | -0.669437 |

(`example/heisenberg_sse_example.jl` runs this scaling and plots it.)

## The one hard requirement: bipartite lattices only

The AFM Heisenberg model is sign-problem-free (and this SSE is exact) **only on
a bipartite lattice** — one with no odd cycles, so the two sublattices can be
2-colored. A square $L_x\times L_y$ lattice with periodic boundaries is
bipartite iff both $L_x$ and $L_y$ are even; an odd periodic ring (e.g. a 2×3
cylinder) is frustrated and non-bipartite. On a non-bipartite lattice the model
has a genuine sign problem and this code — which tracks no signs — would return
*silently wrong* answers. `run_sse` therefore checks `is_bipartite(lat)` and
throws a clear error rather than mislead you. Open boundaries are always
bipartite for square/chain lattices.

## What's under the hood

- **`build_heisenberg_square(Lx, Ly; J, pbc)`** / **`build_heisenberg_chain(Ns;
  J, pbc)`** build the `HeisenbergLattice` (just a bond list, so a new geometry
  is a new bond builder). `is_bipartite(lat)` / `nbonds(lat)` are helpers.
- **`run_sse(lat; beta, num_thermal, num_measure, seed)`** runs the SSE:
  alternating a *diagonal update* (insert/remove diagonal bond operators along
  the operator string) and an *operator-loop update* (build closed loops
  through the vertex-leg linked list, flip each with probability 1/2, toggling
  diagonal ↔ off-diagonal operators — the loops are deterministic and always
  flippable at the isotropic Heisenberg point). Returns `energy`,
  `energy_per_site` (each with a blocked statistical error), and the mean
  operator count. Energy estimator: $E = J\,N_{\rm bonds}/4 - \langle n\rangle/\beta$.

Validation lives in `test/heisenberg.jl`: SSE vs a from-scratch $S_z=0$-sector
exact diagonalization (`test/heisenberg_ed_reference.jl`) on small bipartite
lattices, the exact −2 J energy of the 4-site ring, the bipartite guard, and
the 4×4 literature value.
