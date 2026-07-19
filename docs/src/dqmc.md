# Determinant QMC (finite-temperature auxiliary-field) for the Hubbard model

`LatticeMC.DQMC` implements the Blankenbecler–Scalapino–Sugar (BSS)
determinant QMC algorithm for the repulsive Hubbard model,

$$H = -t\sum_{\langle i,j\rangle,\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow},$$

at the particle-hole-symmetric point (**half filling**, $\mu = U/2$). This is
the method that is genuinely **sign-problem-free** for the half-filled Hubbard
model, so it is *exact* -- no constrained-path/trial bias, only Trotter
($\Delta\tau$) and statistical error -- and reaches the ground state as the
temperature is lowered ($\beta$ large). It complements the projector
`AFQMC` (see below).

## Quick start

```julia
using LatticeMC

lat = LatticeMC.build_hubbard_square(6, 6, 1.0, 4.0; pbc=true)   # reuses the AFQMC lattice
r = LatticeMC.run_dqmc(lat; beta=8.0, num_thermal=200, num_sweeps=800)
r.energy_per_site     # ~ -0.86 J (sign-free; extrapolate dtau -> 0 to refine)
r.double_occupancy    # <n_up n_down>
r.neg_move_fraction   # 0.0 at half filling -> no sign problem
```

`run_dqmc(lat; beta, L, num_thermal, num_sweeps, nwrap, seed)` returns a named
tuple with `energy`/`energy_per_site` (mean + blocked error),
`double_occupancy`, `mean_sign`, and `neg_move_fraction`. `L` is the number of
imaginary-time slices (default gives $\Delta\tau \approx 0.1$); the energy has
an $O(\Delta\tau^2)$ Trotter bias, so extrapolate $\Delta\tau\to0$ for a
publication-quality number (`example/dqmc_hubbard_example.jl` does this).

## Why this, and how it compares

The half-filled 6×6 Hubbard model is the case where the projector methods
struggle in exactly the ways this one doesn't:

| method | 6×6 half-filled U/t=4 | issue |
|---|---|---|
| AFQMC constrained-path | E/site ≈ -0.833 | trial-dependent constrained-path bias |
| AFQMC free projection | E/site ≈ -0.69, mean_sign ≈ 0.3 | projector sign problem (not sign-free!) |
| **DQMC (this)** | **E/site ≈ -0.86, neg_moves = 0** | **exact, sign-free (only Trotter/stat error)** |

The key point (and a correction to a common confusion): "half-filled bipartite
Hubbard is sign-problem-free" is a statement about *this finite-temperature
determinant* algorithm, where the weight per configuration is
$\det(I+B^\uparrow_L\cdots B^\uparrow_1)\det(I+B^\downarrow_L\cdots B^\downarrow_1)$
and particle-hole symmetry makes that product non-negative. It is **not**
inherited by the zero-temperature projector AFQMC with a trial wavefunction
(whose sign genuinely decays). So for exact half-filled results, DQMC is the
right tool.

## What's under the hood

- $Z=\mathrm{Tr}\,e^{-\beta H}$ is Trotter-split into $L$ slices; the
  interaction is decoupled by the discrete Hirsch (spin-channel) HS transform,
  giving an auxiliary Ising field $s\in\{\pm1\}^{L\times N_s}$ with
  $B^\sigma_l = e^{-\Delta\tau K}\,\mathrm{diag}(e^{\sigma\lambda s_{i,l}})$,
  $\cosh\lambda = e^{\Delta\tau U/2}$.
- The field is sampled by Metropolis single-spin flips; each flip's
  determinant ratio and the equal-time Green's function update are $O(1)$/
  $O(N_s^2)$ via Sherman–Morrison (`update_greens!`).
- The equal-time Green's function $G=(I+B_L\cdots B_1)^{-1}$ is computed with a
  **UDT (stabilized-QR) factorization** (`stable_greens`) -- essential,
  because a naive $B$-product is catastrophically ill-conditioned at large
  $\beta$. The single most important implementation subtlety: re-stabilization
  mid-sweep must use the *cyclic* $B$-string starting at the current slice, not
  the fixed slice-1 order (getting this wrong makes the energy diverge with
  $\beta$ -- it did, until fixed).

Validated in `test/dqmc.jl`: `stable_greens` against a naive dense inverse
(to $10^{-10}$), the energy against a from-scratch finite-temperature grand-
canonical exact diagonalization (`test/dqmc_ed_reference.jl`), the Trotter
error vanishing as $\Delta\tau\to0$, and `neg_move_fraction == 0` (sign-free)
at half filling.

## Scope

Half filling only (the particle-hole-symmetric, sign-free point). Away from
half filling the Hubbard model has a genuine sign problem for this method too,
and `run_dqmc` reports `neg_move_fraction` / `mean_sign` so a departure from
the sign-free point is visible rather than silent. Repulsive `U >= 0` (the
spin-channel HS transform).
