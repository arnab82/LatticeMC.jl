# LatticeMC.jl

==============================

[![CI](https://github.com/arnab82/LatticeMC.jl/actions/workflows/blank.yml/badge.svg)](https://github.com/arnab82/LatticeMC.jl/actions/workflows/blank.yml)
[![Docs](https://github.com/arnab82/LatticeMC.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/arnab82/LatticeMC.jl/actions/workflows/documentation.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://arnab82.github.io/LatticeMC.jl/)


## Overview

A Julia package for Monte Carlo simulation in statistical mechanics and
condensed matter/quantum chemistry, organized as several submodules:

- **`LatticeMC.Ising`** -- classical Metropolis Monte Carlo for the 2D Ising
  model: spin configurations, magnetization, energy, heat capacity, and
  related thermodynamic properties.
- **`LatticeMC.Heisenberg`** -- Stochastic Series Expansion (SSE) QMC for the
  S=1/2 antiferromagnetic Heisenberg model on bipartite lattices. Sign-
  problem-free and exact (no determinants/auxiliary fields -- it samples the
  power series of `exp(-beta H)`), so an 8x8 = 64-site ground state (a 2^64
  Hilbert space, hopeless for exact diagonalization) is a few-second run.
  Reproduces the known square-lattice energies (4x4 -0.7015, 6x6 -0.6790,
  8x8 -0.6735 J/site) essentially exactly.
- **`LatticeMC.AFQMC`** -- phaseless (constrained-path) Auxiliary-Field
  Quantum Monte Carlo for the fermionic Hubbard model on 1D chains and 2D
  square lattices: imaginary-time projection of a population of
  Slater-determinant walkers, importance-sampled from a free-fermion trial
  wavefunction via a discrete Hubbard-Stratonovich transform of the on-site
  interaction, with population control and a constrained-path approximation
  to control the fermion sign problem.
- **`LatticeMC.AFQMC.AbInitio`** -- the same AFQMC family for general
  (molecular) Hamiltonians: complex walkers, a Cholesky-decomposed
  two-electron tensor, a continuous Hubbard-Stratonovich transform, a
  Hartree-Fock trial, and a self-contained STO-3G integral engine for
  hydrogen chains (no external quantum chemistry dependency).
- **`LatticeMC.DQMC`** -- determinant (BSS finite-temperature auxiliary-field)
  QMC for the half-filled Hubbard model, sign-problem-free at the particle-
  hole-symmetric point. This is the *exact* method for half filling (only
  Trotter + statistical error): on 6x6 U/t=4 it gives E/site ~ -0.86 with
  zero negative-weight moves, where the projector AFQMC is either constrained-
  path-biased (-0.833) or hits a sign problem (free projection, mean_sign~0.3).

All submodules re-export their public API at the top level, so
`LatticeMC.IsingModel(...)`, `LatticeMC.run_sse(...)`,
`LatticeMC.run_afqmc(...)`, and `LatticeMC.run_afqmc_ab_initio(...)` all work
directly.

**New here?** Start with the [documentation site](https://arnab82.github.io/LatticeMC.jl/)
or [`docs/src/tutorial.md`](docs/src/tutorial.md) -- a hands-on walkthrough --
rather than this README.

## Features

### Ising (classical MC)

- Monte Carlo simulation of a 2D Ising model.
- Visualization of the spin configurations.
- Calculation of thermodynamic properties such as magnetization, energy, and heat capacity.

### AFQMC (quantum MC)

- Hubbard model on 1D chains and 2D square lattices (open or periodic boundaries).
- Free-fermion (tight-binding) trial wavefunction, used for importance sampling and the constrained-path gate.
- Discrete Hirsch Hubbard-Stratonovich transform of the on-site `U` term (repulsive Hubbard, `U >= 0`).
- Population control (comb/systematic resampling) and periodic QR reorthonormalization for numerical stability.
- Mixed-estimator ground-state energy with a blocked statistical error estimate.
- Rank-1 (Sherman-Morrison) Green's-function updates for the auxiliary-field
  sweep, the same fast-update trick used in production DQMC/CPMC codes:
  O(1) per candidate field and O(N^2) per accepted site update, instead of an
  O(N^3) determinant recompute -- see `docs/src/afqmc_theory.md` and
  `docs/src/afqmc_implementation.md` for the full derivation and where it lives
  in the code.
- A second trial wavefunction, `build_uhf_trial`, self-consistent unrestricted
  Hartree-Fock with an antiferromagnetic (Neel) seed -- a drop-in alternative
  to `build_trial_wavefunction`'s paramagnetic free-fermion trial, useful at
  large `U` where AFM correlations dominate and the paramagnetic trial is a
  poor reference.

Known, standard AFQMC approximations: constrained-path bias, population-control
bias, and finite-`dtau` Trotter error -- not implementation bugs, see
`docs/src/afqmc_theory.md`. Overall cost is O(Ns^3) per full propagation step
(dominated by the two dense K/2 half-steps and the one Green's-function
recompute per step), which is practical up to roughly Ns ~ 30-50 sites for a
full production run (hundreds of walkers, thousands of steps) in well under a
few minutes; beyond that, runtime grows quickly and would need further
optimization (delayed updates, sparser K, GPU) not implemented here.

### Ab initio AFQMC (quantum MC, molecular)

- General one-/two-electron integrals: plain Julia arrays
  (`MolecularIntegrals`) or `read_fcidump` for the standard text format.
- A self-contained STO-3G integral engine for hydrogen chains
  (`build_h_chain_sto3g`) -- analytical Gaussian integrals + Loewdin
  orthogonalization, no PySCF/Python dependency, works in CI as-is.
- Pivoted Cholesky decomposition of the two-electron tensor
  (`cholesky_decompose_eri`) and the associated one-body exchange
  correction, feeding a continuous (Gaussian-field) Hubbard-Stratonovich
  transform -- one matrix exponential per step, not one per Cholesky vector.
- Restricted (closed-shell) Hartree-Fock trial (`build_rhf_trial`), and an
  unrestricted (symmetry-broken) alternative (`build_uhf_trial_ab_initio`)
  for geometries where RHF stops being qualitatively adequate (the standard
  bond-dissociation instability) -- collapses back to RHF where that
  instability hasn't kicked in yet, so it's always safe to use.
- **Multi-determinant / CASCI trials** (`build_casci_trial`,
  `MultiDetTrial`, `multidet_from_ci`) -- the one knob that systematically
  closes the phaseless-approximation accuracy gap: adding determinants drives
  the result monotonically to FCI (on equilibrium H4: top-1 ~8%, top-5 ~73%,
  top-10 ~92%, full expansion exact to ~1e-15). A full-FCI trial makes AFQMC
  exact, the strongest end-to-end validation of the machinery. Built in the
  RHF MO basis (required for a well-behaved truncation).
- Complex walkers and the general complex phaseless gate (`w *= |I|*max(0,
  cos(arg I))`), of which the Hubbard side's real sign-flip gate is the
  real/discrete special case.
- Optional force-bias (mean-field-shifted) sampling
  (`run_afqmc_ab_initio(...; force_bias=true)`) for a modest (~7% measured)
  variance reduction. Off by default -- it's a genuine but small effect, and
  it does *not* fix the equilibrium-H4 accuracy gap below (that was the
  natural hypothesis; measured and ruled out, see the theory doc §7.1 -- the
  multi-determinant trial above is what fixes it).

**Read before trusting a result**: correct, but accuracy is strongly
trial-quality-dependent, more so than you'd guess from the code. Measured on
H4: near equilibrium, ~11% of the correlation energy is recovered -- this
is the phaseless approximation's own asymptotic bias for this trial (not a
sampling-variance issue: doesn't respond to more walkers, more steps,
smaller `dtau`, force bias, *or* the UHF trial, which correctly collapses
back to RHF there since RHF is already the locally optimal single
determinant). Stretched, the plain RHF trial actually gets *worse than
mean-field* (RHF's own dissociation-catastrophe instability poisoning the
trial) while the UHF trial recovers ~75%, a real fix for that different
failure mode. H2, which stays single-reference throughout, recovers ~93%
with just the RHF trial. See
[`docs/src/afqmc_ab_initio_theory.md`](docs/src/afqmc_ab_initio_theory.md) sections
6-7 for the full account (including the cross-validation that ruled out a
bug) before drawing conclusions on a new system.

## Documentation

Rendered and searchable at **<https://arnab82.github.io/LatticeMC.jl/>**
(built from `docs/src/` by Documenter.jl on every push to `master`). The
source pages:

- [`docs/src/afqmc_theory.md`](docs/src/afqmc_theory.md) -- step-by-step derivation of
  the AFQMC algorithm implemented here: the Hubbard Hamiltonian, symmetric
  Trotter splitting, the discrete Hirsch Hubbard-Stratonovich transform,
  importance sampling from a trial wavefunction, the constrained-path
  approximation, population control, the mixed-estimator energy, and the
  rank-1 Green's-function fast-update formulas.
- [`docs/src/afqmc_implementation.md`](docs/src/afqmc_implementation.md) -- maps each
  piece of that theory to the actual code (file by file, function by
  function), for when you're reading or modifying `src/afqmc/`.
- [`docs/src/afqmc_algorithm.md`](docs/src/afqmc_algorithm.md) -- formal, paper-style
  pseudocode for the outer population loop, the per-walker Trotter step
  (including the rank-1-update inner loop), and population control, each
  cross-referenced to the theory section and the implementing function.
- [`docs/src/afqmc_ab_initio_theory.md`](docs/src/afqmc_ab_initio_theory.md) --
  addendum for the ab initio case: the general Hamiltonian, orbital
  orthonormality, the Cholesky decomposition and its one-body exchange
  correction, the continuous Hubbard-Stratonovich transform, the complex
  phaseless gate, and the measured trial-quality dependence (section 6).
- [`docs/src/afqmc_implementation.md`](docs/src/afqmc_implementation.md) -- maps each
  piece of both theory docs to the actual code (file by file, function by
  function), for when you're reading or modifying `src/afqmc/`.
- [`docs/src/tutorial.md`](docs/src/tutorial.md) -- hands-on getting-started
  walkthrough of all three methods; start here if you're new.

## Examples

- `example/example.jl` -- classical Ising Monte Carlo (original example).
- `example/heisenberg_sse_example.jl` -- SSE ground-state energy of the 2D
  Heisenberg model for LxL lattices up to 8x8, with finite-size scaling
  toward the thermodynamic limit.
- `example/dqmc_hubbard_example.jl` -- sign-problem-free determinant QMC for
  the half-filled 6x6 Hubbard model, with a dtau -> 0 Trotter extrapolation.
- `example/afqmc_example.jl` -- AFQMC energy trace on a Hubbard chain.
- `example/afqmc_phase_diagram.jl` -- ground-state energy vs `U/t` for a
  small chain, AFQMC overlaid on an exact-diagonalization reference.
- `example/afqmc_convergence_study.jl` -- Trotter error (energy vs `dtau`)
  and statistical convergence (`energy_err` vs `num_steps`).
- `example/afqmc_square_lattice.jl` -- a 2D square-lattice (not just chain)
  AFQMC run.
- `example/afqmc_h4_example.jl` -- ab initio AFQMC on an H4 chain (STO-3G),
  reporting AFQMC/RHF/FCI together and the recovered-correlation percentage
  directly (see the ab initio theory doc before reading too much into it).

## Testing

`test/runtests.jl` runs, beyond the original Ising regression test:
`test/ising.jl` (closed-form and statistical checks for the classical
model), `test/afqmc.jl` (Hubbard AFQMC vs. a from-scratch exact-
diagonalization reference across multiple geometries/fillings/`U` values),
`test/afqmc_units.jl` (component-level correctness -- notably that the
rank-1 fast update exactly matches a brute-force recompute),
`test/afqmc_convergence.jl` (statistical-error and population-control-bias
behavior), and `test/ab_initio.jl` (STO-3G integral sanity checks against a
known textbook value, Cholesky reconstruction, the RHF variational bound
and dissociation-catastrophe trend, and AFQMC vs. FCI on H2/H4 with
tolerances calibrated to each system's measured, honest behavior --  not
uniformly tight). `test/ed_reference.jl` (Hubbard-specialized) and
`test/ab_initio_fci_reference.jl` (general Slater-Condon, cross-validated
against the former as a special case) hold the shared exact-diagonalization
code reused across the test suite and the phase-diagram/H4 examples.

## Installation
```bash
git clone https://github.com/arnab82/LatticeMC.jl.git
cd LatticeMC.jl
julia --project=./ -tauto
using Pkg
Pkg.instantiate()
Pkg.precompile()
using LatticeMC
Pkg.test()
```

## Quick start

```julia
using LatticeMC

# Ising
lattice = LatticeMC.IsingModel(50, 1.0, true)
e = LatticeMC.energy_manual(lattice)

# Heisenberg SSE (8x8 antiferromagnet ground state, ~seconds)
lat = LatticeMC.build_heisenberg_square(8, 8; pbc=true)
r = LatticeMC.run_sse(lat; beta=32.0)
println(r.energy_per_site)   # ~ -0.6735 J

# AFQMC (4-site Hubbard chain, open boundaries, U/t = 4)
h = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
trial = LatticeMC.build_trial_wavefunction(h, 2, 2)
result = LatticeMC.run_afqmc(h, trial, 2, 2)
println(result.energy_mean, " +/- ", result.energy_err)

# AFQMC with the self-consistent AFM/UHF trial instead, e.g. at larger U
h2 = LatticeMC.build_hubbard_chain(6, 1.0, 8.0; pbc=false)
afm_trial = LatticeMC.build_uhf_trial(h2, 3, 3)
result2 = LatticeMC.run_afqmc(h2, afm_trial, 3, 3)
println(result2.energy_mean, " +/- ", result2.energy_err)

# Ab initio AFQMC (H2, STO-3G, near equilibrium -- no external QC package needed)
mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)
rhf_trial = LatticeMC.build_rhf_trial(mi, 1, 1)
result3 = LatticeMC.run_afqmc_ab_initio(mi, rhf_trial, 1, 1)
println(result3.energy_mean, " +/- ", result3.energy_err)

# ...at a stretched geometry, use the UHF trial instead (RHF's own
# dissociation-catastrophe instability otherwise poisons the AFQMC result)
mi_stretched = LatticeMC.build_h_chain_sto3g(3.0; n_atoms=4)
uhf_trial = LatticeMC.build_uhf_trial_ab_initio(mi_stretched, 2, 2)
result4 = LatticeMC.run_afqmc_ab_initio(mi_stretched, uhf_trial, 2, 2)
println(result4.energy_mean, " +/- ", result4.energy_err)
```

## Copyright

Copyright (c) 2023, Arnab Bachhar
