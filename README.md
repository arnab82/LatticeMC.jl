# LatticeMC.jl

==============================

[![CI](https://github.com/arnab82/LatticeMC.jl/actions/workflows/blank.yml/badge.svg)](https://github.com/arnab82/LatticeMC.jl/actions/workflows/blank.yml)


## Overview

A Julia package for Monte Carlo simulation of lattice models in statistical
mechanics and condensed matter physics, organized as two submodules:

- **`LatticeMC.Ising`** -- classical Metropolis Monte Carlo for the 2D Ising
  model: spin configurations, magnetization, energy, heat capacity, and
  related thermodynamic properties.
- **`LatticeMC.AFQMC`** -- phaseless (constrained-path) Auxiliary-Field
  Quantum Monte Carlo for the fermionic Hubbard model on 1D chains and 2D
  square lattices: imaginary-time projection of a population of
  Slater-determinant walkers, importance-sampled from a free-fermion trial
  wavefunction via a discrete Hubbard-Stratonovich transform of the on-site
  interaction, with population control and a constrained-path approximation
  to control the fermion sign problem.

Both submodules re-export their public API at the top level, so
`LatticeMC.IsingModel(...)` and `LatticeMC.run_afqmc(...)` both work directly.

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
  O(N^3) determinant recompute -- see `docs/afqmc_theory.md` and
  `docs/afqmc_implementation.md` for the full derivation and where it lives
  in the code.
- A second trial wavefunction, `build_uhf_trial`, self-consistent unrestricted
  Hartree-Fock with an antiferromagnetic (Neel) seed -- a drop-in alternative
  to `build_trial_wavefunction`'s paramagnetic free-fermion trial, useful at
  large `U` where AFM correlations dominate and the paramagnetic trial is a
  poor reference.

Known, standard AFQMC approximations: constrained-path bias, population-control
bias, and finite-`dtau` Trotter error -- not implementation bugs, see
`docs/afqmc_theory.md`. Overall cost is O(Ns^3) per full propagation step
(dominated by the two dense K/2 half-steps and the one Green's-function
recompute per step), which is practical up to roughly Ns ~ 30-50 sites for a
full production run (hundreds of walkers, thousands of steps) in well under a
few minutes; beyond that, runtime grows quickly and would need further
optimization (delayed updates, sparser K, GPU) not implemented here.

## Documentation

- [`docs/afqmc_theory.md`](docs/afqmc_theory.md) -- step-by-step derivation of
  the AFQMC algorithm implemented here: the Hubbard Hamiltonian, symmetric
  Trotter splitting, the discrete Hirsch Hubbard-Stratonovich transform,
  importance sampling from a trial wavefunction, the constrained-path
  approximation, population control, the mixed-estimator energy, and the
  rank-1 Green's-function fast-update formulas.
- [`docs/afqmc_implementation.md`](docs/afqmc_implementation.md) -- maps each
  piece of that theory to the actual code (file by file, function by
  function), for when you're reading or modifying `src/afqmc/`.
- [`docs/afqmc_algorithm.md`](docs/afqmc_algorithm.md) -- formal, paper-style
  pseudocode for the outer population loop, the per-walker Trotter step
  (including the rank-1-update inner loop), and population control, each
  cross-referenced to the theory section and the implementing function.

## Examples

- `example/example.jl` -- classical Ising Monte Carlo (original example).
- `example/afqmc_example.jl` -- AFQMC energy trace on a Hubbard chain.
- `example/afqmc_phase_diagram.jl` -- ground-state energy vs `U/t` for a
  small chain, AFQMC overlaid on an exact-diagonalization reference.
- `example/afqmc_convergence_study.jl` -- Trotter error (energy vs `dtau`)
  and statistical convergence (`energy_err` vs `num_steps`).
- `example/afqmc_square_lattice.jl` -- a 2D square-lattice (not just chain)
  AFQMC run.

## Testing

`test/runtests.jl` runs, beyond the original Ising regression test:
`test/ising.jl` (closed-form and statistical checks for the classical
model), `test/afqmc.jl` (AFQMC vs. a from-scratch exact-diagonalization
reference across multiple geometries/fillings/`U` values),
`test/afqmc_units.jl` (component-level correctness -- notably that the
rank-1 fast update exactly matches a brute-force recompute), and
`test/afqmc_convergence.jl` (statistical-error and population-control-bias
behavior). `test/ed_reference.jl` holds the shared bitmask exact-
diagonalization code used by several of these and by
`example/afqmc_phase_diagram.jl`.

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
```

## Copyright

Copyright (c) 2023, Arnab Bachhar
