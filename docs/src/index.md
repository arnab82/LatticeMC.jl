# LatticeMC.jl

A Julia package for Monte Carlo simulation in statistical mechanics and
condensed matter / quantum chemistry, organized as three submodules:

- **`LatticeMC.Ising`** — classical Metropolis Monte Carlo for the 2D Ising
  model.
- **`LatticeMC.AFQMC`** — phaseless (constrained-path) Auxiliary-Field
  Quantum Monte Carlo for the fermionic Hubbard model on 1D chains and 2D
  square lattices, with rank-1 Green's-function fast updates, population
  control, a free-fermion trial, and a self-consistent antiferromagnetic
  (UHF) trial.
- **`LatticeMC.AFQMC.AbInitio`** — the same AFQMC family for general
  (molecular) Hamiltonians: complex walkers, a Cholesky-decomposed
  two-electron tensor, a continuous Hubbard-Stratonovich transform, RHF and
  UHF trials, optional force-bias sampling, an energy shift for numerical
  stability, and a self-contained STO-3G integral engine for hydrogen chains
  (plus a FCIDUMP reader for integrals from anywhere else).

All three re-export their public API at the top level, so
`LatticeMC.IsingModel(...)`, `LatticeMC.run_afqmc(...)`, and
`LatticeMC.run_afqmc_ab_initio(...)` all work directly.

## Where to start

- **New here?** [Tutorial](tutorial.md) — a hands-on walkthrough of all three
  methods, from an Ising run to ab initio AFQMC on H2/H4.
- **Understanding the method:** [AFQMC theory](afqmc_theory.md) (Hubbard) and
  [ab initio AFQMC theory](afqmc_ab_initio_theory.md) (molecular) derive
  everything step by step; [algorithm](afqmc_algorithm.md) is the pseudocode.
- **Reading or modifying the code:**
  [implementation notes](afqmc_implementation.md) map every function to the
  theory section that derives it.

## Quick start

```julia
using LatticeMC

# Classical: 2D Ising
lattice = LatticeMC.IsingModel(50, 1.0, true)
e = LatticeMC.energy_manual(lattice)

# Quantum: Hubbard AFQMC (4-site chain, open boundaries, U/t = 4)
h = LatticeMC.build_hubbard_chain(4, 1.0, 4.0; pbc=false)
trial = LatticeMC.build_trial_wavefunction(h, 2, 2)
result = LatticeMC.run_afqmc(h, trial, 2, 2)

# Ab initio AFQMC (H2, STO-3G, near equilibrium)
mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)
rhf_trial = LatticeMC.build_rhf_trial(mi, 1, 1)
result = LatticeMC.run_afqmc_ab_initio(mi, rhf_trial, 1, 1)
```

## A note on accuracy

The ab initio AFQMC's accuracy is strongly trial-quality-dependent — more so
than the code alone suggests. See
[ab initio theory §6–7](afqmc_ab_initio_theory.md) for the measured,
honestly-reported behavior (H2 recovers ~93% of the correlation energy with a
plain RHF trial; H4 near equilibrium only ~10%, a phaseless-approximation
limit that neither symmetry-breaking nor force bias fixes) before drawing
conclusions on a new system.
