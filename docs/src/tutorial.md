# Getting started with LatticeMC.jl

A hands-on walkthrough, from installing the package to running each of its
three methods. For the derivations and algorithmic detail behind what's
happening here, see [`afqmc_theory.md`](afqmc_theory.md),
[`afqmc_algorithm.md`](afqmc_algorithm.md),
[`afqmc_ab_initio_theory.md`](afqmc_ab_initio_theory.md), and
[`afqmc_implementation.md`](afqmc_implementation.md).

## Install

```bash
git clone https://github.com/arnab82/LatticeMC.jl.git
cd LatticeMC.jl
julia --project=./ -e 'using Pkg; Pkg.instantiate()'
```

```julia
julia> using LatticeMC
```

## 1. Classical Monte Carlo: the 2D Ising model

```julia
lattice = LatticeMC.IsingModel(20, 1.0, true)   # 20x20, J=1.0, mostly-positive initial spins
e0 = LatticeMC.energy_manual(lattice)

Bj = 1.0 / 1.5   # J / (k*T), T=1.5 (below Tc=2.269 -> ordered phase)
spin_per_site, net_energy, net_spin = LatticeMC.metropolis(lattice, 50_000, Bj, e0)

using Statistics
mean(spin_per_site[10_000:end])   # should be close to +-1 -- ordered
```

Try `T` above `2.269` (e.g. `T=3.0`) and the same average should collapse
toward 0 -- the paramagnetic phase. This is the whole classical side of the
package; see `example/example.jl` for the original demo and
`test/ising.jl` for closed-form sanity checks (fully-aligned and checkerboard
lattices have exact, hand-computable energies).

## 2. Quantum Monte Carlo: AFQMC on the Hubbard model

The Hubbard model is a *lattice* model like Ising, but fermionic and quantum
-- AFQMC finds its ground-state energy via imaginary-time projection instead
of finite-temperature sampling.

```julia
# 6-site chain, open boundaries, half filling, moderate repulsion
lattice = LatticeMC.build_hubbard_chain(6, 1.0, 4.0; pbc=false)   # Ns=6, t=1.0, U=4.0
trial = LatticeMC.build_trial_wavefunction(lattice, 3, 3)          # Nup=Ndown=3

result = LatticeMC.run_afqmc(lattice, trial, 3, 3;
                              dtau=0.01, num_walkers=200, num_steps=3000,
                              equilibration_steps=500)
println(result.energy_mean, " +/- ", result.energy_err)
```

At large `U`, the paramagnetic trial above is a poor reference -- swap in
the self-consistent antiferromagnetic trial instead, same call signature:

```julia
afm_trial = LatticeMC.build_uhf_trial(lattice, 3, 3)
result2 = LatticeMC.run_afqmc(lattice, afm_trial, 3, 3; dtau=0.01, num_walkers=200, num_steps=3000)
```

For small lattices you can check the answer against exact diagonalization
directly -- `test/ed_reference.jl`'s `ed_ground_state_energy` is exactly
this, reused across the test suite and `example/afqmc_phase_diagram.jl`.

Worth running once to get a feel for the method: `example/afqmc_example.jl`
(a single run's energy trace), `example/afqmc_phase_diagram.jl` (energy vs.
`U/t`, AFQMC overlaid on exact diagonalization), and
`example/afqmc_convergence_study.jl` (how the answer depends on `dtau` and
sample count).

## 3. Quantum Monte Carlo: ab initio AFQMC on a real molecule

Same algorithmic family, general (molecular) Hamiltonian instead of Hubbard
-- complex arithmetic, a Cholesky-decomposed interaction, and a Hartree-Fock
trial instead of a lattice trial.

```julia
# H2, STO-3G, near equilibrium bond length (self-contained integral engine,
# no external quantum chemistry package needed)
mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=2)   # bohr
trial = LatticeMC.build_rhf_trial(mi, 1, 1)           # 1 electron per spin (H2, closed shell)

result = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1;
                                        dtau=0.01, num_walkers=300, num_steps=2000,
                                        equilibration_steps=500)
println(result.energy_mean, " +/- ", result.energy_err)
```

Or hand it integrals from elsewhere via FCIDUMP:

```julia
mi = LatticeMC.read_fcidump("path/to/FCIDUMP")
```

At a stretched/dissociating geometry, RHF itself becomes qualitatively
unreliable (the standard bond-dissociation instability) and poisons the
AFQMC trial -- swap in the unrestricted (symmetry-broken) trial, same call
signature, and it's always safe to use (it collapses back to RHF wherever
that instability hasn't kicked in, e.g. near equilibrium):

```julia
uhf_trial = LatticeMC.build_uhf_trial_ab_initio(mi, 1, 1)
```

There's also an optional force-bias (variance-reduction) mode:

```julia
result = LatticeMC.run_afqmc_ab_initio(mi, trial, 1, 1; force_bias=true)
```

**Read this before trusting a result on a new molecule**: ab initio AFQMC's
accuracy here depends heavily on how well the trial describes your system,
in two somewhat different ways. Measured on H4: near equilibrium, ~11% of
the correlation energy is recovered no matter what you do here (UHF
collapses back to RHF there, and force bias doesn't move it either -- this
is the phaseless approximation's own asymptotic bias for this trial, not a
sampling-variance problem); stretched, the plain RHF trial actually does
*worse than mean-field* while the UHF trial recovers ~75%, a real fix for
that different failure mode. H2, which stays single-reference throughout,
recovers ~93% with just RHF. None of this is a bug -- it's measured and
explained in [`afqmc_ab_initio_theory.md`](afqmc_ab_initio_theory.md)
sections 6-7. Read it before drawing conclusions from a result on a system
you suspect has real multi-reference character. `example/afqmc_h4_example.jl`
shows the equilibrium case directly (reports the recovered-correlation
percentage, doesn't hide it).

For small molecules, `test/ab_initio_fci_reference.jl`'s
`ab_initio_fci_ground_state_energy` gives you an exact comparison point the
same way `ed_reference.jl` does for Hubbard.

### Closing the gap: a multi-determinant trial

When a single determinant isn't enough (the equilibrium-H4 case above), the
fix is a **multi-determinant trial** — a short CI expansion instead of one
Slater determinant. `build_casci_trial` does RHF, transforms to the MO basis,
solves a small CI, and keeps the top-`max_dets` determinants. It returns the
MO-basis integrals you must run with (trial and integrals share a basis):

```julia
mi = LatticeMC.build_h_chain_sto3g(1.4; n_atoms=4)
trial, mi_mo, e_ci = LatticeMC.build_casci_trial(mi, 2, 2; max_dets=10)
result = LatticeMC.run_afqmc_ab_initio(mi_mo, trial, 2, 2)
# more determinants -> systematically closer to FCI: on H4, top-1 ~8%,
# top-5 ~73%, top-10 ~92%, full expansion exact.
```

`build_casci_trial`'s internal CI has the same size ceiling as exact
diagonalization; for larger systems, build the trial from an external CASSCF
expansion (e.g. a PySCF CI vector) via `multidet_from_ci`. This is the one
knob that fixes the phaseless-approximation accuracy limit (see
[`afqmc_ab_initio_theory.md`](afqmc_ab_initio_theory.md) section 8) — unlike
symmetry-breaking or force bias, which address different, narrower problems.

## Where to go next

- Modifying or reading `src/afqmc/`: start at
  [`afqmc_implementation.md`](afqmc_implementation.md), which maps every
  function to the theory section that derives it.
- Understanding *why* the algorithm is built the way it is:
  [`afqmc_theory.md`](afqmc_theory.md) (Hubbard) and
  [`afqmc_ab_initio_theory.md`](afqmc_ab_initio_theory.md) (molecular).
- A pseudocode-level view before diving into either:
  [`afqmc_algorithm.md`](afqmc_algorithm.md).
