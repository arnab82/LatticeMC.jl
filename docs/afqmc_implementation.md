# AFQMC implementation notes

This is a guide to `src/afqmc/`, cross-referenced to the derivations in
[`afqmc_theory.md`](afqmc_theory.md) (linked as `theory §N` below). Read the
theory doc first if the formulas here look unmotivated.

Load order (`src/LatticeMC.jl`, inside `module AFQMC`):
`lattice.jl` -> `trial_wavefunction.jl` -> `uhf_trial.jl` -> `walker.jl` ->
`estimators.jl` -> `propagator.jl` -> `population_control.jl` ->
`driver.jl`. Each file's public names are re-exported from `LatticeMC`
directly (e.g. `LatticeMC.run_afqmc` works without `LatticeMC.AFQMC.run_afqmc`).

See also [`afqmc_algorithm.md`](afqmc_algorithm.md) for a pseudocode-level
view of `run_afqmc` / `propagate_step!` / `population_control!` before
diving into the prose below.

## `lattice.jl` -- `HubbardLattice`, `build_hubbard_chain`, `build_hubbard_square`

Builds the hopping matrix $K$ (theory §1) for a 1D chain or 2D square
lattice, open or periodic. `HubbardLattice` bundles `Ns`, `K`, `U`, `t`, and
a `geometry` tag.

**Edge case to know about:** a dimension of size 2 with `pbc=true` gives that
bond a *doubled* hopping element ($-2t$ instead of $-t$), because the "left"
and "right" neighbor coincide. This isn't a bug -- it's what you get from
correctly applying periodic boundary conditions and matches the exact $N=2$
ring dispersion $\varepsilon(k)=-2t\cos k$ at $k=0,\pi$ giving eigenvalues
$\mp2t$. It's just easy to mistake for one, which is why `test/afqmc.jl` uses
open boundaries for its 2x2 validation case instead.

## `trial_wavefunction.jl` -- `TrialWavefunction`, `build_trial_wavefunction`

Diagonalizes $K$ and fills the lowest `Nup`/`Ndown` orbitals -- the
free-fermion trial wavefunction $\psi_T$ used throughout (theory §5, §7,
§8). Takes a `bias` keyword (default `1e-4`): a tiny generic diagonal ramp
added to $K$ *only* for this diagonalization, to deterministically break
exact degeneracies at the Fermi level (e.g. the open 2x2 plaquette at half
filling has a doubly-degenerate HOMO/LUMO). Without it, `eigen` returns an
arbitrary orthonormal basis of the degenerate subspace, which is still a
valid $U=0$ ground state (same energy) but a poor, arbitrarily-oriented
reference for importance sampling and the constrained-path gate once
$U\neq0$ -- this was an actual bug caught by `test/afqmc.jl`'s 2x2-plaquette
case (AFQMC energy off by $\sim1$ in units of $t$ at $U/t=4$ before the fix).
`lattice.K` itself is untouched; only the trial construction sees the bias.

The bias does perturb the trial slightly even in non-degenerate cases, to
second order in energy (see the `atol=1e-6`, not `1e-8`, in
`test/afqmc.jl`'s deterministic $U=0$ check) -- harmless at the default
magnitude, but worth knowing about if you tighten tolerances elsewhere.

## `uhf_trial.jl` -- `bipartite_coloring`, `build_uhf_trial`

A second trial-wavefunction constructor, returning a plain
`TrialWavefunction` (drop-in for `build_trial_wavefunction` everywhere).
Instead of the paramagnetic free-fermion trial, this self-consistently
solves Hubbard UHF (unrestricted Hartree-Fock): each spin channel sees a
one-body potential from the *other* spin's mean-field density,
$H_\uparrow = K + U\operatorname{diag}(n_\downarrow)$,
$H_\downarrow = K + U\operatorname{diag}(n_\uparrow)$, iterated to
self-consistency (density convergence, linear-mixed for stability) from an
antiferromagnetic-seeded starting density. Useful at large $U$, where the
paramagnetic trial is a poor reference and real AFM correlations dominate;
coincides with the paramagnetic trial at $U=0$ (no mean field to seed).

`bipartite_coloring(K)` does a BFS 2-coloring of the lattice graph (from
$K$'s nonzero pattern) to find the sublattice assignment needed for the AFM
seed, generically for any geometry -- not hardcoded to chains or squares.
Returns `nothing` if the graph has an odd cycle (e.g. an odd-length PBC
ring), where a bipartite Neel seed isn't well-defined; `build_uhf_trial`
turns that into a clear `ArgumentError` rather than silently seeding
garbage. Same `bias` degeneracy-lifting trick as `build_trial_wavefunction`
is applied to each spin's mean-field Hamiltonian before diagonalizing.

## `walker.jl` -- `Walker`, `init_walkers`, `orthonormalize!`

`Walker` is just `(phi_up, phi_down, weight)`. `orthonormalize!` does the
thin-QR numerical stabilization from theory §10 -- deliberately *no* weight
compensation; see the theory doc for why that's exact, not an approximation.

## `estimators.jl` -- Green's function, overlaps, local energy, and the fast-update formulas

- `greens_function(phi, psi_trial)` / `overlap(phi, psi_trial)`: theory §5,
  computed directly from the definitions ($G=\phi M^{-1}\psi_T^\dagger$,
  overlap $=\det M$). `walker_overlap` is the up$\times$down product used for
  the constrained-path gate.
- `local_energy` / `mixed_energy_estimator`: theory §6, the
  $\operatorname{tr}(KG_\uparrow)+\operatorname{tr}(KG_\downarrow)+U\sum_iG_{\uparrow,ii}G_{\downarrow,ii}$
  formula, weighted-averaged over the walker population.
- `local_update_ratio(G, i, gamma_factor)` / `rank1_update_greens_function!(G, i, gamma_factor, R)`:
  theory §9, the Sherman-Morrison fast-update pair. **These two always get
  called as a pair**: compute `R = local_update_ratio(G, i, gamma_factor)`
  first (needs the *pre*-update `G[i,i]`), decide whether to accept the
  candidate field using `R`, and only if accepted, call
  `rank1_update_greens_function!(G, i, gamma_factor, R)` to mutate `G` in
  place to match. Calling the update with a stale `R` (e.g. computed for a
  different `gamma_factor`) silently corrupts `G` -- there's no cross-check,
  by design, since this is the hot inner loop.

## `propagator.jl` -- `build_expK_half`, `hs_gamma`, `propagate_step!`

- `build_expK_half`: $\exp(-\Delta\tau K/2)$ (theory §3), precomputed once
  per `run_afqmc` call and reused every step.
- `hs_gamma`: solves $\cosh\gamma=e^{\Delta\tau U/2}$ (theory §4). Throws for
  `U < 0` (attractive Hubbard needs a different transform, not implemented).
  Returns `0.0` exactly at `U == 0` (no auxiliary field effect, propagation
  reduces to the free-fermion projector).
- `propagate_step!`: the full per-walker Trotter step, theory §3-§9 in
  sequence:
  1. `overlap_ref = walker_overlap(...)` -- reference for the end-of-step gate.
  2. Apply `expK_half` to both spins (first $K/2$ half-step).
  3. Compute `G_up`, `G_down` from scratch (`greens_function`) -- the *only*
     full $O(N^3)$ recompute in the step.
  4. Loop over sites `i = 1:Ns`: for both candidate fields, get the ratio via
     `local_update_ratio` ($O(1)$ each), force-bias-sample which field to
     keep (theory §7), apply it to the walker's rows, and update `G_up`,
     `G_down` via `rank1_update_greens_function!` ($O(N^2)$ each).
     `walker.weight *= 0.5 * (|ratio_plus| + |ratio_minus|)` accumulates the
     importance weight -- note this is *already* a ratio relative to the
     state just before this site (no division needed, unlike a from-scratch
     overlap would require), because `R` from `local_update_ratio` already
     *is* $\langle\psi_T|\phi'\rangle/\langle\psi_T|\phi\rangle$.
  5. Apply `expK_half` again (second $K/2$ half-step).
  6. `overlap_new = walker_overlap(...)`; if its sign differs from
     `overlap_ref`'s (or `overlap_ref == 0`), zero the weight -- the
     constrained-path gate (theory §8).

  `walker.weight` is guaranteed to stay $\ge0$ throughout: every factor
  multiplied into it in step 4 is a magnitude (`abs(...)`), and only the
  explicit sign-flip check in step 6 can zero it. The *sign* of the
  determinant ratio never leaks into the stored weight -- it's used only to
  decide the gate.

  If `walker.weight` is already `0.0` on entry, the function returns
  immediately without doing any work -- dead walkers stay dead until the
  next `population_control!` call replaces them.

**Why fast-update instead of brute force**: an earlier version of this file
recomputed `walker_overlap` from scratch (a fresh $N\times N$ determinant) for
every candidate field at every site -- correct, but $O(Ns\cdot N^3)$ per
step, which at half filling ($N=Ns/2$) is $O(Ns^4)$. The current version is
$O(Ns^3)$ per step (dominated by the two $K/2$ multiplies and the one $G$
recompute), which is the standard DQMC/CPMC asymptotic complexity. Rough
numbers on this machine (chain, half filling, `dtau=0.01`, `U/t=4`, time per
`propagate_step!` call): `Ns=16` ~0.015 ms, `Ns=32` ~0.056 ms, `Ns=64`
~0.34 ms, `Ns=96` ~0.95 ms -- a full run (200 walkers x 2000 steps) at
`Ns=96` takes on the order of 5-10 minutes; `Ns` in the low tens is
comfortably under a minute. Numbers will vary by machine; rerun the
benchmark loop in git history / a scratch script if you need current figures.

## `population_control.jl` -- `population_control!`

Comb/systematic resampling (theory §11): one random offset, evenly spaced
sampling positions through the cumulative weight, total weight preserved
exactly, new population all equal weight. Mutates the `walkers` vector's
elements in place (replaces each walker's `phi_up`/`phi_down`/`weight`, not
the vector itself, so external references to the `Vector{Walker}` stay
valid).

## `driver.jl` -- `run_afqmc`, `block_average`

`run_afqmc` is the outer loop (theory §12): propagate every walker, then on
the configured cadences reorthonormalize (`stabilize_every`) and
population-control (`pop_control_every`), then record the mixed-estimator
energy once past `equilibration_steps`. Returns a named tuple
`(energy_trace, energy_mean, energy_err, walkers)`.

`block_average` does simple fixed-block blocking: split the trace into
`num_blocks` contiguous chunks, treat block means as independent samples,
report `mean(block_means)` and `std(block_means)/sqrt(nblocks)`. This is a
coarse estimator (no automatic block-size selection to find the
autocorrelation-safe regime) -- fine as a default, but if you need a
tighter error bar or suspect the blocks are still autocorrelated, increase
`equilibration_steps` and/or post-process `result.energy_trace` yourself
with a proper blocking/autocorrelation analysis.

## Adding a new lattice geometry

1. Write a `build_hubbard_<shape>(...)::HubbardLattice` in `lattice.jl` that
   constructs `K` (theory §1) -- everything downstream (`trial_wavefunction.jl`
   through `driver.jl`) only depends on `lattice.Ns`, `lattice.K`,
   `lattice.U`, so nothing else needs to change.
2. Watch for degenerate single-particle levels at your target filling (see
   the `trial_wavefunction.jl` section above) -- if in doubt, compare
   `eigen(Symmetric(lattice.K)).values` for repeated values near the Fermi
   level before trusting a new geometry's AFQMC results.

## `ab_initio/` -- general (molecular) Hamiltonian AFQMC

A separate nested submodule (`AFQMC.AbInitio`, re-exported up to `LatticeMC`
directly, e.g. `LatticeMC.run_afqmc_ab_initio`), parallel to the Hubbard
files above and touching none of them. See
[`afqmc_ab_initio_theory.md`](afqmc_ab_initio_theory.md) for the derivations
-- this section is the file-by-file map, same style as above. Naming
convention: every ab initio name is either a distinct noun
(`AbInitioWalker`, `AbInitioTrial`, `MolecularIntegrals`) or has an
`_ab_initio` suffix (`local_energy_ab_initio`, `propagate_step_ab_initio!`,
...), so nothing collides with the real/Hubbard names when both are
`using`-imported into the same scope -- deliberate, to avoid the Julia
multiple-dispatch-across-modules ambiguity that sharing names like
`orthonormalize!` would have introduced.

- **`integrals.jl`** -- `MolecularIntegrals` (`Ns`, `h1e`, `h2e`, `E_nuc`)
  and `read_fcidump` (standard FCIDUMP text format parser, real orbitals,
  8-fold ERI symmetry fill-out).
- **`sto3g_hydrogens.jl`** -- the self-contained STO-3G integral engine
  (theory §2): primitive Gaussian overlap/kinetic/nuclear-attraction/ERI
  formulas, contraction normalization, `lowdin_orthogonalize`, and
  `build_h_chain_sto3g`. The Boys-function small-$t$ branch
  (`boys_f0`) matters more than it looks -- same-center integrals hit
  $t\to0$ constantly for a chain of atoms.
- **`cholesky.jl`** -- `cholesky_decompose_eri` (pivoted Cholesky, theory
  §3) and `modified_one_body` (the $h1e^{\rm mod}$ exchange correction).
- **`rhf_trial.jl`** -- `AbInitioTrial` (the complex trial-wavefunction
  struct, defined here since this is its first use site), `rhf_scf` (returns
  orbitals + total energy, used directly in tests for the variational-bound
  check), `build_rhf_trial` (wraps `rhf_scf`, returns the `AbInitioTrial`).
  Closed-shell only (`Nup == Ndown`, throws otherwise).
- **`uhf_trial.jl`** -- `uhf_scf` / `build_uhf_trial_ab_initio`, the ab
  initio analogue of `uhf_trial.jl`'s Hubbard `build_uhf_trial`: separate
  `phi_up`/`phi_down` (unlike `rhf_trial.jl`'s shared orbitals), Fock
  matrices built with Coulomb from the *total* density but exchange only
  from the *same-spin* density (reduces exactly to `rhf_scf`'s Fock matrix
  when `P_up == P_down`, a good sanity check if you touch this code).
  Symmetry-breaking seed is a simple alternating-site-parity density bias --
  simpler than Hubbard's graph-coloring `bipartite_coloring` since
  `build_h_chain_sto3g` only produces linear chains so far; would need
  generalizing for a future non-chain ab initio geometry builder.

  **Measured impact** (this is *why* it exists, not a nice-to-have): on H4,
  UHF collapses back to the RHF solution near equilibrium (zero moment, no
  benefit -- as expected, RHF is adequate there) but spontaneously spin-
  polarizes at stretched bond lengths (the standard RHF-to-UHF dissociation
  instability), where it gives a dramatically better AFQMC trial:

  | bond (bohr) | RHF trial captures | UHF trial captures |
  |---|---|---|
  | 1.4 (near eq.) | ~11% | ~11% (UHF == RHF here) |
  | 2.5 | -50% (worse than RHF!) | ~39% |
  | 3.0 | -66% (worse than RHF!) | ~75% |

  ("captures" = percent of the RHF-to-FCI correlation-energy gap recovered;
  negative means the AFQMC estimate landed *above* RHF, i.e. worse than
  mean-field -- a real failure mode of the plain-RHF-trial Tier-1 algorithm
  at a geometry RHF itself doesn't describe qualitatively correctly.) Doesn't
  help the *equilibrium* H4 case from theory doc §6 -- that gap has a
  different origin (Tier 1's missing force bias, not trial symmetry) --
  see the theory doc for the distinction.
- **`walker.jl`** -- `AbInitioWalker`, `init_ab_initio_walkers`,
  `orthonormalize_ab_initio!`. Complex QR; the "no weight compensation
  needed" argument from theory §10 is unchanged by working over $\mathbb{C}$
  (it only used that $G$ is invariant under $\phi\to\phi A$ for invertible
  $A$, real or complex).
- **`estimators.jl`** -- `greens_function_ab_initio` / `overlap_ab_initio` /
  `walker_overlap_ab_initio` (same Thouless-theorem formulas, complex);
  `local_energy_ab_initio` (the generalized-Wick's-theorem formula, theory
  doc's `estimators.jl` cross-reference above -- verified during development
  against an explicit many-body CI-vector calculation for a generic, fully
  asymmetric complex walker, not just the symmetric trial-equals-walker
  case); `mixed_energy_estimator_ab_initio` (weighted average, `real(...)`
  at the end).
- **`propagator.jl`** -- `apply_one_body_ab_initio!`,
  `propagate_step_ab_initio!` (theory §4-§5: sample the joint Gaussian,
  build `dK`, one matrix exponential, phaseless gate).
- **`population_control.jl`** -- `population_control_ab_initio!`, same comb
  algorithm as the real version, `ComplexF64`-typed.
- **`driver.jl`** -- `run_afqmc_ab_initio`. Builds the Cholesky vectors and
  `h1e_mod` once at the start (not per step), then the same
  propagate/stabilize/population-control/measure loop structure as
  `run_afqmc`; reuses `block_average` directly (imported via `import
  ..block_average` at the top of the submodule -- it's fully generic over
  the energy trace type, no `_ab_initio` variant needed).

### If you're about to trust a result on a new molecule

Read theory doc §6 first. The short version: this implementation's accuracy
is trial-quality-dependent in a way that isn't obvious from the code alone
-- H2 near equilibrium is essentially exact (~93% of correlation energy),
H4 recovers only ~10% with the same settings. Before trusting a number,
either (a) pick a system you have reason to believe is well-described by a
single RHF determinant, or (b) cross-check against
`test/ab_initio_fci_reference.jl`'s FCI if the system is small enough
(`Ns` up to ~12-14, same practical ED ceiling as the Hubbard side).

## Adding attractive-`U` support

`hs_gamma` explicitly rejects `U < 0`. The attractive Hubbard model needs a
charge-channel (rather than spin-channel) discrete transform -- couple the
auxiliary field to $n_{i\uparrow}+n_{i\downarrow}$ instead of
$n_{i\uparrow}-n_{i\downarrow}$ -- which changes the sign structure of the
problem entirely (attractive Hubbard famously has *no* sign problem with the
right transform). Not a small tweak to the current code: budget it as a
separate variant of `hs_gamma` + `propagate_step!`, not a parameter flip.
