# Ab initio molecular AFQMC: what's different from Hubbard

This is an addendum to [`afqmc_theory.md`](afqmc_theory.md), covering what
changes for a general (ab initio) molecular Hamiltonian. Read that document
first -- the Thouless-theorem overlap/Green's-function machinery (§5), the
mixed-estimator energy structure (§6), population control (§11), and
numerical stabilization (§10) all carry over **unchanged**, just with
complex arithmetic. What's genuinely different is the Hamiltonian itself,
how its two-body part is decomposed for the Hubbard-Stratonovich transform,
and the resulting propagator and phaseless gate. Implementation map: the
"ab_initio/" section of [`afqmc_implementation.md`](afqmc_implementation.md).

## 1. The Hamiltonian

Chemist notation, matching FCIDUMP/PySCF convention:

$$H = E_{\rm nuc} + \sum_{pq} h1e_{pq}\,c^\dagger_p c_q + \tfrac12\sum_{pqrs} (pq|rs)\,c^\dagger_p c^\dagger_r c_s c_q$$

Unlike Hubbard's on-site-only $U\,n_{i\uparrow}n_{i\downarrow}$, $(pq|rs)$ is
a general, dense 4-index tensor -- there's no simple per-site discrete field
available.

## 2. Orbital basis must be orthonormal

Every second-quantized formula here (Thouless theorem, Wick's theorem,
Green's functions) implicitly assumes the single-particle basis indexed by
$p,q,r,s$ is orthonormal -- true by construction for Hubbard's site basis,
**not** true for raw atomic-orbital (AO) Gaussian basis functions on
different atoms. `sto3g_hydrogens.jl` builds the AO integrals and then
Loewdin-orthogonalizes ($X=S^{-1/2}$, transform $h1e$ and all four indices
of $(pq|rs)$) before returning `MolecularIntegrals` -- skipping this step
silently gives wrong physics, not an error.

## 3. Cholesky decomposition replaces the discrete Hirsch transform

Reshape $(pq|rs)$ as the matrix $M_{(pq),(rs)}$ over the compound index
$(pq)$; for real orbitals this is symmetric positive semidefinite (a
standard fact), so a pivoted Cholesky decomposition gives

$$(pq|rs) = \sum_\gamma L^\gamma_{pq} L^\gamma_{rs}, \qquad V := \tfrac12\sum_{pqrs}(pq|rs)c^\dagger_pc^\dagger_rc_sc_q = \tfrac12\sum_\gamma v_\gamma^2,\quad v_\gamma = \sum_{pq}L^\gamma_{pq}c^\dagger_pc_q$$

Rewriting $c^\dagger_pc^\dagger_rc_sc_q$ in the $v_\gamma^2 = \sum_{pqrs}L^\gamma_{pq}L^\gamma_{rs}c^\dagger_pc_qc^\dagger_rc_s$ ordering
and normal-ordering via the fermion anticommutator
$c_qc^\dagger_r=\delta_{qr}-c^\dagger_rc_q$ picks up a one-body remainder:

$$\sum_{pqrs}L_{pq}L_{rs}\,c^\dagger_pc^\dagger_rc_sc_q = v^2 - \sum_{ps}(L^2)_{ps}\,c^\dagger_pc_s$$

so the full Hamiltonian becomes $H = E_{\rm nuc} + \sum h1e^{\rm mod}_{pq}c^\dagger_pc_q + \tfrac12\sum_\gamma v_\gamma^2$
with $h1e^{\rm mod} = h1e - \tfrac12\sum_\gamma (L^\gamma)^2$ -- computed
once, up front (`modified_one_body`), not per step. (Each $L^\gamma$ comes
out symmetric because $(pq|rs)$ has the standard real-orbital 8-fold
symmetry; verified numerically in `test/ab_initio.jl`, not just assumed.)

## 4. Continuous Hubbard-Stratonovich transform

For a Hermitian one-body operator $v$, the real-Gaussian characteristic
function identity $E_{x\sim N(0,1)}[e^{iax}] = e^{-a^2/2}$ gives

$$e^{-\Delta\tau v^2/2} = E_{x\sim N(0,1)}[e^{i\sqrt{\Delta\tau}\,x\,v}]$$

and its multivariate generalization (independent $x_\gamma$, still exact
regardless of whether the $v_\gamma$ commute -- verified numerically for a
non-commuting pair in development, not just asserted):

$$e^{-\Delta\tau\sum_\gamma v_\gamma^2/2} = E_{x\sim N(0,I)}\Big[\exp\big(i\sqrt{\Delta\tau}\,\textstyle\sum_\gamma x_\gamma v_\gamma\big)\Big]$$

Since $\sum_\gamma x_\gamma v_\gamma$ is itself one-body (linear in
$c,c^\dagger$), it corresponds to a single-particle matrix
$dK=\sum_\gamma x_\gamma L^\gamma$, and applying the exponentiated operator
to a Slater determinant is exactly left-multiplication by the matrix
$\exp(i\sqrt{\Delta\tau}\,dK)$ -- **one** matrix exponential per step,
sampling the *full* Cholesky-vector-length Gaussian vector $x$ jointly, not
one exponential per Cholesky vector.

## 5. Phaseless gate (complex generalization of the Hubbard sign gate)

With complex walkers, $I(x) = \langle\psi_T|B(x)|\phi\rangle/\langle\psi_T|\phi\rangle$
is a complex number. The phaseless weight update (Zhang & Krakauer's general
formulation, of which Hubbard's real sign-flip gate is the real/discrete
special case):

$$w \mathrel{*}= |I(x)|\cdot\max(0,\cos(\arg I(x)))$$

## 6. Tier 1 scope: what's simplified, and what that costs (measured)

One thing is deliberately simplified relative to production codes, and its
cost was measured directly (not just estimated) during development:

- **Direct-tensor local energy**, not Cholesky-vector-based: fine at the
  `Ns` ~ 4-8 scale this targets, `O(Ns^4)` per measurement.

(A force-bias/mean-field-shifted sampling option *is* available --
`run_afqmc_ab_initio(...; force_bias=true)` -- off by default. See §7.1 for
why: it's a real, modest variance-reduction technique, not the bias fix an
earlier version of this document mistakenly suggested it would be.)

**What this costs in practice**, measured on H4 (4 hydrogens, STO-3G,
bond=1.4 bohr) vs. H2 at the same bond length, both with a plain RHF trial:

| System | RHF-to-FCI gap | AFQMC recovers |
|---|---|---|
| H2 | 0.021 Ha | ~93% of it |
| H4 | 0.041 Ha | ~10% of it |

This is a **real, measured property of Tier 1 + RHF trial**, not a bug --
confirmed during development by an extensive, independent cross-validation
of every static piece (integrals against a known textbook overlap value,
Cholesky reconstruction, RHF's variational bound and dissociation-catastrophe
trend, the mixed-estimator formula against an explicit many-body CI-vector
calculation for a generic complex walker, and the core multivariate Gaussian
identity against direct Monte Carlo for non-commuting Cholesky vectors) —
all matched to near machine precision.

**On force bias specifically** (§7.1 has the full investigation, including
an implementation pitfall worth knowing about): a correctly-implemented
mean-field force bias does not close this gap, confirmed by a statistically
well-powered comparison and a direct mechanistic check of *why* it should
help. It's a variance-reduction technique -- makes the stochastic estimator
more efficient for a fixed accuracy target -- not a bias-reduction one. The
H4-equilibrium gap is not a variance problem (it doesn't shrink with more
walkers, more steps, smaller `dtau`, or force bias); it's the *asymptotic*
($\tau\to\infty$) bias of the phaseless approximation itself for this
specific (Hamiltonian, RHF trial) pair, set by how well the trial's phase/
nodal structure tracks the true ground state's along the imaginary-time
path. No refinement of the sampling scheme changes that -- only a
qualitatively different trial (multi-determinant, not just a different
single determinant) would.

**Why H4 and not H2**: H2's two electrons in one bonding orbital are well
described by a single RHF determinant even somewhat away from equilibrium
(RHF still captures the qualitative physics). `test/ab_initio.jl` tests both
systems: H2 with a tight tolerance (as a correctness check), H4 with a
deliberately loose one (as an honest demonstration of this limitation,
checking "meaningfully better than RHF", not "close to FCI").

**Update, after building `build_uhf_trial_ab_initio`** (the ab initio
analogue of the Hubbard side's `build_uhf_trial`, §7 below): the natural
hypothesis was that H4's gap has the same origin as Hubbard's degenerate/
paramagnetic-trial problem, fixable the same way. **That hypothesis was
wrong for the equilibrium-bond-length H4 case specifically** -- at bond=1.4,
UHF converges back to the exact RHF solution (zero spin polarization; RHF is
already a locally stable, adequate single-reference description there), so
the trial is unchanged and the AFQMC result doesn't move. The gap at that
geometry is the asymptotic phaseless-approximation bias described just
above -- not fixable by a better *single-determinant* trial search, since
RHF already is the (locally) optimal single determinant there.

Where UHF *does* help enormously is stretched H4, a different (and more
classic) failure mode -- the standard RHF-to-UHF bond-dissociation
instability, where RHF stops being even a qualitatively adequate reference
and UHF spontaneously spin-polarizes. Measured on H4 (`Nup=Ndown=2`):

| bond (bohr) | RHF trial captures | UHF trial captures |
|---|---|---|
| 1.4 (equilibrium) | ~11% | ~11% (UHF == RHF here) |
| 2.5 | -50% (worse than RHF) | ~39% |
| 3.0 | -66% (worse than RHF) | ~75% |

("captures" = percent of the RHF-to-FCI gap recovered; negative means the
AFQMC estimate landed *above* RHF -- a real failure mode of the plain-RHF
trial at a geometry it doesn't describe qualitatively correctly, not just a
quantitatively poor one.) So: two genuinely different problems share the
same symptom ("H4 AFQMC is inaccurate") but need different fixes -- trial
symmetry breaking fixes the dissociation-instability case; the equilibrium
case needs a qualitatively different (multi-determinant) trial, not a
sampling refinement (§7.1).

## 7. Unrestricted Hartree-Fock trial (`build_uhf_trial_ab_initio`)

Same idea as `build_uhf_trial` on the Hubbard side, generalized from the
on-site-$U$ mean field to the full two-electron tensor: separate spatial
orbitals per spin, each solving its own Fock equation

$$F_\uparrow = h1e + \sum_{\lambda\sigma}P^{\rm tot}_{\lambda\sigma}(\mu\nu|\lambda\sigma) - \sum_{\lambda\sigma}P^\uparrow_{\lambda\sigma}(\mu\lambda|\nu\sigma), \qquad F_\downarrow \text{ (same, } P^\downarrow\text{ in the exchange term)}$$

($P^{\rm tot}=P^\uparrow+P^\downarrow$; Coulomb sees every electron, exchange
only same-spin ones, i.e. Pauli exclusion). This reduces exactly to
`rhf_scf`'s Fock matrix when $P^\uparrow=P^\downarrow=P^{\rm tot}/2$, so UHF
is a strict variational generalization of RHF: its converged energy can only
be $\le E_{\rm RHF}$, with equality exactly when the unrestricted
optimization finds no benefit to spin-polarizing (verified in
`test/ab_initio.jl`, both the inequality and the H4-specific "collapses at
equilibrium, polarizes when stretched" behavior above).

Symmetry-breaking seed: an alternating-site-parity density bias (odd sites
seeded toward spin-up, even toward spin-down) -- simple because it only
needs to handle the linear chains `build_h_chain_sto3g` produces; would need
a proper geometry-aware coloring (like Hubbard's `bipartite_coloring`, which
works from the lattice graph) for a future non-chain ab initio builder.

### 7.1. Force bias (`force_bias=true`): a variance fix, not a bias fix

The natural next hypothesis after UHF (asked for explicitly, and worth
recording in full since it took two attempts and overturned an earlier claim
in this document): if UHF doesn't fix the equilibrium H4 gap, does a
*properly self-consistent* mean-field force bias -- the standard variance-
reduction technique real AFQMC codes use? Implemented per Cholesky vector as
the textbook prescription: shift the sampled Gaussian field by
$\bar f_\gamma = -\sqrt{\Delta\tau}\,\mathrm{Re}\,\mathrm{tr}(L^\gamma G)$,
and correct the weight by the exact Radon-Nikodym factor for sampling from
the shifted Gaussian instead of the standard one,
$\exp(-\tfrac12|\bar f|^2 - \bar f\cdot\xi)$ ($\xi = x-\bar f$, the realized
fluctuation) -- this factor is derived from an exact change-of-variables
identity, so it introduces no approximation of its own for *any* choice of
$\bar f$; only the choice of $\bar f$ affects variance, never correctness.

**Attempt 1: walker-adaptive shift** ($G$ = each walker's own mixed Green's
function, re-evaluated every step). Result: no improvement, and by two
independent measurements. Mechanistically, does the shift reduce the
per-step phase decorrelation ($\cos(\arg I(x))$, the quantity the phaseless
gate consumes)? No: mean $\cos\theta$ per step was $0.958$ unbiased vs.
$0.955$ shifted (5000-sample, matched seed) -- statistically distinguishable
but in the *wrong* direction. Directly, a well-powered AFQMC run (500
walkers, 4000 steps, matched seeds) gave gap-from-FCI $0.0375\pm0.0002$ Ha
unbiased vs. $0.0365\pm0.0007$ Ha shifted (no significant bias change --
both comfortably within about $1\sigma$ of each other) -- and the error bar
was over $3\times$ *larger* with the shift, the opposite of what a
variance-reduction technique should do.

**Why attempt 1 backfired**: re-evaluating $\bar f$ from each walker's own
(noisy, increasingly decorrelated-from-trial) state every step makes the
shift itself a fluctuating quantity, correlated with the walker's own noise
-- adding a new noise source rather than removing one.

**Attempt 2: static shift**, $\bar f$ computed once from the *trial's* own
density (`force_bias_shift`, evaluated before the walk starts, reused every
step for every walker) rather than adaptively per walker per step. This
fixes the variance problem: measured error bar (same system, dtau, walker/
step count, one seed, all three variants run back to back for a fair
comparison) $\pm0.000318$ unbiased vs. $\pm0.000296$ static-shifted -- a
modest (~7%) but genuine reduction, not a regression. The bias is unchanged
across all three ($0.0370$ unbiased / $0.0377$ static-shifted / $0.0374$
walker-adaptive, all statistically consistent with each other given their
error bars) -- consistent with, and confirming, the conclusion below.

**Conclusion, and why it makes sense in hindsight**: force bias reduces the
*variance* of the stochastic estimator for a fixed sample size -- it does
not change the *asymptotic* ($\tau\to\infty$, infinite-sample) bias of the
phaseless approximation itself, which is fixed by the trial's phase/nodal
structure relative to the true ground state along the imaginary-time path.
The equilibrium H4 gap doesn't respond to more walkers, more steps, smaller
`dtau`, *or* either force-bias variant, precisely because none of those are
variance knobs for *this* quantity. The only lever that changes the
phaseless bias itself is the trial's *qualitative* structure -- a
multi-determinant trial, not a better single determinant (UHF already
established RHF is the locally optimal single determinant at this
geometry) -- genuine future work, and a materially bigger undertaking than
either of the trial-wavefunction additions made so far.

**What shipped**: the static-shift version, as `run_afqmc_ab_initio(...;
force_bias=true)` / `propagate_step_ab_initio_force_bias!` +
`force_bias_shift` in `propagator.jl`. Off by default (`force_bias=false`)
since the effect is modest and it costs one extra Green's-function
evaluation up front; worth turning on for production runs where every bit
of variance reduction helps, not worth it for a quick check.

## 8. Multi-determinant trial: the actual fix for the phaseless bias

§6 and §7.1 established that the equilibrium-H4 gap is the phaseless
approximation's own asymptotic bias, fixed only by a *qualitatively* better
trial — not a different single determinant (UHF), not better sampling (force
bias). The lever that does work: a **multi-determinant trial**,

$$|\psi_T\rangle = \sum_{i=1}^{K} c_i\,|D_i\rangle,\qquad |D_i\rangle = |D_i^\uparrow\rangle\otimes|D_i^\downarrow\rangle$$

each $|D_i\rangle$ a Slater determinant (a pair of up/down orbital matrices),
with complex coefficients $c_i$. In the limit where the expansion is the
exact ground state, the phaseless bias vanishes entirely; truncated, it
systematically interpolates between a single determinant and exact.

**Everything the AFQMC needs is an overlap-weighted average over the
determinants.** Writing $O_i = \langle D_i|\phi\rangle = \det(D_i^{\uparrow\dagger}\phi^\uparrow)\det(D_i^{\downarrow\dagger}\phi^\downarrow)$
for the $i$-th determinant's overlap with a walker $|\phi\rangle$:

- **Overlap**: $\langle\psi_T|\phi\rangle = \sum_i c_i^*\,O_i$ — just a sum of
  single-determinant Thouless overlaps (§5), each a product of two dets.
- **Mixed Green's function**: $G^\sigma = \big(\sum_i c_i^* O_i\,G_i^\sigma\big)\big/\big(\sum_i c_i^* O_i\big)$,
  where $G_i^\sigma$ is the *single*-determinant Green's function between
  $D_i$ and $\phi$ — the overlap-weighted average of the per-determinant
  Green's functions.
- **Mixed local energy**: $E_{\rm loc} = \big(\sum_i c_i^* O_i\,E_i\big)\big/\big(\sum_i c_i^* O_i\big)$,
  where $E_i$ is the single-determinant mixed local energy (§6) computed with
  $G_i$. This is exact because $\langle\psi_T|H|\phi\rangle = \sum_i c_i^*\langle D_i|H|\phi\rangle = \sum_i c_i^* O_i E_i$.

So the whole implementation is: run the existing single-determinant machinery
once per determinant, and combine by overlap-weighting. Cost is $O(K)\times$
the single-determinant cost — cheap for the modest $K$ (tens of determinants)
that a truncated CI needs.

### Where the determinants come from, and the one basis subtlety that bites

The natural source is a (truncated) CI / CASSCF expansion. The key practical
point, learned the hard way in development: **the truncation is only
well-behaved in the MO basis.** In the RHF molecular-orbital basis, a
determinant is simply a choice of occupied orbitals (its orbital matrix is
the corresponding columns of the identity), and the CI vector is *dominated*
by the HF determinant ($|c_0|\approx0.99$ for H4), so keeping the top-$K$
determinants is a sensible approximation. In a raw AO basis the CI vector has
no such dominant determinant; a truncated AO-basis expansion is a *poor,
ill-behaved* trial that gives unphysical (below-FCI) phaseless energies. A
*full* expansion is basis-independent and works either way — which is exactly
why the bug only shows up on truncation. `build_casci_trial` therefore always
transforms to the RHF MO basis internally and returns the MO-basis integrals
to run AFQMC with (trial and integrals must share a basis).

### Measured: this is what closes the gap

On the equilibrium-H4 case that defeated UHF and force bias, adding
determinants monotonically closes the phaseless gap (`dtau=0.01`, correlation
energy recovered):

| trial | corr. recovered |
|---|---|
| top-1 determinant (= HF) | ~8% |
| top-2 | ~21% |
| top-5 | ~73% |
| top-10 | ~92% |
| full CI (36 dets) | exact (to ~1e-15) |

The top-1 case reproduces the single-RHF-determinant limit from §6; the full
expansion gives AFQMC = FCI to machine precision, which is the strongest
possible end-to-end validation of the multi-determinant overlap / Green's-
function / local-energy machinery (if any of those were wrong, the exact-
ground-state trial would not give the exact energy).

**What shipped**: `MultiDetTrial` (a drop-in `AbstractAbInitioTrial`, so the
same `run_afqmc_ab_initio` propagates it), `build_casci_trial` (internal
RHF + MO transform + full-CI solve + top-$K$ truncation, small systems only),
and `multidet_from_ci` (assemble a trial from an externally-supplied
expansion, e.g. a PySCF CASSCF CI vector, for systems past the internal-CI
size ceiling). Force bias is not yet implemented for multi-determinant trials
(it errors if requested). See
[`afqmc_implementation.md`](afqmc_implementation.md) for the file/function
map.
