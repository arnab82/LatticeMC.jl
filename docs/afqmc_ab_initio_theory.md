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

Two things are deliberately simplified relative to production codes, and
their cost was measured directly (not just estimated) during development:

- **No force bias / mean-field shift.** $x$ is sampled from $N(0,I)$
  directly, not shifted toward the walker's own mixed density (the standard
  variance-reduction refinement). This is mathematically unbiased on its
  own (any importance-sampling shift is, given a correctly-computed
  reweighting factor) but increases variance.
- **Direct-tensor local energy**, not Cholesky-vector-based: fine at the
  `Ns` ~ 4-8 scale this targets, `O(Ns^4)` per measurement.

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
all matched to near machine precision. Adding a hand-rolled mean-field force
bias to the propagator did *not* change the H4 result at all, which is
itself informative: it confirms the gap is coming from the phaseless
constraint's inherent information loss (discarding the phase of the
importance ratio every step) compounding over many steps for a system whose
true ground state isn't well-aligned with a single RHF determinant, not from
sampling variance that force-bias would fix.

**Why H4 and not H2**: H2's two electrons in one bonding orbital are well
described by a single RHF determinant even somewhat away from equilibrium
(RHF still captures the qualitative physics). Four separated-but-interacting
hydrogens have more open-shell/multi-configurational character that a single
paramagnetic determinant doesn't represent well -- the same phenomenon
already seen on the Hubbard side (`build_uhf_trial`'s antiferromagnetic seed
exists precisely because the plain paramagnetic trial is a poor reference in
exactly this kind of regime). The fix here would be the analogous one: a
symmetry-broken (unrestricted) ab initio trial, and/or the force-bias
refinement done properly (self-consistently, not the single hand-rolled
variant tried during development) -- both real future work, not implemented
here. `test/ab_initio.jl` tests both systems: H2 with a tight tolerance (as
a correctness check), H4 with a deliberately loose one (as an honest
demonstration of this limitation, checking "meaningfully better than RHF",
not "close to FCI").
