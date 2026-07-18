# AFQMC theory, step by step

This document derives, from the Hubbard Hamiltonian down to the exact
formulas used in the code, the phaseless (constrained-path) Auxiliary-Field
Quantum Monte Carlo algorithm implemented in `src/afqmc/`. See
[`afqmc_implementation.md`](afqmc_implementation.md) for where each piece of
this derivation lives in the source, and
[`afqmc_algorithm.md`](afqmc_algorithm.md) for the same content distilled
into formal pseudocode.

## 1. The Hamiltonian

The Hubbard model on `Ns` sites:

$$H = -t\sum_{\langle i,j\rangle,\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow} \equiv \hat K + \hat V$$

$\hat K$ is one-body (hopping matrix $K_{ij}$, `-t` on bonds, 0 elsewhere,
built in `lattice.jl`). $\hat V$ is two-body (on-site repulsion, `U >= 0`
here).

## 2. Ground state via imaginary-time projection

For any trial state $|\psi_T\rangle$ not orthogonal to the ground state
$|\Psi_0\rangle$,

$$|\Psi_0\rangle \propto \lim_{\tau\to\infty} e^{-\tau H}|\psi_T\rangle$$

AFQMC represents $e^{-\tau H}|\psi_T\rangle$ stochastically as a population
of walkers and evolves it in small imaginary-time steps $\Delta\tau$,
$\tau = n\,\Delta\tau$.

## 3. Trotter splitting

Since $\hat K$ and $\hat V$ don't commute, split each step symmetrically
(smaller Trotter error than a naive split, $O(\Delta\tau^3)$ vs $O(\Delta\tau^2)$):

$$e^{-\Delta\tau H} \approx e^{-\Delta\tau \hat K/2}\,e^{-\Delta\tau \hat V}\,e^{-\Delta\tau \hat K/2}$$

The one-body factors are easy: $e^{-\Delta\tau \hat K/2}$ acting on a Slater
determinant just left-multiplies its orbital matrix by the `Ns x Ns` matrix
$\exp(-\Delta\tau K/2)$ (`build_expK_half`).

## 4. Hubbard-Stratonovich transform of the interaction

$\hat V$ is two-body and can't be applied directly to a determinant. Use the
fermion identity $n_{i\uparrow}^2 = n_{i\uparrow}$ (occupation is 0 or 1) to
write, per site,

$$n_{i\uparrow}n_{i\downarrow} = \tfrac12(n_{i\uparrow}+n_{i\downarrow}) - \tfrac12(n_{i\uparrow}-n_{i\downarrow})^2$$

so

$$e^{-\Delta\tau U n_{i\uparrow}n_{i\downarrow}} = e^{-\Delta\tau U(n_{i\uparrow}+n_{i\downarrow})/2}\; e^{+\Delta\tau U(n_{i\uparrow}-n_{i\downarrow})^2/2}$$

The remaining piece is now the exponential of a *square* of a one-body
operator, so it admits the **discrete Hirsch transform**: for $U\ge0$, define
$\gamma$ by $\cosh\gamma = e^{\Delta\tau U/2}$, then

$$e^{+\Delta\tau U(n_{i\uparrow}-n_{i\downarrow})^2/2} = \tfrac12\sum_{s=\pm1} e^{\gamma s(n_{i\uparrow}-n_{i\downarrow})}$$

(a one-line check: expand both sides in the two eigenvalues of
$n_{i\uparrow}-n_{i\downarrow}$, which are $0,\pm1$ for the four occupation
states of site $i$.) This only works for $U\ge0$; attractive Hubbard needs a
different (charge-channel) transform, not implemented here (`hs_gamma`
throws for `U < 0`).

Putting it together, for each site $i$ there is a two-valued auxiliary field
$s_i=\pm1$, and applying the field multiplies row $i$ of the orbital matrix:

$$\phi_\uparrow[i,:] \mathrel{*}= e^{-\Delta\tau U/2}\,e^{\gamma s_i}, \qquad \phi_\downarrow[i,:] \mathrel{*}= e^{-\Delta\tau U/2}\,e^{-\gamma s_i}$$

The full interaction propagator is the product over independent site fields:

$$e^{-\Delta\tau\hat V} = \prod_i \Big[\tfrac12\sum_{s_i=\pm1}(\cdots)\Big]$$

Summing over all $2^{Ns}$ field configurations exactly is intractable, so
AFQMC samples one configuration stochastically per step (Monte Carlo instead
of exact summation) -- this is where the walk becomes a walk.

## 5. Slater determinants, overlaps, and Thouless' theorem

A walker is a Slater determinant, represented as an `Ns x N` matrix $\phi$
(orthonormal columns = occupied orbitals). For two determinants $\phi,\psi_T$
with the same particle number, Thouless' theorem gives the overlap and mixed
one-body density matrix in closed form:

$$\langle\psi_T|\phi\rangle = \det(\psi_T^\dagger\phi) =: \det M$$

$$G := \phi\,M^{-1}\psi_T^\dagger \quad\Rightarrow\quad G_{qp} = \frac{\langle\psi_T|c^\dagger_p c_q|\phi\rangle}{\langle\psi_T|\phi\rangle}$$

($G$ is `Ns x Ns`; note the transposed index convention -- $G_{qp}$, not
$G_{pq}$, equals $\langle c^\dagger_p c_q\rangle$. The diagonal is convention-
independent: $G_{ii} = \langle n_i\rangle$.) This is `greens_function` /
`overlap` in `estimators.jl`, and it is the single most important object in
the algorithm -- essentially everything below is either computing $G$, using
$G$, or updating $G$.

## 6. Mixed estimator: the local energy

By Wick's theorem, and because up and down determinants factorize
($|\phi\rangle=|\phi_\uparrow\rangle\otimes|\phi_\downarrow\rangle$), the
Hubbard energy's mixed estimator is exactly

$$E_{\rm loc}(\phi) = \frac{\langle\psi_T|H|\phi\rangle}{\langle\psi_T|\phi\rangle} = \operatorname{tr}(K G_\uparrow) + \operatorname{tr}(K G_\downarrow) + U\sum_i G_{\uparrow,ii}\,G_{\downarrow,ii}$$

(`local_energy`). Averaging over the walker population, weighted, gives the
mixed-estimator ground-state energy (`mixed_energy_estimator`):

$$E = \frac{\sum_k w_k\,E_{\rm loc}(\phi_k)}{\sum_k w_k}$$

This is exact once $|\phi\rangle \propto |\Psi_0\rangle$; at finite $\tau$
it's a variational-quality estimate that converges as the population
equilibrates.

## 7. Importance sampling

Applying the bare projector with fields sampled uniformly ($s_i=\pm1$ each
with probability $1/2$) is *free projection*: unbiased in principle, but
$\langle\psi_T|\phi\rangle$ can wander toward zero or blow up, and variance
grows badly with $\tau$. Instead, **importance sample**: bias the field
distribution toward configurations that keep the walker aligned with
$\psi_T$, and correct with an importance weight so the estimator stays
unbiased (before the sign-problem approximation of step 8):

For a single site's two-valued field, the *exact* identity is
$\langle\psi_T|e^{-\Delta\tau\hat V_i}|\phi\rangle = \sum_{s=\pm1}\tfrac12\,\langle\psi_T|B_i(s)|\phi\rangle$.
Sampling $s$ from $P(s)\propto\tfrac12|\langle\psi_T|B_i(s)|\phi\rangle|$
("force bias") instead of uniformly, and reweighting by
$\tfrac12\big(|\langle\psi_T|B_i(+1)|\phi\rangle| + |\langle\psi_T|B_i(-1)|\phi\rangle|\big) / \langle\psi_T|\phi\rangle$,
reproduces the exact sum in expectation over the random choice, while
concentrating samples where the trial wavefunction says they matter. This
per-site ratio is exactly what `local_update_ratio` computes (see step 9);
`propagate_step!` samples each site's field this way and accumulates the
weight factor into `walker.weight`.

## 8. The sign problem and the constrained-path approximation

$\langle\psi_T|\phi\rangle$ is a real number that can be positive or
negative (there is no complex phase here, since the trial and the discrete
fields are both real) -- but the *walker weight* used for population
dynamics must stay $\ge0$ to make sense as a branching probability. Left
unconstrained, importance-sampled walkers eventually sample both signs and
the positive/negative contributions cancel in a sum whose *individual terms*
grow exponentially in magnitude with $\tau$ -- the fermion sign problem,
here in "real" (as opposed to complex-phase) form.

The **constrained-path approximation** (Zhang & Krakauer, *Phys. Rev. Lett.*
**90**, 136401 (2003)) controls this: after a full propagation step, if the
walker's overlap with $\psi_T$ has flipped sign relative to before the step,
its weight is set to 0 (the walker is discarded). This is exact if $\psi_T$
were the true ground state (whose overlap with $\Phi(\tau)$ never changes
sign along the correct path); for an approximate $\psi_T$ it introduces a
controlled, well-documented systematic bias, not a correctness bug. This is
what `propagate_step!` does at the end of each step (compare
`overlap_ref` to `overlap_new`). For complex trial wavefunctions / continuous
Hubbard-Stratonovich fields (not needed here, since the discrete Hirsch
transform and a real trial keep everything real), the general version of
this idea is the *phaseless* approximation, constraining the complex phase
rather than just the sign.

## 9. Fast local updates: Sherman-Morrison for a single-site field

A candidate field at site $i$ scales *only* row $i$ of $\phi$ by a factor
$\gamma$: $\phi' = B\phi$ with $B = I + (\gamma-1)e_ie_i^\top$, a rank-1
perturbation of the identity. Recomputing $\det(\psi_T^\dagger\phi')$ from
scratch is $O(N^3)$ -- expensive to do twice per site (once per candidate
field) for every walker at every step. But since $B$ is rank-1, so is the
update to $M=\psi_T^\dagger\phi$, and Sherman-Morrison gives the ratio in
$O(1)$ once $G$ is already known:

$$M' = M + (\gamma-1)\,u v^\top,\quad u=\psi_T[i,:]^\top,\ v=\phi[i,:]^\top$$

$$R := \frac{\det M'}{\det M} = 1+(\gamma-1)\,v^\top M^{-1}u = 1 + (\gamma-1)\,G_{ii}$$

using $v^\top M^{-1}u = [\phi M^{-1}\psi_T^\dagger]_{ii} = G_{ii}$ from the
definition of $G$ in step 5. (Sanity checks: $G_{ii}=1\Rightarrow R=\gamma$,
$G_{ii}=0\Rightarrow R=1$ -- scaling an orbital component the trial doesn't
see at all can't change the overlap.) This is `local_update_ratio`.

Once a field is chosen, $G$ itself must be updated to stay consistent with
the new $\phi'$. Applying the same Sherman-Morrison expansion to
$G=\phi M^{-1}\psi_T^\dagger$ (full derivation: expand
$G'=(\phi+(\gamma-1)e_iv^\top)(M^{-1}-\tfrac{\gamma-1}{R}M^{-1}uv^\top M^{-1})\psi_T^\dagger$
and simplify using $\phi M^{-1}\psi_T^\dagger=G$, $\phi M^{-1}u = G[:,i]$,
$v^\top M^{-1}\psi_T^\dagger = G[i,:]$) gives a rank-1, $O(N^2)$ update:

$$G'_{kl} = G_{kl} - \frac{\gamma-1}{R}\big(G_{ki}-\delta_{ki}\big)\,G_{il}$$

This is `rank1_update_greens_function!`. Together, `local_update_ratio` +
`rank1_update_greens_function!` replace an $O(N^3)$ determinant recompute per
candidate field with an $O(1)$ ratio and an $O(N^2)$ update -- the same
fast-update trick that makes production DQMC/CPMC codes practical. $G$ is
recomputed from scratch (the $O(N^3)$ way) only once per full step, right
after the first $K/2$ half-step (`propagate_step!`); the site sweep then uses
only the $O(1)$/$O(N^2)$ formulas above. The two dense $K/2$ half-steps
(matrix multiplies, $O(Ns^3)$ each at half filling) and that one $G$
recompute dominate the per-step cost, giving $O(Ns^3)$ total instead of the
$O(Ns\cdot N^3)$ of a naive per-site recompute.

## 10. Numerical stabilization

Repeated matrix multiplication of $\phi$ by non-unitary propagators makes
its columns increasingly ill-conditioned (some orbital components grow,
others shrink, over many steps) even though the column *space* they span is
what actually matters. Periodic thin-QR reorthonormalization
(`orthonormalize!`) fixes this. It requires **no weight compensation**: if
$\phi_{\rm raw}=QR$ and we store $Q$ going forward, then for the *next*
step's ratio, $\langle\psi_T|B(x)Q\rangle/\langle\psi_T|Q\rangle =
\langle\psi_T|B(x)\phi_{\rm raw}\rangle/\langle\psi_T|\phi_{\rm raw}\rangle$
exactly -- the $\det R$ factors introduced by discarding $R$ cancel between
numerator and denominator of the next ratio, *provided* every overlap is
always recomputed fresh from whatever matrix is currently stored (never
mixing a pre- and post-orthonormalization reference).

## 11. Population control

Over many steps, walker weights drift apart (some become negligible, a few
dominate), wasting population on near-zero-weight walkers. Periodic **comb
(systematic) resampling** (`population_control!`) replaces the population
with a new one of the same size, total weight preserved exactly, drawn with
a single random offset and evenly spaced "teeth" through the cumulative
weight -- unbiased and lower-variance than resampling each walker
independently. This introduces its own well-known (small, controllable via
resampling frequency) *population-control bias*, distinct from the
constrained-path bias.

## 12. Putting it together: `run_afqmc`

Per step: propagate every walker one Trotter step (§3-9, with the
constrained-path gate from §8), periodically reorthonormalize (§10) and
population-control (§11), then record the mixed-estimator energy (§6) after
an equilibration period. The reported `energy_mean`/`energy_err` use a
simple blocking analysis (`block_average`): split the post-equilibration
trace into blocks, treat block means as (approximately) independent samples,
and report their mean and standard error -- a standard way to get an honest
error bar from a correlated Markov-chain time series.

## Known approximations, for the record

- **Constrained-path bias** (§8): controlled, from the trial wavefunction not
  being exact.
- **Population-control bias** (§11): controlled by resampling frequency /
  population size.
- **Finite-$\Delta\tau$ Trotter error** (§3): $O(\Delta\tau^2)$ per step for
  the symmetric split; reduce `dtau` to check convergence.

None of these are implementation bugs -- they are the standard, documented
tradeoffs of the method, and are exactly why AFQMC results are always quoted
with a $\Delta\tau\to0$ / trial-wavefunction-quality discussion in the
literature.
