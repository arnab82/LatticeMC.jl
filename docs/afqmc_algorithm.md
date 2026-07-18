# AFQMC algorithm (pseudocode)

Formal statement of the algorithm derived in [`afqmc_theory.md`](afqmc_theory.md)
and implemented in `src/afqmc/` (file-by-file map in
[`afqmc_implementation.md`](afqmc_implementation.md)). Each block below is
self-contained; read them top to bottom (Algorithm 1 calls Algorithm 2 calls
Algorithm 3).

## Algorithm 1: `run_afqmc` -- outer population loop

*Theory §12. Implemented in `driver.jl`.*

```
Input:  lattice (Ns, K, U), trial psi_T, Nup, Ndown,
        dtau, num_walkers, num_steps, equilibration_steps,
        stabilize_every, pop_control_every
Output: energy_trace, energy_mean, energy_err

 1: walkers <- num_walkers copies of (phi_up, phi_down) <- (psi_T.phi_up, psi_T.phi_down), weight <- 1
 2: expK_half <- exp(-dtau * K / 2)                          # Alg 2, once
 3: gamma <- hs_gamma(lattice, dtau)                          # theory §4
 4: energy_trace <- []
 5: for step = 1 .. num_steps:
 6:     for each walker in walkers:
 7:         propagate_step!(walker, lattice, psi_T, expK_half, gamma, dtau)   # Algorithm 2
 8:     if step mod stabilize_every == 0:
 9:         for each walker with weight > 0: orthonormalize!(walker)          # theory §10
10:     if step mod pop_control_every == 0:
11:         population_control!(walkers)                                     # Algorithm 3
12:     if step > equilibration_steps:
13:         append mixed_energy_estimator(lattice, walkers, psi_T) to energy_trace
14: (energy_mean, energy_err) <- block_average(energy_trace)
15: return energy_trace, energy_mean, energy_err
```

## Algorithm 2: `propagate_step!` -- one walker, one Trotter step

*Theory §3-§9. Implemented in `propagator.jl`, using the fast-update pair
from `estimators.jl`.*

```
Input:  walker (phi_up, phi_down, weight), lattice, psi_T, expK_half, gamma, dtau
Output: walker mutated in place

 1: if walker.weight == 0: return                              # dead walker, skip

 2: overlap_ref <- walker_overlap(walker, psi_T)                # theory §5, reference for line 20

 3: phi_up   <- expK_half * phi_up                              # first K/2 half-step
 4: phi_down <- expK_half * phi_down

 5: G_up   <- greens_function(phi_up, psi_T.phi_up)              # ONE O(N^3) recompute per step
 6: G_down <- greens_function(phi_down, psi_T.phi_down)

 7: charge <- exp(-dtau * U / 2)
 8: (gamma_up[+1], gamma_down[+1]) <- (charge * e^{+gamma}, charge * e^{-gamma})
 9: (gamma_up[-1], gamma_down[-1]) <- (charge * e^{-gamma}, charge * e^{+gamma})

10: for i = 1 .. Ns:                                             # auxiliary-field sweep
11:     for s in {+1, -1}:
12:         R_up[s]   <- local_update_ratio(G_up,   i, gamma_up[s])     # O(1), theory §9
13:         R_down[s] <- local_update_ratio(G_down, i, gamma_down[s])   # O(1)
14:         ratio[s]  <- R_up[s] * R_down[s]
15:     total <- |ratio[+1]| + |ratio[-1]|
16:     if total == 0: walker.weight <- 0; return                # degenerate, discard
17:     s_chosen <- +1 with prob |ratio[+1]| / total, else -1     # force-bias sample, theory §7
18:     phi_up[i,:]   *= gamma_up[s_chosen]
19:     phi_down[i,:] *= gamma_down[s_chosen]
20:     rank1_update_greens_function!(G_up,   i, gamma_up[s_chosen],   R_up[s_chosen])    # O(N^2)
21:     rank1_update_greens_function!(G_down, i, gamma_down[s_chosen], R_down[s_chosen])  # O(N^2)
22:     walker.weight *= 0.5 * total                              # magnitude only, stays >= 0

23: phi_up   <- expK_half * phi_up                              # second K/2 half-step
24: phi_down <- expK_half * phi_down

25: overlap_new <- walker_overlap(walker, psi_T)
26: if overlap_ref == 0 or sign(overlap_new) != sign(overlap_ref):
27:     walker.weight <- 0                                       # constrained-path gate, theory §8
```

Cost: line 5-6 and lines 3-4/23-24 are each $O(Ns^3)$ (at half filling);
the sweep (lines 10-22) is $Ns$ iterations of $O(1)+O(N^2)$ work, i.e.
$O(Ns \cdot N^2) \le O(Ns^3)$. Total: $O(Ns^3)$ per step, matching production
DQMC/CPMC codes.

## Algorithm 3: `population_control!` -- comb resampling

*Theory §11. Implemented in `population_control.jl`.*

```
Input:  walkers (length Nw, weights w_1..w_Nw)
Output: walkers mutated in place, same length, weights all equal

 1: W <- sum(w_k)
 2: if W <= 0: return                                            # everything died
 3: xi <- uniform(0, 1)                                          # one shared random offset
 4: cumulative <- cumsum(w_1, ..., w_Nw)
 5: idx <- 1
 6: for j = 1 .. Nw:
 7:     target <- (j - 1 + xi) / Nw * W
 8:     while idx < Nw and cumulative[idx] < target: idx <- idx + 1
 9:     new_walkers[j] <- copy(phi_up[idx], phi_down[idx]), weight <- W / Nw
10: walkers <- new_walkers
```

Total weight $W$ is preserved exactly (up to floating point); every walker
in the new population has weight $W/N_w$.

## Complexity summary

| Step | Cost | Where |
|---|---|---|
| $K/2$ half-step (x2) | $O(Ns^3)$ | `propagator.jl`, `apply_one_body!` |
| $G$ recompute (x1/step) | $O(Ns^3)$ | `estimators.jl`, `greens_function` |
| Per-site ratio (x2, x$Ns$) | $O(1)$ each | `estimators.jl`, `local_update_ratio` |
| Per-site $G$ update (x1, x$Ns$) | $O(N^2)$ each | `estimators.jl`, `rank1_update_greens_function!` |
| Orthonormalize (periodic) | $O(Ns \cdot N^2)$ (thin QR) | `walker.jl` |
| Population control (periodic) | $O(N_w \log N_w)$ (sort-free comb) | `population_control.jl` |

**Total per full propagation step, per walker: $O(Ns^3)$.**
