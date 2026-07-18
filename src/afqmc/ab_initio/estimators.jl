function greens_function_ab_initio(phi::Matrix{ComplexF64}, psi_trial::Matrix{ComplexF64})
    Ns = size(phi, 1)
    N = size(phi, 2)
    N == 0 && return zeros(ComplexF64, Ns, Ns)
    M = psi_trial' * phi
    return phi * (M \ psi_trial')
end

function overlap_ab_initio(phi::Matrix{ComplexF64}, psi_trial::Matrix{ComplexF64})
    N = size(phi, 2)
    N == 0 && return one(ComplexF64)
    return det(psi_trial' * phi)
end

function walker_overlap_ab_initio(walker::AbInitioWalker, trial::AbInitioTrial)
    return overlap_ab_initio(walker.phi_up, trial.phi_up) * overlap_ab_initio(walker.phi_down, trial.phi_down)
end

# Generalized Wick's-theorem mixed-estimator local energy for a general
# two-electron tensor (theory: same Thouless-theorem G as the Hubbard case,
# generalized two-body contraction). For p,q,r,s spatial orbitals and G_up,
# G_down the per-spin Green's functions:
#   <c+_p c+_r c_s c_q>_same-spin  = G[q,p]G[s,r] - G[s,p]G[q,r]
#   <c+_p c+_r c_s c_q>_cross-spin = G_up[q,p]G_down[s,r]  (no exchange term:
#     different spins never contract against each other)
# Summed over both same-spin channels and both cross-spin orderings, this
# collapses to the compact form below with Gt = G_up + G_down (derivation
# in docs/src/afqmc_ab_initio_theory.md; cross-checked to match the Hubbard
# on-site special case in test/ab_initio.jl).
function local_energy_ab_initio(mi::MolecularIntegrals, walker::AbInitioWalker, trial::AbInitioTrial)
    G_up = greens_function_ab_initio(walker.phi_up, trial.phi_up)
    G_down = greens_function_ab_initio(walker.phi_down, trial.phi_down)
    Gt = G_up .+ G_down

    e1 = tr(mi.h1e * Gt)

    Ns = mi.Ns
    e2 = zero(ComplexF64)
    for p in 1:Ns, q in 1:Ns, r in 1:Ns, s in 1:Ns
        v = mi.h2e[p, q, r, s]
        v == 0.0 && continue
        e2 += v * (Gt[q, p] * Gt[s, r] - G_up[s, p] * G_up[q, r] - G_down[s, p] * G_down[q, r])
    end
    e2 *= 0.5

    return mi.E_nuc + e1 + e2
end

function mixed_energy_estimator_ab_initio(mi::MolecularIntegrals, walkers::Vector{AbInitioWalker}, trial::AbInitioTrial)
    numerator = zero(ComplexF64)
    denominator = 0.0
    for walker in walkers
        w = walker.weight
        w == 0.0 && continue
        numerator += w * local_energy_ab_initio(mi, walker, trial)
        denominator += w
    end
    denominator == 0.0 && return NaN
    return real(numerator / denominator)
end
