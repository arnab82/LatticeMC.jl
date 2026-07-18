function greens_function(phi::Matrix{Float64}, psi_trial::Matrix{Float64})
    Ns = size(phi, 1)
    N = size(phi, 2)
    N == 0 && return zeros(Ns, Ns)
    M = psi_trial' * phi
    return phi * (M \ psi_trial')
end

function overlap(phi::Matrix{Float64}, psi_trial::Matrix{Float64})
    N = size(phi, 2)
    N == 0 && return 1.0
    return det(psi_trial' * phi)
end

function walker_overlap(walker::Walker, trial::TrialWavefunction)
    return overlap(walker.phi_up, trial.phi_up) * overlap(walker.phi_down, trial.phi_down)
end

# --- fast (rank-1) local updates ---
#
# A single-site diagonal update phi'[i,:] = gamma_factor * phi[i,:] (all other
# rows unchanged) is the one-body operator B = I + (gamma_factor-1) e_i e_i^T.
# Writing M = psi_trial' * phi, this is a rank-1 update of M, so by the
# Sherman-Morrison formula the overlap ratio is available in O(1) once G is
# known (no need to recompute a fresh N x N determinant for every candidate
# field):
#   R := <psi_trial|phi'> / <psi_trial|phi> = 1 + (gamma_factor - 1) * G[i, i]
# with G[i,i] evaluated *before* the update.
function local_update_ratio(G::Matrix{Float64}, i::Int, gamma_factor::Float64)
    return 1.0 + (gamma_factor - 1.0) * G[i, i]
end

# Updates G in place to match the row-i scaling used in local_update_ratio,
# given the R already computed for the same gamma_factor. Applying the same
# Sherman-Morrison expansion to G = phi * M^{-1} * psi_trial' gives an O(N^2)
# rank-1 (outer product) update in place of an O(N^3) recompute:
#   G'_{kl} = G_{kl} - [(gamma_factor - 1)/R] * (G_{ki} - delta_{ki}) * G_{il}
function rank1_update_greens_function!(G::Matrix{Float64}, i::Int, gamma_factor::Float64, R::Float64)
    coeff = (gamma_factor - 1.0) / R
    col = G[:, i]
    col[i] -= 1.0
    row = G[i, :]
    G .-= coeff .* (col * row')
    return G
end

function local_energy(lattice::HubbardLattice, walker::Walker, trial::TrialWavefunction)
    G_up = greens_function(walker.phi_up, trial.phi_up)
    G_down = greens_function(walker.phi_down, trial.phi_down)

    e_kinetic = tr(lattice.K * G_up) + tr(lattice.K * G_down)
    e_interaction = lattice.U * sum(G_up[i, i] * G_down[i, i] for i in 1:lattice.Ns)

    return e_kinetic + e_interaction
end

function mixed_energy_estimator(lattice::HubbardLattice, walkers::Vector{Walker}, trial::TrialWavefunction)
    numerator = 0.0
    denominator = 0.0
    for walker in walkers
        w = walker.weight
        w == 0.0 && continue
        numerator += w * local_energy(lattice, walker, trial)
        denominator += w
    end
    denominator == 0.0 && return NaN
    return numerator / denominator
end
