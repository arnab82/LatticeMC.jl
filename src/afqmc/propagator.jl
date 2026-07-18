function build_expK_half(lattice::HubbardLattice, dtau::Float64)
    return exp(-dtau / 2 .* lattice.K)
end

# Discrete Hirsch (Hubbard-Stratonovich) transform of the on-site interaction:
#   n_i^up n_i^down = (n_i^up + n_i^down)/2 - (n_i^up - n_i^down)^2 / 2
# exp(-dtau*U*n_i^up*n_i^down) = exp(-dtau*U/2*(n_i^up+n_i^down)) *
#     (1/2) sum_{s=+-1} exp(gamma*s*(n_i^up - n_i^down)),   cosh(gamma) = exp(dtau*U/2)
# Only valid for the repulsive case U >= 0; the attractive Hubbard model needs
# a different (charge-channel) transform, not implemented here.
function hs_gamma(lattice::HubbardLattice, dtau::Float64)
    lattice.U >= 0 || throw(ArgumentError("discrete Hirsch transform here requires U >= 0 (repulsive Hubbard)"))
    lattice.U == 0 && return 0.0
    return acosh(exp(dtau * lattice.U / 2))
end

apply_one_body!(phi::Matrix{Float64}, expK_half::Matrix{Float64}) = (phi .= expK_half * phi)

# Advances a single walker by one symmetric-Trotter imaginary-time step
# exp(-dtau*K/2) * exp(-dtau*V) * exp(-dtau*K/2), sampling one discrete
# auxiliary field per site with force-bias importance sampling from the trial
# wavefunction, then applying the constrained-path gate: if the walker's
# overlap with the trial wavefunction has flipped sign over the full step,
# its weight is zeroed (this is the real-valued analogue of the general
# complex phaseless approximation).
#
# The auxiliary-field sweep uses the rank-1 Green's-function update
# (local_update_ratio / rank1_update_greens_function! in estimators.jl): G is
# computed once per step (right after the first K/2 half-step, O(N^3)), and
# each site's candidate-field overlap ratios and the resulting G update are
# then O(1) and O(N^2) respectively, instead of an O(N^3) determinant
# recompute per candidate. The two dense K/2 half-steps (O(Ns^3) each) and
# the single G recompute dominate, giving O(Ns^3) total per full step instead
# of the O(Ns * N^3) of a naive per-site determinant recompute.
function propagate_step!(walker::Walker, lattice::HubbardLattice, trial::TrialWavefunction,
                          expK_half::Matrix{Float64}, gamma::Float64, dtau::Float64)
    walker.weight == 0.0 && return walker

    overlap_ref = walker_overlap(walker, trial)

    apply_one_body!(walker.phi_up, expK_half)
    apply_one_body!(walker.phi_down, expK_half)

    G_up = greens_function(walker.phi_up, trial.phi_up)
    G_down = greens_function(walker.phi_down, trial.phi_down)

    charge_factor = exp(-dtau * lattice.U / 2)
    gamma_up_plus = charge_factor * exp(gamma)
    gamma_down_plus = charge_factor * exp(-gamma)
    gamma_up_minus = gamma_down_plus
    gamma_down_minus = gamma_up_plus

    for i in 1:lattice.Ns
        R_up_plus = local_update_ratio(G_up, i, gamma_up_plus)
        R_down_plus = local_update_ratio(G_down, i, gamma_down_plus)
        ratio_plus = R_up_plus * R_down_plus

        R_up_minus = local_update_ratio(G_up, i, gamma_up_minus)
        R_down_minus = local_update_ratio(G_down, i, gamma_down_minus)
        ratio_minus = R_up_minus * R_down_minus

        w_plus = abs(ratio_plus)
        w_minus = abs(ratio_minus)
        total = w_plus + w_minus
        if total == 0.0
            walker.weight = 0.0
            return walker
        end

        if rand() < w_plus / total
            walker.phi_up[i, :] .*= gamma_up_plus
            walker.phi_down[i, :] .*= gamma_down_plus
            rank1_update_greens_function!(G_up, i, gamma_up_plus, R_up_plus)
            rank1_update_greens_function!(G_down, i, gamma_down_plus, R_down_plus)
        else
            walker.phi_up[i, :] .*= gamma_up_minus
            walker.phi_down[i, :] .*= gamma_down_minus
            rank1_update_greens_function!(G_up, i, gamma_up_minus, R_up_minus)
            rank1_update_greens_function!(G_down, i, gamma_down_minus, R_down_minus)
        end

        walker.weight *= 0.5 * total
    end

    apply_one_body!(walker.phi_up, expK_half)
    apply_one_body!(walker.phi_down, expK_half)

    overlap_new = walker_overlap(walker, trial)
    if overlap_ref == 0.0 || sign(overlap_new) != sign(overlap_ref)
        walker.weight = 0.0
    end

    return walker
end
