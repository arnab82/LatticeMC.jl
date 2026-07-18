apply_one_body_ab_initio!(phi::Matrix{ComplexF64}, expH1_half::Matrix{Float64}) = (phi .= expH1_half * phi)

# Advances one walker by a symmetric-Trotter step
# exp(-dtau*h1e_mod/2) * exp(-dtau*V) * exp(-dtau*h1e_mod/2), where the
# two-body factor uses the continuous Hubbard-Stratonovich transform: sample
# the full Cholesky-vector-length Gaussian x ~ N(0, I), form the single
# random one-body rotation dK = sum_gamma x_gamma * L^gamma, and apply
# exp(i*sqrt(dtau)*dK) as one matrix exponential -- exact (a multivariate
# Gaussian-integral identity), not a per-vector product of exponentials.
#
# Deliberately unbiased: x is sampled directly from N(0, I), not shifted
# toward a mean field. See propagate_step_ab_initio_force_bias! below for
# the variance-reduced alternative, and docs/afqmc_ab_initio_theory.md
# section 7.1 for why force bias helps variance but not the phaseless
# approximation's own asymptotic bias (measured, not assumed). Correctness
# here is carried entirely by the phaseless gate below (the complex
# generalization of the real sign-flip gate used for Hubbard, theory doc
# section 8): w *= |I(x)| * max(0, cos(arg(I(x)))), I(x) = overlap_new/overlap_ref.
function propagate_step_ab_initio!(walker::AbInitioWalker, mi::MolecularIntegrals, trial::AbInitioTrial,
                                    expH1_half::Matrix{Float64}, Ls::Vector{Matrix{Float64}}, dtau::Float64)
    walker.weight == 0.0 && return walker

    overlap_ref = walker_overlap_ab_initio(walker, trial)

    apply_one_body_ab_initio!(walker.phi_up, expH1_half)
    apply_one_body_ab_initio!(walker.phi_down, expH1_half)

    Ns = mi.Ns
    Nchol = length(Ls)
    x = randn(Nchol)
    dK = zeros(Float64, Ns, Ns)
    for gamma in 1:Nchol
        dK .+= x[gamma] .* Ls[gamma]
    end

    prop = exp(im * sqrt(dtau) .* dK)
    walker.phi_up = prop * walker.phi_up
    walker.phi_down = prop * walker.phi_down

    apply_one_body_ab_initio!(walker.phi_up, expH1_half)
    apply_one_body_ab_initio!(walker.phi_down, expH1_half)

    overlap_new = walker_overlap_ab_initio(walker, trial)

    if overlap_ref == 0
        walker.weight = 0.0
        return walker
    end

    I_ratio = overlap_new / overlap_ref
    walker.weight *= abs(I_ratio) * max(0.0, cos(angle(I_ratio)))

    return walker
end

# Static (trial-based) force-bias shift, one value per Cholesky vector:
# f_gamma = -sqrt(dtau) * Re[tr(L^gamma * G_trial)], G_trial the trial's own
# mixed Green's function (evaluated once; NOT re-evaluated per walker per
# step -- see the note on propagate_step_ab_initio_force_bias! below for why
# that distinction matters).
function force_bias_shift(trial::AbInitioTrial, Ls::Vector{Matrix{Float64}}, dtau::Float64)
    G_up = greens_function_ab_initio(trial.phi_up, trial.phi_up)
    G_down = greens_function_ab_initio(trial.phi_down, trial.phi_down)
    Gt = G_up .+ G_down
    return [-sqrt(dtau) * real(tr(L * Gt)) for L in Ls]
end

# Variance-reduced variant of propagate_step_ab_initio!: samples the
# Cholesky-vector Gaussian from N(fbar, I) instead of N(0, I), reweighting
# by the exact Radon-Nikodym factor exp(-|fbar|^2/2 - fbar.xi) for that
# change of sampling distribution (xi = x - fbar, the realized fluctuation
# around the shift) -- unbiased for *any* fbar, by a change-of-variables
# identity that introduces no approximation of its own; only the choice of
# fbar affects variance, never correctness (theory doc section 7.1).
#
# fbar must be precomputed by force_bias_shift and passed in -- deliberately
# a *static* shift (computed once from the trial, not from each walker's own
# evolving state every step). A walker-adaptive shift was tried first and
# measured to *increase* variance rather than reduce it: the shift itself
# becomes a fluctuating quantity correlated with the walker's own noise,
# adding a new noise source rather than removing one. The static shift
# avoids that and gives a modest, genuine variance reduction (theory doc
# section 7.1) -- it does not change the phaseless approximation's own
# asymptotic bias, which is not a variance effect.
function propagate_step_ab_initio_force_bias!(walker::AbInitioWalker, mi::MolecularIntegrals, trial::AbInitioTrial,
                                                expH1_half::Matrix{Float64}, Ls::Vector{Matrix{Float64}},
                                                dtau::Float64, fbar::Vector{Float64})
    walker.weight == 0.0 && return walker

    overlap_ref = walker_overlap_ab_initio(walker, trial)

    apply_one_body_ab_initio!(walker.phi_up, expH1_half)
    apply_one_body_ab_initio!(walker.phi_down, expH1_half)

    Ns = mi.Ns
    Nchol = length(Ls)
    xi = randn(Nchol)
    x = fbar .+ xi
    dK = zeros(Float64, Ns, Ns)
    for gamma in 1:Nchol
        dK .+= x[gamma] .* Ls[gamma]
    end

    prop = exp(im * sqrt(dtau) .* dK)
    walker.phi_up = prop * walker.phi_up
    walker.phi_down = prop * walker.phi_down

    apply_one_body_ab_initio!(walker.phi_up, expH1_half)
    apply_one_body_ab_initio!(walker.phi_down, expH1_half)

    overlap_new = walker_overlap_ab_initio(walker, trial)

    if overlap_ref == 0
        walker.weight = 0.0
        return walker
    end

    I_ratio = overlap_new / overlap_ref
    radon_nikodym = exp(-0.5 * sum(abs2, fbar) - dot(fbar, xi))
    walker.weight *= abs(I_ratio) * max(0.0, cos(angle(I_ratio))) * radon_nikodym

    return walker
end
