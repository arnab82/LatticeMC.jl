apply_one_body_ab_initio!(phi::Matrix{ComplexF64}, expH1_half::Matrix{Float64}) = (phi .= expH1_half * phi)

# Advances one walker by a symmetric-Trotter step
# exp(-dtau*h1e_mod/2) * exp(-dtau*V) * exp(-dtau*h1e_mod/2), where the
# two-body factor uses the continuous Hubbard-Stratonovich transform: sample
# the full Cholesky-vector-length Gaussian x ~ N(0, I), form the single
# random one-body rotation dK = sum_gamma x_gamma * L^gamma, and apply
# exp(i*sqrt(dtau)*dK) as one matrix exponential -- exact (a multivariate
# Gaussian-integral identity), not a per-vector product of exponentials.
#
# Deliberately unbiased (Tier 1): x is sampled directly from N(0, I), not
# force-biased/mean-field-shifted toward the current walker's density (the
# standard variance-reduction refinement used in production codes, noted as
# future work in docs/afqmc_ab_initio_theory.md). Correctness is instead
# carried entirely by the phaseless gate below (the complex generalization
# of the real sign-flip gate used for Hubbard, theory doc section 8):
# w *= |I(x)| * max(0, cos(arg(I(x)))), I(x) = overlap_new/overlap_ref.
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
