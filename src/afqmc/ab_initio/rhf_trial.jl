# Common supertype for ab initio trial wavefunctions. The propagator and
# driver are written against this, so they work with either a single-
# determinant trial (AbInitioTrial) or a multi-determinant one
# (MultiDetTrial, multidet_trial.jl) as long as each provides the
# overlap / mixed-Green's-function / mixed-local-energy methods.
abstract type AbstractAbInitioTrial end

struct AbInitioTrial <: AbstractAbInitioTrial
    phi_up::Matrix{ComplexF64}
    phi_down::Matrix{ComplexF64}
end

# Uniform accessor used by the driver for walker seeding / size checks / the
# default energy shift, so it doesn't have to special-case the trial type.
reference_determinant(trial::AbInitioTrial) = (trial.phi_up, trial.phi_down)

# Closed-shell restricted Hartree-Fock SCF. F[mu,nu] = h1e[mu,nu] +
# sum_{lambda,sigma} P[lambda,sigma] * (h2e[mu,nu,lambda,sigma]
# - 0.5*h2e[mu,lambda,nu,sigma]), density P = 2*C_occ*C_occ' (double
# occupancy), iterated to density convergence with linear mixing -- same
# iterate-to-self-consistency pattern as uhf_trial.jl's Hubbard UHF loop,
# generalized to the full two-electron tensor. Requires a closed-shell
# system (Nup == Ndown == Nocc).
# Returns `(C, E)` where C is the *full* Ns x Ns matrix of MO coefficients
# (all orbitals, occupied first -- the occupied block is C[:, 1:Nocc]), so it
# also serves as the basis transform to the RHF MO basis, and E is the total
# RHF energy.
function rhf_scf(mi::MolecularIntegrals, Nocc::Int; max_iter::Int=200, tol::Float64=1e-9, mix::Float64=0.5)
    Ns = mi.Ns
    F = copy(mi.h1e)
    P = zeros(Ns, Ns)
    C = Matrix{Float64}(I, Ns, Ns)

    for _ in 1:max_iter
        C = Matrix(eigen(Symmetric(F)).vectors)
        Cocc = C[:, 1:Nocc]
        P_new = 2.0 .* (Cocc * Cocc')

        delta = maximum(abs.(P_new .- P))
        P = mix .* P_new .+ (1 - mix) .* P

        F = copy(mi.h1e)
        for lam in 1:Ns, sig in 1:Ns
            Plsig = P[lam, sig]
            Plsig == 0.0 && continue
            for mu in 1:Ns, nu in 1:Ns
                F[mu, nu] += Plsig * (mi.h2e[mu, nu, lam, sig] - 0.5 * mi.h2e[mu, lam, nu, sig])
            end
        end

        delta < tol && break
    end

    E_elec = 0.5 * sum(P .* (mi.h1e .+ F))
    return C, E_elec + mi.E_nuc
end

# Returns a complex TrialWavefunction (real RHF orbitals promoted to
# ComplexF64) with phi_up == phi_down == the occupied RHF orbitals, the
# standard closed-shell trial for ab initio AFQMC.
function build_rhf_trial(mi::MolecularIntegrals, Nup::Int, Ndown::Int; kwargs...)
    Nup == Ndown || throw(ArgumentError("build_rhf_trial requires a closed-shell system (Nup == Ndown)"))
    C, _ = rhf_scf(mi, Nup; kwargs...)
    phi = Matrix{ComplexF64}(C[:, 1:Nup])
    return AbInitioTrial(phi, copy(phi))
end
