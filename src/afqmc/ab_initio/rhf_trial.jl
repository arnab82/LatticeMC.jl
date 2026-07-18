struct AbInitioTrial
    phi_up::Matrix{ComplexF64}
    phi_down::Matrix{ComplexF64}
end

# Closed-shell restricted Hartree-Fock SCF. F[mu,nu] = h1e[mu,nu] +
# sum_{lambda,sigma} P[lambda,sigma] * (h2e[mu,nu,lambda,sigma]
# - 0.5*h2e[mu,lambda,nu,sigma]), density P = 2*C_occ*C_occ' (double
# occupancy), iterated to density convergence with linear mixing -- same
# iterate-to-self-consistency pattern as uhf_trial.jl's Hubbard UHF loop,
# generalized to the full two-electron tensor. Requires a closed-shell
# system (Nup == Ndown == Nocc).
function rhf_scf(mi::MolecularIntegrals, Nocc::Int; max_iter::Int=200, tol::Float64=1e-9, mix::Float64=0.5)
    Ns = mi.Ns
    F = copy(mi.h1e)
    P = zeros(Ns, Ns)
    C = zeros(Ns, Nocc)

    for _ in 1:max_iter
        C = Matrix(eigen(Symmetric(F)).vectors[:, 1:Nocc])
        P_new = 2.0 .* (C * C')

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
    phi = Matrix{ComplexF64}(C)
    return AbInitioTrial(phi, copy(phi))
end
