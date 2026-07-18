# Unrestricted Hartree-Fock SCF: separate spatial orbitals per spin, unlike
# rhf_scf's shared phi_up == phi_down. General Fock matrices (Coulomb from
# the *total* density, exchange only from the *same-spin* density -- Pauli
# exclusion only pairs same-spin electrons):
#   F_up[mu,nu]   = h1e[mu,nu] + sum_{lam,sig} P_total[lam,sig]*h2e[mu,nu,lam,sig]
#                                - sum_{lam,sig} P_up[lam,sig]*h2e[mu,lam,nu,sig]
#   F_down[mu,nu] = (same, P_down in the exchange term)
# Reduces exactly to rhf_scf's Fock matrix when P_up == P_down == P_total/2
# (verified in test/ab_initio.jl) -- UHF is a strict variational
# generalization of RHF, so its converged energy can only be <= RHF's.
#
# Symmetry-breaking seed: currently a simple alternating-site-parity density
# bias (site i favors up for odd i, down for even i), specific to the linear
# H-chains build_h_chain_sto3g produces -- analogous to uhf_trial.jl's
# bipartite_coloring seed for Hubbard, but simplified since chains are the
# only ab initio geometry builder so far. Would need generalizing (e.g. a
# geometry-aware coloring) for a future non-chain ab initio builder.
function uhf_scf(mi::MolecularIntegrals, Nup::Int, Ndown::Int;
                  m0::Float64=0.3, max_iter::Int=300, tol::Float64=1e-9, mix::Float64=0.4)
    Ns = mi.Ns
    seed = [(-1)^(i + 1) for i in 1:Ns]
    n_avg = (Nup + Ndown) / (2 * Ns)
    n_up_diag = clamp.(n_avg .+ 0.5 .* m0 .* seed, 0.0, 1.0)
    n_down_diag = clamp.(n_avg .- 0.5 .* m0 .* seed, 0.0, 1.0)

    P_up = Matrix(Diagonal(n_up_diag))
    P_down = Matrix(Diagonal(n_down_diag))
    C_up = zeros(Ns, Nup)
    C_down = zeros(Ns, Ndown)

    for _ in 1:max_iter
        P_total = P_up .+ P_down
        F_up = copy(mi.h1e)
        F_down = copy(mi.h1e)
        for lam in 1:Ns, sig in 1:Ns
            Pt = P_total[lam, sig]
            Pu = P_up[lam, sig]
            Pd = P_down[lam, sig]
            (Pt == 0.0 && Pu == 0.0 && Pd == 0.0) && continue
            for mu in 1:Ns, nu in 1:Ns
                coulomb = Pt * mi.h2e[mu, nu, lam, sig]
                F_up[mu, nu] += coulomb - Pu * mi.h2e[mu, lam, nu, sig]
                F_down[mu, nu] += coulomb - Pd * mi.h2e[mu, lam, nu, sig]
            end
        end

        C_up = Matrix(eigen(Symmetric(F_up)).vectors[:, 1:Nup])
        C_down = Matrix(eigen(Symmetric(F_down)).vectors[:, 1:Ndown])

        P_up_new = C_up * C_up'
        P_down_new = C_down * C_down'

        delta = max(maximum(abs.(P_up_new .- P_up); init=0.0),
                    maximum(abs.(P_down_new .- P_down); init=0.0))

        P_up = mix .* P_up_new .+ (1 - mix) .* P_up
        P_down = mix .* P_down_new .+ (1 - mix) .* P_down

        delta < tol && break
    end

    P_total = P_up .+ P_down
    F_up = copy(mi.h1e)
    F_down = copy(mi.h1e)
    for lam in 1:Ns, sig in 1:Ns
        Pt = P_total[lam, sig]
        Pu = P_up[lam, sig]
        Pd = P_down[lam, sig]
        for mu in 1:Ns, nu in 1:Ns
            coulomb = Pt * mi.h2e[mu, nu, lam, sig]
            F_up[mu, nu] += coulomb - Pu * mi.h2e[mu, lam, nu, sig]
            F_down[mu, nu] += coulomb - Pd * mi.h2e[mu, lam, nu, sig]
        end
    end
    E_elec = 0.5 * (sum(P_up .* (mi.h1e .+ F_up)) + sum(P_down .* (mi.h1e .+ F_down)))

    return C_up, C_down, E_elec + mi.E_nuc
end

# Returns a complex AbInitioTrial with (generally different) phi_up, phi_down
# -- drop-in for build_rhf_trial everywhere an AbInitioTrial is expected.
# Useful when a single RHF determinant is a poor reference (see
# docs/afqmc_ab_initio_theory.md section 6 for the measured H4 case this was
# built to address).
function build_uhf_trial_ab_initio(mi::MolecularIntegrals, Nup::Int, Ndown::Int; kwargs...)
    C_up, C_down, _ = uhf_scf(mi, Nup, Ndown; kwargs...)
    return AbInitioTrial(Matrix{ComplexF64}(C_up), Matrix{ComplexF64}(C_down))
end
