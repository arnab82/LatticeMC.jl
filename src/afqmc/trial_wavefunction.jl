struct TrialWavefunction
    phi_up::Matrix{Float64}
    phi_down::Matrix{Float64}
end

# bias is a tiny generic (non-bipartite-specific) diagonal perturbation used
# only to diagonalize K for the trial: many small lattices (e.g. an open
# 2x2 plaquette at half filling) have an exactly degenerate single-particle
# level at the Fermi surface, so "the lowest Nup orbitals" is not unique.
# An arbitrary pick within a degenerate shell is still a valid U=0 ground
# state (same energy), but it is a poor, arbitrarily-oriented reference for
# importance sampling / the constrained-path gate once U != 0. The bias
# lifts the degeneracy deterministically without perturbing propagation
# (lattice.K itself is untouched).
function build_trial_wavefunction(lattice::HubbardLattice, Nup::Int, Ndown::Int; bias::Float64=1e-4)
    0 <= Nup <= lattice.Ns || throw(ArgumentError("Nup must be between 0 and Ns"))
    0 <= Ndown <= lattice.Ns || throw(ArgumentError("Ndown must be between 0 and Ns"))

    K_trial = lattice.K + Diagonal(bias .* (1:lattice.Ns))
    F = eigen(Symmetric(K_trial))
    phi_up = Matrix(F.vectors[:, 1:Nup])
    phi_down = Matrix(F.vectors[:, 1:Ndown])
    return TrialWavefunction(phi_up, phi_down)
end
