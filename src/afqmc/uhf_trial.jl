# BFS 2-coloring of the lattice graph defined by K's nonzero pattern.
# Returns a +-1 sublattice assignment, or `nothing` if the graph has an odd
# cycle (e.g. an odd-length PBC ring), where a bipartite Neel/AFM seed isn't
# well-defined.
function bipartite_coloring(K::Matrix{Float64})
    Ns = size(K, 1)
    color = zeros(Int, Ns)
    for start in 1:Ns
        color[start] != 0 && continue
        color[start] = 1
        queue = [start]
        while !isempty(queue)
            u = popfirst!(queue)
            for v in 1:Ns
                (v == u || K[u, v] == 0.0) && continue
                if color[v] == 0
                    color[v] = -color[u]
                    push!(queue, v)
                elseif color[v] == color[u]
                    return nothing
                end
            end
        end
    end
    return color
end

# Self-consistent unrestricted Hartree-Fock trial wavefunction with an
# antiferromagnetic (Neel) seed: each spin species sees a one-body potential
# from the *other* spin's mean-field density, U*n_down for the up channel and
# U*n_up for the down channel. Returns a plain TrialWavefunction, a drop-in
# for build_trial_wavefunction at every call site (run_afqmc, tests,
# examples). Useful at large U, where the paramagnetic free-fermion trial
# from build_trial_wavefunction is known to be a poor reference and AFM
# correlations dominate; at U=0 the two coincide (no mean field to seed).
function build_uhf_trial(lattice::HubbardLattice, Nup::Int, Ndown::Int;
                          m0::Float64=0.5, max_iter::Int=200, tol::Float64=1e-8,
                          bias::Float64=1e-4)
    0 <= Nup <= lattice.Ns || throw(ArgumentError("Nup must be between 0 and Ns"))
    0 <= Ndown <= lattice.Ns || throw(ArgumentError("Ndown must be between 0 and Ns"))

    Ns = lattice.Ns
    sublattice = bipartite_coloring(lattice.K)
    sublattice === nothing && throw(ArgumentError(
        "lattice graph is not bipartite (e.g. an odd-length PBC ring); " *
        "an antiferromagnetic seed is not well-defined for this geometry"))

    n_up = [0.5 + 0.5 * m0 * sublattice[i] for i in 1:Ns]
    n_down = [0.5 - 0.5 * m0 * sublattice[i] for i in 1:Ns]

    bias_diag = Diagonal(bias .* (1:Ns))
    phi_up = zeros(Ns, Nup)
    phi_down = zeros(Ns, Ndown)

    for _ in 1:max_iter
        H_up = lattice.K + Diagonal(lattice.U .* n_down) + bias_diag
        H_down = lattice.K + Diagonal(lattice.U .* n_up) + bias_diag

        phi_up = Matrix(eigen(Symmetric(H_up)).vectors[:, 1:Nup])
        phi_down = Matrix(eigen(Symmetric(H_down)).vectors[:, 1:Ndown])

        n_up_new = vec(sum(phi_up .^ 2, dims=2))
        n_down_new = vec(sum(phi_down .^ 2, dims=2))

        delta = max(maximum(abs.(n_up_new .- n_up); init=0.0),
                    maximum(abs.(n_down_new .- n_down); init=0.0))

        n_up = 0.5 .* n_up_new .+ 0.5 .* n_up
        n_down = 0.5 .* n_down_new .+ 0.5 .* n_down

        delta < tol && break
    end

    return TrialWavefunction(phi_up, phi_down)
end
