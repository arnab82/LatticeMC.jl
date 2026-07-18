using LinearAlgebra

# --- tiny exact-diagonalization reference for small Hubbard clusters ---
# Fock-space states for one spin species are represented as Ns-bit integers
# (bit i set <=> site i occupied). Fermion signs use the standard convention
# sign = (-1)^(number of occupied orbitals with index < i).
#
# Shared by test/afqmc.jl, test/afqmc_convergence.jl, and
# example/afqmc_phase_diagram.jl -- kept in one place since it's genuinely
# reused, not duplicated per file.

function bit_states(Ns::Int, Nparticles::Int)
    return [s for s in 0:(2^Ns - 1) if count_ones(s) == Nparticles]
end

function annihilate(s::Int, i::Int)
    ((s >> i) & 1) == 0 && return (0, 0)
    sgn = isodd(count_ones(s & ((1 << i) - 1))) ? -1 : 1
    return (sgn, s & ~(1 << i))
end

function create(s::Int, i::Int)
    ((s >> i) & 1) == 1 && return (0, 0)
    sgn = isodd(count_ones(s & ((1 << i) - 1))) ? -1 : 1
    return (sgn, s | (1 << i))
end

function ed_ground_state_energy(lattice, Nup::Int, Ndown::Int)
    Ns = lattice.Ns
    up_states = bit_states(Ns, Nup)
    down_states = bit_states(Ns, Ndown)
    up_index = Dict(s => k for (k, s) in enumerate(up_states))
    down_index = Dict(s => k for (k, s) in enumerate(down_states))
    ndown = length(down_states)
    dim = length(up_states) * ndown
    H = zeros(Float64, dim, dim)
    combined(iu, id) = (iu - 1) * ndown + id

    for (iu, su) in enumerate(up_states), (id, sd) in enumerate(down_states)
        n = combined(iu, id)
        e_int = 0.0
        for site in 0:Ns-1
            nu = (su >> site) & 1
            nd = (sd >> site) & 1
            e_int += lattice.U * nu * nd
        end
        H[n, n] += e_int
    end

    for i in 1:Ns, j in 1:Ns
        i == j && continue
        Kij = lattice.K[i, j]
        Kij == 0.0 && continue

        for (iu, su) in enumerate(up_states)
            sgn1, s1 = annihilate(su, j - 1)
            sgn1 == 0 && continue
            sgn2, s2 = create(s1, i - 1)
            sgn2 == 0 && continue
            haskey(up_index, s2) || continue
            iu2 = up_index[s2]
            for id in 1:ndown
                H[combined(iu2, id), combined(iu, id)] += Kij * sgn1 * sgn2
            end
        end

        for (id, sd) in enumerate(down_states)
            sgn1, s1 = annihilate(sd, j - 1)
            sgn1 == 0 && continue
            sgn2, s2 = create(s1, i - 1)
            sgn2 == 0 && continue
            haskey(down_index, s2) || continue
            id2 = down_index[s2]
            for iu in 1:length(up_states)
                H[combined(iu, id2), combined(iu, id)] += Kij * sgn1 * sgn2
            end
        end
    end

    return eigmin(Symmetric(H))
end
