using LinearAlgebra

# Exact ground-state energy of the S=1/2 Heisenberg model H = J sum_<ij> S_i.S_j
# for a given bond list, by dense diagonalization in the Sz=0 sector (which
# contains the singlet ground state for a bipartite AFM). Bit i (0-based) of
# an Ns-bit state = spin up at site i+1. Small systems only (Sz=0 sector
# dimension C(Ns, Ns/2)); comfortable up to Ns ~ 12-14.
#
# S_i.S_j = S^z_i S^z_j + 1/2 (S^+_i S^-_j + S^-_i S^+_j):
#   diagonal    S^z_i S^z_j = +1/4 (parallel) or -1/4 (antiparallel)
#   off-diagonal flips an antiparallel pair with matrix element 1/2.

function heisenberg_ed_energy(bonds::Vector{Tuple{Int,Int}}, Ns::Int; J::Float64=1.0)
    iseven(Ns) || throw(ArgumentError("Sz=0 sector needs an even number of sites"))
    states = [s for s in 0:(2^Ns - 1) if count_ones(s) == Ns ÷ 2]
    index = Dict(s => k for (k, s) in enumerate(states))
    dim = length(states)
    H = zeros(Float64, dim, dim)

    bit(s, i) = (s >> (i - 1)) & 1        # 1-based site i -> its bit
    for (k, s) in enumerate(states)
        for (i, j) in bonds
            si = bit(s, i)
            sj = bit(s, j)
            if si == sj
                H[k, k] += 0.25 * J
            else
                H[k, k] -= 0.25 * J
                # off-diagonal: flip both spins (antiparallel -> the other order)
                s2 = s ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                k2 = index[s2]
                H[k2, k] += 0.5 * J
            end
        end
    end

    return eigmin(Symmetric(H))
end
