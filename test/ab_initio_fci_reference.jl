using LinearAlgebra

# --- general (not Hubbard-diagonal-only) exact diagonalization for small
# ab initio Hamiltonians, via direct second-quantized operator application
# on the same bitmask Fock-space representation as ed_reference.jl. ---
#
# H = E_nuc + sum_pq h1e[p,q] c+_p c_q
#           + 1/2 sum_pqrs h2e[p,q,r,s] (c+_p c+_r c_s c_q)_same-spin, both spins
#           +     sum_pqrs h2e[p,q,r,s] (c+_p c_q)_up (c+_r c_s)_down
#
# (the cross-spin up/down terms of the general spin-orbital sum combine,
# using h2e[p,q,r,s] = h2e[r,s,p,q], into a single un-halved density-density
# term -- this is the standard decomposition, and exactly reproduces the
# Hubbard model's U*n_up*n_down interaction as the special case
# h2e[i,i,i,i] = U, all other entries zero, which is used as a cross-check
# against ed_reference.jl's independently-validated Hubbard ED below).

function ab_initio_fci_ground_state_energy(h1e::Matrix{Float64}, h2e::Array{Float64,4},
                                            Nup::Int, Ndown::Int; E_nuc::Float64=0.0)
    Ns = size(h1e, 1)
    up_states = bit_states(Ns, Nup)
    down_states = bit_states(Ns, Ndown)
    up_index = Dict(s => k for (k, s) in enumerate(up_states))
    down_index = Dict(s => k for (k, s) in enumerate(down_states))
    nup = length(up_states)
    ndown = length(down_states)
    dim = nup * ndown
    H = zeros(Float64, dim, dim)
    combined(iu, id) = (iu - 1) * ndown + id

    # one-body, both spins
    for (iu, su) in enumerate(up_states), p in 0:Ns-1, q in 0:Ns-1
        h1e[p+1, q+1] == 0.0 && continue
        sgn1, s1 = annihilate(su, q)
        sgn1 == 0 && continue
        sgn2, s2 = create(s1, p)
        sgn2 == 0 && continue
        haskey(up_index, s2) || continue
        iu2 = up_index[s2]
        coeff = h1e[p+1, q+1] * sgn1 * sgn2
        for id in 1:ndown
            H[combined(iu2, id), combined(iu, id)] += coeff
        end
    end
    for (id, sd) in enumerate(down_states), p in 0:Ns-1, q in 0:Ns-1
        h1e[p+1, q+1] == 0.0 && continue
        sgn1, s1 = annihilate(sd, q)
        sgn1 == 0 && continue
        sgn2, s2 = create(s1, p)
        sgn2 == 0 && continue
        haskey(down_index, s2) || continue
        id2 = down_index[s2]
        coeff = h1e[p+1, q+1] * sgn1 * sgn2
        for iu in 1:nup
            H[combined(iu, id2), combined(iu, id)] += coeff
        end
    end

    # two-body, same spin (up-up and down-down), with the 1/2 prefactor
    for (iu, su) in enumerate(up_states), p in 0:Ns-1, q in 0:Ns-1, r in 0:Ns-1, s in 0:Ns-1
        v = h2e[p+1, q+1, r+1, s+1]
        v == 0.0 && continue
        sgn1, s1 = annihilate(su, q)
        sgn1 == 0 && continue
        sgn2, s2 = annihilate(s1, s)
        sgn2 == 0 && continue
        sgn3, s3 = create(s2, r)
        sgn3 == 0 && continue
        sgn4, s4 = create(s3, p)
        sgn4 == 0 && continue
        haskey(up_index, s4) || continue
        iu2 = up_index[s4]
        coeff = 0.5 * v * sgn1 * sgn2 * sgn3 * sgn4
        for id in 1:ndown
            H[combined(iu2, id), combined(iu, id)] += coeff
        end
    end
    for (id, sd) in enumerate(down_states), p in 0:Ns-1, q in 0:Ns-1, r in 0:Ns-1, s in 0:Ns-1
        v = h2e[p+1, q+1, r+1, s+1]
        v == 0.0 && continue
        sgn1, s1 = annihilate(sd, q)
        sgn1 == 0 && continue
        sgn2, s2 = annihilate(s1, s)
        sgn2 == 0 && continue
        sgn3, s3 = create(s2, r)
        sgn3 == 0 && continue
        sgn4, s4 = create(s3, p)
        sgn4 == 0 && continue
        haskey(down_index, s4) || continue
        id2 = down_index[s4]
        coeff = 0.5 * v * sgn1 * sgn2 * sgn3 * sgn4
        for iu in 1:nup
            H[combined(iu, id2), combined(iu, id)] += coeff
        end
    end

    # two-body, cross spin (up density * down density), no extra prefactor
    for (iu, su) in enumerate(up_states), p in 0:Ns-1, q in 0:Ns-1
        sgn1, s1 = annihilate(su, q)
        sgn1 == 0 && continue
        sgn2, s2 = create(s1, p)
        sgn2 == 0 && continue
        haskey(up_index, s2) || continue
        iu2 = up_index[s2]
        for (id, sd) in enumerate(down_states), r in 0:Ns-1, s in 0:Ns-1
            v = h2e[p+1, q+1, r+1, s+1]
            v == 0.0 && continue
            sgn3, t1 = annihilate(sd, s)
            sgn3 == 0 && continue
            sgn4, t2 = create(t1, r)
            sgn4 == 0 && continue
            haskey(down_index, t2) || continue
            id2 = down_index[t2]
            H[combined(iu2, id2), combined(iu, id)] += v * sgn1 * sgn2 * sgn3 * sgn4
        end
    end

    return eigmin(Symmetric(H)) + E_nuc
end
