# Pivoted (modified) Cholesky decomposition of the two-electron integral
# tensor h2e[p,q,r,s] = (pq|rs), reshaped as the symmetric positive-
# semidefinite matrix M[(pq),(rs)] over the compound index (pq). Returns a
# vector of Ns x Ns matrices L^gamma with sum_gamma L^gamma_pq L^gamma_rs
# ~= h2e[p,q,r,s]; each L^gamma comes out symmetric because h2e has the
# standard real-orbital 8-fold permutation symmetry (verified numerically
# in test/ab_initio.jl, not just assumed).
function cholesky_decompose_eri(h2e::Array{Float64,4}; threshold::Float64=1e-8)
    Ns = size(h2e, 1)
    npq = Ns * Ns
    pq_to_idx(p, q) = (p - 1) * Ns + q

    M = zeros(npq, npq)
    for p in 1:Ns, q in 1:Ns, r in 1:Ns, s in 1:Ns
        M[pq_to_idx(p, q), pq_to_idx(r, s)] = h2e[p, q, r, s]
    end

    D = [M[n, n] for n in 1:npq]
    vecs = Vector{Float64}[]

    while true
        pivot = argmax(D)
        D[pivot] < threshold && break
        residual = zeros(npq)
        for lk in vecs
            residual .+= lk[pivot] .* lk
        end
        l = (M[:, pivot] .- residual) ./ sqrt(D[pivot])
        push!(vecs, l)
        D .-= l .^ 2
    end

    Ls = Matrix{Float64}[]
    for l in vecs
        Lmat = zeros(Ns, Ns)
        for p in 1:Ns, q in 1:Ns
            Lmat[p, q] = l[pq_to_idx(p, q)]
        end
        push!(Ls, Lmat)
    end
    return Ls
end

# The h1e_mod = h1e - 1/2 sum_gamma (L^gamma)^2 correction: rewriting the
# two-body term as (c+ L c)^2 via the Cholesky vectors picks up a one-body
# piece from the fermion anticommutator when normal-ordering
# c+_p c+_r c_s c_q -> c+_p c_q c+_r c_s - delta_qr c+_p c_s. Folded into
# h1e once, up front, rather than handled per-step in the propagator.
function modified_one_body(h1e::Matrix{Float64}, Ls::Vector{Matrix{Float64}})
    h1e_mod = copy(h1e)
    for L in Ls
        h1e_mod .-= 0.5 .* (L * L)
    end
    return h1e_mod
end
