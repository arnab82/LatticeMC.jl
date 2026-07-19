# Determinant (BSS / finite-temperature auxiliary-field) QMC for the repulsive
# Hubbard model, H = -t sum_<ij>s c+_is c_js + U sum_i n_i^up n_i^down, at the
# particle-hole-symmetric point (half filling, mu = U/2). There the up/down
# determinant product is non-negative, so the method is *sign-problem-free* --
# exact (no constrained-path/trial bias), unlike the projector AFQMC. The
# ground state is reached by cooling (large beta).
#
# Z = Tr e^{-beta H} is Trotter-split into L slices of width dtau = beta/L, the
# interaction is decoupled by the discrete Hirsch (spin-channel) HS transform,
# and the auxiliary field s in {+-1}^{L x Ns} is sampled by Metropolis single-
# spin flips. Per spin the weight is det(I + B_L...B_1) with
# B^s_l = e^{-dtau K} diag(e^{s*lambda*field[l,:]}), cosh(lambda) = e^{dtau U/2}.
# Reuses AFQMC's HubbardLattice (its K matrix and U).

# cosh(lambda) = exp(dtau U / 2); U >= 0 (repulsive) required for a real field.
function dqmc_lambda(U::Float64, dtau::Float64)
    U >= 0 || throw(ArgumentError("DQMC here uses the spin-channel HS transform, which needs U >= 0"))
    return acosh(exp(dtau * U / 2))
end

# One-slice propagator for spin sigma (+1 up / -1 down): expK * diag(exp(sigma*lambda*field_row)).
function dqmc_B(expK::Matrix{Float64}, lambda::Float64, field_row::AbstractVector{Int}, sigma::Int)
    return expK * Diagonal(exp.(sigma * lambda .* field_row))
end

# Stable equal-time Green's function G = (I + B_L B_{L-1} ... B_1)^{-1} via an
# accumulated QR (UDT) factorization of the B-string, which separates the
# widely different scales that make a naive matrix product hopelessly ill-
# conditioned at large beta. `Bs` is ordered [B_1, ..., B_L]; the product is
# formed left-to-right as P <- B_l P.
function stable_greens(Bs::Vector{Matrix{Float64}})
    Ns = size(Bs[1], 1)
    U = Matrix{Float64}(I, Ns, Ns)
    D = ones(Ns)
    T = Matrix{Float64}(I, Ns, Ns)
    for B in Bs
        M = (B * U) * Diagonal(D)
        F = qr(M)
        d = diag(F.R)
        s = sign.(d)
        Dnew = abs.(d)
        U = Matrix(F.Q) * Diagonal(s)
        Rpos = Diagonal(s) * F.R                 # upper-triangular, positive diagonal
        T = (Diagonal(1.0 ./ Dnew) * Rpos) * T   # unit-diagonal upper-triangular * old T
        D = Dnew
    end
    # G = (I + U D T)^{-1}, split D into big/small parts for a stable inverse.
    Db = max.(D, 1.0)
    Ds = min.(D, 1.0)
    Ut = Diagonal(1.0 ./ Db) * U'
    M = Ut + Diagonal(Ds) * T
    return M \ Ut
end

# Advance the equal-time Green's function from one slice to the next:
# G_{l+1} = B_l G_l B_l^{-1}.
wrap_greens(G::Matrix{Float64}, B::Matrix{Float64}, Binv::Matrix{Float64}) = B * G * Binv

# In-place rank-1 Sherman-Morrison update of G after flipping the field at site
# i (spin sigma), whose diagonal propagator entry is multiplied by
# (1 + delta), delta = exp(-2*sigma*lambda*field[l,i]) - 1. R = 1 + delta*(1 - G[i,i]).
#   G'_{jk} = G_{jk} - (delta/R) (delta_{ji} - G_{ji}) G_{ik}
function update_greens!(G::Matrix{Float64}, i::Int, delta::Float64, R::Float64)
    Ns = size(G, 1)
    coeff = delta / R
    col = G[:, i]            # G_{ji}
    row = G[i, :]            # G_{ik}
    for k in 1:Ns, j in 1:Ns
        G[j, k] -= coeff * ((j == i ? 1.0 : 0.0) - col[j]) * row[k]
    end
    return G
end

# Energy per the equal-time Green's functions (G^sigma_{ij} = <c_is c+_js>):
#   <K> = - sum_sigma tr(K G^sigma)      (K has zero diagonal)
#   <V> = U sum_i (1 - G^up_ii)(1 - G^down_ii)      (n_is = 1 - G^sigma_ii)
function dqmc_energy(lattice, Gup::Matrix{Float64}, Gdown::Matrix{Float64})
    K = lattice.K
    e_kin = -(tr(K * Gup) + tr(K * Gdown))
    e_int = 0.0
    for i in 1:lattice.Ns
        e_int += lattice.U * (1 - Gup[i, i]) * (1 - Gdown[i, i])
    end
    return e_kin + e_int
end

"""
    run_dqmc(lattice; beta, L=..., num_thermal=500, num_sweeps=5000, nwrap=8, seed=nothing)

Determinant QMC for the half-filled (particle-hole-symmetric) repulsive
Hubbard model on `lattice` (an `AFQMC`-style `HubbardLattice`) at inverse
temperature `beta`, split into `L` imaginary-time slices (default chosen so
`dtau = beta/L ~ 0.1`). Sign-problem-free here, hence exact up to Trotter
(`dtau`) and statistical error; take `beta` large for the ground state.

Returns a named tuple with `energy` / `energy_per_site` (mean and blocked
error), `double_occupancy`, and `mean_sign` (should be ~1 at half filling; a
value away from 1 flags a sign problem / a departure from the sign-free point).
"""
function run_dqmc(lattice; beta::Float64, L::Union{Int,Nothing}=nothing,
                   num_thermal::Int=500, num_sweeps::Int=5000, nwrap::Int=8,
                   seed::Union{Int,Nothing}=nothing)
    seed !== nothing && Random.seed!(seed)
    Ns = lattice.Ns
    L === nothing && (L = max(1, round(Int, beta / 0.1)))
    dtau = beta / L
    lambda = dqmc_lambda(lattice.U, dtau)
    expK = exp(-dtau .* lattice.K)
    expKinv = exp(dtau .* lattice.K)

    field = rand((-1, 1), L, Ns)

    # G_l = (I + A_l)^{-1}, A_l = B_{l-1}...B_1 B_L...B_l, obtained by stabilizing
    # the *cyclic* B-string [B_l, B_{l+1}, ..., B_L, B_1, ..., B_{l-1}] (recompute
    # from scratch at slice l with the right ordering -- the crux of DQMC
    # stability; using the fixed slice-1 order at every slice silently corrupts G).
    function fresh_greens(l::Int, sigma::Int)
        order = vcat(l:L, 1:l-1)
        Bs = [dqmc_B(expK, lambda, view(field, m, :), sigma) for m in order]
        return stable_greens(Bs)
    end

    energies = Float64[]
    docc = Float64[]
    neg_moves = 0
    total_moves = 0

    total_sweeps = num_thermal + num_sweeps
    for sweep in 1:total_sweeps
        Gup = fresh_greens(1, 1)
        Gdown = fresh_greens(1, -1)

        for l in 1:L
            # re-stabilize G at the correct cyclic ordering for the current slice
            if l > 1 && (l - 1) % nwrap == 0
                Gup = fresh_greens(l, 1)
                Gdown = fresh_greens(l, -1)
            end

            for i in 1:Ns
                s = field[l, i]
                delta_up = exp(-2 * 1 * lambda * s) - 1
                delta_down = exp(-2 * (-1) * lambda * s) - 1
                R_up = 1 + delta_up * (1 - Gup[i, i])
                R_down = 1 + delta_down * (1 - Gdown[i, i])
                R = R_up * R_down
                total_moves += 1
                R < 0 && (neg_moves += 1)
                if rand() < min(1.0, abs(R))
                    update_greens!(Gup, i, delta_up, R_up)
                    update_greens!(Gdown, i, delta_down, R_down)
                    field[l, i] = -s
                end
            end

            # wrap G_l -> G_{l+1} = B_l G_l B_l^{-1}
            fr = view(field, l, :)
            Bup = dqmc_B(expK, lambda, fr, 1)
            Bdown = dqmc_B(expK, lambda, fr, -1)
            Bup_inv = Diagonal(exp.(-lambda .* fr)) * expKinv
            Bdown_inv = Diagonal(exp.(lambda .* fr)) * expKinv
            Gup = wrap_greens(Gup, Bup, Bup_inv)
            Gdown = wrap_greens(Gdown, Bdown, Bdown_inv)
        end

        if sweep > num_thermal
            Gu = fresh_greens(1, 1)
            Gd = fresh_greens(1, -1)
            push!(energies, dqmc_energy(lattice, Gu, Gd))
            d = 0.0
            for i in 1:Ns
                d += (1 - Gu[i, i]) * (1 - Gd[i, i])
            end
            push!(docc, d / Ns)
        end
    end

    e_mean, e_err = _dqmc_block(energies)
    d_mean, _ = _dqmc_block(docc)
    # half filling is particle-hole symmetric -> weight product >= 0; the
    # fraction of proposed moves with R < 0 is the sign-problem diagnostic.
    neg_frac = total_moves == 0 ? 0.0 : neg_moves / total_moves
    return (energy=e_mean, energy_err=e_err,
            energy_per_site=e_mean / Ns, energy_per_site_err=e_err / Ns,
            double_occupancy=d_mean,
            mean_sign=1.0 - 2.0 * neg_frac,
            neg_move_fraction=neg_frac)
end

function _dqmc_block(xs::Vector{Float64}; num_blocks::Int=20)
    n = length(xs)
    n == 0 && return (NaN, NaN)
    nb = min(num_blocks, n)
    bs = max(1, div(n, nb))
    k = div(n, bs)
    bm = [sum(@view xs[(b-1)*bs+1:b*bs]) / bs for b in 1:k]
    m = sum(bm) / k
    e = k > 1 ? sqrt(sum((bm .- m) .^ 2) / (k * (k - 1))) : NaN
    return (m, e)
end
