struct HubbardLattice
    Ns::Int
    K::Matrix{Float64}
    U::Float64
    t::Float64
    geometry::Symbol
end

# Note: a dimension of size 2 with pbc=true doubles the hopping element on that
# bond (the "left" and "right" neighbor coincide) -- this matches the correct
# periodic dispersion at N=2, but is easy to mistake for a bug. Prefer pbc=false
# (or a dimension >= 3) for small clusters used in exact-diagonalization checks.
function build_hubbard_chain(Ns::Int, t::Float64, U::Float64; pbc::Bool=true)
    Ns >= 2 || throw(ArgumentError("chain needs at least 2 sites"))
    K = zeros(Float64, Ns, Ns)
    for i in 1:Ns
        j = i + 1
        if j <= Ns
            K[i, j] -= t
            K[j, i] -= t
        elseif pbc
            K[i, 1] -= t
            K[1, i] -= t
        end
    end
    return HubbardLattice(Ns, K, U, t, :chain)
end

function build_hubbard_square(Lx::Int, Ly::Int, t::Float64, U::Float64; pbc::Bool=true)
    Lx >= 2 && Ly >= 2 || throw(ArgumentError("square lattice needs at least 2x2 sites"))
    Ns = Lx * Ly
    site_index(x, y) = (y - 1) * Lx + x
    K = zeros(Float64, Ns, Ns)

    for y in 1:Ly, x in 1:Lx
        i = site_index(x, y)

        xr = x + 1
        if xr <= Lx
            j = site_index(xr, y)
            K[i, j] -= t
            K[j, i] -= t
        elseif pbc
            j = site_index(1, y)
            K[i, j] -= t
            K[j, i] -= t
        end

        yu = y + 1
        if yu <= Ly
            j = site_index(x, yu)
            K[i, j] -= t
            K[j, i] -= t
        elseif pbc
            j = site_index(x, 1)
            K[i, j] -= t
            K[j, i] -= t
        end
    end

    return HubbardLattice(Ns, K, U, t, :square)
end
