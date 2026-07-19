using LinearAlgebra

# Finite-temperature grand-canonical exact reference for the Hubbard model, to
# validate DQMC. Full Fock space over 2*Ns spin-orbitals (bit (i-1) = site i
# up, bit (Ns+i-1) = site i down). H = sum_ij,sigma K_ij c+_is c_js
# + U sum_i n_i^up n_i^down; grand-canonical weight e^{-beta (H - mu N)}.
# Returns the thermal energy <H> = Tr(H e^{-beta(H-mu N)}) / Tr(e^{-beta(H-mu N)}).
# Small Ns only (dense 2^(2Ns) space): Ns <= 6.

function _dqmc_ed_sign_annihilate(state::Int, orb::Int)
    ((state >> (orb - 1)) & 1) == 0 && return (0, 0)
    sgn = isodd(count_ones(state & ((1 << (orb - 1)) - 1))) ? -1 : 1
    return (sgn, state & ~(1 << (orb - 1)))
end
function _dqmc_ed_sign_create(state::Int, orb::Int)
    ((state >> (orb - 1)) & 1) == 1 && return (0, 0)
    sgn = isodd(count_ones(state & ((1 << (orb - 1)) - 1))) ? -1 : 1
    return (sgn, state | (1 << (orb - 1)))
end

function hubbard_thermal_energy(K::Matrix{Float64}, U::Float64, Ns::Int, beta::Float64; mu::Float64=U / 2)
    norb = 2 * Ns
    dim = 2^norb
    up(i) = i            # orbital index for site i, spin up
    dn(i) = Ns + i       # orbital index for site i, spin down

    H = zeros(Float64, dim, dim)
    N = zeros(Float64, dim)
    for st in 0:dim-1
        col = st + 1
        # diagonal: interaction U n_up n_down, and particle number
        nocc = count_ones(st)
        N[col] = nocc
        for i in 1:Ns
            ni_up = (st >> (up(i) - 1)) & 1
            ni_dn = (st >> (dn(i) - 1)) & 1
            H[col, col] += U * ni_up * ni_dn
        end
        # kinetic: sum_ij,sigma K_ij c+_is c_js
        for i in 1:Ns, j in 1:Ns
            Kij = K[i, j]
            Kij == 0.0 && continue
            for orb_pair in ((up(i), up(j)), (dn(i), dn(j)))
                oi, oj = orb_pair
                sgn1, s1 = _dqmc_ed_sign_annihilate(st, oj)
                sgn1 == 0 && continue
                sgn2, s2 = _dqmc_ed_sign_create(s1, oi)
                sgn2 == 0 && continue
                H[s2 + 1, col] += Kij * sgn1 * sgn2
            end
        end
    end

    Hgc = Symmetric(H - mu .* Diagonal(N))
    F = eigen(Hgc)
    w = exp.(-beta .* (F.values .- minimum(F.values)))   # shift for numerical stability
    Z = sum(w)
    # <H> = sum_a w_a <a|H|a>, with |a> the eigenvectors of Hgc (which commute with H)
    Hexpect = [dot(F.vectors[:, a], H * F.vectors[:, a]) for a in 1:dim]
    return sum(w .* Hexpect) / Z
end
