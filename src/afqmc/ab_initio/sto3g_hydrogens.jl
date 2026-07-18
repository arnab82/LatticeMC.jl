# Self-contained, dependency-free (beyond SpecialFunctions.erf) STO-3G
# integral engine for chains of hydrogen atoms: s-orbitals only, hydrogen
# only, deliberately not a general basis-set engine -- exactly enough to
# generate real MolecularIntegrals for the H-chain AFQMC validation targets
# without depending on an external quantum chemistry package.
#
# Published STO-3G contraction for H 1s (Hehre, Stewart & Pople 1969).
const STO3G_H_ALPHA = (3.42525091, 0.62391373, 0.16885540)
const STO3G_H_D = (0.15432897, 0.53532814, 0.44463454)

# Primitive-normalizes each Gaussian, then renormalizes the whole contracted
# function so that its self-overlap is exactly 1.
function normalized_contraction(alphas, d)
    c = [d[i] * (2 * alphas[i] / pi)^0.75 for i in eachindex(alphas)]
    S = 0.0
    for i in eachindex(alphas), j in eachindex(alphas)
        p = alphas[i] + alphas[j]
        S += c[i] * c[j] * (pi / p)^1.5
    end
    return c ./ sqrt(S)
end

# Boys function of order 0 (the only order needed for s-orbital integrals),
# closed form via erf; small-t Taylor expansion avoids 0/0 cancellation
# error near t=0 (exact limit F0(0) = 1).
function boys_f0(t::Float64)
    t < 1e-10 && return 1.0 - t / 3
    return 0.5 * sqrt(pi / t) * erf(sqrt(t))
end

_r2(Ra, Rb) = sum(abs2, Ra .- Rb)

function primitive_overlap(a, Ra, b, Rb)
    p = a + b
    K = exp(-a * b / p * _r2(Ra, Rb))
    return (pi / p)^1.5 * K
end

function primitive_kinetic(a, Ra, b, Rb)
    p = a + b
    r2 = _r2(Ra, Rb)
    S = primitive_overlap(a, Ra, b, Rb)
    return a * b / p * (3 - 2 * a * b / p * r2) * S
end

function primitive_nuclear(a, Ra, b, Rb, Rc, Zc)
    p = a + b
    K = exp(-a * b / p * _r2(Ra, Rb))
    Rp = (a .* Ra .+ b .* Rb) ./ p
    t = p * _r2(Rp, Rc)
    return -Zc * (2 * pi / p) * K * boys_f0(t)
end

function primitive_eri(a, Ra, b, Rb, c, Rc, d, Rd)
    p = a + b
    q = c + d
    Kab = exp(-a * b / p * _r2(Ra, Rb))
    Kcd = exp(-c * d / q * _r2(Rc, Rd))
    Rp = (a .* Ra .+ b .* Rb) ./ p
    Rq = (c .* Rc .+ d .* Rd) ./ q
    t = p * q / (p + q) * _r2(Rp, Rq)
    return (2 * pi^2.5) / (p * q * sqrt(p + q)) * Kab * Kcd * boys_f0(t)
end

# Loewdin symmetric orthogonalization: X = S^(-1/2), transforming the
# (non-orthogonal) atomic-orbital-basis integrals to an orthonormal basis.
# Required because every second-quantized piece downstream (RHF, FCI,
# AFQMC's Thouless-theorem overlaps and Green's functions) assumes an
# orthonormal single-particle basis, same as the Hubbard model's site basis
# -- but STO-3G orbitals on different atoms are not orthogonal (S != I), so
# skipping this step would silently give wrong physics.
function lowdin_orthogonalize(S::Matrix{Float64}, h1e_ao::Matrix{Float64}, h2e_ao::Array{Float64,4})
    F = eigen(Symmetric(S))
    X = F.vectors * Diagonal(1.0 ./ sqrt.(F.values)) * F.vectors'

    h1e = X' * h1e_ao * X

    Ns = size(S, 1)
    tmp1 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, i in 1:Ns
        tmp1[p, q, r, s] += X[i, p] * h2e_ao[i, q, r, s]
    end
    tmp2 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, j in 1:Ns
        tmp2[p, q, r, s] += X[j, q] * tmp1[p, j, r, s]
    end
    tmp3 = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, k in 1:Ns
        tmp3[p, q, r, s] += X[k, r] * tmp2[p, q, k, s]
    end
    h2e = zeros(Ns, Ns, Ns, Ns)
    for s in 1:Ns, r in 1:Ns, q in 1:Ns, p in 1:Ns, l in 1:Ns
        h2e[p, q, r, s] += X[l, s] * tmp3[p, q, r, l]
    end

    return h1e, h2e
end

# n_atoms hydrogens on a line, spaced by bond_length (bohr), STO-3G basis.
function build_h_chain_sto3g(bond_length::Float64; n_atoms::Int=4)
    centers = [[(i - 1) * bond_length, 0.0, 0.0] for i in 1:n_atoms]
    Z = 1.0
    Ns = n_atoms

    alphas = STO3G_H_ALPHA
    d = normalized_contraction(collect(alphas), collect(STO3G_H_D))

    S = zeros(Ns, Ns)
    T = zeros(Ns, Ns)
    V = zeros(Ns, Ns)
    for A in 1:Ns, B in 1:Ns
        for i in eachindex(alphas), j in eachindex(alphas)
            cc = d[i] * d[j]
            S[A, B] += cc * primitive_overlap(alphas[i], centers[A], alphas[j], centers[B])
            T[A, B] += cc * primitive_kinetic(alphas[i], centers[A], alphas[j], centers[B])
            for C in 1:Ns
                V[A, B] += cc * primitive_nuclear(alphas[i], centers[A], alphas[j], centers[B], centers[C], Z)
            end
        end
    end

    h2e_ao = zeros(Ns, Ns, Ns, Ns)
    for A in 1:Ns, B in 1:Ns, C in 1:Ns, D in 1:Ns
        acc = 0.0
        for i in eachindex(alphas), j in eachindex(alphas), k in eachindex(alphas), l in eachindex(alphas)
            cc = d[i] * d[j] * d[k] * d[l]
            acc += cc * primitive_eri(alphas[i], centers[A], alphas[j], centers[B],
                                       alphas[k], centers[C], alphas[l], centers[D])
        end
        h2e_ao[A, B, C, D] = acc
    end

    h1e_ao = T + V
    h1e, h2e = lowdin_orthogonalize(S, h1e_ao, h2e_ao)

    E_nuc = 0.0
    for A in 1:Ns, B in A+1:Ns
        E_nuc += Z * Z / sqrt(_r2(centers[A], centers[B]))
    end

    return MolecularIntegrals(Ns, h1e, h2e, E_nuc)
end
