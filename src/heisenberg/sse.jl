# Stochastic Series Expansion (SSE) QMC for the S=1/2 antiferromagnetic
# Heisenberg model (Sandvik). The partition function Tr e^{-beta H} is
# expanded as a series in beta; each term is a fixed-length string of L
# bond operators (diagonal or off-diagonal), padded with identities, sampled
# by (1) a diagonal update that inserts/removes diagonal operators and (2) an
# operator-loop update that toggles diagonal<->off-diagonal along closed loops.
# For the AFM Heisenberg on a bipartite lattice all weights are non-negative
# (sign-problem-free), so this is essentially exact at low temperature.
#
# Operator string encoding, op[p]:
#   0            -> identity
#   2*b - 1      -> diagonal    operator on bond b   (odd)
#   2*b          -> off-diagonal operator on bond b   (even)
# so bond(o) = (o + 1) >> 1 and o is diagonal iff isodd(o).

_bond_of(o::Int) = (o + 1) >> 1
_toggle(o::Int) = isodd(o) ? o + 1 : o - 1   # diagonal <-> off-diagonal, same bond
# horizontal within-vertex partner (0<->1, 2<->3 in local 0-based leg index):
_horiz(leg::Int) = let base = ((leg - 1) & ~3); base + (((leg - 1) & 3) ⊻ 1) + 1 end

# One diagonal update sweep over the whole operator string. Propagates a
# working spin state through the string; where it meets an identity it may
# insert a diagonal operator (only on an antiparallel bond), where it meets a
# diagonal operator it may remove it, and off-diagonal operators just flip the
# working spins. Returns the (possibly changed) operator count n.
function _diagonal_update!(spins::Vector{Int}, op::Vector{Int}, lat::HeisenbergLattice,
                            beta::Float64, n::Int)
    L = length(op)
    Nb = length(lat.bonds)
    J = lat.J
    s = copy(spins)
    for p in 1:L
        o = op[p]
        if o == 0
            b = rand(1:Nb)
            i, j = lat.bonds[b]
            if s[i] != s[j]
                if rand() < (beta * Nb * J * 0.5) / (L - n)
                    op[p] = 2 * b - 1
                    n += 1
                end
            end
        elseif isodd(o)
            if rand() < (L - n + 1) / (beta * Nb * J * 0.5)
                op[p] = 0
                n -= 1
            end
        else
            b = _bond_of(o)
            i, j = lat.bonds[b]
            s[i] = -s[i]
            s[j] = -s[j]
        end
    end
    return n
end

# One operator-loop update. Builds the linked list of vertex legs (in
# imaginary time), constructs closed loops via the horizontal within-vertex
# move + vertical links, flips each loop with probability 1/2, then reads the
# new operator types and spin configuration back out.
function _loop_update!(spins::Vector{Int}, op::Vector{Int}, lat::HeisenbergLattice)
    L = length(op)
    Ns = lat.Ns
    first = zeros(Int, Ns)
    last = zeros(Int, Ns)
    vlink = zeros(Int, 4L)

    # vertical (imaginary-time) links between consecutive operators on a site
    for p in 1:L
        o = op[p]
        o == 0 && continue
        b = _bond_of(o)
        i, j = lat.bonds[b]
        base = 4 * (p - 1)
        # legs: base+1 = i before, base+2 = j before, base+3 = i after, base+4 = j after
        for (before, after, site) in ((base + 1, base + 3, i), (base + 2, base + 4, j))
            l = last[site]
            if l != 0
                vlink[l] = before
                vlink[before] = l
            else
                first[site] = before
            end
            last[site] = after
        end
    end
    for site in 1:Ns
        if first[site] != 0
            vlink[last[site]] = first[site]
            vlink[first[site]] = last[site]
        end
    end

    # build and flip loops
    flipped = falses(4L)
    visited = falses(4L)
    for start in 1:4L
        vlink[start] == 0 && continue
        visited[start] && continue
        doflip = rand(Bool)
        l = start
        while true
            visited[l] = true
            doflip && (flipped[l] = true)
            h = _horiz(l)
            visited[h] = true
            doflip && (flipped[h] = true)
            l = vlink[h]
            l == start && break
        end
    end

    # apply: an operator toggles diag<->offdiag iff exactly one of its two
    # same-site legs (before/after on site i) flipped.
    for p in 1:L
        o = op[p]
        o == 0 && continue
        base = 4 * (p - 1)
        if flipped[base + 1] ⊻ flipped[base + 3]
            op[p] = _toggle(o)
        end
    end
    # update the stored spin configuration for the next diagonal update
    for site in 1:Ns
        if first[site] == 0
            rand(Bool) && (spins[site] = -spins[site])   # free spin, flip w.p. 1/2
        elseif flipped[first[site]]
            spins[site] = -spins[site]
        end
    end
    return nothing
end

function _block_stats(xs::Vector{Float64}; num_blocks::Int=20)
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

"""
    run_sse(lat; beta, num_thermal=2000, num_measure=20000, seed=nothing)

Ground-state-oriented SSE for the S=1/2 AFM Heisenberg model on `lat` at
inverse temperature `beta` (take `beta` large — a few times the linear size —
to converge to the ground state). Returns a named tuple with `energy` and
`energy_per_site` (mean and blocked statistical error) and the mean operator
count `n_mean`. Energy estimator: E = J*Nbonds/4 - <n>/beta.
"""
function run_sse(lat::HeisenbergLattice; beta::Float64, num_thermal::Int=2000,
                  num_measure::Int=20000, seed::Union{Int,Nothing}=nothing)
    is_bipartite(lat) || throw(ArgumentError(
        "run_sse requires a bipartite lattice: the AFM Heisenberg model has a " *
        "sign problem on a frustrated (non-bipartite) lattice, where this " *
        "sign-problem-free SSE would give silently wrong answers. A square " *
        "Lx x Ly lattice with PBC is bipartite iff both Lx and Ly are even."))
    seed !== nothing && Random.seed!(seed)
    Ns = lat.Ns
    spins = rand((-1, 1), Ns)
    L = max(20, 4 * Ns)
    op = zeros(Int, L)
    n = 0

    for _ in 1:num_thermal
        n = _diagonal_update!(spins, op, lat, beta, n)
        _loop_update!(spins, op, lat)
        # expand the string if it's getting full (thermalization only)
        newL = n + div(n, 3)
        if newL > L
            append!(op, zeros(Int, newL - L))
            L = newL
        end
    end

    n_samples = Float64[]
    for _ in 1:num_measure
        n = _diagonal_update!(spins, op, lat, beta, n)
        _loop_update!(spins, op, lat)
        push!(n_samples, Float64(n))
    end

    n_mean = sum(n_samples) / length(n_samples)
    offset = lat.J * length(lat.bonds) / 4
    energies = offset .- n_samples ./ beta
    e_mean, e_err = _block_stats(energies)
    return (energy=e_mean, energy_err=e_err,
            energy_per_site=e_mean / Ns, energy_per_site_err=e_err / Ns,
            n_mean=n_mean)
end
