# Spin-1/2 antiferromagnetic Heisenberg model on a bipartite lattice,
# H = J sum_<ij> S_i . S_j, J > 0. `bonds` is the list of (i, j) site-pairs;
# everything downstream (SSE, ED) works off this bond list, so a new geometry
# is just a new bond builder.
struct HeisenbergLattice
    Ns::Int
    bonds::Vector{Tuple{Int,Int}}
    J::Float64
    dims::Tuple{Int,Int}   # (Lx, Ly); (Ns, 1) for a chain
end

nbonds(lat::HeisenbergLattice) = length(lat.bonds)

# BFS 2-coloring of the bond graph. Returns true iff the lattice is bipartite
# (no odd cycles). This SSE is sign-problem-free *only* on a bipartite lattice
# -- on a frustrated (non-bipartite) one the AFM Heisenberg has a genuine sign
# problem and this code, which tracks no signs, silently gives wrong answers.
# A square Lx x Ly lattice with PBC is bipartite iff both dimensions are even.
function is_bipartite(lat::HeisenbergLattice)
    adj = [Int[] for _ in 1:lat.Ns]
    for (i, j) in lat.bonds
        push!(adj[i], j)
        push!(adj[j], i)
    end
    color = zeros(Int, lat.Ns)
    for start in 1:lat.Ns
        color[start] != 0 && continue
        color[start] = 1
        queue = [start]
        while !isempty(queue)
            u = popfirst!(queue)
            for v in adj[u]
                if color[v] == 0
                    color[v] = -color[u]
                    push!(queue, v)
                elseif color[v] == color[u]
                    return false
                end
            end
        end
    end
    return true
end

# 1D chain of Ns sites, nearest-neighbor bonds, open or periodic. (PBC needs
# Ns >= 3 to avoid a doubled bond between sites 1 and 2.)
function build_heisenberg_chain(Ns::Int; J::Float64=1.0, pbc::Bool=true)
    Ns >= 2 || throw(ArgumentError("chain needs at least 2 sites"))
    bonds = Tuple{Int,Int}[]
    for i in 1:Ns-1
        push!(bonds, (i, i + 1))
    end
    if pbc && Ns >= 3
        push!(bonds, (Ns, 1))
    end
    return HeisenbergLattice(Ns, bonds, J, (Ns, 1))
end

# Lx x Ly square lattice, nearest-neighbor bonds, open or periodic. Bipartite
# (so the AFM model is sign-problem-free for SSE) when both dimensions are
# even under PBC. (PBC needs each dimension >= 3 to avoid doubled bonds.)
function build_heisenberg_square(Lx::Int, Ly::Int; J::Float64=1.0, pbc::Bool=true)
    Lx >= 2 && Ly >= 2 || throw(ArgumentError("square lattice needs at least 2x2 sites"))
    Ns = Lx * Ly
    idx(x, y) = (y - 1) * Lx + x
    bonds = Tuple{Int,Int}[]
    for y in 1:Ly, x in 1:Lx
        if x < Lx
            push!(bonds, (idx(x, y), idx(x + 1, y)))
        elseif pbc && Lx >= 3
            push!(bonds, (idx(Lx, y), idx(1, y)))
        end
        if y < Ly
            push!(bonds, (idx(x, y), idx(x, y + 1)))
        elseif pbc && Ly >= 3
            push!(bonds, (idx(x, Ly), idx(x, 1)))
        end
    end
    return HeisenbergLattice(Ns, bonds, J, (Lx, Ly))
end
