# Complex analogue of population_control.jl (real/Hubbard) -- same comb
# resampling, same guarantees (total weight preserved exactly, population
# size unchanged).
function population_control_ab_initio!(walkers::Vector{AbInitioWalker})
    Nw = length(walkers)
    weights = [w.weight for w in walkers]
    total_weight = sum(weights)
    total_weight <= 0.0 && return walkers

    cumulative = cumsum(weights)
    xi = rand()
    new_phi_up = Vector{Matrix{ComplexF64}}(undef, Nw)
    new_phi_down = Vector{Matrix{ComplexF64}}(undef, Nw)

    idx = 1
    for j in 1:Nw
        target = (j - 1 + xi) / Nw * total_weight
        while idx < Nw && cumulative[idx] < target
            idx += 1
        end
        new_phi_up[j] = copy(walkers[idx].phi_up)
        new_phi_down[j] = copy(walkers[idx].phi_down)
    end

    equal_weight = total_weight / Nw
    for j in 1:Nw
        walkers[j].phi_up = new_phi_up[j]
        walkers[j].phi_down = new_phi_down[j]
        walkers[j].weight = equal_weight
    end

    return walkers
end
