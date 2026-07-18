# Comb (systematic) resampling: replaces the walker population with a new
# population of the same size, all equal weight, total weight preserved.
# Lower variance than independent multinomial resampling of each walker, and
# is the standard population-control step used to prevent AFQMC walker
# weights from collapsing to zero or blowing up over many propagation steps.
function population_control!(walkers::Vector{Walker})
    Nw = length(walkers)
    weights = [w.weight for w in walkers]
    total_weight = sum(weights)
    total_weight <= 0.0 && return walkers

    cumulative = cumsum(weights)
    xi = rand()
    new_phi_up = Vector{Matrix{Float64}}(undef, Nw)
    new_phi_down = Vector{Matrix{Float64}}(undef, Nw)

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
