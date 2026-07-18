function block_average(trace::Vector{Float64}; num_blocks::Int=20)
    n = length(trace)
    n == 0 && return (NaN, NaN)
    num_blocks = min(num_blocks, n)
    block_size = max(1, div(n, num_blocks))
    nblocks = div(n, block_size)
    block_means = [mean(trace[(b-1)*block_size+1:b*block_size]) for b in 1:nblocks]
    m = mean(block_means)
    e = nblocks > 1 ? std(block_means) / sqrt(nblocks) : NaN
    return (m, e)
end

"""
Phaseless (constrained-path) AFQMC ground-state energy for a Hubbard lattice.
Known approximations: constrained-path bias, population-control bias, and
finite-dtau Trotter error -- standard, documented AFQMC approximations.
"""
function run_afqmc(lattice::HubbardLattice, trial::TrialWavefunction, Nup::Int, Ndown::Int;
                    dtau::Float64=0.01, num_walkers::Int=100, num_steps::Int=2000,
                    equilibration_steps::Int=200, stabilize_every::Int=5, pop_control_every::Int=10)
    size(trial.phi_up, 2) == Nup || throw(ArgumentError("trial.phi_up must have Nup columns"))
    size(trial.phi_down, 2) == Ndown || throw(ArgumentError("trial.phi_down must have Ndown columns"))

    walkers = init_walkers(trial, num_walkers)
    expK_half = build_expK_half(lattice, dtau)
    gamma = hs_gamma(lattice, dtau)

    energy_trace = Float64[]

    for step in 1:num_steps
        for walker in walkers
            propagate_step!(walker, lattice, trial, expK_half, gamma, dtau)
        end

        if step % stabilize_every == 0
            for walker in walkers
                walker.weight > 0.0 && orthonormalize!(walker)
            end
        end

        if step % pop_control_every == 0
            population_control!(walkers)
        end

        if step > equilibration_steps
            push!(energy_trace, mixed_energy_estimator(lattice, walkers, trial))
        end
    end

    energy_mean, energy_err = block_average(energy_trace)
    return (energy_trace=energy_trace, energy_mean=energy_mean, energy_err=energy_err, walkers=walkers)
end
