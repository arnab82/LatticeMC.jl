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

# Mean sign of a signed walker population: <s> = sum(w) / sum(|w|). Stays ~1
# for a sign-problem-free free-projection run, decays toward 0 when the sign
# problem is present (the free-projection estimator becomes unusable there).
function mean_sign(walkers::Vector{Walker})
    num = sum(w.weight for w in walkers)
    den = sum(abs(w.weight) for w in walkers)
    return den == 0.0 ? NaN : num / den
end

"""
AFQMC ground-state energy for a Hubbard lattice.

`constrained=true` (default): phaseless / constrained-path AFQMC. Robust
everywhere (walkers stay positive), at the cost of a controlled, trial-
dependent constrained-path bias (plus population-control and finite-dtau
Trotter errors) -- the standard approximations.

`constrained=false`: free projection (unconstrained), with uniform (unguided)
field sampling and signed walker weights. Exact (unbiased) in principle, but
only *practical* where the mean sign stays close to 1. Note: this zero-
temperature projector method is NOT automatically sign-problem-free at half
filling -- that result is for finite-temperature determinant QMC, a different
algorithm. Here the mean sign stays 1 only for small clusters (where free
projection is exact, validated against ED); a 6x6 half-filled Hubbard already
shows a real sign problem (mean sign ~ 0.3), so its free-projection energy is
unreliable. The returned named tuple includes `mean_sign` (and its error);
trust the energy only while `mean_sign` stays well away from 0.
"""
function run_afqmc(lattice::HubbardLattice, trial::TrialWavefunction, Nup::Int, Ndown::Int;
                    dtau::Float64=0.01, num_walkers::Int=100, num_steps::Int=2000,
                    equilibration_steps::Int=200, stabilize_every::Int=5, pop_control_every::Int=10,
                    constrained::Bool=true)
    size(trial.phi_up, 2) == Nup || throw(ArgumentError("trial.phi_up must have Nup columns"))
    size(trial.phi_down, 2) == Ndown || throw(ArgumentError("trial.phi_down must have Ndown columns"))

    walkers = init_walkers(trial, num_walkers)
    expK_half = build_expK_half(lattice, dtau)
    gamma = hs_gamma(lattice, dtau)

    energy_trace = Float64[]
    sign_trace = Float64[]

    for step in 1:num_steps
        for walker in walkers
            if constrained
                propagate_step!(walker, lattice, trial, expK_half, gamma, dtau)
            else
                propagate_step_free!(walker, lattice, trial, expK_half, gamma, dtau)
            end
        end

        if step % stabilize_every == 0
            for walker in walkers
                walker.weight != 0.0 && orthonormalize!(walker)
            end
        end

        if step % pop_control_every == 0
            constrained ? population_control!(walkers) : population_control_signed!(walkers)
        end

        if step > equilibration_steps
            push!(energy_trace, mixed_energy_estimator(lattice, walkers, trial))
            constrained || push!(sign_trace, mean_sign(walkers))
        end
    end

    energy_mean, energy_err = block_average(energy_trace)
    sign_mean, sign_err = constrained ? (1.0, 0.0) : block_average(sign_trace)
    return (energy_trace=energy_trace, energy_mean=energy_mean, energy_err=energy_err,
            mean_sign=sign_mean, mean_sign_err=sign_err, walkers=walkers)
end
