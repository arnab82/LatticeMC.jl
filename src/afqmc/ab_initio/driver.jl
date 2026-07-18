"""
Phaseless AFQMC ground-state energy for a general (ab initio) molecular
Hamiltonian, using a Cholesky-decomposed two-electron tensor and complex
walkers/trial. `force_bias=true` uses the static (trial-based) mean-field
shift for a modest variance reduction (propagate_step_ab_initio_force_bias!);
does not change the phaseless approximation's own asymptotic bias, which is
not a variance effect -- see docs/afqmc_ab_initio_theory.md section 7.1.
Local energy is direct-tensor, not Cholesky-vector-based -- see the theory
doc for what that means and what's deferred as future work.
"""
function run_afqmc_ab_initio(mi::MolecularIntegrals, trial::AbInitioTrial, Nup::Int, Ndown::Int;
                              dtau::Float64=0.01, num_walkers::Int=100, num_steps::Int=2000,
                              equilibration_steps::Int=200, stabilize_every::Int=5, pop_control_every::Int=10,
                              cholesky_threshold::Float64=1e-8, force_bias::Bool=false)
    size(trial.phi_up, 2) == Nup || throw(ArgumentError("trial.phi_up must have Nup columns"))
    size(trial.phi_down, 2) == Ndown || throw(ArgumentError("trial.phi_down must have Ndown columns"))

    Ls = cholesky_decompose_eri(mi.h2e; threshold=cholesky_threshold)
    h1e_mod = modified_one_body(mi.h1e, Ls)
    expH1_half = exp(-dtau / 2 .* h1e_mod)
    fbar = force_bias ? force_bias_shift(trial, Ls, dtau) : Float64[]

    walkers = init_ab_initio_walkers(trial, num_walkers)
    energy_trace = Float64[]

    for step in 1:num_steps
        for walker in walkers
            if force_bias
                propagate_step_ab_initio_force_bias!(walker, mi, trial, expH1_half, Ls, dtau, fbar)
            else
                propagate_step_ab_initio!(walker, mi, trial, expH1_half, Ls, dtau)
            end
        end

        if step % stabilize_every == 0
            for walker in walkers
                walker.weight > 0.0 && orthonormalize_ab_initio!(walker)
            end
        end

        if step % pop_control_every == 0
            population_control_ab_initio!(walkers)
        end

        if step > equilibration_steps
            push!(energy_trace, mixed_energy_estimator_ab_initio(mi, walkers, trial))
        end
    end

    energy_mean, energy_err = block_average(energy_trace)
    return (energy_trace=energy_trace, energy_mean=energy_mean, energy_err=energy_err, walkers=walkers)
end
