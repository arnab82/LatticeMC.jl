using LatticeMC
using Plots
using Random

include(joinpath(@__DIR__, "..", "test", "ed_reference.jl"))

Random.seed!(7)

Ns = 4
t = 1.0
U = 4.0
Nup, Ndown = 2, 2

lattice = LatticeMC.build_hubbard_chain(Ns, t, U; pbc=false)
trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)
e_ed = ed_ground_state_energy(lattice, Nup, Ndown)

# --- (a) Trotter error: energy vs dtau at fixed walkers/steps ---
dtaus = [0.04, 0.02, 0.01, 0.005, 0.0025]
dtau_energies = Float64[]
dtau_errors = Float64[]
for dtau in dtaus
    result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown; dtau=dtau, num_walkers=200,
                                  num_steps=3000, equilibration_steps=500)
    push!(dtau_energies, result.energy_mean)
    push!(dtau_errors, result.energy_err)
    println("dtau=$dtau: E = $(result.energy_mean) +/- $(result.energy_err)")
end

plt1 = plot(dtaus, dtau_energies, yerror=dtau_errors, marker=:circle, label="AFQMC",
            xlabel="dtau", ylabel="Energy", title="Trotter error", grid=true)
hline!(plt1, [e_ed], label="Exact", linestyle=:dash)

# --- (b) statistical convergence: energy_err vs number of post-equilibration samples ---
step_counts = [800, 1500, 3000, 6000, 12000]
err_by_steps = Float64[]
for num_steps in step_counts
    result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown; dtau=0.01, num_walkers=200,
                                  num_steps=num_steps, equilibration_steps=300)
    push!(err_by_steps, result.energy_err)
    println("num_steps=$num_steps: energy_err = $(result.energy_err)")
end

plt2 = plot(step_counts, err_by_steps, marker=:circle, xscale=:log10, yscale=:log10,
            xlabel="num_steps", ylabel="energy_err", title="Statistical convergence",
            grid=true, label="AFQMC")

plt = plot(plt1, plt2, layout=(1, 2), size=(1000, 400), dpi=800)
savefig(plt, "afqmc_convergence_study.png")
display(plt)
