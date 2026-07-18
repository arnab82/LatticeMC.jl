using LatticeMC
using Plots
using Random

Random.seed!(3)

Lx, Ly = 4, 4
t = 1.0
U = 4.0
Nup, Ndown = 8, 8

lattice = LatticeMC.build_hubbard_square(Lx, Ly, t, U; pbc=true)
trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)

result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown;
                              dtau=0.01, num_walkers=200, num_steps=3000,
                              equilibration_steps=500, stabilize_every=5,
                              pop_control_every=10)

println("$(Lx)x$(Ly) Hubbard (PBC), U/t=$(U/t): E = $(result.energy_mean) +/- $(result.energy_err)  " *
        "(per site: $(result.energy_mean / lattice.Ns))")

plt = plot(result.energy_trace, xlabel="Sample index (post-equilibration)", ylabel="Mixed-estimator energy",
           label="$(Lx)x$(Ly), U/t=$(U/t)", grid=true, dpi=800)
hline!(plt, [result.energy_mean], label="mean", linestyle=:dash)
savefig(plt, "afqmc_square_lattice_trace.png")
display(plt)
