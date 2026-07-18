using LatticeMC
using Plots

Ns = 8
t = 1.0
U = 4.0
Nup, Ndown = 4, 4

lattice = LatticeMC.build_hubbard_chain(Ns, t, U; pbc=true)
trial = LatticeMC.build_trial_wavefunction(lattice, Nup, Ndown)

result = LatticeMC.run_afqmc(lattice, trial, Nup, Ndown;
                              dtau=0.01, num_walkers=200, num_steps=4000,
                              equilibration_steps=500, stabilize_every=5,
                              pop_control_every=10)

println("E = $(result.energy_mean) +/- $(result.energy_err)  (per site: $(result.energy_mean / Ns))")

plt = plot(result.energy_trace, xlabel="Sample index (post-equilibration)", ylabel="Mixed-estimator energy",
           label="Ns=$Ns, U/t=$(U/t)", grid=true, dpi=800)
hline!([result.energy_mean], label="mean", linestyle=:dash)
savefig(plt, "afqmc_energy_trace.png")
display(plt)
