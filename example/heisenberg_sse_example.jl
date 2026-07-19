using LatticeMC
using Plots

# Ground-state energy per site of the 2D square-lattice S=1/2 antiferromagnetic
# Heisenberg model via SSE QMC, for L x L lattices up to 8x8 (a 2^64 Hilbert
# space -- far beyond exact diagonalization), extrapolated toward the
# thermodynamic limit (-0.669437 J/site, Sandvik). beta is taken ~ several
# times L so the finite-temperature result has converged to the ground state.
E_TD = -0.669437   # thermodynamic-limit reference

Ls = [2, 4, 6, 8]
energies = Float64[]
errors = Float64[]
for L in Ls
    lat = LatticeMC.build_heisenberg_square(L, L; pbc=true)
    beta = 4.0 * L
    r = LatticeMC.run_sse(lat; beta=beta, num_thermal=5000, num_measure=50000, seed=2024)
    push!(energies, r.energy_per_site)
    push!(errors, r.energy_per_site_err)
    println("$(L)x$(L) (Ns=$(lat.Ns), beta=$beta): E/site = " *
            "$(round(r.energy_per_site, digits=5)) +/- $(round(r.energy_per_site_err, digits=5))")
end
println("thermodynamic-limit reference: E/site = $E_TD")

# finite-size scaling plot vs 1/L^3 (the leading spin-wave finite-size
# correction for the square-lattice AFM goes ~ 1/L^3). Plot only the
# asymptotic-regime sizes L >= 4; the 2x2 lattice is too small to lie on the
# scaling line (reported above, omitted from the fit/plot).
mask = Ls .>= 4
x = (1 ./ Ls[mask]) .^ 3
plt = plot(x, energies[mask], yerror=errors[mask], marker=:circle, label="SSE (LxL, PBC)",
           xlabel="1 / L^3", ylabel="ground-state energy per site (J)",
           title="2D Heisenberg AFM: finite-size scaling to the TD limit",
           grid=true, dpi=800)
hline!(plt, [E_TD], label="thermodynamic limit (-0.669437)", linestyle=:dash)
savefig(plt, "heisenberg_finite_size_scaling.png")
display(plt)
