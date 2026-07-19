using LatticeMC
using Plots
using Printf

# Ground-state energy per site of the half-filled 2D Hubbard model via
# determinant QMC (DQMC), which is SIGN-PROBLEM-FREE at the particle-hole-
# symmetric point -- exact up to Trotter (dtau) and statistical error, with no
# constrained-path/trial bias. Shown for a 6x6 lattice at U/t=4 with a
# dtau -> 0 extrapolation, plus the sign-free diagnostic. Compare to the
# projector-AFQMC numbers for the same system: constrained-path (biased) and
# free projection (has a sign problem here) -- see afqmc docs.
lat = LatticeMC.build_hubbard_square(6, 6, 1.0, 4.0; pbc=true)
beta = 8.0

dtaus = [0.125, 0.1, 0.0625, 0.05]
energies = Float64[]
errors = Float64[]
for dtau in dtaus
    L = round(Int, beta / dtau)
    r = LatticeMC.run_dqmc(lat; beta=beta, L=L, num_thermal=200, num_sweeps=800, nwrap=8, seed=7)
    push!(energies, r.energy_per_site)
    push!(errors, r.energy_per_site_err)
    @printf("dtau=%.4f (L=%d): E/site = %.4f +/- %.4f   double_occ=%.4f   neg_moves=%.4g\n",
            dtau, L, r.energy_per_site, r.energy_per_site_err, r.double_occupancy, r.neg_move_fraction)
end

# linear extrapolation in dtau^2 (leading Trotter order) to dtau -> 0
x = dtaus .^ 2
A = [ones(length(x)) x]
coef = A \ energies          # [intercept, slope]
e0 = coef[1]
@printf("dtau -> 0 extrapolated ground-state energy per site (beta=%.1f): %.4f\n", beta, e0)

plt = plot(x, energies, yerror=errors, marker=:circle, label="DQMC (6x6, U/t=4, half filling)",
           xlabel="dtau^2", ylabel="ground-state energy per site (J)",
           title="Half-filled 2D Hubbard: DQMC (sign-problem-free), dtau -> 0",
           grid=true, dpi=800)
scatter!(plt, [0.0], [e0], marker=:star5, markersize=8, label=@sprintf("extrapolated %.4f", e0))
savefig(plt, "dqmc_trotter_extrapolation.png")
display(plt)
