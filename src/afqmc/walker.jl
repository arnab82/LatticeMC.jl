mutable struct Walker
    phi_up::Matrix{Float64}
    phi_down::Matrix{Float64}
    weight::Float64
end

function init_walkers(trial::TrialWavefunction, num_walkers::Int)
    return [Walker(copy(trial.phi_up), copy(trial.phi_down), 1.0) for _ in 1:num_walkers]
end

# Thin-QR reorthonormalization for numerical stability. No weight compensation
# is needed: as long as trial overlaps are always recomputed from whatever
# matrix is currently stored (never mixed with a pre-orthonormalization cache),
# the det(R) factors removed here cancel exactly against the same factors in
# the next step's overlap ratio.
function orthonormalize!(walker::Walker)
    Nup = size(walker.phi_up, 2)
    Ndown = size(walker.phi_down, 2)
    if Nup > 0
        Qu = qr(walker.phi_up).Q
        walker.phi_up = Matrix(Qu)[:, 1:Nup]
    end
    if Ndown > 0
        Qd = qr(walker.phi_down).Q
        walker.phi_down = Matrix(Qd)[:, 1:Ndown]
    end
    return walker
end
