mutable struct AbInitioWalker
    phi_up::Matrix{ComplexF64}
    phi_down::Matrix{ComplexF64}
    weight::Float64
end

function init_ab_initio_walkers(trial::AbInitioTrial, num_walkers::Int)
    return [AbInitioWalker(copy(trial.phi_up), copy(trial.phi_down), 1.0) for _ in 1:num_walkers]
end

# Complex analogue of orthonormalize! (real/Hubbard walker.jl): same
# argument for why no weight compensation is needed (theory doc section 10)
# applies unchanged with complex QR.
function orthonormalize_ab_initio!(walker::AbInitioWalker)
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
