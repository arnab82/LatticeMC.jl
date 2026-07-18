mutable struct IsingModel
    """
    parameters:
    lattice: lattice of the Ising model
    N: size of the lattice
    J: coupling constant
    _________________________
    returns:
    IsingModel object
    """
    lattice::Matrix{Int}
    N::Int
    J::Float64

    function IsingModel(N, J::Float64=1.0,positive_spins=true,percent_spin_type::Float64=0.75)
        init_random = rand(N, N)
        lattice = zeros(Int, N, N)

        for i in 1:N
            for j in 1:N
                if positive_spins
                    lattice[i, j] = init_random[i, j] <= percent_spin_type ? 1 : -1
                    lattice[i, j] = init_random[i, j] > percent_spin_type ? -1 : 1

                else
                    lattice[i, j] = init_random[i, j] < percent_spin_type ? -1 : 1
                    lattice[i, j] = init_random[i, j] >= percent_spin_type ? +1 : -1
                end
            end
        end

        new(lattice, N, J)
    end
end

function display_lattice(model::IsingModel)
    println("Lattice:\n")
    display(model.lattice)
end

# Example :
# N = 50
# lattice_p = IsingModel(N, true)
# # display_lattice(lattice_p)

# lattice_n= IsingModel(N, false)
# display_lattice(lattice_n)
# heatmap(lattice_n.lattice, xlabel="Column", ylabel="Row", title="Lattice with 75% Negative spins)", aspect_ratio=1)

function energy_manual(lattice_ising::IsingModel)
    """
    Calculates energy of the lattice manually
    parameters:
    lattice_ising: IsingModel object
    _________________________
    returns:
    energy: energy of the lattice

    """
    energy = 0
    lattice=lattice_ising.lattice
    N=lattice_ising.N
    J=lattice_ising.J

    for i in 1:N
        for j in 1:N
            # periodic boundary conditions
            i_p1 = i == N ? 1 : i + 1
            j_m1 = j == 1 ? N : j - 1
            i_m1 = i == 1 ? N : i - 1
            j_p1 = j == N ? 1 : j + 1

            energy += J*(-lattice[i, j] * (lattice[i_p1, j] + lattice[i, j_m1] + lattice[i_m1, j] + lattice[i, j_p1]))/2
        end
    end

    return energy
end


function energy_periodic(lattice_ising::IsingModel)
    """
    Calculates energy of the lattice manually
    parameters:
    lattice_ising: IsingModel object
    _________________________
    returns:
    energy: energy of the lattice
    
    """
    lattice=lattice_ising.lattice
    J=lattice_ising.J
    energy = 0
    N=lattice_ising.N

    for i in 1:N
        for j in 1:N
            i_p1 = mod1(i + 1, N)
            j_m1 = mod1(j - 1, N)
            i_m1 = mod1(i - 1, N)
            j_p1 = mod1(j + 1, N)

            energy += J*(-lattice[i, j] * (lattice[i_p1, j] + lattice[i, j_m1] + lattice[i_m1, j] + lattice[i, j_p1]))/2
        end
    end

    return energy
end
# energy_manual(lattice_n)
# energy_periodic(lattice_n)


