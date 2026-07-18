struct MolecularIntegrals
    Ns::Int
    h1e::Matrix{Float64}
    h2e::Array{Float64,4}
    E_nuc::Float64
end

# Parses the standard FCIDUMP text format (Knowles & Handy convention, real
# orbitals): a namelist header giving NORB, followed by lines
# `value i j k l` where k=l=0 gives a one-electron integral h1e[i,j], all
# indices zero gives the nuclear-repulsion/core energy, and otherwise gives
# a two-electron integral h2e[i,j,k,l] = (ij|kl) in chemist notation, filled
# out using the real-orbital 8-fold permutation symmetry.
function read_fcidump(path::AbstractString)
    lines = readlines(path)
    header_end = findfirst(l -> occursin("&END", uppercase(l)) || strip(l) == "/", lines)
    header_end === nothing && throw(ArgumentError("could not find end of FCIDUMP header (&END or /)"))
    header = uppercase(join(lines[1:header_end], " "))

    norb_match = match(r"NORB\s*=\s*(\d+)", header)
    norb_match === nothing && throw(ArgumentError("FCIDUMP header missing NORB"))
    Ns = parse(Int, norb_match.captures[1])

    h1e = zeros(Float64, Ns, Ns)
    h2e = zeros(Float64, Ns, Ns, Ns, Ns)
    E_nuc = 0.0

    for line in lines[header_end+1:end]
        toks = split(strip(line))
        isempty(toks) && continue
        val = parse(Float64, toks[1])
        i, j, k, l = (parse(Int, toks[m]) for m in 2:5)

        if i == 0
            E_nuc = val
        elseif k == 0 && l == 0
            h1e[i, j] = val
            h1e[j, i] = val
        else
            for (p, q, r, s) in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                  (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i))
                h2e[p, q, r, s] = val
            end
        end
    end

    return MolecularIntegrals(Ns, h1e, h2e, E_nuc)
end
