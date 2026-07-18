using Documenter
using LatticeMC

makedocs(;
    sitename="LatticeMC.jl",
    authors="Arnab Bachhar",
    modules=[LatticeMC],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://arnab82.github.io/LatticeMC.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "AFQMC theory (Hubbard)" => "afqmc_theory.md",
        "AFQMC algorithm (pseudocode)" => "afqmc_algorithm.md",
        "Ab initio AFQMC theory" => "afqmc_ab_initio_theory.md",
        "Implementation notes" => "afqmc_implementation.md",
        "API reference" => "api.md",
    ],
    # These docs are hand-written narrative pages with cross-links and no
    # @docs docstring blocks; don't hard-fail CI on Documenter's stricter
    # link/reference checks -- warnings are enough here.
    warnonly=true,
)

deploydocs(;
    repo="github.com/arnab82/LatticeMC.jl.git",
    devbranch="master",
)
