# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, FitPopulations

makedocs(
    modules = [FitPopulations],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "FitPopulations.jl",
    pages = Any["Home" => "index.md",
                "Reference" => "reference.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/jbrea/FitPopulations.jl.git",
    push_preview = true
)
