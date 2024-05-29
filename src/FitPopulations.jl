"""
Placeholder for a short summary about FitPopulations.
"""
module FitPopulations
using Random, LinearAlgebra
using Enzyme, ForwardDiff
using ConcreteStructs, ComponentArrays, Distributions
using DocStringExtensions, Printf
using NLopt, Optimisers
import Optim

include("api.jl")
include("simulate.jl")
include("populations.jl")
include("helper.jl")
include("diff.jl")
include("fit.jl")

end # module
