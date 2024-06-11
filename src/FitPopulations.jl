module FitPopulations
using Random, LinearAlgebra
using Enzyme, ForwardDiff
using ConcreteStructs, ComponentArrays, Distributions
using DocStringExtensions, Printf
using NLopt, Optimisers
using JLD2
import Optim

export initialize!, parameters, logp, sample, PopulationModel, maximize_logp, simulate, logp_tracked, mc_marginal_logp, BIC_int, gradient_logp, hessian_logp, Callback, CheckPointSaver, EventTrigger, LogProgress, OptimisersOptimizer, Optimizer, IterationTrigger, Evaluator, NLoptOptimizer, TimeTrigger, LaplaceEM

include("api.jl")
include("simulate.jl")
include("populations.jl")
include("helper.jl")
include("diff.jl")
include("fit.jl")

end # module
