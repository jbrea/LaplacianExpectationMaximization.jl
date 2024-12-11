module LaplacianExpectationMaximization
using Random, LinearAlgebra
using ADTypes
using ForwardDiff
using ConcreteStructs, ComponentArrays, Distributions
using DocStringExtensions, Printf
using OptimizationCallbacks
import OptimizationCallbacks: trigger!
import Optim, Optimisers

export initialize!, parameters, logp, sample, PopulationModel, maximize_logp, simulate, logp_tracked, mc_marginal_logp, BIC_int, gradient_logp, hessian_logp, OptimisersOptimizer, Optimizer, NLoptOptimizer, OptimizationOptimizer, OptimOptimizer, LaplaceEM

include("api.jl")
include("simulate.jl")
include("populations.jl")
include("helper.jl")
include("diff.jl")
include("fit.jl")

end # module
