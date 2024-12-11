using LaplacianExpectationMaximization
import LaplacianExpectationMaximization as F
import LaplacianExpectationMaximization: parameters, logp, sample, simulate, initialize!
using Random, Statistics
using Optimization, OptimizationOptimJL, Enzyme, Optimisers
using ComponentArrays, Distributions, ConcreteStructs, LogExpFunctions
using FiniteDiff, ForwardDiff, ADTypes
using Test

include("examples.jl")
include("diff.jl")
include("priors.jl")
include("helper.jl")
include("fit.jl")
include("populations.jl")
include("model_recovery.jl")

## NOTE add JET to the test environment, then uncomment
using JET
@testset "static analysis with JET.jl" begin
    @test isempty(JET.get_reports(report_package(LaplacianExpectationMaximization, target_modules=(LaplacianExpectationMaximization,))))
end

## NOTE add Aqua to the test environment, then uncomment
@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(LaplacianExpectationMaximization; ambiguities = false)
    # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_ambiguities(LaplacianExpectationMaximization)
end
