using FitPopulations
import FitPopulations as F
import FitPopulations: parameters, logp, sample, simulate, initialize!
using Random, Statistics
using ComponentArrays, Distributions, ConcreteStructs, LogExpFunctions
using FiniteDiff, ForwardDiff
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
    @test isempty(JET.get_reports(report_package(FitPopulations, target_modules=(FitPopulations,))))
end

## NOTE add Aqua to the test environment, then uncomment
@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(FitPopulations; ambiguities = false)
    # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_ambiguities(FitPopulations)
end
