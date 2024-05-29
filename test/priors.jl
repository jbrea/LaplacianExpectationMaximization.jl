@testset "NormalPrior" begin
    p = ComponentVector(a = 2., b = 3., c = 1.,
                        pop_params = (μ = (a = 3., b = 3.), σ = (a = .3, b = 1.)))
    logp = logpdf(Normal(3, .3), 2.) + logpdf(Normal(3, 1.), 3.)
    @test F.logp(p, F.DiagonalNormalPrior(), p.pop_params) ≈ logp
end
