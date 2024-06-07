@testset "single sample recovery" begin
    Random.seed!(123)
    # BiasedCoin
    m = BiasedCoin()
    p = ComponentArray(F.parameters(m))
    p.w = 1.4
    data, lp = F.simulate(m, p, n_steps = 10^4)
    @test sum(data)/10^4 ≈ sigmoid(p.w) atol = .05
    data, lp = F.simulate(m, p, n_steps = 100)
    fit = F.maximize_logp(data, m, maxeval = 10^4, verbosity = 0)
    @test fit.logp > lp
    @test sigmoid(fit.parameters.w) ≈ sum(data)/100
    m = HabituatingBiasedCoin()
    fit2 = F.maximize_logp(data, m, fixed = (; η = 0), maxeval = 10^4, verbosity = 0)
    @test fit2.parameters.w₀ ≈ fit.parameters.w
    @test fit2.logp ≈ fit.logp
end

