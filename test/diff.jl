@testset "gradlogp & hesslogp" begin
    # Const
    m = BiasedCoin()
    p = ComponentArray(F.parameters(m))
    dp = zero(p)
    data, = F.simulate(m, p)
    g! = F.GradLogP(data, m)
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p) atol = 1e-8
    h! = F.HessLogP(data, m)
    H = zeros(length(p), length(p))
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    # HabituatingBiasedCoin
    m = HabituatingBiasedCoin()
    p = ComponentArray(F.parameters(m))
    dp = zero(p)
    data, = F.simulate(m, p)
    g! = F.GradLogP(data, m)
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h! = F.HessLogP(data, m)
    H = zeros(length(p), length(p))
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    p.w₀ = .1
    p.η = -.01
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    # DiagonalNormalPrior, no sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.DiagonalNormalPrior(), ())
    p = ComponentArray(F.parameters(m))
    dp = zero(p)
    data, = F.simulate(m.model, p)
    g! = F.GradLogP(data, m)
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h! = F.HessLogP(data, m)
    H = zeros(length(p), length(p))
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    p.population_parameters.μ .+= .1
    p.η = .01
    p.w₀ = -.2
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    # DiagonalNormalPrior, sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.DiagonalNormalPrior(), (:η,))
    p = ComponentArray(F.parameters(m))
    dp = zero(p)
    data, = F.simulate(m.model, p)
    g! = F.GradLogP(data, m)
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h! = F.HessLogP(data, m)
    H = zeros(length(p), length(p))
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
    p.population_parameters.μ .+= .1
    p.η = .01
    p.w₀ = -.2
    g!(dp, p)
    @test dp ≈ FiniteDiff.finite_difference_gradient(p -> F.logp(data, m, p), p)
    h!(H, p)
    @test H ≈ FiniteDiff.finite_difference_hessian(p -> F.logp(data, m, p), p) rtol = 1e-4
end

@testset "ForwardDiff & Enzyme" begin
    m = BiasedCoin()
    p = ComponentArray(F.parameters(m))
    dp = zero(p)
    data, = F.simulate(m, p)
    @test F.gradient_logp(data, m, p, ad = :ForwardDiff) == F.gradient_logp(data, m, p, ad = :Enzyme)
    @test_broken F.hessian_logp(data, m, p, ad = :ForwardDiff) == F.hessian_logp(data, m, p, ad = :Enzyme) # see e.g. https://github.com/EnzymeAD/Enzyme.jl/issues/1385
end
