@testset "populations" begin
    # flat prior, no sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.FlatPrior(), ())
    p = F.parameters(m)
    data, logp = F.simulate(m.model, p)
    @test F.logp(data, m, p) ≈ logp
    # flat prior, sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.FlatPrior(), (:w₀,))
    p = ComponentVector(F.parameters(m))
    p.η = .01
    p.w₀ = .1
    @test keys(p)[end] === :w₀
    data, logp = F.simulate(m.model, p)
    @test F.logp(data, m, p) ≈ logp
    # flat prior, sharing, grad
    g = F.GradLogP(Val(:Enzyme), data, m)
    dp = zero(p)
    g(dp, p)
    # normal prior, no sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.DiagonalNormalPrior(), ())
    p = F.parameters(m)
    data, logp = F.simulate(m.model, p)
    @test F.logp(data, m, p) ≈ logp + logpdf(Normal(0, 1), p.w₀) + logpdf(Normal(0, 1), p.η)
    # normal prior, sharing
    m = F.PopulationModel(HabituatingBiasedCoin(), F.DiagonalNormalPrior(), (:w₀,))
    p = ComponentVector(F.parameters(m))
    p.η = .01
    p.w₀ = .1
    data, logp = F.simulate(m.model, p)
    @test F.logp(data, m, p) ≈ logp + logpdf(Normal(0, 1), p.η)
    # flat prior, sharing, grad
    g = F.GradLogP(Val(:Enzyme), data, m)
    dp = zero(p)
    g(dp, p)
end

@testset "PopGradLogP" begin
    m = F.PopulationModel(HabituatingBiasedCoin(), prior = F.DiagonalNormalPrior())
    p = F.parameters(m)
    data = [rand((true, false), 10) for _ in 1:3]
    g!, optp = F.gradient_function(data, m, p, (;), (), 0)
    @test g!.mask_idxs == [[(1, 1), (2, 2)], [(3, 1), (4, 2)], [(5, 1), (6, 2)]]
    @test length(optp) == 6
    @test isa(F.default_optimizer(m, p; fixed = (;)), F.LaplaceEM)
    g!, optp = F.gradient_function(data, m, p, (; η = 0), (), 0)
    @test g!.mask_idxs == [[(1, 1)], [(2, 1)], [(3, 1)]]
    @test length(optp) == 3
    m = F.PopulationModel(HabituatingBiasedCoin(), prior = F.DiagonalNormalPrior(), shared = :η)
    p = F.parameters(m)
    g!, optp = F.gradient_function(data, m, p, (;), (), 0)
    @test g!.mask_idxs == [[(1, 1), (4, 2)], [(2, 1), (4, 2)], [(3, 1), (4, 2)]]
    @test length(optp) == 4
    m = F.PopulationModel(HabituatingBiasedCoin(), prior = F.DiagonalNormalPrior(), shared = :η)
    p = F.parameters(m)
    g!, optp = F.gradient_function(data, m, p, (; w₀ = 0.), (), 0)
    @test g!.mask_idxs == [[(1, 1)], [(1, 1)], [(1, 1)]]
    @test length(optp) == 1
    m = F.PopulationModel(HabituatingBiasedCoin(), prior = F.DiagonalNormalPrior(), shared = (:η, :w₀))
    p = F.parameters(m)
    g!, optp = F.gradient_function(data, m, p, (;), (), 0)
    @test g!.mask_idxs == [[(1, 1), (2, 2)], [(1, 1), (2, 2)], [(1, 1), (2, 2)]]
    @test length(optp) == 2
end

@testset "BIC_int" begin
    m = F.PopulationModel(GaussianModel(), prior = F.DiagonalNormalPrior(), shared = (:μ, :σ))
    p = ComponentArray(parameters(m))
    flat_data, lp = simulate(m.model, p, n_steps = 200)
    n = length(flat_data)
    data = [flat_data[i*20+1:(i+1)*20] for i in 0:9]
    @test mean(F.BIC_int(data, m, p)) ≈ -2 * lp + length(m.shared) * log(n)
end

@testset "population recovery" begin
    # identical coins
    Random.seed!(123)
    m1 = F.PopulationModel(BiasedCoin(), prior = F.DiagonalNormalPrior())
    data = [simulate(m1.model, (; w = .3), n_steps = 20)[1] for _ in 1:20]
    res1 = F.maximize_logp(data, m1, verbosity = 0)
    res1fw = F.maximize_logp(data, m1, verbosity = 0, gradient_ad = Val(:ForwardDiff))
    m2 = F.PopulationModel(BiasedCoin(), prior = F.DiagonalNormalPrior(), shared = :w)
    res2 = F.maximize_logp(data, m2, verbosity = 0)
    @test sigmoid(res2.parameters[1].w) ≈ mean(vcat(data...))
    @test mean(F.BIC_int(data, m1, res1.population_parameters, n_samples = 10^3)) > mean(F.BIC_int(data, m2, res2.population_parameters))
    # different coins
    m1 = F.PopulationModel(BiasedCoin(), prior = F.DiagonalNormalPrior())
    data = [simulate(m1.model, (; w = .4*randn() + .3), n_steps = 50)[1] for _ in 1:50]
    res1 = F.maximize_logp(data, m1, verbosity = 0)
    @test sigmoid(res1.population_parameters.population_parameters.μ.w) ≈ mean(mean(data)) atol = 1e-2
    @test res1.population_parameters.population_parameters.σ.w ≈ .4 atol = .1
    m2 = F.PopulationModel(BiasedCoin(), prior = F.DiagonalNormalPrior(), shared = :w)
    res2 = F.maximize_logp(data, m2, verbosity = 0)
    @test mean(F.BIC_int(data, m1, res1.population_parameters, n_samples = 10^3)) < mean(F.BIC_int(data, m2, res2.population_parameters))
end
