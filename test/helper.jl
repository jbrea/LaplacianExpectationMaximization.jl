@testset "copy_elements" begin
    x = ComponentVector(a = 2., b = rand(3), c = (a = rand(2, 2),))
    y = ComponentVector(a = 5, d = 4, b = randn(3), c = (a = randn(2, 2), b = 3))
    F.copy_elements!(y, x)
    @test y.a == x.a
    @test y.b == x.b
    @test y.c.a == x.c.a
    @test y.d == 4
    @test y.c.b == 3
end

@testset "distribute_shared" begin
    p1 = ComponentArray(sample1 = (a = 3, b = 2), sample2 = (a = 4, b = 3))
    @test F.distribute_shared(p1) == p1
    p2 = ComponentArray(sample1 = (a = 3, b = 2), sample2 = (a = 4, b = 3), __shared = [])
    @test F.distribute_shared(p2) == p1
    p3 = ComponentArray(sample1 = (a = 3, b = 2), sample2 = (a = 4, b = 3), __shared = (c = 7,))
    @test F.distribute_shared(p3) == ComponentArray(sample1 = (a = 3, b = 2, c = 7), sample2 = (a = 4, b = 3, c = 7))
    p4 = ComponentArray(sample1 = (a = 3,), sample2 = (a = 4,), __shared = (b = 0, c = 7))
    @test F.distribute_shared(p4) == ComponentArray(sample1 = (a = 3, b = 0, c = 7), sample2 = (a = 4, b = 0, c = 7))
end

@testset "drop" begin
    m = F.PopulationModel(HabituatingBiasedCoin(), F.DiagonalNormalPrior(), ())
    p = F.parameters(m)
    @test F.drop(p, (:η, :population_parameters)) == (; w₀ = 0)
    p = ComponentArray(p)
    @test F.drop(p, (:η, :population_parameters)) == ComponentArray(w₀ = 0)
    p2 = F.drop_population_parameters(m.prior, p, (; η = 0.))
    @test p2.population_parameters.μ == ComponentArray(w₀ = 0)
    @test p2.population_parameters.σ == ComponentArray(w₀ = 1)
    m = F.PopulationModel(BiasedCoin(), F.DiagonalNormalPrior(), ())
    p = F.parameters(m)
    @test F.drop(p, (:population_parameters,)) == (; w = p.w)
    p = ComponentArray(p)
    @test F.drop(p, (:population_parameters,)) == ComponentArray(w = p.w)
    p2 = F.drop_population_parameters(m.prior, p, (; w = 0.))
    @test p2.population_parameters.μ == []
    @test p2.population_parameters.σ == []
end
