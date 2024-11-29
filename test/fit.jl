@testset "fix" begin
    # HabituatingBiasedCoin
    m = HabituatingBiasedCoin()
    p = ComponentArray(F.parameters(m))
    p.w₀ = .1; p.η = -.2
    p0 = copy(p)
    data, = F.simulate(m, p, n_steps = 20)
    f = x -> F.logp(data, m, x)
    g! = F.GradLogP(AutoEnzyme(), data, m)
    h! = F.HessLogP(AutoForwardDiff(), data, m)
    p.η = .1
    fix, = F.fix(data, m, p, (; η = -.2), (), 0)
    p.η = .3
    @test fix.x.η == -.2
    @test fix(true, nothing, nothing, p[KeepIndex(:w₀)]) == F.logp(data, m, p0)
    @test fix(true, nothing, nothing, [.1]) == F.logp(data, m, p0)
    @test fix.x.η == -.2
    dp = [0.]
    fix(true, dp, nothing, [.1])
    @test dp[1] ≈ F.gradient_logp(data, m, p0).w₀
    @test fix.x.η == -.2
    # repeat to be sure nothing changed in the state
    fix(true, dp, nothing, [.1])
    @test dp[1] ≈ F.gradient_logp(data, m, p0).w₀
    @test fix.x.η == -.2
    # same game with w₀
    fix, = F.fix(data, m, p, (; w₀ = .1), (), 0)
    p.w₀ = .3
    p.η = -.2
    @test fix.x.w₀ == .1
    @test fix(true, nothing, nothing, p[KeepIndex(:η)]) == F.logp(data, m, p0)
    @test fix(true, nothing, nothing, [-.2]) == F.logp(data, m, p0)
    @test fix.x.w₀ == .1
    dp = [0.]
    fix(true, dp, nothing, [-.2])
    @test dp[1] ≈ F.gradient_logp(data, m, p0).η
    @test fix.x.w₀ == .1
    # repeat to be sure nothing changed in the state
    fix(true, dp, nothing, [-.2])
    @test dp[1] ≈ F.gradient_logp(data, m, p0).η
    @test fix.x.w₀ == .1
    # Hessian
    fix, = F.fix(data, m, p, (; η = -.2), (), 0)
    H = zeros(1, 1)
    fix(false, nothing, H, [.1])
    @test H ≈ F.hessian_logp(data, m, p0)[1:1, 1:1]
    # HabituatingMarkovChain
    m = HabituatingMarkovChain(3)
    p = ComponentArray(F.parameters(m))
    p.c = 4.
    p.η₀ = .5
    p0 = copy(p)
    data, = F.simulate(m, p, n_steps = 20, init = [(1, 1)])
    f = x -> F.logp(data, m, x)
    g! = F.GradLogP(AutoForwardDiff(), data, m) # Enzyme 0.13.16 seems broken on this one.
    h! = F.HessLogP(AutoForwardDiff(), data, m)
    p.c = 0.
    fix, = F.fix(data, m, p, (; c = 4., η₀ = .5), (), 0, gradient_ad = AutoForwardDiff())
    p.c = 1.
    @test fix.x.c .== 4.
    @test fix(true, nothing, nothing, p[KeepIndex(:w₀)]) == F.logp(data, m, p0)
    x = vcat(p0.w₀...)
    @test fix(true, nothing, nothing, x) == F.logp(data, m, p0)
    dp = zero(x)
    fix(true, dp, nothing, x)
    @test dp ≈ vcat(F.gradient_logp(data, m, p0).w₀...)
    fix, = F.fix(data, m, p0, (; w₀ = collect(p0.w₀), η₀ = .5), (), 0, gradient_ad = AutoForwardDiff())
    p.c = 1.
    p.w₀[1] .= randn(3)
    @test fix(true, nothing, nothing, p0[KeepIndex(:c)]) == F.logp(data, m, p0)
    x = [p0.c]
    @test fix(true, nothing, nothing, x) == F.logp(data, m, p0)
    dp = zero(x)
    fix(true, dp, nothing, x)
    @test dp ≈ [F.gradient_logp(data, m, p0).c]
    # L2
    m = HabituatingBiasedCoin()
    p = ComponentArray(F.parameters(m))
    p.w₀ = .1; p.η = -.2
    data, = F.simulate(m, p, n_steps = 20)
    f = x -> F.logp(data, m, x)
    g! = F.GradLogP(AutoEnzyme(), data, m)
    h! = F.HessLogP(AutoForwardDiff(), data, m)
    λ = .1
    fix, = F.fix(data, m, p, (;), (), λ)
    f_fd = p -> logp(data, F._convert_eltype(eltype(p), m), p) - λ/2 * sum(abs2, p)
    dp = zero(p)
    H = zeros(length(p), length(p))
    lp = fix(true, dp, H, p)
    @test lp ≈ f_fd(p)
    @test dp ≈ ForwardDiff.gradient(f_fd, p)
    @test H ≈ ForwardDiff.hessian(f_fd, p)
    # one fixed
    p.η = 1.
    fix, = F.fix(data, m, p, (; η = p.η), (), λ)
    f_fd = p -> logp(data, F._convert_eltype(eltype(p), m), p) - λ/2 * p.w₀^2
    dp = [0.]
    H = zeros(1, 1)
    lp = fix(true, dp, H, [p.w₀])
    @test lp ≈ f_fd(p)
    @test dp[1] ≈ ForwardDiff.gradient(f_fd, p).w₀
    @test H[1, 1] ≈ ForwardDiff.hessian(f_fd, p)[1, 1]
    # HabituatingBiasedCoin coupled
    m = HabituatingBiasedCoin()
    p = ComponentArray(F.parameters(m))
    p.w₀ = .1; p.η = .1
    p0 = copy(p)
    data, = F.simulate(m, p, n_steps = 20)
    fix, _p = F.fix(data, m, p, (;), ((:w₀, :η),), 0)
    @test length(_p) == 1
    @test fix(true, nothing, nothing, _p) == F.logp(data, m, p0)
    dp = zero(_p)
    H = zeros(length(_p), length(_p))
    lp = fix(true, dp, H, _p)
    f_fd = p -> logp(data, F._convert_eltype(eltype(p), m), ComponentArray(; w₀ = p.w₀, η = p.w₀))
    @test dp ≈ ForwardDiff.gradient(f_fd, _p)
    @test H ≈ ForwardDiff.hessian(f_fd, _p)
end

using JLD2
@testset "maximize_logp" begin
    Random.seed!(123)
    m = HabituatingBiasedCoin()
    p = ComponentArray(F.parameters(m))
    p.w₀ = .1; p.η = -.2
    data, = F.simulate(m, p, n_steps = 200)
    res1 = F.maximize_logp(data, m,
                           gradient_ad = AutoForwardDiff(),
                           verbosity = 0)
    res2 = F.maximize_logp(data, m,
                           gradient_ad = AutoEnzyme(),
                           verbosity = 0)
    res3 = F.maximize_logp(data, m, F.parameters(m),
                           gradient_ad = AutoEnzyme(),
                           verbosity = 0)
    res4 = F.maximize_logp(data, m, F.parameters(m),
                           gradient_ad = AutoEnzyme(),
                           optimizer = F.OptimizationOptimizer(LBFGS(), AutoForwardDiff(), (;)),
                           verbosity = 0)
    @test res1.logp ≈ res2.logp atol = 1e-1
    @test res3.logp ≈ res2.logp atol = 1e-1
    popdata = [F.simulate(m, p, n_steps = 20).data for _ in 1:30]
    popm = PopulationModel(m)
    filename = tempname() * ".jld2"
    res4 = F.maximize_logp(popdata, popm, evaluate_training = true, print_interval = 1, callbacks = [F.Callback((F.EventTrigger((:end,)), F.TimeTrigger(2)), F.CheckPointSaver(filename, overwrite = true))])
    res5 = load(filename)
    @test res4.logp == res5[string(last(sort(parse.(Int, keys(res5)))))].fmax
end
