# Definition
struct GaussianModel end
parameters(::GaussianModel) = (μ = 0., σ = 0.)
function logp(data, ::GaussianModel, parameters)
    logp = 0.
    (; μ, σ) = parameters
    for d in data
        logp += logpdf(Normal(μ, log1pexp(σ)), d)
    end
    logp
end
function sample(rng, data, ::GaussianModel, parameters)
    (; μ, σ) = parameters
    rand(rng, Normal(μ, log1pexp(σ)))
end
@testset "GaussianModel" begin
    data = .3*randn(20) .+ 3
    m = GaussianModel()
    p = parameters(m)
    lp = sum(logpdf.(Normal(0, log1pexp(0)), data))
    @test lp ≈ logp(data, m, p)
    simdata, lp = simulate(m, ComponentArray(μ = .2, σ = .7), n_steps = 10^5)
    @test mean(simdata) ≈ .2 atol = 5e-2
    @test std(simdata) ≈ log1pexp(.7) atol = 5e-2
end

# Definition
struct BiasedCoin end
parameters(::BiasedCoin) = (; w = 0.)
sigmoid(w) = 1/(1 + exp(-w))
logsigmoid(x) = -log1pexp(-x)
function logp(data, ::BiasedCoin, parameters)
    logp = 0.
    w = parameters.w
    for d in data
        logp += logsigmoid((2d-1)*w)
    end
    logp
end
sample(rng, data, ::BiasedCoin, parameters) = rand(rng) ≤ sigmoid(parameters.w)
@testset "BiasedCoin" begin
    # logp and simulate
    data = rand(20) .≤ .7
    m = BiasedCoin()
    p = parameters(m)
    ρ = sigmoid(p.w)
    lp = sum(log(ρ) * data + log(1 - ρ) * (1 .- data))
    @test lp ≈ logp(data, m, p)
    simp = (; w = 4)
    simdata, lp = simulate(m, simp, n_steps = 100)
    @test lp ≈ logp(simdata, m, simp)
    @test sum(simdata) > 70
end

# Definition
@concrete struct HabituatingBiasedCoin
    w
end
HabituatingBiasedCoin() = HabituatingBiasedCoin(Base.RefValue(0.))
function initialize!(m::HabituatingBiasedCoin, parameters)
    m.w[] = parameters.w₀
end
parameters(::HabituatingBiasedCoin) = (; w₀ = 0., η = 0.)
function logp(data, m::HabituatingBiasedCoin, parameters)
    initialize!(m, parameters)
    η = parameters.η
    logp = 0.
    for d in data
        ρ = sigmoid(m.w[])
        logp += logpdf(Bernoulli(ρ), d)
        m.w[] += η * (d - ρ)
    end
    logp
end
function sample(rng, ::Any, m::HabituatingBiasedCoin, ::Any)
    rand(rng) ≤ sigmoid(m.w[])
end
@testset "HabituatingBiasedCoin" begin
    # logp and simulate
    data = [false, true, false]
    m = HabituatingBiasedCoin()
    @test length(parameters(m)) == 2
    p = (w₀ = .5, η = .1)
    w0 = p.w₀
    w1 = w0 - p.η * sigmoid(w0)
    w2 = w1 + p.η * (1 - sigmoid(w1))
    lp = log(sigmoid(-w0)) + log(sigmoid(w1)) + log(sigmoid(-w2))
    @test lp ≈ logp(data, m, p)
    simdata, lp = simulate(m, p, n_steps = 100)
    @test lp ≈ logp(simdata, m, p)
end

# Definition
softmax(x) = softmax!(copy(x))
@views function softmax!(x)
    for i in axes(x, 2)
        x[:, i] .-= maximum(x[:, i])
    end
    @. x = exp(x)
    for i in axes(x, 2)
        n = sum(x[:, i])
        x[:, i] ./= n
    end
    x
end
struct MarkovChain end
parameters(::MarkovChain) = (; w = randn(3, 3))
function logp(data, ::MarkovChain, parameters)
    logp = 0.
    w = softmax(parameters.w)
    for (d, d′) in data
        logp += log(w[d′, d])
    end
    logp
end
function wsample(rng, w)
    t = rand(rng)
    s = 0.
    for (i, wi) in pairs(w)
        s += wi
        s ≥ t && return i
    end
    return length(w)
end
function sample(rng, data, ::MarkovChain, parameters)
    d = data[end][2]
    d′ = wsample(rng, softmax(parameters.w[:, d]))
    (d, d′)
end
@testset "MarkovChain" begin
    m = MarkovChain()
    p = parameters(m)
    simdata, lp = simulate(m, p, n_steps = 20, init = [(1, 1)])
    @test lp ≈ logp(simdata[2:end], m, p)
end

# Definition
function softmax!(π, x)
    π .= x
    π .-= maximum(π)
    @. π = exp(π)
    π ./= sum(π)
end
function logsumexp!(π, x)
    _max = maximum(x)
    π .= x
    π .-= _max
    @. π = exp(π)
    _max + log(sum(π))
end
@concrete struct HabituatingMarkovChain
    w
    π
    η
    t
end
function HabituatingMarkovChain(n_states)
    HabituatingMarkovChain([zeros(n_states) for _ in 1:n_states],
                           zeros(n_states),
                           Ref(0.),
                           Ref(0))
end
function initialize!(m::HabituatingMarkovChain, parameters)
    for i in eachindex(m.w)
        m.w[i] .= parameters.w₀[i]
    end
    m.t[] = 0
    m.η[] = parameters.η₀
    m
end
function parameters(::HabituatingMarkovChain)
    (; w₀ = [rand(3) for _ in 1:3], η₀ = 0., c = 1)
end
function logp(data, m::HabituatingMarkovChain, parameters)
    initialize!(m, parameters)
    logp = 0.
    w = m.w
    π = m.π
    t = m.t
    η = m.η
    c = parameters.c
    for (d, d′) in data
        logp += w[d][d′] - logsumexp!(π, w[d])
        η[] = 1/(t[] + c)
        softmax!(π, w[d])
        w[d] .-= η[] * π
        w[d][d′] += η[]
        t[] += 1
    end
    logp
end
function sample(rng, data, m::HabituatingMarkovChain, parameters)
    d = data[end][2]
    d′ = wsample(rng, softmax!(m.π, m.w[d]))
    (d, d′)
end
@testset "HabituatingMarkovChain" begin
    m = HabituatingMarkovChain(3)
    p = parameters(m)
    simdata, lp = simulate(m, p, n_steps = 20, init = [(1, 1)])
    @test lp ≈ logp(simdata[2:end], m, p)
end
