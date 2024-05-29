# Priors
struct FlatPrior end
parameters(::FlatPrior, ::Any) = (;)
logp(::Any, ::FlatPrior, ::Any) = 0.
struct DiagonalNormalPrior end
function parameters(::DiagonalNormalPrior, params)
    (μ = _zero(params), σ = _one(params))
end
function logp(params, ::DiagonalNormalPrior, pop_params)
    μ = pop_params.μ
    σ = pop_params.σ
    _logp = 0.
    for k in eachindex(μ) # keys(μ) would be less fragile, but allocating
        _logp += logpdf(Normal(μ[k], σ[k]), params[k])
    end
    _logp
end
function resample!(rng, ::DiagonalNormalPrior, params)
    μ = params.population_parameters.μ
    σ = params.population_parameters.σ
    for k in eachindex(μ) # keys(μ) would be less fragile, but allocating
        params[k] = rand(rng, Normal(μ[k], σ[k]))
    end
    params
end
struct NormalPrior end
function parameters(::NormalPrior, params)
    (μ = _zero(params), Σ = ones(length(params), length(params)))
end
function logp(params, ::NormalPrior, pop_params)
    μ = pop_params.μ
    Σ = pop_params.Σ
    logpdf(MvNormal(μ, Σ), params(keys(μ)))
end
function resample!(rng, ::NormalPrior, params)
    μ = params.population_parameters.μ
    Σ = params.population_parameters.Σ
    params[keys(μ)] .= rand(rng, MvNormal(μ, Σ))
    params
end

# PopulationModel
@concrete terse struct PopulationModel
    model
    prior
    shared
end
PopulationModel(model, prior, shared::Symbol) = PopulationModel(model, prior, (shared,))
"""
    PopulationModel(model; prior = DiagonalNormalPrior(), shared = ())

Wrap a model for estimating population parameters. Shared parameters should be given as a tuple of symbols.
"""
PopulationModel(model; prior = DiagonalNormalPrior(), shared = ()) = PopulationModel(model, prior, shared)
function parameters(m::PopulationModel)
    params = parameters(m.model)
    params_nonshared = drop(params, m.shared)
    population_parameters = parameters(m.prior, params_nonshared)
    merge(params_nonshared, (; population_parameters), params[m.shared]) # order matter! see above
end
function logp(data, model::PopulationModel, params)
    logp(data, model.model, params) + logp(params, model.prior, params.population_parameters)
end

# marginal
function logmeanexp(x)
    _max = maximum(x)
    _max + log(1/length(x) * sum(exp.(x .- _max)))
end
function _mc_marginal_logp(data, model::PopulationModel, params;
        n_samples = 10^4, rng = Random.default_rng())
    _params = copy(params)
    sum(logmeanexp([logp(d, model.model, resample!(rng, model.prior, _params))
                    for _ in 1:n_samples]) for d in data)
end
"""
    mc_marginal_logp(data, model::PopulationModel, params;
                     repetitions = 20, n_samples = 10^4, rng = Random.default_rng())

Estimate the marginal log probability of the data given a model by sampling from the population.
"""
function mc_marginal_logp(data, model::PopulationModel, params;
        repetitions = 20, n_samples = 10^4, rng = Random.default_rng())
    n_pop_params = length(params.population_parameters)
    if n_pop_params == 0 # no sampling needed
        n_samples = 1
        repetitions = 1
    end
    [_mc_marginal_logp(data, model, params; n_samples, rng)
     for _ in 1:repetitions] |> sort
end
BIC(logp, k, n) = -2*logp + k * log(n)
"""
$SIGNATURES
Estimate the Bayesian Information Criterion by sampling from the population.
Keyword arguments `kw` are passed to [`mc_marginal_logp`](@ref).
"""
function BIC_int(data, model::PopulationModel, params; kw...)
    k = length(params.population_parameters) + length(model.shared)
    n = sum(length.(data))
    logp = mc_marginal_logp(data, model, params; kw...)
    BIC.(logp, k, n)
end
