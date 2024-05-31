# FitPopulations

## Example

### Model Definition

For a sequence of binary values ``y = (y_1, \ldots, y_T)`` we define a habituating biased coin model with probability ``P(y) = \prod_{t=1}^TP(y_t|w_{t-1})`` with ``w_t = w_{t-1} + \eta (y_{t-1} - \sigma(w_{t-1}))``, where ``w_0`` and ``\eta`` are parameters of the model and ``\sigma(w) = 1/(1 + \exp(-w))``.

We define the model `HabituatingBiasedCoin` with state variable `w` and extend the functions `parameters, initialize!, logp` and `sample` from `FitPopulations`.

```@example hbc
using FitPopulations
using ConcreteStructs, Distributions

@concrete struct HabituatingBiasedCoin
    w
end
HabituatingBiasedCoin() = HabituatingBiasedCoin(Base.RefValue(0.))

function FitPopulations.initialize!(m::HabituatingBiasedCoin, parameters)
    m.w[] = parameters.w₀
end

FitPopulations.parameters(::HabituatingBiasedCoin) = (; w₀ = 0., η = 0.)

sigmoid(w) = 1/(1 + exp(-w))

function FitPopulations.logp(data, m::HabituatingBiasedCoin, parameters)
    initialize!(m, parameters)
    η = parameters.η
    logp = 0.
    for yₜ in data
        ρ = sigmoid(m.w[])
        logp += logpdf(Bernoulli(ρ), yₜ)
        m.w[] += η * (yₜ - ρ)
    end
    logp
end

function FitPopulations.sample(rng, ::Any, m::HabituatingBiasedCoin, ::Any)
    rand(rng) ≤ sigmoid(m.w[])
end
```

### Generating data

Let us generate 5 sequences of 30 steps with this model.

```@example hbc
import FitPopulations: simulate

model = HabituatingBiasedCoin()
params = (; w₀ = .3, η = .1)
data = [simulate(model, params, n_steps = 30).data for _ in 1:5]
```

### Fitting a single model

First we check, if gradients are properly computed for our model.

```@example hbc
import FitPopulations: gradient_logp
gradient_logp(data[1], model, params)
```

If this fails, it is recommended to check that `logp` does not allocate, e.g. with
```@example hbc
using BenchmarkTools
@benchmark logp($(data[1]), $model, $params)
```

We also check if Hessians are properly computed.

```@example hbc
import FitPopulations: hessian_logp
hessian_logp(data[1], model, params)
```

This may fail, if the model is too restrictive in its type parameters.

If everything works fine we run the optimizer:

```@example hbc
import FitPopulations: maximize_logp
result = maximize_logp(data[1], model)
```

To inspect the state of the fitted model we can run
```@example hbc
import FitPopulations: logp_tracked
logp_tracked(data[1], model, result.parameters).history
```

We can also fit with some fixed parameters.
```@example hbc
result = maximize_logp(data[1], model, fixed = (; η = 0.))
result.parameters
```
or with coupled parameters
```@example hbc
result = maximize_logp(data[1], model, coupled = [(:w₀, :η)])
result.parameters
```

### Fitting a population model

Now we fit all data samples with approximate EM, assuming a diagonal normal prior over the parameters.

```@example hbc
import FitPopulations: PopulationModel
pop_model1 = PopulationModel(model)
result1 = maximize_logp(data, pop_model1)
result1.population_parameters
```

Let us compare this to a model where all samples are assumed to be generated from the same parameters, i.e. the variance of the normal prior is zero.

```@example hbc
pop_model2 = PopulationModel(model, shared = (:w₀, :η))
result2 = maximize_logp(data, pop_model2)
result2.population_parameters
```

To compare the models we look at the approximate BIC
```@example hbc
import FitPopulations: BIC_int
(model1 = BIC_int(data, pop_model1, result1.population_parameters, repetitions = 1),
 model2 = BIC_int(data, pop_model2, result2.population_parameters))
```
We see that the second model without variance of the prior has the lower BIC. This is not surprising, given that the data was generated with identical parameters.
