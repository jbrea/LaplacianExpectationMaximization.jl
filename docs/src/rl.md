# Fitting a Q-Learner

We define a Q-Learner that explores an environment with 10 states and 3 actions with softmax policy.

```@example rl
using FitPopulations
using ConcreteStructs
import LogExpFunctions: softplus, logistic

@concrete struct QLearner
    q
end
QLearner() = QLearner(zeros(10, 3))

function FitPopulations.initialize!(m::QLearner, parameters)
    m.q .= parameters.q₀
end

FitPopulations.parameters(::QLearner) = (; q₀ = zeros(10, 3), η = 0., β_real = 1., γ_logit = 10.)

function FitPopulations.logp(data, m::QLearner, parameters)
    initialize!(m, parameters)
    (; η, β_real, γ_logit) = parameters
    β = softplus(β_real)
    γ = logistic(γ_logit)
    q = m.q
    logp = 0.
    for (; s, a, s′, r, done) in data
        logp += logsoftmax(β, q, s, a)
        td_error = r + γ * findmax(view(q, s′, :))[1] - q[s, a]
        q[s, a] += η * td_error
    end
    logp
end

function FitPopulations.sample(rng, data, m::QLearner, parameters; environment)
    (; η, β_real) = parameters
    q = m.q
    (; s′, done) = data[end]
    if done
        s′ = initial_state(rng, environment)
    end
    a′ = randsoftmax(rng, softplus(β_real), q, s′)
    s′′ = transition(rng, environment, s′, a′)
    r′ = reward(rng, environment, s′, a′, s′′)
    (s = s′, a = a′, s′ = s′′, r = r′, done = isdone(rng, environment, s′′))
end
```

Let us define the helper functions `logsoftmax`, `randsoftmax` and the environment functions `initial_state`, `transition`, `reward`, `isdone`.

```@example rl
using Distributions, LinearAlgebra

logsoftmax(β, q, s, a) = β * q[s, a] - logsumexp(β, view(q, s, :))
function logsumexp(β, v)
    m = β * maximum(v)
    sumexp = zero(eltype(v))
    for vᵢ in v
        sumexp += exp(β * vᵢ - m)
    end
    m + log(sumexp)
end
function randsoftmax(rng, β, q, s)
    m = maximum(view(q, s, :))
    p = exp.(q[s, :] .- m)
    p ./= sum(p)
    rand(rng, Categorical(p))
end
@concrete struct Environment
    t
    r
end
function Environment(; rng = Random.default_rng())
    Environment([normalize(rand(rng, 10), 1) for s in 1:10, a in 1:3],
                randn(rng, 10, 3))
end
initial_state(rng, ::Environment) = rand(rng, 1:10)
transition(rng, e::Environment, s, a) = rand(rng, Categorical(e.t[s, a]))
reward(rng, e::Environment, s, a, s′) = e.r[s, a]
isdone(rng, e::Environment, s) = s == 10
```

With this we can generate some artificial data and fit it.

```@example rl
using ComponentArrays, Random

model = QLearner()
p = ComponentArray(parameters(model))
p.η = .15
p.β_real = 1.6
p.γ_logit = 1.8
rng = Xoshiro(17)
tmp = [simulate(model, p;
                n_steps = 200, rng,
                init = [(s = 1, a = 1, s′ = 1, r = 0., done = false)],
                environment = Environment(; rng))
       for _ in 1:100]
data = first.(tmp)
data_logp = sum(last.(tmp))
```
This is the probability with which the data was generated.
Let us check the data probability under the default parameters.

```@example rl
population_model = PopulationModel(model, shared = (:q₀, :η, :β_real, :γ_logit))
p0 = ComponentArray(parameters(population_model))
mc_marginal_logp(data, population_model, p0)
```
We see that the data probability under the default parameters is higher than under the parameter with which the data was generated. Now we maximize the log-likelihood, to find the optimal parameters for this data.

```@example rl
result = maximize_logp(data, population_model)
result.logp
```
The resulting data probability is indeed the highest. The fitted parameters are, however, not super close to `p`:
```@example rl
result.population_parameters
```

We can try fixing `q₀`:

```@example rl
result2 = maximize_logp(data, population_model, fixed = (; q₀ = zeros(10, 3)))
result2.logp
```
The data probability is a bit lower, as expected.
```@example rl
result2.population_parameters
```
The fitted parameters, however, are closer to `p`.
