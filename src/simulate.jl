@concrete terse struct Sampler
    model
    parameters
    rng
    stop
    data
    kw
end
function Base.iterate(s::Sampler, i = 0)
    d = sample(s.rng, s.data, s.model, s.parameters; s.kw...)
    s.stop(s.data, i) && return nothing
    push!(s.data, d)
    d, i+1
end

struct FixedSteps
    T::Int
end
(s::FixedSteps)(::Any, i) = i ≥ s.T

"""
$SIGNATURES
Returns a named tuple `(; data, logp)`.
`stop(data, i)` is a boolean function that depends on the sequence of simulated `data` and the iteration counter `i`. If `tracked = true` the state of the model is saved for every step in the simulation.
Additional keyword arguments `kw` are passed to the `sample` function.
"""
function simulate(model, parameters;
        n_steps = 20,
        stop = FixedSteps(n_steps),
        init = [],
        tracked = false,
        rng = Random.default_rng(),
        kw...)
    initialize!(model, parameters)
    sampler = Sampler(model, parameters, rng, stop, deepcopy(init), kw)
    if tracked
        sampler = TrackModel(sampler, model)
    end
    _logp = logp(sampler, model, parameters)
    (data = eltype(sampler.data) === Any ? [sampler.data...] :
            sampler.data,
     logp = _logp)
end

###
### Tracker
###
@concrete terse struct TrackModel
    data
    model
    history
end
TrackModel(data, model) = TrackModel(data, model, [deepcopy(model)])
function Base.iterate(t::TrackModel, next = iterate(t.data))
    next === nothing && return nothing
    push!(t.history, deepcopy(t.model))
    first(next), iterate(t.data, last(next))
end
"""
$SIGNATURES
Returns a names tuple `(; history, logp)`.
In the history the state of the model is saved for every step.
"""
function logp_tracked(data, model, parameters)
    initialize!(model, parameters)
    tracker = TrackModel(data, model)
    _logp = logp(tracker, model, parameters)
    (history = tracker.history, logp = _logp)
end
