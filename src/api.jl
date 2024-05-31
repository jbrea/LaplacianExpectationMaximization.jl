"""
    parameters(model)

Returns a named tuple.
"""
function parameters end

"""
    initialize!(model, parameters)

Initalizes the state of the model.
"""
function initialize!(model, parameters) end

"""
    logp(data, model, parameters)

Returns ``\\log P(data|model, parameters)``.
"""
function logp end

"""
    sample(rng, data, model, parameters; kw...)

Takes already generated `data` as input and returns a new data point.
This function is called in the `simulate` function.
Keyword arguments are passed through the `simulate` function.
"""
function sample end

