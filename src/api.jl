"""
$SIGNATURES
Returns a named tuple.
"""
function parameters(model) end

"""
$SIGNATURES
Initalizes the state of the model.
"""
function initialize!(model, parameters) end

"""
$SIGNATURES
Returns ``\\log P(data|model, parameters)``.
"""
function logp(data, model, parameters) end

"""
    sample(rng, data, model, parameters; kw...)

Takes already generated `data` as input and returns a new data point.
This function is called in the `simulate` function.
Keyword arguments are passed through the `simulate` function.
"""
function sample(rng, data, model, parameters) end

