###
### Enzyme
###
@concrete terse struct EnzGradLogP
    logp
    data
    model
    parameters
end
Base.show(io::IO, ::EnzGradLogP) = println(io, "EnzGradLogP{...}")
# hessian
@concrete terse struct EnzHessLogP
    logp
    dlogp
    data
    model
    dmodel
    parameters
    dparameters
end
Base.show(io::IO, ::EnzHessLogP) = println(io, "EnzHessLogP{...}")

###
### ForwardDiff
###

@concrete struct FWHessLogP
    f
    cfg
end
Base.show(io::IO, ::FWHessLogP) = println(io, "FWHessLogP{...}")
function HessLogP(::AutoForwardDiff, data, model,
        parameters = ComponentArray(parameters(model)))
    f = p -> logp(data, _convert_eltype(eltype(p), model), p)
    cfg = ForwardDiff.HessianConfig(f, parameters)
    FWHessLogP(f, cfg)
end
function (h::FWHessLogP)(ddx, x)
    ddx .= ForwardDiff.hessian(h.f, x, h.cfg)
end
@concrete struct FWGradLogP
    f
    cfg
end
Base.show(io::IO, ::FWGradLogP) = println(io, "FWGradLogP{...}")
function GradLogP(::AutoForwardDiff, data, model,
        parameters = ComponentArray(parameters(model)))
    f = p -> logp(data, _convert_eltype(eltype(p), model), p)
    cfg = ForwardDiff.DiffResults.GradientResult(parameters)
    FWGradLogP(f, cfg)
end
function (h::FWGradLogP)(dx, x)
    ForwardDiff.gradient!(h.cfg, h.f, x)
    dx .= ForwardDiff.DiffResults.gradient(h.cfg)
    ForwardDiff.DiffResults.value(h.cfg)
end

"""
    gradient_logp(data, model, parameters; ad = AutoEnzyme())

Compute the gradient of `logp`.
"""
function gradient_logp(data, model, parameters; ad = AutoForwardDiff())
    g! = GradLogP(ad, data, model, parameters)
    dp = zero(parameters)
    g!(dp, parameters)
    dp
end
gradient_logp(data, model, p::NamedTuple; kw...) = gradient_logp(data, model, ComponentArray(p); kw...)
"""
    hessian_logp(data, model, parameters; ad = AutoForwardDiff())

Compute the hessian of `logp`.
"""
function hessian_logp(data, model, parameters; ad = AutoForwardDiff())
    h! = HessLogP(ad, data, model, parameters)
    H = zeros(length(parameters), length(parameters))
    h!(H, parameters)
end
hessian_logp(data, model, p::NamedTuple; kw...) = hessian_logp(data, model, ComponentArray(p); kw...)
