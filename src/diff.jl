###
### Enzyme
###
function logp!(_logp, data, model, parameters)
    _logp[] = logp(data, model, parameters)
    nothing
end
@concrete terse struct GradLogP
    logp
    data
    model
    parameters
end
Base.show(io::IO, ::GradLogP) = println(io, "GradLogP{...}")
function GradLogP(data, model, parameters = ComponentArray(parameters(model)))
    dd = _zero(data)
    GradLogP(Duplicated([0.], [1.]),
                 Duplicated(convert(typeof(dd), data), dd),
                 _deep_ismutable(model) ? Duplicated(model, _zero(model)) :
                                          Const(model),
                 Duplicated(parameters, _zero(parameters)))
end
function (g::GradLogP)(dx, x)
    g.parameters.val .= x
    g.parameters.dval .= 0
    g.logp.dval .= 1
    g.logp.val .= 0
    autodiff(Reverse, logp!, g.logp, g.data, g.model, g.parameters)
    dx .= g.parameters.dval
    g.logp.val[]
end
# hessian
@concrete terse struct EnzHessLogP
    logp
    dlogp
    data
    ddata
    model
    dmodel
    parameters
    dparameters
end
Base.show(io::IO, ::EnzHessLogP) = println(io, "EnzHessLogP{...}")
function _grad1(logp, dlogp, data, ddata, model, params, dparams)
    autodiff_deferred(Reverse, logp!,
                      DuplicatedNoNeed(logp, dlogp),
                      DuplicatedNoNeed(data, ddata),
                      model,
                      Duplicated(params, dparams))
    nothing
end
function _grad2(logp, dlogp, data, ddata, model, dmodel, params, dparams)
    autodiff_deferred(Reverse, logp!,
                      DuplicatedNoNeed(logp, dlogp),
                      DuplicatedNoNeed(data, ddata),
                      DuplicatedNoNeed(model, dmodel),
                      Duplicated(params, dparams))
    nothing
end
function _hess(h::EnzHessLogP{<:Const})
    autodiff(Forward, _grad1, h.logp, h.dlogp, h.data, h.ddata,
                              h.model, h.parameters, h.dparameters)
end
function _hess(h::EnzHessLogP)
    autodiff(Forward, _grad2, h.logp, h.dlogp, h.data, h.ddata,
                             h.model, h.dmodel, h.parameters, h.dparameters)
end
function _EnzymeHessLogP(data, model, parameters = ComponentArray(parameters(model)))
    dd = _zero(data)
    _data = convert(typeof(dd), data)
    n = length(parameters)
    vx = ntuple(i -> begin t = _zero(parameters); t[i] = 1; t end, n)
    ddx = ntuple(_ -> _zero(parameters), n)
    EnzHessLogP(Const([0.]), Const([1.]),
              BatchDuplicated(_data, ntuple(_ -> _zero(dd), n)), BatchDuplicated(_zero(dd), ntuple(_ -> _zero(dd), n)),
              _deep_ismutable(model) ? BatchDuplicated(model, ntuple(_ -> _zero(model), n)) : Const(model),
              _deep_ismutable(model) ? BatchDuplicated(_zero(model), ntuple(_ -> _zero(model), n)) : Const(model),
              BatchDuplicated(parameters, vx), BatchDuplicated(_zero(parameters), ddx))
end
function (h::EnzHessLogP)(ddx, x)
    h.parameters.val .= x
    h.dlogp.val .= 1
    _hess(h)
    for (i, v) in pairs(h.dparameters.dval)
        ddx[:, i] .= v
        v .= 0
    end
    ddx
end

###
### ForwardDiff
###

@concrete struct FWHessLogP
    f
    cfg
end
Base.show(io::IO, ::FWHessLogP) = println(io, "FWHessLogP{...}")
function _ForwardDiffHessLogP(data, model, parameters = ComponentArray(parameters(model)))
    f = p -> logp(data, _convert_eltype(eltype(p), model), p)
    cfg = ForwardDiff.HessianConfig(f, parameters)
    FWHessLogP(f, cfg)
end
function (h::FWHessLogP)(ddx, x)
    ddx .= ForwardDiff.hessian(h.f, x, h.cfg)
end

###
### Generic
###
function HessLogP(data, model, parameters = ComponentArray(parameters(model)); ad = :ForwardDiff)
    constructor = if ad === :ForwardDiff
        _ForwardDiffHessLogP
    elseif ad === :Enzyme
        _EnzymeHessLogP
    end
    constructor(data, model, parameters)
end
"""
$SIGNATURES
Compute the gradient of `logp`.
"""
function gradient_logp(data, model, parameters)
    g! = GradLogP(data, model, parameters)
    dp = zero(parameters)
    g!(dp, parameters)
    dp
end
gradient_logp(data, model, p::NamedTuple) = gradient_logp(data, model, ComponentArray(p))
"""
    hessian_logp(data, model, parameters; ad = :ForwardDiff)

Compute the hessian of `logp`.
"""
function hessian_logp(data, model, parameters; ad = :ForwardDiff)
    h! = HessLogP(data, model, parameters; ad)
    H = zeros(length(parameters), length(parameters))
    h!(H, parameters)
end
hessian_logp(data, model, p::NamedTuple) = hessian_logp(data, model, ComponentArray(p))
