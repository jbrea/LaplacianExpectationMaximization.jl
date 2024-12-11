module EnzymeExt

using Enzyme
import LaplacianExpectationMaximization: _zero, _deep_ismutable, ComponentArray, parameters, AutoEnzyme, EnzGradLogP, EnzHessLogP, GradLogP, logp

function logp!(_logp, data, model, parameters)
    _logp[] = logp(data, model, parameters)
    nothing
end
function GradLogP(::AutoEnzyme, data, model,
        parameters = ComponentArray(parameters(model)))
    EnzGradLogP(Duplicated([0.], [1.]),
                  Const(data),
                  _deep_ismutable(model) ? Duplicated(model, _zero(model)) :
                                             Const(model),
                  Duplicated(parameters, _zero(parameters)))
end
function (g::EnzGradLogP)(dx, x)
    g.parameters.val .= x
    g.parameters.dval .= 0
    g.logp.dval .= 1
    g.logp.val .= 0
    autodiff(Reverse, logp!, g.logp, g.data, g.model, g.parameters)
    dx .= g.parameters.dval
    g.logp.val[]
end


function _grad1(logp, dlogp, data, model, params, dparams)
    autodiff_deferred(Reverse, logp!,
                      DuplicatedNoNeed(logp, dlogp),
                      Const(data),
                      Const(model),
                      Duplicated(params, dparams))
    nothing
end
function _grad2(logp, dlogp, data, model, dmodel, params, dparams)
    autodiff_deferred(Reverse, logp!,
                      DuplicatedNoNeed(logp, dlogp),
                      Const(data),
                      DuplicatedNoNeed(model, dmodel),
                      Duplicated(params, dparams))
    nothing
end
function _hess(h::EnzHessLogP{<:Const})
    autodiff(Forward, _grad1, h.logp, h.dlogp, h.data,
                              h.model, h.parameters, h.dparameters)
end
function _hess(h::EnzHessLogP)
    autodiff(Forward, _grad2, h.logp, h.dlogp, h.data,
                             h.model, h.dmodel, h.parameters, h.dparameters)
end
function HessLogP(::AutoEnzyme, data, model,
        parameters = ComponentArray(parameters(model)))
    n = length(parameters)
    vx = ntuple(i -> begin t = _zero(parameters); t[i] = 1; t end, n)
    ddx = ntuple(_ -> _zero(parameters), n)
    EnzHessLogP(Const([0.]), Const([1.]),
                Const(data),
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
# TODO: EnzDiagHessLogP with forward over forward
end
