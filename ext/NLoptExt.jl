module NLoptExt

using NLopt, FitPopulations

function FitPopulations.maximize(opt::FitPopulations.NLoptOptimizer, g!, params)
    if opt.opt ∈ (:G_MLSL, :G_MLSL_LDS)
        o = Opt(opt.opt, length(params))
        _lopt = Opt(opt.options.local_optimizer, length(params))
        o.local_optimizer = _lopt
    else
        o = Opt(opt.opt, length(params))
        o.max_objective = (params, dparams) -> g!(true, dparams, nothing, params)
    end
    for (k, v) in pairs(FitPopulations.drop(NamedTuple(opt.options), (:local_optimizer,)))
        setproperty!(o, k, v)
    end
    o.max_objective = (params, dparams) -> g!(true, dparams, nothing, params)
    _, _, extra = NLopt.optimize(o, params)
    (; extra)
end

end
