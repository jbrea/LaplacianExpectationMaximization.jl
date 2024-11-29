module OptimizationExt

using FitPopulations, Optimization

function FitPopulations.maximize(opt::FitPopulations.OptimizationOptimizer, g!, params)
    of = OptimizationFunction((p, _) -> g!(true, nothing, nothing, p), opt.adtype,
                              grad = (dp, p, _) -> g!(true, dp, nothing, p),
                              hess = (H, p, _) -> g!(true, nothing, H, p))
    op = OptimizationProblem(of, params, sense = MaxSense)
    sol = solve(op, opt.optimizer; opt.options...)
    (; extra = sol)
end

end
