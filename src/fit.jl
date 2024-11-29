###
### OptimizationTracker
###

evaluator_function(data, model; kw...) = x -> logp(data, model, x.xmax)
evaluator_function(data, model::PopulationModel; kw...) = x -> mc_marginal_logp(data, model, x.xmax[1]; kw...)
return_result(ev::Evaluator) = NamedTuple{(ev.label,)}((ev.evaluations,))
return_result(cb::Callback) = return_result(cb.func)
return_result(::Any) = (;)

@concrete terse mutable struct OptimizationTracker
    i
    g
    fmax
    xmax
    callbacks
end
function wrap_tracker(g, params; callbacks = [])
    OptimizationTracker(0, g, -Inf, copy(params), callbacks)
end
function (t::OptimizationTracker)(f, g, H, x)
    t.i += 1
    ret = t.g(f, g, H, x)
    if ret !== nothing && ret > t.fmax
        t.fmax = ret
        t.xmax .= x
    end
    state = (i = t.i, t = time(), f = ret, g, H, x,
             fmax = t.fmax,
             xmax = _parameters(t.g),
             g! = t)
    for cb in t.callbacks
        cb(state, ret)
    end
    ret
end
population_parameters(o::OptimizationTracker) = population_parameters(o.g)
_parameters(o::OptimizationTracker) = _parameters(o.g)
function results(o, extra = (;))
    res = merge((; logp = o.fmax[], parameters = _parameters(o)), extra)
    pp = population_parameters(o)
    if !isnothing(pp)
        res = merge((; population_parameters = pp), res)
    end
    for cb_return in return_result.(o.callbacks)
        res = merge(res, cb_return)
    end
    res
end

###
### Fix
###

@concrete struct Fix
    mask_idxs
    h!
    H
    g!
    g
    f
    x
    λ
end
Base.show(io::IO, ::Fix) = println(io, "Fix{...}")
function (g::Fix)(f, dx, H, x)
    _g = g.g
    _x = g.x
    _H = g.H
    λ = g.λ
    for (i, j) in g.mask_idxs
        _x[j] = x[i]
    end
    if dx !== nothing && length(dx) > 0
        dx .= 0
        res = g.g!(_g, _x)
        for (i, j) in g.mask_idxs
            dx[i] += _g[j]
        end
        if λ > 0
            @. dx -= λ * x
        end
    else
        if f !== nothing
            res = g.f(_x)
        end
    end
    if H !== nothing
        H .= 0
        g.h!(_H, _x)
        for (i, j) in g.mask_idxs
            for (k, l) in g.mask_idxs
                H[i, k] += _H[j, l]
            end
        end
        if λ > 0
            for i in axes(H, 1)
                H[i, i] -= λ
            end
        end
    end
    if f !== nothing
        if λ > 0
            res - λ/2 * sum(abs2, x)
        else
            res
        end
    end
end
_indices(x::ComponentArray, label) = getindex(getaxes(x)[1], label).idx
function fix(parameters, fixed, coupled)
    if !isempty(fixed)
        x = ComponentArray(; drop(parameters, keys(fixed))..., fixed...) # this is because of fragile logp in populations!! Urgently need a less fragile approach!
        mask_idxs = collect(pairs(vcat(_indices.(Ref(x), filter(∉(keys(fixed)), keys(x)))...)))
    else
        x = parameters
        mask_idxs = collect(pairs(1:length(x)))
    end
    if !isempty(coupled)
        mapping = Pair{Int,Int}[]
        for (_first, _rest...) in coupled
            target_idxs = _indices(x, _first)
            for r in _rest
                source_idxs = _indices(x, r)
                if length(source_idxs) == 1
                    x[source_idxs] = x[target_idxs]
                    push!(mapping, source_idxs => target_idxs)
                else
                    x[source_idxs] .= x[target_idxs]
                    append!(mapping, source_idxs .=> target_idxs)
                end
            end
        end
        mapping_dict = Dict(mapping...)
        mask_idxs = [get(mapping_dict, i, i) for i in first.(mask_idxs)] .=> last.(mask_idxs)
    end
    (x, mask_idxs, drop(parameters, union(keys(fixed), Base.tail.(coupled)...)))
end
function fix(data, model, parameters, fixed, coupled, λ;
        gradient_ad = AutoEnzyme(), hessian_ad = AutoForwardDiff())
    x, mask_idxs, params = fix(parameters, fixed, coupled)
    (Fix(mask_idxs,
         HessLogP(hessian_ad, data, model, x),
         zeros(length(x), length(x)),
         GradLogP(gradient_ad, data, model, x),
         zero(x),
         x -> logp(data, model, x),
         copy(x),
         λ),
     params)
end
function _set_params_to_population_mean!(x)
    μ = x.population_parameters.μ
    for i in eachindex(μ)
        x[i] = μ[i]
    end
    x
end
function population_parameters(f::Fix)
    haskey(f.x, :population_parameters) || return nothing
    _set_params_to_population_mean!(copy(f.x))
end
_parameters(f::Fix) = f.x

###
### evaluate
###

function evaluate(data_train, data_test, model::PopulationModel, parameters; kw...)
    (train = mc_marginal_logp(data_train, model, parameters; kw...),
     test = mc_marginal_logp(data_test, model, parameters; kw...))
end
function evaluate(data_train, ::Nothing, model::PopulationModel, parameters; kw...)
    (train = mc_marginal_logp(data_train, model, parameters; kw...),)
end
function evaluate(data_train, data_test, model, parameters; kw...)
    (train = logp(data_train, model, parameters; kw...),
     test = logp(data_test, model, parameters; kw...))
end
function evaluate(data_train, ::Nothing, model, parameters; kw...)
    (train = logp(data_train, model, parameters; kw...),)
end

###
### PopGradLogP
###
@concrete struct PopGradLogP
    mask_idxs
    g_funcs
    gs
    ps
    shared
end
Base.show(io::IO, ::PopGradLogP) = println(io, "PopGradLogP{...}")
function PopGradLogP(gs, ps, shared)
    lbls = labels(ps[1])
    shared_idxs = findall(l -> split(l, ('[', '.'))[1] ∈ string.(shared), lbls)
    nonshared_idxs = setdiff(1:length(lbls), shared_idxs)
    n_nonshared = length(nonshared_idxs)
    target_idxs = [nonshared_idxs; shared_idxs]
    source_shared_idxs = (1:length(shared_idxs)) .+ length(ps) * n_nonshared
    mask_idxs = [collect(zip([i*n_nonshared+1:(i+1)*n_nonshared; source_shared_idxs], target_idxs))
                 for i in 0:length(ps)-1]
    PopGradLogP(mask_idxs, gs, zero.(ps), ps, shared)
end
function (g::PopGradLogP)(f, dx, H, x)
    isnothing(H) || @warn "Cannot compute Hessians for PopulationModels."
    if !isnothing(dx) && length(dx) > 0
        dx .= 0
    end
    logp = 0.
    for k in eachindex(g.mask_idxs)
        p = g.ps[k]; dp = g.gs[k]
        for (i, j) in g.mask_idxs[k]
            p[j] = x[i]
        end
        if !isnothing(dx) && length(dx) > 0
            logp += g.g_funcs[k](f, dp, nothing, p)
            for (i, j) in g.mask_idxs[k]
                dx[i] += dp[j]
            end
        else
            logp += g.g_funcs[k](f, nothing, nothing, p)
        end
    end
    f === nothing || return logp
end
population_parameters(p::PopGradLogP) = population_parameters(p.g_funcs[1])
_parameters(p::PopGradLogP) = [_parameters(g) for g in p.g_funcs]

###
### gradient function
###

function gradient_function(data, model, parameters, fixed, coupled, λ; kw...)
    fix(data, model, parameters, fixed, coupled, λ; kw...)
end
function all_parameters(ps, shared)
    _ps = Array.(drop.(ps, Ref((shared..., :population_parameters))))
    float.(vcat(_ps..., Array(ps[1][shared])))
end
function gradient_function(data, model::PopulationModel, parameters, fixed, coupled, λ; kw...)
    _parameters = drop_population_parameters(model.prior, ComponentArray(parameters), fixed)
    fixed = merge(fixed, (; population_parameters = _parameters.population_parameters))
    tmp = [fix(d, model, _parameters, fixed, coupled, λ; kw...) for d in data]
    gs = first.(tmp); ps = last.(tmp)
    (PopGradLogP(gs, ps, model.shared),
     all_parameters(ps, drop(model.shared,
                             union(keys(fixed), Base.tail.(coupled)...))))
end


###
### Optimizers
###
"""
    OptimizationOptimizer(optimizer, adtype, options)

Uses [Optimization.jl](https://docs.sciml.ai/Optimization/stable). `options` are passed to `Optimization.solve`

### Example
```
using FitPopulations, Optimization, OptimizationOptimJL, ADTypes
OptimizationOptimizer(LBFGS(), AutoForwardDiff(), (;))
"""
@concrete struct OptimizationOptimizer
    optimizer
    adtype
    options
end
"""
    NLoptOptimizer(optimizer; options...)

The `optimizer` is a symbol (e.g. `:LD_LBGFS`) as specified [here](https://github.com/JuliaOpt/NLopt.jl?tab=readme-ov-file#the-opt-type). For options, see [NLopt options](https://github.com/JuliaOpt/NLopt.jl?tab=readme-ov-file#using-with-mathoptinterface).
"""
@concrete struct NLoptOptimizer
    opt::Symbol
    options
end
function NLoptOptimizer(name; kw...)
    NLoptOptimizer(name, kw)
end
"""
    OptimOptimizer(optimizer; options...)

Optimizer can be anything from `subtypes.(subtypes(Optim.AbstractOptimizer))`.
For options, see [Optim Options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/#General-Options).
"""
@concrete struct OptimOptimizer
    opt
    options
end
OptimOptimizer(opt; iterations = 10^6, kw...) = OptimOptimizer(opt, Optim.Options(; iterations, kw...))
@concrete terse struct SwapSign
    g!
end
function (s::SwapSign)(f, g, H, x)
    res = s.g!(f, g, H, x)
    g === nothing || (g .*= -1)
    H === nothing || (H .*= -1)
    f === nothing || return -res
    nothing
end
function maximize(opt::OptimOptimizer, g!, params)
    (; extra = Optim.optimize(Optim.only_fgh!(SwapSign(g!)), params,
                              opt.opt, opt.options))
end
"""
    OptimisersOptimizer(opt; maxeval = 10^5, maxtime = Inf, min_grad_norm = 1e-8, lower_bounds = -Inf, upper_bounds = Inf)

Optimizer `opt` can be anything from `subtypes(Optimisers.AbstractRule)`.
Optimization stops, when the L2-norm of the gradient falls below `min_grad_norm` or `maxeval` or `maxtime` is reached. See also [Optimisers](https://fluxml.ai/Optimisers.jl/dev/api/).
"""
Base.@kwdef @concrete struct OptimisersOptimizer
    opt = Optimisers.Adam()
    maxeval = 10^5
    maxtime = Inf
    min_grad_norm = 1e-8
    lower_bounds = -Inf
    upper_bounds = Inf
end
function maximize(opt::OptimisersOptimizer, g!, params)
    tstart = time()
    _opt = opt.opt
    ub = opt.upper_bounds
    lb = opt.lower_bounds
    maxtime = opt.maxtime
    min_grad_norm = opt.min_grad_norm
    state = Optimisers.init(_opt, params)
    dparams = zero(params)
    for _ in 1:opt.maxeval
        logp = g!(true, dparams, nothing, params)
        state, dp = Optimisers.apply!(opt.opt, state, params, dparams)
        params .+= dp
        @. params = max(min(params, ub), lb)
        (time() - tstart > maxtime || sqrt(sum(abs2, dparams)) < min_grad_norm) && break
    end
    (;)
end
"""
    Optimizer(; optimizer = OptimOptimizer(Optim.LBFGS()), finetuner = nothing, fallback = OptimisersOptimizer())

Default optimizer. `finetuner` can be another optimizer that is called after the first `optimizer` finished. The `fallback` optimizer is called, if the `optimizer` or `finetuner` fails.

Can also be constructed as

    Optimizer(optimizer; finetuner = nothing, fallback = OptimisersOptimizer(), kw...)

where `optimizer` can be a symbol (to use NLopt), or an optimiser from Optim or Optimiser.
See also [`NLoptOptimizer`](@ref), [`OptimOptimizer`](@ref), [`OptimisersOptimizer`](@ref).
"""
Base.@kwdef @concrete struct Optimizer
    optimizer = OptimOptimizer(Optim.LBFGS())
    finetuner = nothing
    fallback = OptimisersOptimizer()
end
function Optimizer(optimizer; finetuner = nothing, fallback = OptimisersOptimizer(maxeval = 10^5), kw...)
    if isa(optimizer, Symbol)
        opt = NLoptOptimizer(optimizer, kw)
    elseif isa(optimizer, Optimisers.AbstractRule)
        opt = OptimisersOptimizer(optimizer; kw...)
    else
        opt = OptimOptimizer(optimizer; kw...)
    end
    Optimizer(opt, finetuner, fallback)
end
maximize(::Nothing, ::Any, ::Any) = nothing
function maximize(opt::Optimizer, g!, params)
    try
        result = maximize(opt.optimizer, g!, params)
        if !isnothing(opt.finetuner)
            trigger!.(g!.callbacks, :start_finetuner)
            result_finetuner = maximize(opt.finetuner, g!, g!.xmax)
            result = merge(result, (; result_finetuner))
        end
        result
    catch e
        @warn e
        @info "Optimizing with fallback $(opt.fallback)."
        trigger!.(g!.callbacks, :start_fallback)
        maximize(opt.fallback, g!, g!.xmax)
    end
end

###
### maximize_logp
###

function default_optimizer(::Any, parameters = nothing; fixed = (;))
    Optimizer()
end
function default_optimizer(model::PopulationModel,
        parameters = parameters(model); fixed = (;))
    pp = parameters.population_parameters
    if length(pp) > 0 && length(pp.μ) > 0 && length(setdiff(keys(pp.μ), keys(fixed))) > 0
        LaplaceEM(; model)
    else
        default_optimizer(model.model)
    end
end
# TODO: Mostly done. Check if this can be futher improved. Use normal arrays everywhere except when calling logp to speed up compilation.
# TODO: Would it be possible to make this super generic, such that it runs on CPU/GPU, whatever your model runs on?
# TODO: Only compute diagonal of Hessian if only diagonal is needed. Could this easily be done with Enzyme?
# TODO: Don't require user to specify PopulationModel(model, ...)?
# TODO: document new interface (callbacks, triggers, stopper in LaplaceEM)
"""
    maximize_logp(data, model, parameters = parameters(model);
                  fixed = (;)
                  coupled = [],
                  optimizer = default_optimizer(model, parameters, fixed),
                  lambda_l2 = 0.,
                  hessian_ad = AutoForwardDiff(),
                  gradient_ad = AutoForwardDiff(),
                  evaluate_training = false,
                  evaluate_test_data = nothing,
                  evaluation_trigger = EventTrigger(),
                  evaluation_options = (;),
                  callbacks = [],
                  verbosity = 1, print_interval = 3,
                  return_g! = false,
                  )

"""
function maximize_logp(data, model, parameters = parameters(model);
        verbosity = 1, print_interval = 3, fixed = (;), return_g! = false,
        lambda_l2 = 0.,
        coupled = [],
        evaluate_training = false,
        evaluate_test_data = nothing,
        evaluation_trigger = EventTrigger((:start, :end, :iteration_end, :start_finetuner, :start_fallback)),
        evaluation_options = (;),
        optimizer = default_optimizer(model, parameters; fixed),
        hessian_ad = AutoForwardDiff(),
        gradient_ad = AutoForwardDiff(),
        callbacks = [],
    )
    gfunc, params = gradient_function(data, model, ComponentArray(parameters),
                                      NamedTuple(fixed),
                                      coupled, lambda_l2;
                                      gradient_ad, hessian_ad)
    if verbosity > 0
        callbacks = [callbacks; Callback(TimeTrigger(print_interval), LogProgress())]
    end
    if evaluate_training
        callbacks = [callbacks; Callback(evaluation_trigger,
                                         Evaluator(evaluator_function(data, model; evaluation_options...),
                                                   T = Vector{Float64}, label = :training_logp))]
    end
    if !isnothing(evaluate_test_data)
        callbacks = [callbacks; Callback(evaluation_trigger,
                                         Evaluator(evaluator_function(evaluate_test_data, model;
                                                                      evaluation_options...);
                                                   T = Vector{Float64}, label = :test_logp))]
    end
    if verbosity > 0 && isa(optimizer, LaplaceEM)
        callbacks = [callbacks; Callback(EventTrigger((:iteration_end,)), (_, _, _, _) -> println("Starting M-Step."))]
    end
    g! = wrap_tracker(gfunc, params; callbacks)
    trigger!.(callbacks, :start)
    extra = maximize(optimizer, g!, params)
    trigger!.(callbacks, :end)
    g!(true, nothing, nothing, g!.xmax) # run once with the optimal parameters such that g.x in the next line is certainly set to the optimum
    res = results(g!, extra)
    if return_g!
        res = merge(res, (; g!))
    end
    res
end

# TODO: better accessors
"""
    LaplaceEM(model, Estep_optimizer = Optimizer(), derivative_threshold = 1e-3, iterations = 10, stopper = () -> false)

Implements the Expectation-Maximization (EM) method with Laplace approximation, as described e.g. in [Huys et al. (2012)](http://dx.doi.org/10.1371/journal.pcbi.1002410).
"""
Base.@kwdef @concrete struct LaplaceEM
    model
    Estep_optimizer = Optimizer()
    derivative_threshold = 1e-3
    iterations = 10
    stopper = () -> false
end
function mstep!(::DiagonalNormalPrior, g!, Hs)
    ps = g!.g.ps
    N = length(ps)
    μ = sum(ps)/N
    second_moment_samples = sum(p.^2 for p in ps)/N
    free_idxs = 1:length(ps[1])
    second_moment_laplace = -sum(1 ./ clamp.(diag(Hs[i][free_idxs, free_idxs]), -Inf, -eps()) for i in eachindex(ps))/N
    d = second_moment_samples + second_moment_laplace - μ .^ 2
    σ = sqrt.(max.(eps(), d))
    for g in g!.g.g_funcs
        popp = g.x.population_parameters
        for i in eachindex(popp.μ) # shared and fixed parameters are at the end
            popp.μ[i] = μ[i]
            popp.σ[i] = σ[i]
        end
    end
end
function maximize(opt::LaplaceEM, g!, params)
    dp = zero(g!.xmax)
    g!.xmax .= params
    for i in 1:opt.iterations
        # E-step
        maximize(opt.Estep_optimizer, g!, g!.xmax)
        logp = g!(true, dp, nothing, g!.xmax) # sets parameters to optimum
        dp_max = maximum(abs, dp[1:end-length(opt.model.shared)])
        if dp_max > opt.derivative_threshold
            @warn "Skipping M-Step: E-Step may not have converged.\nThe partial derivative with largest absolute value is $dp_max > derivative_threshold = $(opt.derivative_threshold)."
        else
            Hs = [begin
                      H = zero(g.H)
                      g(nothing, nothing, H, g.x) # compute Hessians
                      H
                  end
                  for g in g!.g.g_funcs]
            # M-step
            mstep!(opt.model.prior, g!, Hs)
        end
        trigger!.(g!.callbacks, :iteration_end)
        opt.stopper() && break
    end
    (;)
end
