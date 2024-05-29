###
### OptimizationTracker
###

@concrete terse struct OptimizationTracker
    t0
    i
    g
    fmax
    xmax
    verbosity
    print_interval
end
function wrap_tracker(g, params; verbosity = 1, print_interval = 2)
    OptimizationTracker(Ref(0.), Ref(0), g, Ref(-Inf), copy(params), verbosity, float(print_interval))
end
function (t::OptimizationTracker)(f, g, H, x)
    if t.i[] == 0
        if t.verbosity > 0
            println(" eval   | current    | best")
            println("_"^33)
        end
        t.t0[] = time()
    end
    t.i[] += 1
    ret = t.g(f, g, H, x)
    if ret !== nothing && ret > t.fmax[]
        t.fmax[] = ret
        t.xmax .= x
    end
    t1 = time()
    if t.verbosity > 0 && t1 - t.t0[] > t.print_interval
        t.t0[] = t1
        @printf "%7i | %10.6g | %10.6g\n" t.i[] ret t.fmax[]
    end
    ret
end
population_parameters(o::OptimizationTracker) = population_parameters(o.g)
_parameters(o::OptimizationTracker) = _parameters(o.g)

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
# TODO: write tests for coupled and fix tests for fix (new argument)
# TODO: adapt PopGradLogP
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
function fix(data, model, parameters, fixed, coupled, λ)
    x, mask_idxs, params = fix(parameters, fixed, coupled)
    (Fix(mask_idxs,
        HessLogP(data, model, x),
        zeros(length(x), length(x)),
        GradLogP(data, model, x),
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

function gradient_function(data, model, parameters, fixed, coupled, λ)
    fix(data, model, parameters, fixed, coupled, λ)
end
function all_parameters(ps, shared)
    _ps = Array.(drop.(ps, Ref((shared..., :population_parameters))))
    float.(vcat(_ps..., Array(ps[1][shared])))
end
function gradient_function(data, model::PopulationModel, parameters, fixed, coupled, λ)
    _parameters = drop_population_parameters(model.prior, ComponentArray(parameters), fixed)
    fixed = merge(fixed, (; population_parameters = _parameters.population_parameters))
    tmp = [fix(d, model, _parameters, fixed, coupled, λ) for d in data]
    gs = first.(tmp); ps = last.(tmp)
    (PopGradLogP(gs, ps, model.shared),
     all_parameters(ps, drop(model.shared,
                             union(keys(fixed), Base.tail.(coupled)...))))
end

###
### maximize_logp
###

function default_optimizer(::Any, parameters, fixed)
    Optim.LBFGS()
#     length(parameters) - length(fixed) < 50 ? :LD_SLSQP : :LD_LBFGS
end
function default_optimizer(model::PopulationModel, parameters, fixed)
    pp = parameters.population_parameters
    if length(pp) > 0 && length(pp.μ) > 0 && length(setdiff(keys(pp.μ), keys(fixed))) > 0
        LaplaceEM()
    else
        default_optimizer(model.model, parameters, fixed)
    end
end
# TODO: Use normal arrays everywhere except when calling logp to speed up compilation. Mostly done. Check if this can be futher improved.
# TODO: Would it be possible to make this super generic, such that it runs on CPU/GPU, whatever your model runs on?
# TODO: Only compute diagonal of Hessian if only diagonal is needed. Could this easily be done with Enzyme?
# TODO: add kwarg equal_parameters = list of tuples; implement with mask_idxs
# TODO: Don't require user to specify PopulationModel(model, ...)?
# TODO: Should I create a separate handling for shared, coupled, fixed, transformed parameters or can I use e.g. ParameterHandling?
"""
    maximize_logp(data, model, parameters = ComponentArray(parameters(model));
                  fixed = (;)
                  coupled = [],
                  optimizer = default_optimizer(model, parameters, fixed),
                  lambda_l2 = 0.,
                  verbosity = 1, print_interval = 10,
                  return_g! = false,
                  evaluation = (;),
                  kw...)

"""
function maximize_logp(data, model, parameters = ComponentArray(parameters(model));
        verbosity = 1, print_interval = 10, fixed = (;), return_g! = false,
        lambda_l2 = 0.,
        coupled = [],
        optimizer = default_optimizer(model, parameters, fixed),
        evaluation = (;),
        kw...)
    gfunc, params = gradient_function(data, model, parameters,
                                      NamedTuple(fixed),
                                      coupled, lambda_l2)
    g! = wrap_tracker(gfunc, params; verbosity, print_interval)
    res = if isa(optimizer, LaplaceEM)
        maximize(optimizer, model, g!, params; verbosity, evaluation, kw...)
    else
        if !isempty(evaluation)
            evaluations = [evaluate(data, evaluation.test_data, model,
                                    population_parameters(g!);
                                    drop(evaluation, (:test_data,))...)]
        end
        _res = maximize_failsafe(optimizer, model, g!, params; verbosity, kw...)
        if !isempty(evaluation)
            push!(evaluations, evaluate(data, evaluation.test_data, model,
                                        population_parameters(g!);
                                        drop(evaluation, (:test_data,))...))
            merge(_res, (; evaluations, population_parameters = population_parameters(g!)))
        else
            pp = population_parameters(g!)
            if isnothing(pp)
                _res
            else
                merge(_res, (; population_parameters = pp))
            end
        end
    end
    g!(true, nothing, nothing, g!.xmax) # run once with the optimal parameters such that g.x in the next line is certainly set to the optimum
    res = merge((; logp = g!.fmax[], parameters = _parameters(g!)), res)
    if return_g!
        res = merge(res, (; g!))
    end
    res
end
maximize_logp(data, model, p::NamedTuple; kw...) = maximize_logp(data, model, ComponentArray(p); kw...)

###
### maximize
###
function maximize_failsafe(opt, model, g!, params; kw...)
    try
        # reset fmax
        g!.fmax[] = -Inf
        maximize(opt, model, g!, params; kw...)
    catch e
        @error e
        @info "Optimizing with Adam instead of $opt."
        g!.fmax[] = -Inf
        maximize(Adam(), model, g!, params; kw...)
    end
end
function maximize(opt::Optimisers.AbstractRule, model, g!, params;
                  maxeval = 10^6, maxtime = 3600, min_grad_norm = 1e-8,
                  lb = -Inf, ub = Inf, verbosity = 1)
    tstart = time()
    lc = Float64[]
    gns = Float64[]
    state = Optimisers.init(opt, params)
    dparams = zero(params)
    for _ in 1:maxeval
        logp = g!(true, dparams, nothing, params)
        push!(lc, logp)
        state, dp = Optimisers.apply!(opt, state, params, dparams)
        params .+= dp
        clamp!(params, lb, ub)
        (time() - tstart > maxtime || sqrt(sum(abs2, dparams)) < min_grad_norm) && break
    end
    (; opt, extra = (; lc, gns, dparams))
end
function maximize(opt::Symbol, model, g!, params;
                  lb = -Inf, ub = Inf, maxeval = 10^6,
                  maxtime = 3600, lopt = :LD_LBFGS, verbosity = 1)
    if opt === :MLSL
        o = Opt(:G_MLSL_LDS, length(params))
        o.lower_bounds = lb
        o.upper_bounds = ub
        _lopt = Opt(lopt, length(params))
#         _lopt.xtol_rel = 1e-3
#         _lopt.ftol_rel = 1e-5
        o.local_optimizer = _lopt
        o.max_objective = (params, dparams) -> g!(true, dparams, nothing, params)
        o.maxtime = maxtime
        o.maxeval = maxeval
        logp, xsol, extra = NLopt.optimize(o, params)
    else
        o = Opt(opt, length(params))
        o.lower_bounds = lb
        o.upper_bounds = ub
        o.max_objective = (params, dparams) -> g!(true, dparams, nothing, params)
        o.maxtime = maxtime
        o.maxeval = maxeval
        logp, xsol, extra = NLopt.optimize(o, params)
    end
    (; extra)
end
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
function maximize(opt::Optim.AbstractOptimizer, model, g!, params; maxeval = 10^6, iterations = 10^6, verbosity = 1, kw...)
    (; extra = Optim.optimize(Optim.only_fgh!(SwapSign(g!)), params, opt, Optim.Options(; iterations, f_calls_limit = maxeval, kw...)))
end

# TODO: better accessors
struct LaplaceEM end
function mstep!(::DiagonalNormalPrior, g!, Hs)
    ps = g!.g.ps
    N = length(ps)
    μ = sum(ps)/N
    second_moment_samples = sum(p.^2 for p in ps)/N
    free_idxs = 1:length(ps[1])
    second_moment_laplace = -sum(1 ./ clamp.(diag(Hs[i][free_idxs, free_idxs]), -Inf, 0) for i in eachindex(ps))/N
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
function maximize(::LaplaceEM, model, g!, params;
        iterations = 10, Estep = (;), evaluation = (;),
        verbosity = 1, derivative_threshold = 1e-3)
    Estep = merge((; opt = default_optimizer(model.model, params, ())), Estep)
    evaluation = merge((; test_data = nothing), evaluation)
    dp = zero(g!.xmax)
    g!.xmax .= params
    train_data = [g.g!.data.val for g in g!.g.g_funcs]
    _population_parameters = population_parameters(g!)
    evaluations = [evaluate(train_data, evaluation.test_data, model,
                            _population_parameters;
                            drop(evaluation, (:test_data,))...)]
    m_old, s_old = map(f -> f(evaluations[1].train), [mean, std])
    for i in 1:iterations
        # E-step
        (; extra) = maximize_failsafe(Estep.opt, model, g!, g!.xmax;
                                      drop(Estep, (:opt,))...)
        if verbosity > 1
            @show extra
        end
        logp = g!(true, dp, nothing, g!.xmax) # sets parameters to optimum
        if verbosity > 0
            println("Finished EM iteration $i at logp = $logp.")
        end
        dp_max = maximum(abs, dp[1:end-length(model.shared)])
        if dp_max > derivative_threshold
            @warn "Skipping M-Step: E-Step may not have converged.\nThe partial derivative with largest absolute value is $dp_max > derivative_threshold = $derivative_threshold."
        else
            Hs = [begin
                      H = zero(g.H)
                      g(nothing, nothing, H, g.x) # compute Hessians
                      H
                  end
                  for g in g!.g.g_funcs]
            # M-step
            mstep!(model.prior, g!, Hs)
        end
        # evaluation
        _evaluations =
              evaluate(train_data, evaluation.test_data, model,
                       g!.g.g_funcs[1].x;
                       drop(evaluation, (:test_data,))...)
        m_new, s_new = map(f -> f(_evaluations.train), [mean, std])
        if i > 1 && m_new - m_old < -(s_new + s_old)/2
            @info "Stopping EM at iteration $(i-1), because the MC estimate of the training logp decreased."
            break
        end
        push!(evaluations, _evaluations)
        _population_parameters .= population_parameters(g!)
        m_old = m_new
        s_old = s_new
    end
    (; population_parameters = _population_parameters, evaluations)
end
