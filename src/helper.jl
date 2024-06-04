### Zero
_zero(x::AbstractArray) = _zero.(x)
_zero(x::Number) = zero(x)
_zero(x::Base.RefValue) = Ref(_zero(x[]))
_zero(x::Tuple) = _zero.(x)
_zero(x::NamedTuple{K}) where K = NamedTuple{K}(_zero.(values(x)))
function _zero(x::D) where D
    D.name.wrapper((_zero(getfield(x, f)) for f in fieldnames(D))...)
end
_one(x::Number) = one(x)
_one(x::NamedTuple{K}) where K = NamedTuple{K}(_one.(values(x)))

### drop
drop(nt::NamedTuple, k) = Base.structdiff(nt, NamedTuple{k})
drop(t::Tuple, k) = filter(∉(k), t)
function drop(x::ComponentArray, k) # TODO: can this be made better?
    ComponentArray(; (ki => copy.(x[ki]) for ki in drop(keys(x), k))...)
end
drop(x::AbstractVector, k) = isempty(x) ? x : @warn "Trying to drop $k from $x"

drop_population_parameters(::Any, parameters, ::Any) = parameters
function drop_population_parameters(::DiagonalNormalPrior, parameters, fixed)
    pp = parameters.population_parameters
    ComponentArray(; parameters...,
                   population_parameters = (μ = drop(pp.μ, keys(fixed)), σ = drop(pp.σ, keys(fixed))))
end

### mutable
_deep_ismutable(x::T) where T = _deep_ismutable(T)
function _deep_ismutable(x::Type)
    ismutabletype(x) && return true
    any(_deep_ismutable, fieldtypes(x))
end

### convert
_convert_eltype(T, x::AbstractArray) = _convert_eltype.(T, x)
_convert_eltype(T, x::Number) = T(x)
_convert_eltype(::Any, x::Bool) = x
_convert_eltype(::Any, x::Int) = x
_convert_eltype(T, x::Base.RefValue) = Ref(_convert_eltype(T, x[]))
_convert_eltype(T, x::Tuple) = _convert_eltype.(T, x)
_convert_eltype(T, x::NamedTuple{K}) where K = NamedTuple{K}(_convert_eltype.(T, values(x)))
function _convert_eltype(T, x::D) where D
    D.name.wrapper((_convert_eltype(T, getfield(x, f)) for f in fieldnames(D))...)
end

### copy
_deep_getproperty(x, keys) = foldl(getproperty, keys, init = x)
function _copy_element!(target, key, val::Number)
    setproperty!(_deep_getproperty(target, Base.front(key)), last(key), val)
end
function _copy_element!(target, key, val::AbstractArray)
    Base.materialize!(Base.dotgetproperty(_deep_getproperty(target, Base.front(key)),
                                          last(key)),
                      Base.broadcasted(Base.identity, val))
end
function _copy_elements!(target, source, parentkey, ::Any)
    _copy_element!(target, parentkey, _deep_getproperty(source, parentkey))
end
function _copy_elements!(target, source, parentkey, ks::Tuple)
    for k in ks
        newkey = (parentkey..., k)
        _copy_elements!(target, source,
                        newkey,
                        keys(_deep_getproperty(source, newkey)))
    end
end
function copy_elements!(target, source)
    _copy_elements!(target, source, (), keys(source))
    target
end

### distribute_shared
function distribute_shared(parameters)
    haskey(parameters, :__shared) || return parameters
    isempty(parameters.__shared) && return drop(parameters, (:__shared,))
    isempty(parameters.sample1) && return ComponentArray(; [k => parameters.__shared
                                                           for k in keys(parameters)[1:end-1]]...)
    ComponentArray(; [k => ComponentArray(getproperty(parameters, k); parameters.__shared...)
                      for k in keys(parameters)[1:end-1]]...)
end
