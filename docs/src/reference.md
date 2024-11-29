# Reference
## Functions to extend with a new model
```@docs
FitPopulations.initialize!
FitPopulations.parameters
FitPopulations.logp
FitPopulations.sample
```

## Population model
```@docs
FitPopulations.PopulationModel
```

## Fitting
```@docs
FitPopulations.maximize_logp
```

### Optimizers
```@docs
FitPopulations.Optimizer
FitPopulations.NLoptOptimizer
FitPopulations.OptimOptimizer
FitPopulations.OptimisersOptimizer
FitPopulations.OptimizationOptimizer
FitPopulations.LaplaceEM
```

## Simulation
```@docs
FitPopulations.simulate
FitPopulations.logp_tracked
```

## Evaluation
```@docs
FitPopulations.mc_marginal_logp
FitPopulations.BIC_int
```

## Derivatives
```@docs
FitPopulations.gradient_logp
FitPopulations.hessian_logp
```
