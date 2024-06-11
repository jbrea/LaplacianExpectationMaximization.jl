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

### Callbacks
```@docs
FitPopulations.Callback
FitPopulations.LogProgress
FitPopulations.Evaluator
FitPopulations.CheckPointSaver
FitPopulations.TimeTrigger
FitPopulations.IterationTrigger
FitPopulations.EventTrigger
```

### Optimizers
```@docs
FitPopulations.Optimizer
FitPopulations.NLoptOptimizer
FitPopulations.OptimOptimizer
FitPopulations.OptimisersOptimizer
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
