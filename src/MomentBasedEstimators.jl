module MomentBasedEstimators

using PDMats
using ForwardDiff
using DualNumbers
using MathProgBase
using Ipopt
using StatsBase
using Reexport
using Distributions
using Divergences
@reexport using CovarianceMatrices
import MathProgBase.MathProgSolverInterface
import CovarianceMatrices.RobustVariance

const DEFAULT_DIVERGENCE = KullbackLeibler()

## Common
include("common/smoothing.jl")
include("common/derivatives.jl")
## GMM
include("gmm/iteration_managers.jl")

## MD
include("common/interface.jl")
include("common/api.jl")


# include("common/display.jl")
# include("common/post_estimation.jl")


end
