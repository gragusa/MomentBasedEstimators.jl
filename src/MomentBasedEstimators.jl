module MomentBasedEstimators

using ForwardDiff
using MathProgBase
using Ipopt
using StatsBase
using Reexport
using Distributions
@reexport using Divergences
using KNITRO
using Base.LinAlg.BLAS
using Compat
@reexport using CovarianceMatrices
using Calculus
import CovarianceMatrices.RobustVariance

const DEFAULT_DIVERGENCE = KullbackLeibler()

## Common
include("md/smoothing.jl")
#include("common/derivatives.jl")
## GMM
include("gmm/iteration_managers.jl")

## MD
include("common/interface.jl")
include("common/api.jl")
include("md/mdp.jl")


include("gmm/mathprogbase.jl")
include("md/mathprogbase.jl")
# include("common/display.jl")
# include("common/post_estimation.jl")

include("common/util.jl")
include("common/post_estimation.jl")

include("common/display.jl")

export GMMEstimator,
       MDEstimator,
       Unconstrained,
       Constrained,
       TwoStepGMM,
       OneStepGMM,
       optimal_W,
       Weighted,
       Unweighted,
       MomentFunction,
       MomentBasedEstimator,
       MinimumDivergenceProblem,
       status,
       estimate!,
       initialize!,
       solver!,
       solve!,
       setparLB!,
       setparUB!,
       setparbounds!,
       setwtsLB!,
       setwtsUB!,
       setwtsbounds!,
       setmfLB!,
       setmfUB!,
       setmfbounds!,
       objval,
       J_test,
       writemime,
       TruncatedSmoother,
       BartlettSmoother,
       MinimumDivergenceProblem
end
