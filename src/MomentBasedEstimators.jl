module MomentBasedEstimators

using PDMats
using ForwardDiff
using DualNumbers
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

import MathProgBase: MathProgSolverInterface
import MathProgBase.MathProgSolverInterface: eval_f, eval_grad_f, eval_g, eval_jac_g, eval_hesslag

import MathProgBase.MathProgSolverInterface: jac_structure, hesslag_structure
import MathProgBase.MathProgSolverInterface: loadnonlinearproblem!
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

include("gmm/mathprogbase.jl")
include("md/mathprogbase.jl")
# include("common/display.jl")
# include("common/post_estimation.jl")

include("common/util.jl")
include("common/post_estimation.jl")

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
       status,
       estimate!,
       initialize!,
       solver!,
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
       J_test
end
