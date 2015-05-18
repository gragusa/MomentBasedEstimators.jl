module MomentBasedEstimators

using PDMats
using ForwardDiff
using MathProgBase
using Ipopt
using StatsBase
using Reexport
using Distributions
@reexport using CovarianceMatrices
import MathProgBase.MathProgSolverInterface
import CovarianceMatrices.RobustVariance

export gmm, status, coef, objval, momentfunction, jacobian, mfvcov, vcov,
       optimal_W, TwoStepGMM, OneStepGMM, IterativeGMM


# ------------------ #
# Iteration managers #
# ------------------ #

abstract IterationManager

immutable OneStepGMM <: IterationManager
    k::RobustVariance
end

immutable TwoStepGMM <: IterationManager
    k::RobustVariance
end

immutable IterativeGMM <: IterationManager
    k::RobustVariance
    tol::Float64
    maxiter::Int
end

# kwarg constructors with default values
OneStepGMM(;k::RobustVariance=HC0()) = OneStepGMM(k)

TwoStepGMM(;k::RobustVariance=HC0()) = TwoStepGMM(k)

function IterativeGMM(;k::RobustVariance=HC0(), tol::Float64=1e-12,
                       maxiter::Int=500)
    IterativeGMM(k, tol, maxiter)
end

type IterationState
    n::Int
    change::Float64
    prev::Array  # previous value
end

finished(::OneStepGMM, ist::IterationState) = ist.n >= 1
finished(::TwoStepGMM, ist::IterationState) = ist.n >= 2
function finished(mgr::IterativeGMM, ist::IterationState)
    ist.n > mgr.maxiter || abs(ist.change) <= mgr.tol
end

# -------------- #
# Main GMM Types #
# -------------- #

type GMMNLPE <: MathProgSolverInterface.AbstractNLPEvaluator
    mf::Function
    smf::Function
    Dmf::Function
    mgr::IterationManager
    W::Array{Float64, 2}
end

abstract MomentBasedEstimatorResult

type GMMResult <: MomentBasedEstimatorResult
    status::Symbol
    objval::Real
    coef::Array{Float64, 1}
    nmom::Integer
    npar::Integer
    nobs::Integer
end

abstract MomentBasedEstimator

type GMMEstimator <: MomentBasedEstimator
    e::GMMNLPE
    r::GMMResult
end

# --------------- #
# Display methods #
# --------------- #

function show_extra(me::GMMEstimator)
    j, p = J_test(me, me.e.mgr.k)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

# default to nothing
show_extra(me::MomentBasedEstimator) = ""

function Base.writemime{T<:MomentBasedEstimator}(io::IO, ::MIME"text/plain", me::T)
    # get coef table and j-test
    ct = coeftable(me, me.e.mgr.k)

    # show info for our model
    println(io, "$(T): $(npar(me)) parameter(s) with $(nmom(me)) moment(s)")

    # Show extra information for this type
    println(io, show_extra(me))

    # print coefficient table
    println(io, "Coefficients:\n")

    # then show coeftable
    show(io, ct)
end

include("util.jl")
include("gmm_mathprogbase.jl")
include("api.jl")
include("post_estimation.jl")


end
