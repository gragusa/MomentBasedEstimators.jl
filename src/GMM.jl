module GMM

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

abstract MomentEstimatorResult

type GMMResult <: MomentEstimatorResult
    status::Symbol
    objval::Real
    coef::Array{Float64, 1}
    nmom::Integer
    npar::Integer
    nobs::Integer
end

abstract MomentEstimator

type GMMEstimator <: MomentEstimator
    e::GMMNLPE
    r::GMMResult
end


# ----------------------------- #
# MathProgBase solver interface #
# ----------------------------- #

function MathProgSolverInterface.initialize(d::GMMNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end


MathProgSolverInterface.features_available(d::GMMNLPE) = [:Grad, :Jac, :Hess]

function MathProgSolverInterface.eval_f(d::GMMNLPE, theta)
    gn = d.smf(theta)
    (gn'd.W*gn)[1]
end

MathProgSolverInterface.eval_g(d::GMMNLPE, Dg, x) = nothing

function MathProgSolverInterface.eval_grad_f(d::GMMNLPE, grad_f, theta)
    grad_f[:] = 2*d.Dmf(theta)'*(d.W*d.smf(theta))
end

MathProgSolverInterface.jac_structure(d::GMMNLPE) = [],[]
MathProgSolverInterface.eval_jac_g(d::GMMNLPE, J, x) = nothing
MathProgSolverInterface.eval_hesslag(d::GMMNLPE, H, x, σ, μ) = nothing
MathProgSolverInterface.hesslag_structure(d::GMMNLPE) = [],[]

"""
Tells maximum number of arguments for a generic or anonymous function
"""
function max_args(f::Function)
    if isgeneric(f)
        return methods(f).max_args
    else
        # anonymous function
        # NOTE: This might be quite fragile, but works on 0.3.6 and 0.4-dev
        return length(Base.uncompressed_ast(f.code).args[1])
    end
end

# ------------ #
# Main routine #
# ------------ #

"""
TODO: Document the rest of the arguments

`mf` should be a function that computes the empirical moments of the
model. It can have one of two call signatures:

1. `mf(θ)`: computes moments, given only a parameter vector
2. `mf(θ, data)`: computes moments, given a parameter vector and an
   arbitrary object that contains the data necessary to compute the
   moments. Examples of `data` is a matrix, a Dict, or a DataFrame.
   The data argument is not used internally by these routines, but is
   simply here for user convenience.

The `mf` function should return an object of type Array{Float64, 2}
"""
function gmm(mf::Function, theta::Vector, W::Array{Float64, 2};
             solver = IpoptSolver(hessian_approximation="limited-memory"),
             data=nothing,
             mgr::IterationManager=OneStepGMM())
    npar = length(theta)
    theta_l = fill(-Inf, npar)
    theta_u = fill(+Inf, npar)
    gmm(mf, theta, theta_l, theta_u, W,  solver = solver, data=data, mgr=mgr)
end

function gmm(mf::Function, theta::Vector, theta_l::Vector, theta_u::Vector,
             W::Array{Float64, 2};
             solver = IpoptSolver(hessian_approximation="limited-memory"),
             data=nothing,
             mgr::IterationManager=OneStepGMM())

    # NOTE: all handling of data happens right here, because we will use _mf
    #       internally from now on.
    _mf(theta) = max_args(mf) == 1 ? mf(theta): mf(theta, data)

    mf0        = _mf(theta)
    nobs, nmom = size(mf0)
    npar       = length(theta)

    nl         = length(theta_l)
    nu         = length(theta_u)

    @assert nl == nu
    @assert npar == nl
    @assert nobs > nmom
    @assert nmom >= npar

    ## mf is n x m
    smf(theta) = reshape(sum(_mf(theta),1), nmom, 1);
    smf!(θ::Vector, gg) = gg[:] = smf(θ)

    Dsmf = ForwardDiff.forwarddiff_jacobian(smf!, Float64, fadtype=:dual,
                                            n = npar, m = nmom)

    l = theta_l
    u = theta_u
    lb = Float64[]
    ub = Float64[]

    # begin iterations
    ist = IterationState(0, 10.0, theta)

    # Define these outside while loop so they are available after it
    NLPE = GMMNLPE(_mf, smf, Dsmf, mgr, W)
    m = MathProgSolverInterface.model(solver)

    while !(finished(mgr, ist))
        NLPE = GMMNLPE(_mf, smf, Dsmf, mgr, W)
        m = MathProgSolverInterface.model(solver)
        MathProgSolverInterface.loadnonlinearproblem!(m, npar, 0, l, u, lb,
                                                      ub, :Min, NLPE)
        MathProgSolverInterface.setwarmstart!(m, theta)
        MathProgSolverInterface.optimize!(m)

        # update theta and W
        theta = MathProgSolverInterface.getsolution(m)
        W = optimal_W(_mf, theta, mgr.k)

        # update iteration state
        ist.n += 1
        ist.change = maxabs(ist.prev - theta)
        ist.prev = theta
    end

    r = GMMResult(MathProgSolverInterface.status(m),
                  MathProgSolverInterface.getobjval(m),
                  MathProgSolverInterface.getsolution(m),
                  nmom, npar, nobs)
    GMMEstimator(NLPE, r)
end

# --------------------- #
# Post-estimation tools #
# --------------------- #

function optimal_W(mf::Function, theta::Vector, k::RobustVariance)
    h = mf(theta)
    n = size(h, 1)
    S = vcov(h, k) * n
    W = pinv(S)
    W 
end

status(me::MomentEstimator) = me.r.status
StatsBase.coef(me::MomentEstimator) = me.r.coef
objval(me::MomentEstimator) = me.r.objval
momentfunction(me::MomentEstimator) = me.e.mf(coef(me))
jacobian(me::MomentEstimator) = me.e.Dmf(coef(me))
mfvcov(me::MomentEstimator, k::RobustVariance) = vcov(momentfunction(me), k)
nobs(me::MomentEstimator) = me.r.nobs
npar(me::MomentEstimator) = me.r.npar
nmom(me::MomentEstimator) = me.r.nmom
df(me::MomentEstimator) = nmom(me) - npar(me)
z_stats(me::MomentEstimator, k::RobustVariance) = coef(me) ./ stderr(me, k)
p_values(me::MomentEstimator, k::RobustVariance) = 2*ccdf(Normal(), z_stats(me, k))
shat(me::GMMEstimator, k::RobustVariance) = mfvcov(me, k)
optimal_W(me::GMMEstimator, k::RobustVariance) = pinv(full(shat(me, k)*nobs(me)))


StatsBase.vcov(me::MomentEstimator, k::RobustVariance) = vcov(me, k, me.e.mgr)
StatsBase.vcov(me::MomentEstimator) = vcov(me, me.e.mgr.k, me.e.mgr)

function StatsBase.vcov(me::MomentEstimator, k::RobustVariance, mgr::TwoStepGMM)
    G = jacobian(me)
    n = nobs(me)
    p = npar(me)
    S = shat(me, k)
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \not Var(\sqrt{N}
    ## A = pinv(G'*pinv(S)*G)
    ## B = G'*pinv(S)**G
    (n.^2/(n-p))*pinv(G'*pinv(S)*G)
end

function StatsBase.vcov(me::MomentEstimator, k::RobustVariance, mgr::OneStepGMM)
    G = jacobian(me)
    n = nobs(me)
    p = npar(me)
    S = shat(me, k)
    W = me.e.W
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \not Var(\sqrt{N}
    A = pinv(G'*W*G)
    B = G'*W*S*W*G
    (n.^2/(n-p))*A*B*A
end

function StatsBase.stderr(me::MomentEstimator)
    sqrt(diag(vcov(me, me.e.mgr.k, me.e.mgr)))
end

function StatsBase.stderr(me::MomentEstimator, mgr::IterationManager)
    sqrt(diag(vcov(me, mgr.k, mgr)))
end

function StatsBase.stderr(me::MomentEstimator, k::RobustVariance)
    sqrt(diag(vcov(me, k, me.e.mgr)))
end

function J_test(me::GMMEstimator, k::RobustVariance=me.e.mgr.k)
    # NOTE: because objective is sum of mf instead of typical mean of mf,
    #       there is no need to multiply by $T$ here (already done in obj)
    # j = objval(me)
    ## TODO: Decision: should we use objval or what is below
    ##                 that recalculate the S matrix
    g = mean(momentfunction(me), 1)
    S = pinv(shat(me, k))
    j = (nobs(me)*(g*S*g'))[1]
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j, clamp(p, eps(), Inf)
end

# --------------- #
# Display methods #
# --------------- #

function StatsBase.coeftable(me::MomentEstimator,
                             k::RobustVariance=me.e.mgr.k)
    cc = coef(me)
    se = stderr(me, k)
    zz = z_stats(me, k)
    CoefTable(hcat(cc, se, zz, 2.0*ccdf(Normal(), abs(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              ["x$i" for i = 1:npar(me)],
              4)
end

function show_extra(me::GMMEstimator)
    j, p = J_test(me, me.e.mgr.k)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

# default to nothing
show_extra(me::MomentEstimator) = ""


function Base.writemime{T<:MomentEstimator}(io::IO, ::MIME"text/plain", me::T)
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

export gmm, status, coef, objval, momentfunction, jacobian, mfvcov, vcov, optimal_W, TwoStepGMM



end
