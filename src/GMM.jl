module GMM

using PDMats
using ForwardDiff
using MathProgBase
using Ipopt
using StatsBase
using Reexport
@reexport using CovarianceMatrices
import MathProgBase.MathProgSolverInterface

const RobustVariance = CovarianceMatrices.RobustVariance

type GMMNLPE <: MathProgSolverInterface.AbstractNLPEvaluator
    mf::Function
    smf::Function
    Dmf::Function
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
             data=nothing)
    npar = length(theta)
    theta_l = ones(npar)*-Inf
    theta_u = ones(npar)*+Inf
    gmm(mf, theta, theta_l, theta_u, W,  solver = solver, data=data)
end

function max_args(f::Function)
    if isgeneric(f)
        return methods(f).max_args
    else
        # anonymous function
        # NOTE: This might be quite fragile, but works on 0.3.6 and 0.4-dev
        return length(Base.uncompressed_ast(f.code).args[1])
    end
end

function gmm(mf::Function, theta::Vector, theta_l::Vector, theta_u::Vector,
             W::Array{Float64, 2};
             solver = IpoptSolver(hessian_approximation="limited-memory"),
             data=nothing)

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
    NLPE = GMMNLPE(_mf, smf, Dsmf, W)
    m = MathProgSolverInterface.model(solver)
    l = theta_l
    u = theta_u
    lb = Float64[]
    ub = Float64[]
    MathProgSolverInterface.loadnonlinearproblem!(m, npar, 0, l, u, lb, ub,
                                                  :Min, NLPE)
    MathProgSolverInterface.setwarmstart!(m, theta)
    MathProgSolverInterface.optimize!(m)
    r = GMMResult(MathProgSolverInterface.status(m),
                                    MathProgSolverInterface.getobjval(m),
                                    MathProgSolverInterface.getsolution(m),
                                    nmom, npar, nobs)
    GMMEstimator(NLPE, r)
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
Shat(me::MomentEstimator, k::RobustVariance) = PDMat(mfvcov(me, k)) * nobs(me)
optimal_W(me::MomentEstimator, k::RobustVariance) = pinv(full(Shat(me, k)))

function StatsBase.vcov(me::MomentEstimator, k::RobustVariance=HC0())
    G = jacobian(me)
    S = Shat(me, k)
    (nobs(me)/(nobs(me)-npar(me)))*pinv(Xt_invA_X(S, G))
end

function J_test(me::MomentEstimator)
    # NOTE: because objective is sum of mf instead of typical mean of mf,
    #       there is no need to multiply by $T$ here (already done in obj)
    j = objval(me)
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN

    # sometimes p is garbage, so we clamp it to be within reason
    return j, clamp(p, eps(), Inf)
end

function two_step(mf::Function, theta::Vector, W::Array{Float64, 2};
                  k::RobustVariance=BartlettKernel(),
                  solver=IpoptSolver(hessian_approximation="limited-memory"),
                  data=nothing)
    me1 = gmm(mf, theta,  W; solver=solver, data=data)
    theta1 = coef(me1)

    # do gmm one more time with optimal W
    gmm(mf, theta1, optimal_W(me1, k); solver=solver, data=data)
end

## To do: implement show method for MomentEstimator
## function Base.show(io::, me::MomentEstimator)
## end


export gmm, status, coef, objval, momentfunction, jacobian, mfvcov, vcov



end
