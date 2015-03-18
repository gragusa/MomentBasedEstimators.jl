module GMM

using ModelsGenerators
using PDMats
using ForwardDiff
using MathProgBase
using Ipopt
import MathProgBase.MathProgSolverInterface

type GMMNLPE <: MathProgSolverInterface.AbstractNLPEvaluator
    mf::Function
    Dmf::Function
    W::Array{Float64, 2}
end

function MathProgSolverInterface.initialize(d::GMMNLPE, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end


MathProgSolverInterface.features_available(d::GMMNLPE) = [:Grad, :Jac, :Hess]

function MathProgSolverInterface.eval_f(d::GMMNLPE, theta) 
    gn = d.mf(theta)
    (gn'd.W*gn)[1]
end

MathProgSolverInterface.eval_g(d::GMMNLPE, Dg, x) = nothing

function MathProgSolverInterface.eval_grad_f(d::GMMNLPE, grad_f, theta)
    grad_f[:] = 2*d.Dmf(theta)'*(d.W*d.mf(theta))
end

MathProgSolverInterface.jac_structure(d::GMMNLPE) = [],[]
MathProgSolverInterface.eval_jac_g(d::GMMNLPE, J, x) = nothing
MathProgSolverInterface.eval_hesslag(d::GMMNLPE, H, x, σ, μ) = nothing
MathProgSolverInterface.hesslag_structure(d::GMMNLPE) = [],[]

function gmm(mf::Function, theta::Vector, W::Array{Float64, 2};
             solver = IpoptSolver(hessian_approximation="limited-memory"))
    npar = length(theta)
    theta_l = ones(npar)*-Inf
    theta_u = ones(npar)*+Inf    
    gmm(mf, theta, theta_l, theta_u, W,  solver = solver)
end 

function gmm(mf::Function, theta::Vector, theta_l::Vector, theta_u::Vector,
             W::Array{Float64, 2};
             solver = IpoptSolver(hessian_approximation="limited-memory"))
    
    mf0        = mf(theta)
    nobs, nmom = size(mf0)
    npar       = length(theta)
    
    nl         = length(theta_l)
    nu         = length(theta_u)
    
    @assert nl == nu
    @assert npar == nl
    @assert nobs > nmom
    @assert nmom >= npar
    
    ## mf is n x m
    smf(theta) = reshape(sum(mf(theta),1), nmom, 1);

    smf!(θ::Vector, gg) = gg[:] = smf(θ)

    Dsmf = ForwardDiff.forwarddiff_jacobian(smf!, Float64,
                                           fadtype=:dual, n = npar, m = nmom)

    NLPE = GMMNLPE(smf, Dsmf, W)    
    
    m = MathProgSolverInterface.model(solver)
    l = theta_l
    u = theta_u
    lb = Float64[]
    ub = Float64[]
    MathProgSolverInterface.loadnonlinearproblem!(m, length(theta), 0, l, u, lb, ub, :Min, NLPE)

    MathProgSolverInterface.setwarmstart!(m,[0.0])
    MathProgSolverInterface.optimize!(m)

    (MathProgSolverInterface.getobjval(m),
     MathProgSolverInterface.getsolution(m),
     MathProgSolverInterface.status(m))
    
end

end
