abstract GenericMomentBasedEstimator <: MathProgBase.AbstractNLPEvaluator
abstract Constraint
abstract Weighting

abstract MomentFunction

abstract AnaMomFun <: MomentFunction

type FADMomFun <: MomentFunction
    g::Function            ## Moment Function
    s::Function            ## Smoothed moment function
    kern::SmoothingKernel
end

type AnaGradMomFun <: AnaMomFun
    g::Function            ## Moment Function
    s::Function            ## Smoothed moment function
    Dsn::Function
    Dws::Function
    Dsl::Function
    kern::SmoothingKernel
end

type AnaFullMomFun <: AnaMomFun
    g::Function            ## Moment Function
    s::Function            ## Smoothed moment function
    Dsn::Function
    Dws::Function
    Dsl::Function
    Hwsl::Function
    kern::SmoothingKernel
end

type MomentBasedEstimatorResults
    status::Symbol
    objval::Float64
    coef::Array{Float64, 1}
    H::Array{Float64,2}      ## Hessian of the objective function
end

type GMMEstimator{M<:MomentFunction, V<:IterationManager, S<:Constraint, T<:Weighting} <: GenericMomentBasedEstimator
    mf::M
    c::S
    x0::Array{Float64, 1}
    lb::Array{Float64, 1}
    ub::Array{Float64, 1}
    glb::Array{Float64, 1}
    gub::Array{Float64, 1}
    mgr::V
    ist::IterationState
    W::Array{Array{Float64,2},1}
    wtg::T
    gele::Int64
    hele::Int64
    nobs::Int64
    npar::Int64
    nmom::Int64
end

type MDEstimator{M<:MomentFunction, V<:Divergence, S<:Constraint, T<:Weighting} <: GenericMomentBasedEstimator
    mf::M
    c::S
    x0::Array{Float64, 1}
    lb::Array{Float64, 1}
    ub::Array{Float64, 1}
    glb::Array{Float64, 1}
    gub::Array{Float64, 1}
    wlb::Array{Float64, 1}
    wub::Array{Float64, 1}
    div::V
    wtg::T
    gele::Int64
    hele::Int64
    nobs::Int64
    npar::Int64
    nmom::Int64
end

type MomentBasedEstimatorOptions
    ##options
    ##maybe optimization options?
end

type Unweighted <: Weighting end

type Weighted <: Weighting
    wtg::WeightVec{Float64}
end

type Unconstrained <: Constraint end

type Constrained <: Constraint
    h::Function
    Dh::Function
    Hh::Function
    hlb::Array{Float64, 1}
    hub::Array{Float64, 1}
    nc::Int64  ## Number of constraints: row of h(θ)
end

type DEFAULT_SOLVER{T <: GenericMomentBasedEstimator}
    s::MathProgBase.AbstractMathProgSolver
end

DEFAULT_SOLVER(::GMMEstimator) = IpoptSolver(hessian_approximation="limited-memory", print_level=2)
DEFAULT_SOLVER(::MDEstimator)  = IpoptSolver(print_level=2)

type MomentBasedEstimator{T<:GenericMomentBasedEstimator}
    e::T
    r::MomentBasedEstimatorResults
    s::MathProgBase.AbstractMathProgSolver
    m::MathProgBase.AbstractMathProgModel
    status::Symbol
end

## Basic MomentBasedEstimator constructor
function MomentBasedEstimator(e::GenericMomentBasedEstimator)
    MomentBasedEstimator(e,
        MomentBasedEstimatorResults(
            :Unsolved,
            0.0,
            Array(Float64, npar(e)),
            Array(Float64, npar(e), npar(e))),
        DEFAULT_SOLVER(e),
        model(DEFAULT_SOLVER(e)),
        :Uninitialized)
end

setW0(mgr::TwoStepGMM, m::Int64) = [Array(Float64, m, m) for i=1:2]
setW0(mgr::OneStepGMM, m::Int64) = [Array(Float64, m, m) for i=1:1]
setW0(mgr::IterativeGMM, m::Int64) = [Array(Float64, m, m) for i=1:mgr.maxiter+1]

function make_fad_mom_fun(g::Function, 
                          kernel::SmoothingKernel = IdentitySmoother())
    FADMomFun(g, θ->smooth(g(θ), kernel), kernel)
end

function make_ana_mom_fun(::Type{GMMEstimator}, g::Function, ∇g::Function)    
    AnaGradMomFun(g, g, ∇g, identity, identity, IdentitySmoother())
end

function make_ana_mom_fun(::Type{MDEstimator}, g::Function, ∇g::Tuple{Function, Function, Function})
    AnaGradMomFun(g, g, ∇g..., IdentitySmoother())
end

function make_ana_mom_fun(::Type{MDEstimator}, g::Function, ∇g::Tuple{Function, Function, Function, Function})
    AnaFullMomFun(g, g, ∇g..., IdentitySmoother())
end
