abstract GenericMomentBasedEstimator <: MathProgBase.MathProgSolverInterface.AbstractNLPEvaluator
abstract Constraint
abstract Weighting

type MomentFunction
    g::Function            ## Moment Function
    s::Function            ## Smoothed moment function
    ws::Function           ## (m×1) ∑pᵢ sᵢ
    sn::Function           ## ∑sᵢ(θ)
    Dws::Function          ## (k×m)
    Dsn::Function          ## (k×m)
    Dsl::Function          ## (n×k)
    Dwsl::Function         ## (k×1)
    Hwsl::Function         ## (kxk)
    kern::SmoothingKernel
    nobs::Int64
    npar::Int64
    nmom::Int64
end

function MomentFunction(g::Function, dtype::Symbol;
                        kernel::SmoothingKernel = IdentitySmoother(),
                        nobs = Nothing,
                        npar = Nothing,
                        nmom = Nothing)
    ## Default is no smoothing
    MomentFunction(get_mom_deriv(g, dtype, kernel,
                                 nobs, nmom, npar)..., kernel, nobs, npar, nmom)
end

function get_mom_deriv(g::Function, dtype::Symbol, k::SmoothingKernel, nobs, nmom, npar)
    ## Smoothed moment condition
    s(θ::Vector)  = smooth(g(θ), k)
    ## Average smoothed moment condition
    sn(θ::Vector)  = vec(sum(s(θ), 1))
    ## Average unsmoothed moment condition
    ## gn(θ::Vector)  = sum(g(θ), 1)
    ## Weighted moment conditions
    ws(θ::Vector, p::Vector) = s(θ)'*p
    sl(θ::Vector, λ::Vector) = s(θ)*λ
    wsl(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)
    wsls(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]
    ## Weighted moment conditions (in-place versions)
    sn!(θ::Vector, jac_out) = jac_out[:] = sum(s(θ), 1)
    ws!(θ::Vector, jac_out, p::Vector) = jac_out[:] = s(θ)'*p
    sl!(θ::Vector, jac_out, λ::Vector) = jac_out[:] =s(θ)*λ
    wsl!(θ::Vector, jac_out,  p::Vector, λ::Vector) = jac_out[:] = (p'*s(θ)*λ)
    ## typed, dual, or finite difference based derivatives
    if dtype==:dual
        ## First derivative
        Dsn  = ForwardDiff.dual_fad_jacobian(sn!,  Float64, n = npar, m = nmom)
        Dws  = args_dual_fad_jacobian(ws!,  Float64, n = npar, m = nmom)
        Dsl  = args_dual_fad_jacobian(sl!,  Float64, n = npar, m = nobs)
        Dwsl = args_dual_fad_jacobian(wsl!, Float64, n = npar, m = 1)
        ## Second derivative
        ## Uses wsls() which is the scalar version of wsl
        ## because fad_hessian expect a scalar valued function
        Hwsl = args_typed_fad_hessian(wsls, Float64)
    elseif dtype==:typed
        Dsn  = ForwardDiff.typed_fad_gradient(sn,  Float64)
        Dws  = args_typed_fad_gradient(ws,  Float64)
        Dsl  = args_typed_fad_gradient(sl,  Float64)
        Dwsl = args_typed_fad_gradient(wsl, Float64)
        ## Second derivative
        ## Uses wsls() which is the scalar version of wsl
        ## because fad_hessian expect a scalar valued function
        Hwsl = args_typed_fad_hessian(wsls, Float64)
    elseif dtype==:diff
        Dsn(θ::Vector, args...)  = fd_jacobian(sn,  θ, :central, args...)
        Dws(θ::Vector, args...)  = fd_jacobian(ws,  θ, :central, args...)
        Dsl(θ::Vector, args...)  = fd_jacobian(sl,  θ, :central, args...)
        Dwsl(θ::Vector, args...) = fd_jacobian(wsl, θ, :central, args...)
        ## Uses wsls() which is the scalar version of wsl
        ## because fd_hessian expect a scalar valued function
        Hwsl(θ::Vector, args...) = fd_hessian(wsls,  θ, args...)
    end
    return (g, s, ws, sn, Dws, Dsn, Dsl, Dwsl, Hwsl)
end

type MomentBasedEstimatorResults
    status::Symbol
    objval::Float64
    coef::Array{Float64, 1}
    H::Array{Float64,2}      ## Hessian of the objective function
end

type GMMEstimator{V<:IterationManager, S<:Constraint, T<:Weighting} <: GenericMomentBasedEstimator
    mf::MomentFunction
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
end

type MDEstimator{V<:Divergence, S<:Constraint, T<:Weighting} <: GenericMomentBasedEstimator
    mf::MomentFunction
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
            0.,
            Array(Float64, e.mf.npar),
            Array(Float64, e.mf.npar, e.mf.npar)),
        DEFAULT_SOLVER(e),
        MathProgSolverInterface.model(DEFAULT_SOLVER(e)),
        :Uninitialized)
end

setW0(mgr::TwoStepGMM, m::Int64) = [Array(Float64, m,m) for i=1:2]
setW0(mgr::OneStepGMM, m::Int64) = [Array(Float64, m,m) for i=1:1]
setW0(mgr::IterativeGMM, m::Int64) = [Array(Float64, m,m) for i=1:mgr.maxiter+1]
