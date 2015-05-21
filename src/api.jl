# ------------ #
# Main routine #
# ------------ #

const DEFAULT_GMM_SOLVER = IpoptSolver(hessian_approximation="limited-memory",
                                       print_level=2)


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


type GMMProblem
    e::MomentBasedEstimators.GMMNLPE
    r::MomentBasedEstimators.GMMResult
    c::Constraint
    s::AbstractMathProgSolver
    f::ProblemStatus
    function GeneralizedMethodOfMomentsProblem()
        nn   = new()
        nn.s = DEFAULT_SOLVER
        nn.f = :NotInitialized
        nn
    end
end

 
function addMomentFunction!(g::GMMProblem, f::Function, theta::Vector; data = nothing)
    _mf(theta) = data == nothing ? f(theta) : f(theta, data)
    g0   = _mf(theta); n, m = size(g0)
    mf = MomentFunction(_mf, f, data, n, k, m)
    g.e.mf  = mf
    g.e.mgr = defaultItarationManager
    g.e.W   = eye(m)
    g.e.x0  = theta
end


addVarBounds!(g::GMMProblem, 
addInstruments!(g::Function, Z::Matrix{Float64}) = nothing   
setSolver(g::GMM, s::solver) = g.s = Solver    
setSolverOptions!(g::GMM, args...) = setOptions!(g.s, args...)


function solve!(g::GMMProblem)
    ## Solve or resolve
    if init_status(g) == :NotInitialized
        initialize!(g)
    end

    _solve!(g)
end 

        
    









function gmm(mf::Function, theta::Vector, W::Array{Float64, 2};
             solver=DEFAULT_SOLVER, data=nothing,            
             mgr::IterationManager=OneStepGMM())
    npar = length(theta)
    theta_l = fill(-Inf, npar)
    theta_u = fill(+Inf, npar)
    gmm(mf, theta, theta_l, theta_u, W, instruments, solver=solver, data=data, mgr=mgr)
end

function _gmm(mf::Function,
              theta::Vector,
              theta_l::Vector,
              theta_u::Vector,
              W::Array{Float64, 2},
              Z::Matrix{Float64},
              h::Function
              solver=DEFAULT_SOLVER,
              data=nothing,
              mgr::IterationManager=OneStepGMM())

    # NOTE: all handling of data happens right here, because we will use _mf
    #       internally from now on.
    if max_args(mf) == 1
        _mf(theta) = mf(theta)
    else
        _mf(theta) = mf(theta, data)
    end

    # Instrument handling here... use _mfi internally afterwards
    if instruments === nothing
        _mfi(theta) = _mf(theta)
    else
        _mfi(theta) = row_kron(_mf(theta), instruments)
    end

    mf0        = _mfi(theta)
    nobs, nmom = size(mf0)
    npar       = length(theta)

    nl         = length(theta_l)
    nu         = length(theta_u)

    @assert nl == nu
    @assert npar == nl
    @assert nobs > nmom
    @assert nmom >= npar

    ## mf is n x m
    smf(theta) = reshape(sum(_mfi(theta),1), nmom, 1);
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
    NLPE = GMMNLPE(_mfi, smf, Dsmf, mgr, W)
    m = MathProgSolverInterface.model(solver)

    while !(finished(mgr, ist))
        NLPE = GMMNLPE(_mfi, smf, Dsmf, mgr, W)
        m = MathProgSolverInterface.model(solver)
        MathProgSolverInterface.loadnonlinearproblem!(m, npar, 0, l, u, lb,
                                                      ub, :Min, NLPE)
        MathProgSolverInterface.setwarmstart!(m, theta)
        MathProgSolverInterface.optimize!(m)

        # update theta and W
        theta = MathProgSolverInterface.getsolution(m)
        W = optimal_W(_mfi, theta, mgr.k)

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
