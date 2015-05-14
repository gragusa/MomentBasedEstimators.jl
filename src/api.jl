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
             solver=IpoptSolver(hessian_approximation="limited-memory", print_level=2),
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
