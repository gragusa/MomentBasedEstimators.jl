startingval(e::GenericMomentBasedEstimator) = e.x0
startingval(g::MomentBasedEstimator) = startingval(g.e)

npar(e::GenericMomentBasedEstimator) = e.npar
nmom(e::GenericMomentBasedEstimator) = e.nmom
StatsBase.nobs(e::GenericMomentBasedEstimator) = e.nobs
Base.size(e::GenericMomentBasedEstimator) = (nobs(e), npar(e), nmom(e))

StatsBase.nobs(g::MomentBasedEstimator) = nobs(g.e)
npar(g::MomentBasedEstimator) = npar(g.e)
nmom(g::MomentBasedEstimator) = nmom(g.e)
Base.size(g::MomentBasedEstimator) = (nobs(g.e), npar(g.e), nmom(g.e))

objval(e::MomentBasedEstimator) = e.r.objval

# StatsBase.nobs(m::MomentFunction) = m.nobs
# npar(m::MomentFunction) = m.npar
# nmom(m::MomentFunction) = m.nmom
# Base.size(m::MomentFunction) = (nobs(m), npar(m), nmom(m))

################################################################################
## Constructor with function and x0
################################################################################
function GMMEstimator(f::Function, theta::Vector;
                      grad = nothing,
                      data = nothing,
                      initialW = nothing,
                      wts = nothing,
                      mgr::IterationManager = TwoStepGMM(),
                      constraints = Unconstrained(),
                      s::MathProgBase.SolverInterface.AbstractMathProgSolver = IpoptSolver(hessian_approximation = "limited-memory", print_level=2, sb = "yes"))
    ## Set Moment Function
    g(theta) = data == nothing ? f(theta) : f(theta, data)
    ## Evaluate Moment Function
    g₀ = g(theta)
    n, m, p = (size(g₀)..., length(theta))
    ## Initial Weighting Matrix
    W₀ = initialW == nothing ? eye(Float64, m) : initialW
    W  = setW0(mgr, m);
    W[1] = W₀
    ## Weighting
    w = wts == nothing ? Unweighted() : Weighted(float(wts))
    ## Set Default Bounds
    lb  = [-Inf for j=1:p]
    ub  = [+Inf for j=1:p]
    nf  = Float64[]
    ni  = 0::Int64

    if grad == nothing
        mf  = make_fad_mom_fun(g, IdentitySmoother())
    else
        ∇f(theta) = data == nothing ? grad(theta) : grad(theta, data)
        mf  = make_ana_mom_fun(GMMEstimator, g, ∇f)
    end

    MomentBasedEstimator(GMMEstimator(mf, constraints, theta, lb, ub, nf, nf,
                                      mgr, IterationState([1], [10.0], theta), W,
                                      w, ni, ni, n, p, m), s)
end

@compat const GradTuple=Union{Void, Tuple{Function, Function, Function}, Tuple{Function, Function, Function, Function}}


function MDEstimator(f::Function, theta::Vector;
                     grad::GradTuple = nothing,
                     data = nothing, wts = nothing,
                     div::Divergence = DEFAULT_DIVERGENCE,
                     kernel::SmoothingKernel = IdentitySmoother(),
                     s::MathProgBase.SolverInterface.AbstractMathProgSolver = IpoptSolver(print_level=2, sb = "yes"))
    ## Set Moment Function
    g(theta) = data == nothing ? f(theta) : f(theta, data)
    ## Evaluate Moment Function
    g₀ = g(theta)
    n, m, p = (size(g₀)..., length(theta))
    ## Weighting
    w = wts == nothing ? Unweighted() : Weighted(float(wts))
    #=
    Set Default bounds
    =#

    ## -- Bounds on theta           -- ##
    lb  = [-Inf for j=1:p]
    ub  = [+Inf for j=1:p]
    ## -- Bounds on mdweights   -- ##
    wlb = zeros(Float64, n)
    wub = ones(Float64, n)*n
    ## -- Bounds on constraints -- ##
    glb = [zeros(m); n];
    gub = [zeros(m); n];
    ni  = 0::Int64
    if grad == nothing
        mf  = make_fad_mom_fun(g, kernel)
    else
        ff = Array{Function}(length(grad))
        if data != nothing
            for (i, f) in enumerate(grad)
                _f = copy(f)
                _g(theta) = _f(theta, data)
                ff[i] = _g(theta)
            end
            grad = (ff...)
        end
        mf  = make_ana_mom_fun(MDEstimator, g, grad)
    end

    MomentBasedEstimator(MDEstimator(mf, Unconstrained(), theta, lb, ub, glb, gub,
                                     wlb, wub, div, w, ni, ni, n, p, m))
end

################################################################################
## Solve methods
################################################################################
# function solve!(g::MomentBasedEstimator)
# 	   if status(g) == :Uninitialized
# 		      initialize!(g)
# 	   end
# end

function initialize!{M<:MomentFunction, V<:Divergence, S<:Unconstrained, T<:Weighting}(g::MomentBasedEstimator{MDEstimator{M, V, S, T}})
	   n, p, m = size(g)
	   ξ₀ = [ones(n); startingval(g)]
	   g.e.gele = Int((n+p)*(m+1)-p)
	   g.e.hele = Int(n*p + n + (p+1)*p/2)
	   g_L = getmfLB(g)
	   g_U = getmfUB(g)
	   u_L = [getwtsLB(g); getparLB(g)]
	   u_U = [getwtsUB(g); getparUB(g)]
	   MathProgBase.loadproblem!(g.m, n+p, m+1, u_L, u_U, g_L, g_U, :Min, g.e)
	   MathProgBase.setwarmstart!(g.m, ξ₀)
	   g.status[1] = :Initialized
end

function initialize!{M<:MomentFunction, V<:IterationManager, S<:Unconstrained, T<:Weighting}(g::MomentBasedEstimator{GMMEstimator{M, V, S, T}})
    n, p, m = size(g)
	   ξ₀ = startingval(g)
	   g.e.gele = @compat Int(p)
	   g.e.hele = @compat Int(2*p)
	   g_L = Float64[]
	   g_U = Float64[]
	   u_L = getparLB(g)
	   u_U = getparUB(g)
    MathProgBase.loadproblem!(g.m, p, 0, u_L, u_U, g_L, g_U, :Min, g.e)
	   MathProgBase.setwarmstart!(g.m, ξ₀)
	   g.status[1] = :Initialized
end

function initialize!{M<:MomentFunction, V<:IterationManager, S<:Constrained, T<:Weighting}(g::MomentBasedEstimator{GMMEstimator{M, V, S, T}})
    n, p, m = size(g)
	   ξ₀ = MomentBasedEstimators.startingval(g)
	   g.e.gele = @compat Int(g.e.c.nc*p)
	   g.e.hele = @compat Int(0)
	   g_L = g.e.c.hlb
	   g_U = g.e.c.hub
	   u_L = getparLB(g)
	   u_U = getparUB(g)
    MathProgBase.loadproblem!(g.m, p, g.e.c.nc, u_L, u_U, g_L, g_U, :Min, g.e)
	   MathProgBase.setwarmstart!(g.m, ξ₀)
	   g.status[1] = :Initialized
end

################################################################################
## Getters
################################################################################
getparLB(g::MomentBasedEstimator) = g.e.lb
getparUB(g::MomentBasedEstimator) = g.e.ub

getmfLB(g::MomentBasedEstimator) = g.e.glb
getmfUB(g::MomentBasedEstimator) = g.e.gub

getwtsLB{M, V, T, S}(g::MomentBasedEstimator{MDEstimator{M, V, T, S}}) = g.e.wlb
getwtsUB{M, V, T, S}(g::MomentBasedEstimator{MDEstimator{M, V, T, S}}) = g.e.wub

################################################################################
## Set constraint on parameters
################################################################################
function check_constraint_sanity(k, x0, h::Function, hlb, hub)
	   h0 = h(x0); nc = length(h0)
	   typeof(h0)  <: Vector{Float64} || error("Constraint function must be ::Vector{Float64}")
	   nc == length(hub) && nc == length(hlb) || error("Constraint bounds of wrong dimension")
	   typeof(hlb) <: Vector{Float64} || (hlb = float(hlb))
	   typeof(hub) <: Vector{Float64} || (hub = float(hub))
	   (hlb, hub, nc)
end

## This return a constrained version of MomentBasedEstimator
function constrained(h::Function, hlb::Vector, hub::Vector, g::MomentBasedEstimator)
    p = npar(g); chk = check_constraint_sanity(p, startingval(g), h, hlb, hub)
    r  = MomentBasedEstimatorResults(:Uninitialized, 0., Array{Float64}(p), Array{Float64}(p, p))
    if typeof(g.e.mf) == MomentBasedEstimators.FADMomFun
        mf  = make_fad_mom_fun(g.e.mf.g, IdentitySmoother())
    else
        mf  = make_ana_mom_fun(GMMEstimator, g.e.mf.g, g.e.mf.∇g)
    end
    ce = GMMEstimator(mf,
                      Constrained(h, chk...),
                      g.e.x0,
                      g.e.lb,
                      g.e.ub,
                      g.e.glb,
                      g.e.gub,
                      g.e.mgr,
                      g.e.ist,
                      g.e.W,
                      g.e.wtg,
                      g.e.gele,
                      g.e.hele,
                      size(g)...)

    MomentBasedEstimator(ce, r, g.s, g.m, :Uninitialized)
end

################################################################################
## Update solver
################################################################################
function setsolver!(g::MomentBasedEstimator, s::MathProgBase.SolverInterface.AbstractMathProgSolver)
    g.s = s
    g.m = deepcopy(MathProgBase.NonlinearModel(s))
end

################################################################################
## Update lb and up on g(θ) default: (0,...,0)
################################################################################
function setmfLB!(g::MomentBasedEstimator{MDEstimator}, lb::Vector)
	   nmom(g) == length(lb) || error("Dimension error")
	   g.e.glb[:] = lb
end

function setmfUB!(g::MomentBasedEstimator{MDEstimator}, ub::Vector)
	   nmom(g) == length(ub) || error("Dimension error")
	   g.e.glb[:] = ub
end

function setmfbounds!(g::MomentBasedEstimator{MDEstimator}, lb::Vector, ub::Vector)
    setmfLB!(g, lb)
    setmfUB!(g, ub)
end

################################################################################
## Update initial lb and up on parameters(default -inf, +inf)
################################################################################
function setparLB!{T}(g::MomentBasedEstimator{T}, lb::Vector)
    npar(g) == length(lb) || error("Dimension error")
    copy!(g.e.lb, lb)
end

function setparUB!{T}(g::MomentBasedEstimator{T}, ub::Vector)
    npar(g) == length(ub) || error("Dimension error")
    copy!(g.e.ub, ub)
end

function setparbounds!{T}(g::MomentBasedEstimator{T}, lb::Vector, ub::Vector)
	   setparLB!(g, lb)
	   setparUB!(g, ub)
end

################################################################################
## Update initial lb and up on mdweights (default 0, n)
################################################################################
function setwtsLB!{T <: MDEstimator}(g::MomentBasedEstimator{T}, lb::Vector)
    nobs(g) == length(lb) || error("Dimension error")
    copy!(g.e.wlb, lb)
end

function setwtsUB!{T <: MDEstimator}(g::MomentBasedEstimator{T}, ub::Vector)
    nobs(g) == length(ub) || error("Dimension error")
    copy!(g.e.wub, ub)
end

function setwtsbounds!{T <: MDEstimator}(g::MomentBasedEstimator{T}, lb::Vector, ub::Vector)
    setwtsLB!(g, lb)
    setwtsUB!(g, ub)
end

################################################################################
## Update initial weighting matrix (default is I(m))
################################################################################
## TODO: This should depend on the Iteration Manager
setW0!(g::MomentBasedEstimator{GMMEstimator}, W::Array{Float64, 2}) = copy!(g.e.W , W)

################################################################################
## Iteration
################################################################################
function set_iteration_manager!(g::MomentBasedEstimator{GMMEstimator}, mgr::IterationManager)
    g.e.mgr = mgr
end


################################################################################
## estimate!
################################################################################
function setx0!{S <: GMMEstimator}(g::MomentBasedEstimator{S}, x0::Vector{Float64})
    ## For GMM x0 is the parameter
    length(x0) == npar(g) || throw(DimensionMismatch(""))
    copy!(g.e.x0, x0)
    MathProgBase.setwarmstart!(g.m, x0)
end

function setx0!{S <: MDEstimator}(m::MomentBasedEstimator{S}, x0::Vector{Float64})
    ## For GMM x0 is the parameter
    length(x0) == npar(m) || throw(DimensionMismatch(""))
    copy!(m.e.x0, x0)
    x00 = [m.m.inner.x[1:nobs(m)], x0]
    MathProgBase.setwarmstart!(m.m, x00)
end

function estimate!(g::MomentBasedEstimator)
    ## There are three possible states of g.status
    ## :Uninitialized
    ## :Initialized
    ## :Solved(Success|Failure)
    initialize!(g)
    solve!(g, g.m)
    fill_in_results!(g)
    g
end

function fill_in_results!{T <: MDEstimator}(me::MomentBasedEstimator{T})
    me.r.status = MathProgBase.status(me.m)
    n, p, m = size(me)
    ss = MathProgBase.getsolution(me.m)
    copy!(me.r.coef, ss[n+1:end])
    ## The ϕ_value is
    ## equal to 2*objval/(S_T(k1^2/k2))
    v  = MathProgBase.getobjval(me.m)
    k1 = κ₁(me)
    k2 = κ₂(me)
    S  = bw(me)
    me.r.objval = 2*k2*v/(S*k1^2)
    me.status[1] = :Solved
end

function fill_in_results!{S <: GMMEstimator}(g::MomentBasedEstimator{S})
    g.r.status = MathProgBase.status(g.m)
    n, p, m = size(g)
    copy!(g.r.coef, MathProgBase.getsolution(g.m))
    g.r.objval = MathProgBase.getobjval(g.m)
    g.status[1] = :Solved
end

function solve!{S <: MDEstimator}(g::MomentBasedEstimator{S}, s::KNITRO.KnitroMathProgModel)
    # KNITRO.restartProblem(g.m.inner, startingval(g), g.m.inner.numConstr)
    # KNITRO.solveProblem(g.m.inner)
    MathProgBase.optimize!(g.m)
end

solve!{S <: MDEstimator}(g::MomentBasedEstimator{S}, s::Ipopt.IpoptMathProgModel) = MathProgBase.optimize!(g.m)

function solve!{S <: GMMEstimator}(g::MomentBasedEstimator{S}, s::MathProgBase.SolverInterface.AbstractMathProgModel)
    reset_iteration_state!(g)
    n, p, m = size(g)
    while !(finished(g.e.mgr, g.e.ist))
        if g.e.ist.n[1]>1
            g.e.W[g.e.ist.n[1]][:,:] = optimal_W(g.e.mf, theta, g.e.mgr.k, g.e.mgr.demean)
        end
        MathProgBase.optimize!(g.m)

        theta = MathProgBase.getsolution(g.m)
        update!(g.e.ist, theta)
        next!(g.e.ist)
        if !(finished(g.e.mgr, g.e.ist))
            MathProgBase.setwarmstart!(g.m, theta)
        end
    end
    fill_in_results!(g)
    g.status[1] = :Solved
    g
end

next!(x::IterationState) = x.n[1] += 1

function update!(x::IterationState, v::Vector)
    x.change[:] = maximum(abs, x.prev - v)
    x.prev[:] = v
end

reset_iteration_state!(g::MomentBasedEstimator) =
    g.e.ist = deepcopy(IterationState([1], [10.0], startingval(g)))


function optimal_W(g::Array{Float64, 2}, theta::Vector, k::RobustVariance, demean::Bool)
    h = demean ? g .- mean(g, 1) : g
    n = size(h, 1)
    S = CovarianceMatrices.vcov(h, k) * n
    pinv(S)
end

function optimal_W(mf::MomentFunction, theta::Vector, k::RobustVariance, demean::Bool)
    g = mf.s(theta)
    optimal_W(g, theta, k, demean)
end

function optimal_W(mf::Function, theta::Vector, k::RobustVariance, demean::Bool)
    g = mf(theta)
    optimal_W(g, theta, k, demean)
end

function optimal_W(e::MomentBasedEstimator, k::RobustVariance, demean::Bool = false)
    optimal_W(e.e.mf, coef(e), k, demean)
end
