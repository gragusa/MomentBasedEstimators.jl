
startingval(e::GenericMomentBasedEstimator) = e.x0
startingval(g::MomentBasedEstimator) = startingval(g.e)

npar(e::GenericMomentBasedEstimator) = npar(e.mf)
nmom(e::GenericMomentBasedEstimator) = nmom(e.mf)
nobs(e::GenericMomentBasedEstimator) = nobs(e.mf)
Base.size(e::GenericMomentBasedEstimator) = (nobs(e), npar(e), nmom(e))

objval(e::MomentBasedEstimator) = e.r.objval

nobs(g::MomentBasedEstimator) = nobs(g.e)
npar(g::MomentBasedEstimator) = npar(g.e)
nmom(g::MomentBasedEstimator) = nmom(g.e)
Base.size(g::MomentBasedEstimator) = (nobs(g), npar(g), nmom(g))

nobs(m::MomentFunction) = m.nobs
npar(m::MomentFunction) = m.npar
nmom(m::MomentFunction) = m.nmom
Base.size(m::MomentFunction) = (nobs(m), npar(m), nmom(m))

################################################################################
## Constructor with function and x0
################################################################################
function GMMEstimator(mf::MomentFunction, x0::Vector;
                      data = nothing,
                      initialW = nothing,
                      wts = nothing,
                      mgr::IterationManager = TwoStepGMM(),
                      dtype::Symbol = :dual)
    
    w = wts == nothing ? Unweighted() : Weighted(float(wts))    
    g0  = mf.s(x0); n, m = size(g0); p = length(x0)

    ## Bounds
    bt  = [Inf for j=1:p]
    nf  = Float64[]
    ni  = 0::Int64
    ## Weighting matrix
    initialW  = initialW == nothing ? eye(Float64, m) : initialW
    _W   = setW0(mgr, m); _W[1][:,:] = initialW
    ## Moment function
    ## GMMEstimator
    e   = GMMEstimator(mf, Unconstrained(), x0, -bt, +bt, nf, nf,
                       mgr, IterationState([1], [10.0], x0), _W, w, ni, ni)
    ## MomentBasedEstimator
    g   = MomentBasedEstimator(e)
end

function GMMEstimator(f::Function, x0::Vector;
                      data = nothing,
                      initialW = nothing,
                      wts = nothing,
                      mgr::IterationManager = TwoStepGMM(),
                      dtype::Symbol = :dual)
    
    w = wts == nothing ? Unweighted() : Weighted(float(wts))
    _mf(x0) = data == nothing ? f(x0) : f(x0, data)
    g0  = _mf(x0); n, m = size(g0); p = length(x0)
    ## Bounds
    bt  = [Inf for j=1:p]
    nf  = Float64[]
    ni  = 0::Int64
    ## Weighting matrix
    initialW  = initialW == nothing ? eye(Float64, m) : initialW
    _W   = setW0(mgr, m); _W[1][:,:] = initialW
    ## Moment function
    mf  = MomentFunction(_mf, dtype, nobs = n, npar = p, nmom = m)
    ## GMMEstimator
    e   = GMMEstimator(mf, Unconstrained(), x0, -bt, +bt, nf, nf,
                       mgr, IterationState([1], [10.0], x0), _W, w, ni, ni)
    ## MomentBasedEstimator
    g   = MomentBasedEstimator(e)
end

function MDEstimator(f::Function, x0::Vector; data = nothing, wts = nothing,
                     div::Divergence = DEFAULT_DIVERGENCE,
                     kernel::SmoothingKernel = IdentitySmoother(),
                     dtype::Symbol = :dual)
    _mf(x0) = data == nothing ? f(x0) : f(x0, data)
    w = wts == nothing ? Unweighted() : Weighted(wts)
    g0  = _mf(x0); n, m = size(g0); p = length(x0)
    bt  = [Inf for j=1:p]
    wlb = zeros(Float64, n)
    wub = ones(Float64,  n)*n
    glb = [zeros(m), n];
    gub = [zeros(m), n];
    ni  = 0::Int64
    mf  = MomentFunction(_mf, dtype, kernel = kernel, nobs = n, npar = p, nmom = m)
    e   = MDEstimator(mf, Unconstrained(), x0, -bt, bt, glb, gub, wlb, wub,
                      div, w, ni, ni)
    g   = MomentBasedEstimator(e)
end

################################################################################
## Solve methods
################################################################################
function solve!(g::MomentBasedEstimator)
	if status(g) == :Uninitialized
		initialize!(g)
	end
end

function initialize!{V<:Divergence, S<:Unconstrained, T<:Weighting}(g::MomentBasedEstimator{MDEstimator{V, S, T}})
	n, p, m = size(g)
	ξ₀ = [ones(n), startingval(g)]
	g.e.gele = int((n+p)*(m+1)-p)
	g.e.hele = int(n*p + n + (p+1)*p/2)
	g_L = getmfLB(g)
	g_U = getmfUB(g)
	u_L = [getwtsLB(g), getparLB(g)]
	u_U = [getwtsUB(g), getparUB(g)]
	loadnonlinearproblem!(g.m, n+p, m+1, u_L, u_U, g_L, g_U, :Min, g.e)
	MathProgBase.MathProgSolverInterface.setwarmstart!(g.m, ξ₀)
	g.status = :Initialized
end

function initialize!{V<:IterationManager, S<:Unconstrained, T<:Weighting}(g::MomentBasedEstimator{GMMEstimator{V, S, T}})
	n, p, m = size(g)
	ξ₀ = startingval(g)
	g.e.gele = int(p)
	g.e.hele = int(2*p)
	g_L = Float64[]
	g_U = Float64[]
	u_L = getparLB(g)
	u_U = getparUB(g)
	loadnonlinearproblem!(g.m, p, 0, u_L, u_U, g_L, g_U, :Min, g.e)
	MathProgBase.MathProgSolverInterface.setwarmstart!(g.m, ξ₀)
	g.status = :Initialized
end

getparLB(g::MomentBasedEstimator) = g.e.lb
getparUB(g::MomentBasedEstimator) = g.e.ub

getmfLB(g::MomentBasedEstimator) = g.e.glb
getmfUB(g::MomentBasedEstimator) = g.e.gub

getwtsLB{V, T, S}(g::MomentBasedEstimator{MDEstimator{V, T, S}}) = g.e.wlb
getwtsUB{V, T, S}(g::MomentBasedEstimator{MDEstimator{V, T, S}}) = g.e.wub

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
    Dh = forwarddiff_jacobian(h, Float64, fadtype=:typed)
    lh(x, λ) = λ'*h
    Hh = args_typed_fad_hessian(lh, Float64)
    r  = MomentBasedEstimatorResults(:Uninitialized, 0., Array(Float64, p), Array(Float64, p, p))
    ce = deepcopy(g.e); ce.c  = Constrained(h, Dh, Hh, chk...)
    MomentBasedEstimator(ce, r, g.s, g.m, :Uninitialized)
end

################################################################################
## Update solver
################################################################################
function solver!(g::MomentBasedEstimator, s)
    g.s = s
    g.m = MathProgBase.MathProgSolverInterface.model(s)
    initialize!(g)
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
    g.status == :Uninitialized || initialize!(g)
end


################################################################################
## Update initial lb and up on parameters(default -inf, +inf)
################################################################################
function setparLB!{T}(g::MomentBasedEstimator{T}, lb::Vector)
    npar(g) == length(lb) || error("Dimension error")
    copy!(g.e.lb, lb)
    g.status == :Uninitialized || initialize!(g)
end

function setparUB!{T}(g::MomentBasedEstimator{T}, ub::Vector)
    npar(g) == length(ub) || error("Dimension error")
    copy!(g.e.ub, ub)
    g.status == :Uninitialized || initialize!(g)
end

function setparbounds!{T}(g::MomentBasedEstimator{T}, lb::Vector, ub::Vector)
	setparLB!(g, lb)
	setparUB!(g, ub)
	g.status == :Uninitialized || initialize!(g)
end

################################################################################
## Update initial lb and up on mdweights (default 0, n)
################################################################################
function setwtsLB!{T <: MDEstimator}(g::MomentBasedEstimator{T}, lb::Vector)
    nobs(g) == length(lb) || error("Dimension error")
    copy!(g.e.wlb, lb)
    g.status == :Uninitialized || initialize!(g)
end

function setwtsUB!{T <: MDEstimator}(g::MomentBasedEstimator{T}, ub::Vector)
    nobs(g) == length(ub) || error("Dimension error")
    copy!(g.e.wub, ub)
    g.status == :Uninitialized || initialize!(g)
end

function setbounds_wgt!{T <: MDEstimator}(g::MomentBasedEstimator{T}, lb::Vector, ub::Vector)
    setwtsLB!(g, lb)
    setwtsUB!(g, ub)
    g.status == :Uninitialized || initialize!(g)
end

################################################################################
## Update initial weighting matrix (default is I(m))
################################################################################
function setW0!(g::MomentBasedEstimator{GMMEstimator}, W::Array{Float64, 2})
    copy!(g.e.W , W)
end

################################################################################
## Iteration
################################################################################
function set_iteration_manager!(g::MomentBasedEstimator{GMMEstimator}, mgr::IterationManager)
    g.e.mgr = mgr
    g.status == :Uninitialized || initialize!(g)
end


################################################################################
## estimate!
################################################################################
function setx0!{S <: GMMEstimator}(g::MomentBasedEstimator{S}, x0::Vector{Float64})
    ## For GMM x0 is the parameter
    length(x0) == npar(g) || throw(DimensionMismatch(""))
    copy!(g.e.x0, x0)
    MathProgBase.MathProgSolverInterface.setwarmstart!(g.m, x0)
end

function setx0!{S <: MDEstimator}(m::MomentBasedEstimator{S}, x0::Vector{Float64})
    ## For GMM x0 is the parameter
    length(x0) == npar(m) || throw(DimensionMismatch(""))
    copy!(m.e.x0, x0)
    x00 = [m.m.inner.x[1:nobs(m)], x0]
    MathProgBase.MathProgSolverInterface.setwarmstart!(m.m, x00)
end

estimate!(g::MomentBasedEstimator) = estimate!(g, startingval(g))


function estimate!(g::MomentBasedEstimator, x0::Vector)
    ## There are three possible states of g.status
    ## :Uninitialized
    ## :Initialized
    ## :Solved(Success|Failure)
    setx0!(g, x0)
    g.status == :Uninitialized || initialize!(g)
    ## g.status == :Solved        || resolve!(g, g.m)
    g.status == :Initialized   || optimize!(g.m)
    fill_in_results!(g)
    g
end

function fill_in_results!{S <: MDEstimator}(g::MomentBasedEstimator{S})
    g.r.status = MathProgBase.MathProgSolverInterface.status(g.m)
    n, p, m = size(g)
    ss = MathProgBase.MathProgSolverInterface.getsolution(g.m)
    copy!(g.r.coef, ss[n+1:end])
    #g.r.mdwts  = ss[1:n]
    g.r.objval = MathProgBase.MathProgSolverInterface.getobjval(g.m)
end

function fill_in_results!{S <: GMMEstimator}(g::MomentBasedEstimator{S})
    g.r.status = MathProgBase.MathProgSolverInterface.status(g.m)
    n, p, m = size(g)
    copy!(g.r.coef, MathProgBase.MathProgSolverInterface.getsolution(g.m))
    g.r.objval = MathProgBase.MathProgSolverInterface.getobjval(g.m)
end

function resolve!{S <: MDEstimator}(g::MomentBasedEstimator{S}, s::KNITRO.KnitroMathProgModel)
    KNITRO.restartProblem(g.m.inner, startingval(g), g.m.inner.numConstr)
    KNITRO.solveProblem(g.m.inner)
end

resolve!{S <: MDEstimator}(g::MomentBasedEstimator{S}, s::Ipopt.IpoptMathProgModel) = MathProgBase.MathProgSolverInterface.optimize!(g.m)

function resolve!{S <: GMMEstimator}(g::MomentBasedEstimator{S}, s::Ipopt.IpoptMathProgModel)
    reset_iteration_state!(g)
    n, p, m = size(g)
    while !(finished(g.e.mgr, g.e.ist))
        ## if g.e.ist.n[1] > 1
        ##     g.m = MathProgSolverInterface.model(g.s)
        ##     loadnonlinearproblem!(g.m, p, 0, getparLB(g), getparUB(g), getmfLB(g),
        ##                           getmfUB(g), :Min, g.e)
        ##     MathProgSolverInterface.setwarmstart!(g.m, theta)
    ## end
        if g.e.ist.n[1]>1
            g.e.W[g.e.ist.n[1]][:,:] = optimal_W(g.e.mf, theta, g.e.mgr.k)
        end 
        MathProgSolverInterface.optimize!(g.m)
        # update theta and W
        theta = MathProgSolverInterface.getsolution(g.m)
        update!(g.e.ist, theta)
        next!(g.e.ist)
        ## g.e.ist.n[:] += 1
        ## g.e.ist.change[:] = maxabs(g.e.ist.prev - theta)
        ## g.e.ist.prev[:] = theta
        MathProgSolverInterface.setwarmstart!(g.m, theta)
        # update iteration state        
    end    
    fill_in_results!(g)
    g
end


next!(x::IterationState) = x.n[1] += 1
function update!(x::IterationState, v::Vector)
    x.change[:] = maxabs(x.prev - v)
    x.prev[:] = v
end

reset_iteration_state!(g::MomentBasedEstimator) = g.e.ist = deepcopy(IterationState([1], [10.0], startingval(g)))

function optimal_W(mf::MomentFunction, theta::Vector, k::RobustVariance)
    h = mf.s(theta)
    n = size(h, 1)
    S = vcov(h, k) * n
    W = pinv(S)
    W
end

function optimal_W(mf::Function, theta::Vector, k::RobustVariance)
    h = mf(theta)
    n = size(h, 1)
    S = vcov(h, k) * n
    W = pinv(S)
    W
end

function optimal_W(e::MomentBasedEstimator, k::RobustVariance)
    optimal_W(e.e.mf, coef(e), k)
end

