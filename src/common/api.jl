
startingval(e::GenericMomentBasedEstimator) = e.x0
startingval(g::MomentBasedEstimator) = startingval(g.e)

npar(e::GenericMomentBasedEstimator) = e.mf.npar
nmom(e::GenericMomentBasedEstimator) = e.mf.nmom
nobs(e::GenericMomentBasedEstimator) = e.mf.nobs
Base.size(e::GenericMomentBasedEstimator) = (nobs(e), npar(e), nmom(e))


nobs(g::MomentBasedEstimator) = nobs(g.e)
npar(g::MomentBasedEstimator) = npar(g.e)
nmom(g::MomentBasedEstimator) = nmom(g.e)
Base.size(g::MomentBasedEstimator) = (nobs(g), npar(g), nmom(g))


################################################################################
## Constructor with function and x0
################################################################################
function GMMEstimator(f::Function, x0::Vector; mgr = TwoStepGMM(), data = nothing, dtype = :dual)
	_mf(x0) = data == nothing ? f(x0) : f(x0, data)
	g0  = _mf(x0); n, m = size(g0); np = length(x0)
	bt  = [Inf for j=1:np]
	mf  = MomentFunction(_mf, dtype, nobs = n, npar = np, nmom = m)
	e   = GMMEstimator(mf, x0, -bt, +bt, mgr, eye(m), 0, 0)
	g   = MomentBasedEstimator(e)
end

function MDEstimator(f::Function, x0::Vector; data = nothing,
	div::Divergence = DEFAULT_DIVERGENCE,
	kernel::SmoothingKernel = IdentitySmoother(),
	dtype::Symbol = :dual)
_mf(x0) = data == nothing ? f(x0) : f(x0, data)
g0  = _mf(x0); n, m = size(g0); np = length(x0)
bt  = [Inf for j=1:np]
wlb = zeros(Float64, n)
wub = ones(Float64,  n)*n
glb = [zeros(m), n];
gub = [zeros(m), n];
mf  = MomentFunction(_mf, dtype, kernel = kernel, nobs = n, npar = np, nmom = m)
e   = MDEstimator(mf, x0, -bt, bt, wlb, wub, glb, gub, div, 0, 0)
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

function initialize!(g::MomentBasedEstimator{MDEstimator, Unconstrained})
	n, m, k = size(g)
	ξ₀ = [ones(n), startingval(g)]
	g.e.gele = int((n+k)*(m+1)-k)
	g.e.hele = int(n*k + n + (k+1)*k/2)
	g_L = g.e.glb
	g_U = g.e.gub
	u_L = [g.e.wlb,  g.e.lb]
	u_U = [g.e.wub,  g.e.ub]
	loadnonlinearproblem!(g.m, n+k, m+1, u_L, u_U, g_L, g_U, :Min, g.e)
	setwarmstart!(model, ξ₀)
	g.status = :Initialized
end

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
	#if status(g)==:Uninitialized
	chk = check_constraint_sanity(npar(g), startingval(g), h, hlb, hub)
	Dh = forwarddiff_jacobian(h, Float64, fadtype=:typed)
	lh(x, λ) = λ'*h
	Hh = args_typed_fad_hessian(lh, Float64)
	c   = Constrained(h, Dh, Hh, chk...)
	MomentBasedEstimator(g.e,
		MomentBasedEstimatorResults(
			:Uninitilized,
			0.,
			Array(Float64, npar(g)),
			Array(Float64, npar(g), npar(g))),
		c,
		g.s,
		g.m,
		:Uninitialized)
	#end
end

################################################################################
## Update solver
################################################################################
function solver!(g::MomentBasedEstimator, s)
	g.s = s
	g.status == :Uninitialized || initilize!(g)
end


################################################################################
## Update lb and up on g(θ) default: (0,...,0)
################################################################################
function lower_bounds_mf!(g::MomentBasedEstimator{MDEstimator}, glb::Vector)
	nmom(g) == length(lb) || error("Dimension error")
	g.e.glb[:] = glb
end

function upper_bounds_mf!(g::MomentBasedEstimator{MDEstimator}, gub::Vector)
	nmom(g) == length(lb) || error("Dimension error")
	g.e.glb[:] = glb
end

function setbounds_mf!(g::MomentBasedEstimator{MDEstimator}, glb::Vector, gub::Vector)
	lower_bounds_mf!(g, glb)
	upper_bounds_mf!(g, gub)
	g.status == :Uninitialized || initilize!(g)

end


################################################################################
## Update initial lb and up on parameters(default -inf, +inf)
################################################################################
function lower_bounds!(g::MomentBasedEstimator, lb::Vector)
	npar(g) == length(lb) || error("Dimension error")
	g.e.lb[:] = lb

end

function upper_bounds!(g::MomentBasedEstimator, ub::Vector)
	npar(g) == length(lb) || error("Dimension error")
	g.e.ub[:] = ub
end

function setbounds!(g::MomentBasedEstimator, lb::Vector, ub::Vector)
	lower_bounds!(g, lb)
	upper_bounds!(g, ub)
	g.status == :Uninitialized || initilize!(g)
end

################################################################################
## Update initial lb and up on mdweights (default 0, n)
################################################################################
function lower_bounds_wgt!(g::MomentBasedEstimator{MDEstimator}, wlb::Vector)
	nobs(g) == length(wlb) || error("Dimension error")
	g.e.wlb[:] = wlb
end

function upper_bounds_wgt!(g::MomentBasedEstimator{MDEstimator}, wub::Vector)
	nobs(g) == length(wlb) || error("Dimension error")
	g.e.wub[:] = wub
end

function setbounds_wgt!(g::MomentBasedEstimator{MDEstimator}, wlb::Vector, wub::Vector)
	lower_bounds!(g, wlb)
	upper_bounds!(g, wub)
	g.status == :Uninitialized || initilize!(g)
end

################################################################################
## Update initial weighting matrix (default is I(m))
################################################################################
function initial_weighting!(g::MomentBasedEstimator{GMMEstimator}, W::Array{Float64, 2})
	g.e.W = W
end

################################################################################
## Iteration
################################################################################
function iteration_type!(g::MomentBasedEstimator{GMMEstimator}, mgr::IterationManager)
	g.e.mgr = mgr
end

