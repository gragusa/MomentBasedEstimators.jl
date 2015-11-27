type MomentMatrix{F <: AbstractMatrix}
    X::F ## Unsmoothed
    S::F ## Smoothed
    g_L::Vector
    g_U::Vector
    kern::SmoothingKernel
    n::Int64        ## Rows of X
    m::Int64        ## Cols of X
    m_eq::Int64     ## Cols of X[:,1:m_eq] => G
    m_ineq::Int64   ## Cols of X[:,m_eq+1:end] => H
end

type MDP{F} <: GenericMomentBasedEstimator
    mm::MomentMatrix{F}
    div::Divergence
    gele::Int64
    hele::Int64
    solver::MathProgBase.SolverInterface.AbstractMathProgSolver
end

immutable MinimumDivergenceProblem
    m::MathProgBase.AbstractMathProgModel
    e::MDP
end

function MinimumDivergenceProblem(G::AbstractMatrix,
                                  c::Vector;
                                  wlb = nothing,
                                  wub = nothing,
                                  div::Divergence = KullbackLeibler(),
                                  solver = IpoptSolver(print_level = 2),
                                  k::SmoothingKernel = IdentitySmoother())
    n, m = size(G)
    m_eq = length(c)
    @assert m == m_eq "Inconsistent dimension"
    if wlb == nothing
        wlb = zeros(n)
    end
    if wub == nothing
        wub = ones(n)*n
    end
    MinimumDivergenceProblem(G, c, c, m_eq, 0, wlb, wub, div, solver, k)
end

function MinimumDivergenceProblem(G::AbstractMatrix,
                                  c::Vector,
                                  H::AbstractMatrix,
                                  lwr::Vector,
                                  upp::Vector;
                                  wlb = nothing,
                                  wub = nothing,
                                  div::Divergence = KullbackLeibler(),
                                  solver = IpoptSolver(),
                                  k::SmoothingKernel = IdentitySmoother())
    m_c = length(c); m_lwr = length(lwr); m_upp = length(upp)
    n_g, m_g = size(G)
    n_h, m_h = size(H)

    ## TODO: Throw DimensionMismatch errors
    @assert n_g == n_h "Dimensions of G and H are inconsistent"
    @assert m_lwr == m_upp "Dimensions of lower and upper bounds are inconsistent"
    @assert m_g == m_c "Dimensions of G and c are inconsistent"
    @assert m_h == m_lwr "Dimensions of bounds and H are inconsistent"
    m = m_g + m_lwr
    X_L = [c, lwr]
    X_U = [c, upp]
    if wlb == nothing
        wlb = zeros(n_g)
    end
    if wub == nothing
        wub = ones(n_g)*n_g
    end
    MinimumDivergenceProblem([G H], X_L, X_U, m_g, m_h, wlb, wub, div, solver, k)
end

function MinimumDivergenceProblem(X::AbstractMatrix,
                                  X_L::Vector,
                                  X_U::Vector,
                                  m_eq::Int64,
                                  m_ineq::Int64,
                                  wlb::Vector{Float64},
                                  wub::Vector{Float64},
                                  div::Divergence,
                                  solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
                                  k::SmoothingKernel)
    n, m  = size(X)
    model = MathProgBase.model(solver)
    gele  = round(Int, n*(m+1))
    hele  = round(Int, n)
    # u_L   = zeros(n)
    # u_U   = ones(n)*n
    mm    = MomentMatrix(X, smooth(X, k), X_L, X_U, k, n, m, m_eq, m_ineq)
    e     = MDP(mm, div, gele, hele, solver)
    if wlb == nothing
        wlb = zeros(n)
    end
    if wub == nothing
        wub = ones(n)*n
    end
    MathProgBase.loadnonlinearproblem!(model, n, m+1, wlb, wub, [X_L; n], [X_U; n], :Min, e)
    MathProgBase.setwarmstart!(model, ones(n))
    MinimumDivergenceProblem(model, e)
end

function solve!(mdp::MinimumDivergenceProblem)
    MathProgBase.optimize!(mdp.m)
    return mdp
end

Base.size(e::MDP) = (e.mm.n, e.mm.m, e.mm.m_eq, e.mm.m_ineq)

function MathProgBase.initialize(e::MDP, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(e::MDP) = [:Grad, :Jac, :Hess]
MathProgBase.eval_f(e::MDP, u) = Divergences.evaluate(e.div, u[1:e.mm.n])
MathProgBase.isobjlinear(e::MDP) = false
MathProgBase.isobjquadratic(e::MDP) = false
MathProgBase.isconstrlinear(e::MDP, i::Int64) = false

function MathProgBase.eval_g(e::MDP, g, u)
    n, m, m_eq, m_ineq = size(e)
    p   = u[1:n]
    @inbounds StatsBase.wsum!(view(g, 1:m), e.mm.S, p, 1)
    @inbounds g[m+1]  = sum(p)
end

function MathProgBase.eval_grad_f(e::MDP, grad_f, u)
    n, m, m_eq, m_ineq = size(e)
    for j=1:n
        @inbounds grad_f[j] = Divergences.gradient(e.div, u[j])
    end
end

function MathProgBase.jac_structure(e::MDP)
    n, m, m_eq, m_ineq = size(e)
    rows = Array(Int64, e.gele)
    cols = Array(Int64, e.gele)
    for j = 1:m+1, r = 1:n
        @inbounds rows[r+(j-1)*n] = j
        @inbounds cols[r+(j-1)*n] = r
    end
    rows, cols
end

function MathProgBase.hesslag_structure(e::MDP)
    n, m, m_eq, m_ineq = size(e)
    rows = Array(Int64, e.hele)
    cols = Array(Int64, e.hele)
    for j = 1:n
        @inbounds rows[j] = j
        @inbounds cols[j] = j
    end
  rows, cols
end

function MathProgBase.eval_jac_g(e::MDP, J, u)
    n, m, m_eq, m_ineq = size(e)
    for j=1:m+1, i=1:n
        if(j<=m)
            @inbounds J[i+(j-1)*n] = e.mm.S[i+(j-1)*n]
        else
            @inbounds J[i+(j-1)*n] = 1.0
        end
    end
end

function MathProgBase.eval_hesslag(e::MDP, H, u, σ, λ)
    n, m, m_eq, m_ineq = size(e)
    if σ==0
        for j=1:n
            @inbounds H[j] = 0.0
        end
    else
      for j=1:n
          @inbounds H[j] = σ*Divergences.hessian(e.div, u[j])
      end
    end
end
