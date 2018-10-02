################################################################################
# MathProgBase solver interface - GMMEstimator
################################################################################

function MathProgBase.initialize(d::GMMEstimator, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(e::GMMEstimator) = [:Grad, :Jac, :Hess]

function MathProgBase.eval_f(e::GMMEstimator, theta)
    idx = e.ist.n[1]
    gn = vec(sum(e.mf.s(theta), dims=1))
    Base.dot(gn, e.W[idx]*gn)
end

MathProgBase.eval_g(e::GMMEstimator{M, V, T, S}, g, theta) where {M, V, T<:Unconstrained, S} = nothing

function MathProgBase.eval_g(e::GMMEstimator{M, V, T, S}, g, theta) where {M, V, T<:Constrained, S}
    g[:] .= e.c.h(theta)
end

function MathProgBase.eval_grad_f(e::GMMEstimator{M, V, T, S}, grad_f, θ) where {M<:FADMomFun, V, T, S}
    idx = e.ist.n[1]
    sn(θ) = vec(sum(e.mf.s(θ), dims=1))
    Δsn = ForwardDiff.jacobian(sn, θ)::Matrix{Float64}  # m x k
    Wg  = e.W[idx]*sn(θ)
    mul!(grad_f, Δsn', Wg)
end

function MathProgBase.eval_grad_f(e::GMMEstimator{M, V, T, S}, grad_f, θ) where {M <: AnaMomFun, V, T, S}
    ##grad_f[:] = 2*d.Dsn(theta)'*(d.W*d.sn(theta))
    idx = e.ist.n[1]
    sn = vec(sum(e.mf.s(θ), 1))
    gemm!('T', 'N', 2.0,
          e.mf.Dsn(θ),
          e.W[idx]*sn, 0.0, grad_f)
end

MathProgBase.jac_structure(e::GMMEstimator{M, V, T, S}) where {M, V, T<:Unconstrained, S} = Int[],Int[]
MathProgBase.eval_jac_g(e::GMMEstimator{M, V, T, S}, J, x) where {M, V, T<:Unconstrained, S} = nothing

function MathProgBase.jac_structure(e::GMMEstimator{M, V, T, S}) where {M, V, T<:Constrained, S}
    nc = e.c.nc            ## Number of constraints
    n, k, m = size(e)
    ## The jacobian is a nc x k
    rows = Array(Int64, e.gele)
    cols = Array(Int64, e.gele)
    for j = 1:nc, r = 1:k
        @inbounds rows[r+(j-1)*k] = j
        @inbounds cols[r+(j-1)*k] = r
    end
    rows, cols
end

function MathProgBase.eval_jac_g(e::GMMEstimator{M, V, T, S}, J, θ) where {M, V, T<:Constrained, S}
    h(θ) = e.c.h(θ)
    J[:] = vec((ForwardDiff.jacobian(h, θ)::Matrix{Float64})')
end

MathProgBase.hesslag_structure(d::GMMEstimator) = Int[], Int[]
MathProgBase.eval_hesslag(d::GMMEstimator, H, x, σ, μ) = nothing
