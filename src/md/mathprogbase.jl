################################################################################
# MathProgBase solver interface - MDEstimator
################################################################################

function MathProgSolverInterface.initialize(d::MDEstimator, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.isobjlinear(e::MDEstimator) = false
MathProgBase.isobjquadratic(e::MDEstimator) = false
MathProgBase.isconstrlinear(e::MDEstimator, i::Int64) = false

MathProgSolverInterface.features_available(e::MDEstimator) = [:Grad, :Jac, :Hess]

function eval_f{V, S, T<:Unweighted}(e::MDEstimator{V,S,T}, u)
    Divergences.evaluate(e.div, u[1:e.mf.nobs])
end

function eval_grad_f{V, S, T<:Unweighted}(e::MDEstimator{V,S,T}, grad_f, u)
    n, k, m = size(e.mf)
    @simd for j=1:n
        @inbounds grad_f[j] = Divergences.gradient(e.div, u[j])
    end
    @simd for j=(n+1):(n+k)
        @inbounds grad_f[j] = 0.0
    end
end

function eval_g{V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{V,S,T}, g, u)
    n, k, m = size(e)
    p = u[1:n]
    theta = u[(n+1):(n+k)]
    @inbounds g[1:m]  = e.mf.ws(theta, p)
    @inbounds g[m+1]  = sum(p)
end

function jac_structure{V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{V,S,T})
    n, k, m = size(e.mf)
    rows = Array(Int64, e.gele)
    cols = Array(Int64, e.gele)
    for j = 1:m+1, r = 1:n+k
        if !((r > n) && (j==m+1))
            @inbounds rows[r+(j-1)*(n+k)] = j
            @inbounds cols[r+(j-1)*(n+k)] = r
        end
    end
    rows, cols
end

function eval_jac_g{V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{V,S,T}, J, u)
    n, k, m = size(e.mf)
    p  = u[1:n]
    θ  = u[(n+1):(n+k)]
    g  = e.mf.s(θ)
    Dws = e.mf.Dws(θ, p)
    for j=1:m+1, i=1:n+k
        if(j<=m && i<=n)
            @inbounds J[i+(j-1)*(n+k)] = g[i+(j-1)*n]
        elseif (j<=m && i>n)
            @inbounds J[i+(j-1)*(n+k)] = Dws[j, i-n]
        elseif (j>m && i<=n)
            @inbounds J[i+(j-1)*(n+k)] = 1.0
        end
    end
end


function hesslag_structure{V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{V, S, T})
    n, k, m = size(e.mf)
    rows = Array(Int64, e.hele)
    cols = Array(Int64, e.hele)
    @simd for j = 1:n
        @inbounds rows[j] = j
        @inbounds cols[j] = j
    end
    idx = n+1

    for s = 1:n
        for j = 1:k
            @inbounds rows[idx] = n+j
            @inbounds cols[idx] = s
            idx += 1
        end
    end

    for j = 1:k
        for s = 1:j
            @inbounds rows[idx] = n+j
            @inbounds cols[idx] = n+s
            idx += 1
        end
    end
    rows, cols
end


function eval_hesslag(e::MDEstimator, H, u, σ, λ)
    n, k, m = size(e.mf)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    if σ==0
        for j=1:n
            @inbounds H[j] = 0.0
        end
    else
      for j=1:n
          @inbounds H[j] = σ*Divergences.hessian(e.div, u[j])
      end
  end
    @inbounds H[n+1:n*k+n] = transpose(e.mf.Dsl(θ, λ[1:m]))
    @inbounds H[n*k+n+1:e.hele] = gettril(e.mf.Hwsl(θ, p, λ[1:m]))
end
