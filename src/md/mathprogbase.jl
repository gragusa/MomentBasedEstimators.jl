################################################################################
# MathProgBase solver interface - MDEstimator
################################################################################

function MathProgBase.initialize(d::MDEstimator, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.isobjlinear(e::MDEstimator) = false
MathProgBase.isobjquadratic(e::MDEstimator) = false
MathProgBase.isconstrlinear(e::MDEstimator, i::Int64) = false

MathProgBase.features_available(e::MDEstimator) = [:Grad, :Jac, :Hess]

function MathProgBase.eval_f(e::MDEstimator{M, V, S, T}, u) where {M, V, S, T<:Unweighted}
    Divergences.evaluate(e.div, u[1:nobs(e)])
end

function MathProgBase.eval_grad_f(e::MDEstimator{M, V, S, T}, grad_f, u) where {M, V, S, T<:Unweighted}
    n, k, m = size(e)
    Divergences.gradient!(grad_f, e.div, u)
    @simd for j=(n+1):(n+k)
        @inbounds grad_f[j] = 0.0
    end
end

function MathProgBase.eval_g(e::MDEstimator{M, V,S,T}, g, u) where {M, V, S<:Unconstrained, T<:Unweighted}
    n, k, m = size(e)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    @inbounds g[1:m]  = e.mf.s(θ)'*p
    @inbounds g[m+1]  = sum(p)
end

function MathProgBase.jac_structure(e::MDEstimator{M, V, S, T}) where {M, V, S<:Unconstrained, T<:Unweighted}
    n, k, m = size(e)
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

function MathProgBase.eval_jac_g(e::MDEstimator{M, V,S,T}, J, u) where {M<:FADMomFun, V, S<:Unconstrained, T<:Unweighted}
    n, k, m = size(e)
    p  = u[1:n]
    θ  = u[(n+1):(n+k)]
    g  = e.mf.s(θ)::Matrix{Float64}
    ws(θ) = e.mf.s(θ)'*p
    Dws = ForwardDiff.jacobian(ws, θ)::Matrix{Float64}
    @inbounds for j=1:m+1, i=1:n+k
        if(j<=m && i<=n)
            J[i+(j-1)*(n+k)] = g[i+(j-1)*n]
        elseif (j<=m && i>n)
            J[i+(j-1)*(n+k)] = Dws[j, i-n]
        elseif (j>m && i<=n)
            J[i+(j-1)*(n+k)] = 1.0
        end
     end
end

function MathProgBase.eval_jac_g(e::MDEstimator{M, V,S,T}, J, u) where {M <: AnaMomFun, V, S<:Unconstrained, T<:Unweighted}
    n, k, m = size(e)
    p  = u[1:n]
    θ  = u[(n+1):(n+k)]
    g  = e.mf.s(θ)
    Dws = e.mf.Dws(θ, p)
    @inbounds for j=1:m+1, i=1:n+k
        if(j<=m && i<=n)
            J[i+(j-1)*(n+k)] = g[i+(j-1)*n]
        elseif (j<=m && i>n)
            J[i+(j-1)*(n+k)] = Dws[j, i-n]
        elseif (j>m && i<=n)
            J[i+(j-1)*(n+k)] = 1.0
        end
    end
end

function MathProgBase.hesslag_structure(e::MDEstimator{M, V, S, T}) where {M, V, S<:Unconstrained, T<:Unweighted}
    n, k, m = size(e)
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

function MathProgBase.eval_hesslag(e::MDEstimator{M, V, S, T}, H, u, σ, λ) where {M<:FADMomFun, V, S<:Unconstrained, T}
    n, k, m = size(e)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    if σ==0
        @simd for j=1:n
            @inbounds H[j] = 0.0
        end
    else
        Divergences.hessian!(H, e.div, u[1:n])
        @simd for j in 1:n
            @inbounds H[j] = σ*H[j]
        end
    end
    sl(θ)  = e.mf.s(θ)*λ[1:m]
    wsl(θ) = (p'*e.mf.s(θ)*λ[1:m])[1]
    @inbounds H[n+1:n*k+n] = transpose(ForwardDiff.jacobian(sl, θ)::Matrix{Float64})
    @inbounds H[n*k+n+1:e.hele] = gettril(ForwardDiff.hessian(wsl, θ)::Matrix{Float64})
end

function MathProgBase.eval_hesslag(e::MDEstimator{M, V, S, T}, H, u, σ, λ) where {M<:AnaGradMomFun, V, S<:Unconstrained, T}
    n, k, m = size(e)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    if σ==0
        @simd for j=1:n
            @inbounds H[j] = 0.0
        end
    else
        Divergences.hessian!(H, e.div, u[1:n])
        @simd for j in 1:n
            @inbounds H[j] = σ*H[j]
        end
    end
    Dsl  = e.mf.Dsl(θ, λ[1:m])
    wsl(θ) = (p'*e.mf.s(θ)*λ[1:m])[1]
    @inbounds H[n+1:n*k+n] = Dsl'
    @inbounds H[n*k+n+1:e.hele] = gettril(ForwardDiff.hessian(wsl, θ)::Matrix{Float64})
end

function MathProgBase.eval_hesslag(e::MDEstimator{M, V, S, T}, H, u, σ, λ) where {M<:AnaFullMomFun, V, S<:Unconstrained, T}
    n, k, m = size(e)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    if σ==0
        @simd for j=1:n
            @inbounds H[j] = 0.0
        end
    else
        Divergences.hessian!(H, e.div, u[1:n])
        @simd for j in 1:n
            @inbounds H[j] = σ*H[j]
        end
    end
    Dsl  = e.mf.Dsl(θ, λ[1:m])
    Hwsl = e.mf.Hwsl(θ, p, λ[1:m])
    @inbounds H[n+1:n*k+n] = Dsl'
    @inbounds H[n*k+n+1:e.hele] = gettril(Hwsl)
end
