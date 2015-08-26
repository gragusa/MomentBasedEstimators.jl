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

function eval_f{M, V, S, T<:Unweighted}(e::MDEstimator{M, V, S, T}, u)
    Divergences.evaluate(e.div, u[1:nobs(e)])
end

function eval_grad_f{M, V, S, T<:Unweighted}(e::MDEstimator{M, V, S, T}, grad_f, u)
    n, k, m = size(e)
    Divergences.gradient!(grad_f, e.div, u)
    @simd for j=(n+1):(n+k)
        @inbounds grad_f[j] = 0.0
    end
end

function eval_g{M, V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{M, V,S,T}, g, u)
    n, k, m = size(e)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    @inbounds g[1:m]  = e.mf.s(θ)'*p
    @inbounds g[m+1]  = sum(p)
end

function jac_structure{M, V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{M, V, S, T})
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

function eval_jac_g{M<:FADMomFun, V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{M, V,S,T}, J, u)
    n, k, m = size(e)
    p  = u[1:n]
    θ  = u[(n+1):(n+k)]
    g  = e.mf.s(θ)
    ws(θ) = e.mf.s(θ)'*p
    Dws = ForwardDiff.jacobian(ws, θ, chunk_size = length(θ))
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

function eval_jac_g{M<:AnaMomFun, V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{M, V,S,T}, J, u)
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
    
function hesslag_structure{M, V, S<:Unconstrained, T<:Unweighted}(e::MDEstimator{M, V, S, T})
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

function eval_hesslag{M<:FADMomFun, V, S<:Unconstrained, T}(e::MDEstimator{M, V, S, T}, H, u, σ, λ)
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
    @inbounds H[n+1:n*k+n] = transpose(ForwardDiff.jacobian(sl, θ, chunk_size = length(θ)))
    @inbounds H[n*k+n+1:e.hele] = gettril(ForwardDiff.hessian(wsl, θ, chunk_size = length(θ)))
end

function eval_hesslag{M<:AnaGradMomFun, V, S<:Unconstrained, T}(e::MDEstimator{M, V, S, T}, H, u, σ, λ)
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
    @inbounds H[n*k+n+1:e.hele] = gettril(ForwardDiff.hessian(wsl, θ, chunk_size = length(θ)))
end
    
function eval_hesslag{M<:AnaFullMomFun, V, S<:Unconstrained, T}(e::MDEstimator{M, V, S, T}, H, u, σ, λ)
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
