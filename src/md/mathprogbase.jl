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

MathProgBase.isobjlinear(d::MDEstimator) = false
MathProgBase.isobjquadratic(d::MDEstimator) = false
MathProgBase.isconstrlinear(d::MDEstimator, i::Int64) = false

features_available(d::MDEstimator) = [:Grad, :Jac, :Hess]

eval_f(d::MDEstimator, u) = Divergences.evaluate(d.div, u[1:d.momf.nobs])

function MathProgSolverInterface.eval_g(d::MDEstimator, g, u)
    n, m, k = size(d.momf)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    @inbounds g[1:m]  = d.momf.ws(θ, p)
    @inbounds g[m+1]  = sum(p)
end

function MathProgSolverInterface.eval_grad_f(d::MDEstimator, grad_f, u)
    n, m, k = size(d.momf)
    for j=1:n
        @inbounds grad_f[j] = Divergences.gradient(d.div, u[j])
    end
    for j=(n+1):(n+k)
        @inbounds grad_f[j] = 0.0
    end
end

function MathProgSolverInterface.jac_structure(d::MDEstimator)
    n, m, k = size(d.momf)
    rows = Array(Int64, d.gele)
    cols = Array(Int64, d.gele)
    for j = 1:m+1, r = 1:n+k
        if !((r > n) && (j==m+1))
            @inbounds rows[r+(j-1)*(n+k)] = j
            @inbounds cols[r+(j-1)*(n+k)] = r
        end
    end
    rows, cols
end

function MathProgSolverInterface.hesslag_structure(d::MDEstimator)
    n, m, k = size(d.momf)
    rows = Array(Int64, d.hele)
    cols = Array(Int64, d.hele)
    for j = 1:n
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

function MathProgSolverInterface.eval_jac_g(d::MDEstimator, J, u)
    n, m, k = size(d.momf)
    p  = u[1:n]
    θ  = u[(n+1):(n+k)]
    g  = d.momf.s(θ)
    Dws = d.momf.Dws(θ, p)

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

function MathProgSolverInterface.eval_hesslag(d::MDEstimator, H, u, σ, λ)
    n, m, k = size(d.momf)
    p = u[1:n]
    θ = u[(n+1):(n+k)]
    if σ==0
        for j=1:n
            @inbounds H[j] = 0.0
        end
    else
      for j=1:n
          @inbounds H[j] = σ*Divergences.hessian(d.div, u[j])
      end
  end
    @inbounds H[n+1:n*k+n] = transpose(d.momf.Dsl(θ, λ[1:m]))
    @inbounds H[n*k+n+1:d.hele] = gettril(d.momf.Hwsl(θ, p, λ[1:m]))
end
