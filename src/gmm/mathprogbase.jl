################################################################################
# MathProgBase solver interface - GMMEstimator
################################################################################

function MathProgSolverInterface.initialize(d::GMMEstimator, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgSolverInterface.features_available(e::GMMEstimator) = [:Grad, :Jac, :Hess]

function MathProgSolverInterface.eval_f(e::GMMEstimator, theta)
    idx = e.ist.n[1]   
    gn = e.mf.sn(theta)
    ## TODO: Rewrite this in a more human form
    (gn'e.W[idx]*gn)[1]
end

eval_g{V, T<:Unconstrained, S}(e::GMMEstimator{V, T, S}, g, theta) = nothing

function eval_g{V, T<:Constrained, S}(e::GMMEstimator{V, T, S}, g, theta)
    g[:] = e.c.h(theta)
end

function eval_grad_f(e::GMMEstimator, grad_f, theta)
    ##grad_f[:] = 2*d.Dsn(theta)'*(d.W*d.sn(theta))
    idx = e.ist.n[1]
    gemm!('T', 'N', 2.0, e.mf.Dsn(theta), e.W[idx]*e.mf.sn(theta), 0.0, grad_f)
end

jac_structure{V, T<:Unconstrained, S}(e::GMMEstimator{V, T, S}) = Int[],Int[]

eval_jac_g{V, T<:Unconstrained, S}(e::GMMEstimator{V, T, S}, J, x) = nothing

function jac_structure{V, T<:Constrained, S}(e::GMMEstimator{V, T, S})
    nc = e.c.nc            ## Number of constraints
    n, k, m = size(e.mf)
    ## The jacobian is a nc x k
    rows = Array(Int64, e.gele)
    cols = Array(Int64, e.gele)    
    for j = 1:nc, r = 1:k        
        @inbounds rows[r+(j-1)*k] = j
        @inbounds cols[r+(j-1)*k] = r
    end
    rows, cols
end 

function eval_jac_g{V, T<:Constrained, S}(e::GMMEstimator{V, T, S}, J, theta)
    J[:] = vec(e.c.Dh(theta)')
end
    
    


hesslag_structure(d::GMMEstimator) = Int[], Int[]
eval_hesslag(d::GMMEstimator, H, x, σ, μ) = nothing

