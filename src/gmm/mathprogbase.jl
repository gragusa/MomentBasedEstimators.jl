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

MathProgSolverInterface.features_available(d::GMMEstimator) = [:Grad, :Jac, :Hess]

function MathProgSolverInterface.eval_f(d::GMMEstimator, theta)
    gn = d.smf(theta)
    (gn'd.W*gn)[1]
end

MathProgSolverInterface.eval_g(d::GMMEstimator, Dg, x) = nothing

function MathProgSolverInterface.eval_grad_f(d::GMMEstimator, grad_f, theta)
    grad_f[:] = 2*d.Dmf(theta)'*(d.W*d.smf(theta))
end

MathProgSolverInterface.jac_structure(d::GMMEstimator) = [],[]
MathProgSolverInterface.eval_jac_g(d::GMMEstimator, J, x) = nothing
MathProgSolverInterface.eval_hesslag(d::GMMEstimator, H, x, σ, μ) = nothing
MathProgSolverInterface.hesslag_structure(d::GMMEstimator) = [],[]
