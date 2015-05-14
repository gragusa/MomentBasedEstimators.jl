# ----------------------------- #
# MathProgBase solver interface #
# ----------------------------- #

function MathProgSolverInterface.initialize(d::GMMNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end


MathProgSolverInterface.features_available(d::GMMNLPE) = [:Grad, :Jac, :Hess]

function MathProgSolverInterface.eval_f(d::GMMNLPE, theta)
    gn = d.smf(theta)
    (gn'd.W*gn)[1]
end

MathProgSolverInterface.eval_g(d::GMMNLPE, Dg, x) = nothing

function MathProgSolverInterface.eval_grad_f(d::GMMNLPE, grad_f, theta)
    grad_f[:] = 2*d.Dmf(theta)'*(d.W*d.smf(theta))
end

MathProgSolverInterface.jac_structure(d::GMMNLPE) = [],[]
MathProgSolverInterface.eval_jac_g(d::GMMNLPE, J, x) = nothing
MathProgSolverInterface.eval_hesslag(d::GMMNLPE, H, x, σ, μ) = nothing
MathProgSolverInterface.hesslag_structure(d::GMMNLPE) = [],[]
