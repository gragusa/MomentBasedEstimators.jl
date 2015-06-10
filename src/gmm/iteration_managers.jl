# ------------------ #
# Iteration managers #
# ------------------ #

abstract IterationManager

immutable OneStepGMM <: IterationManager
    k::RobustVariance
end

immutable TwoStepGMM <: IterationManager
    k::RobustVariance
end

immutable IterativeGMM <: IterationManager
    k::RobustVariance
    tol::Float64
    maxiter::Int
end

# kwarg constructors with default values
OneStepGMM(;k::RobustVariance=HC0()) = OneStepGMM(k)

TwoStepGMM(;k::RobustVariance=HC0()) = TwoStepGMM(k)

function IterativeGMM(;k::RobustVariance=HC0(), tol::Float64=1e-12,
                       maxiter::Int=500)
    IterativeGMM(k, tol, maxiter)
end

type IterationState
    n::Int
    change::Float64
    prev::Array  # previous value
end

finished(::OneStepGMM, ist::IterationState) = ist.n >= 1
finished(::TwoStepGMM, ist::IterationState) = ist.n >= 2
function finished(mgr::IterativeGMM, ist::IterationState)
    ist.n > mgr.maxiter || abs(ist.change) <= mgr.tol
end
