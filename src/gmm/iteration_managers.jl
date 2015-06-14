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
    n::Array{Int, 1}
    change::Array{Float64, 1}
    prev::Array{Float64, 1}  # previous value
end

finished(::OneStepGMM, ist::IterationState) = ist.n[1] > 1
finished(::TwoStepGMM, ist::IterationState) = ist.n[1] > 2
function finished(mgr::IterativeGMM, ist::IterationState)
    ist.n[1] > mgr.maxiter[1] || abs(ist.change[1]) <= mgr.tol
end
