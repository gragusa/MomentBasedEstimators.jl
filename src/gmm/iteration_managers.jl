# ------------------ #
# Iteration managers #
# ------------------ #

abstract type IterationManager 

end

struct OneStepGMM <: IterationManager
    k::RobustVariance
    demean::Bool
end

struct TwoStepGMM <: IterationManager
    k::RobustVariance
    demean::Bool
end

struct IterativeGMM <: IterationManager
    k::RobustVariance
    demean::Bool
    tol::Float64
    maxiter::Int
end

# kwarg constructors with default values
OneStepGMM(;k::RobustVariance=HC0(), demean::Bool = false) = OneStepGMM(k, demean)
TwoStepGMM(;k::RobustVariance=HC0(), demean::Bool = false) = TwoStepGMM(k, demean)

OneStepGMM(k::RobustVariance) = OneStepGMM(k, false)
TwoStepGMM(k::RobustVariance) = TwoStepGMM(k, false)



function IterativeGMM(;k::RobustVariance=HC0(), demean::Bool = false, tol::Float64=1e-12,
                       maxiter::Int=500)
    IterativeGMM(k, demean, tol, maxiter)
end

mutable struct IterationState
    n::Array{Int, 1}
    change::Array{Float64, 1}
    prev::Array{Float64, 1}  # previous value
end

finished(::OneStepGMM, ist::IterationState) = ist.n[1] > 1
finished(::TwoStepGMM, ist::IterationState) = ist.n[1] > 2
function finished(mgr::IterativeGMM, ist::IterationState)
    ist.n[1] > mgr.maxiter[1] || abs(ist.change[1]) <= mgr.tol
end
