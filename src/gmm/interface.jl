## Constructor with function and x0
function GMMEstimator(f::Function, x0::Vector; data = nothing, dtype = :dual)
    _mf(x0) = data == nothing ? f(x0) : f(x0, data)
    g0  = _mf(x0); n, m = size(g0)
    mf  = MomentFunction(_mf, dtype, nobs = n, npar = length(x0), nmom = m)
    e   = GMMEstimator(mf, x0, TwoStepGMM(), eye(m), 0, 0)
    MomentBasedEstimator(e)
end



