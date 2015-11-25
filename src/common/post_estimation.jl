# --------------------- #
# Post-estimation tools #
# --------------------- #

## Methods
StatsBase.coef{T}(e::MomentBasedEstimator{T}) = e.r.coef
status{T}(e::MomentBasedEstimator{T}) = e.r.status
objvalstatus{T}(e::MomentBasedEstimator{T}) = e.r.objval

momentfunction{T}(e::MomentBasedEstimator{T}) = momentfunction(e, :Smoothed)
momentfunction{T}(e::MomentBasedEstimator{T}, s::Symbol) = s==:Smoothed ? e.e.mf.s(coef(e)) : e.e.mf.g(coef(e))

function shrinkweight(p, n)
    mp = minimum(p)
    ϵ  = -min(mp, 0)
    (p + ϵ)./(1 + ϵ)    # n/(1 + e) + ne/(1+e) = n(1+e)/(1+e) = n
end

function mdweights{T <: MDEstimator}(e::MomentBasedEstimator{T}, ver::Symbol = :unshrunk)
    n = nobs(e)
    p = e.m.inner.x[1:n]
    ver == :shrunk ? shrinkweight(p, n) : p
end

function jacobian{T <: GMMEstimator}(e::MomentBasedEstimator{T})
    s(θ) = vec(sum(e.e.mf.s(θ), 1))
    θ = coef(e)
    ForwardDiff.jacobian(s, θ, chunk_size = length(θ))
end

function jacobian{T <: MDEstimator}(e::MomentBasedEstimator{T}, ver::Symbol)
    if ver == :weighted
        ws(θ) = e.e.mf.s(θ)'*mdweights(e, :shrunk)
    else
        ws(θ) = vec(sum(e.e.mf.s(θ), 1))
    end
    θ = coef(e)
    ForwardDiff.jacobian(ws, θ, chunk_size = length(θ))
end

jacobian{T <: MDEstimator}(e::MomentBasedEstimator{T}) = jacobian(e, :weighted)

κ₂{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₂
κ₁{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₁
κ₃{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₃
bw{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.S

smoothing_kernel{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = e.e.mgr.k
smoothing_kernel{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern

iteration_manager{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = e.e.mgr

# mfvcov(e::MomentBasedEstimator, k::RobustVariance) = vcov(momentfunction(e), k)
# mfvcov{T <: MDEstimator}(e::MomentBasedEstimator{T}) = vcov(momentfunction(e), smoothing_kernel(e))

function mfvcov{T <: MDEstimator}(e::MomentBasedEstimator{T}, ver::Symbol)
    mf = momentfunction(e)
    if ver==:weighted
        p  = sqrt(mdweights(e, :shrunk))
        broadcast!(*, mf, mf, p)
    end
    S  = bw(e)
    k1 = κ₁(e)
    k2 = κ₂(e)
    Omega = vcov(mf, HC0())
    Omega = scale!(k1^2*S/k2, Omega)
    return Omega
end

mfvcov{T <: MDEstimator}(e::MomentBasedEstimator{T}) = mfvcov(e, :weighted)
mfvcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = vcov(momentfunction(e), smoothing_kernel(e))
mfvcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}, k::RobustVariance) = vcov(momentfunction(e), k)

initial_weighting{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = e.e.W[end]

################################################################
#### GMM vcov/stderr
####
################################################################

StatsBase.vcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = vcov(e, smoothing_kernel(e), iteration_manager(e))

function StatsBase.vcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}, k::RobustVariance, mgr::TwoStepGMM)
    G = jacobian(e)
    n = nobs(e)
    p = npar(e)
    S = mfvcov(e, k)
    (n.^2/(n-p))*pinv(G'*pinv(S)*G)
end

function StatsBase.vcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}, k::RobustVariance, mgr::OneStepGMM)
    n, p, m = size(e)
    G = jacobian(e)
    S = mfvcov(e, k)
    W = initial_weighting(e)
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \neq Var(\sqrt{N}g_N(\theta_0))
    A = pinv(G'*W*G)
    B = G'*W*S*W*G
    (n.^2/(n-p))*A*B*A
end

function StatsBase.vcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}, mgr::IterationManager)
    vcov(e, smoothing_kernel(e), mgr)
end

function StatsBase.vcov{T <: GMMEstimator}(e::MomentBasedEstimator{T}, k::RobustVariance)
    vcov(e, k, iteration_manager(e))
end

function StatsBase.stderr{T <: GMMEstimator}(e::MomentBasedEstimator{T})
    sqrt(diag(vcov(e, smoothing_kernel(e), iteration_manager(e))))
end

function StatsBase.stderr{T <: GMMEstimator}(e::MomentBasedEstimator{T}, mgr::IterationManager)
    sqrt(diag(vcov(e, mgr)))
end

function StatsBase.stderr{T}(e::MomentBasedEstimator{T}, k::RobustVariance)
    sqrt(diag(vcov(e, k)))
end

function StatsBase.stderr{T <: MDEstimator}(e::MomentBasedEstimator{T})
    sqrt(diag(vcov(e)))
end

function StatsBase.stderr{T <: MDEstimator}(e::MomentBasedEstimator{T}, robust::Bool, ver::Symbol)
    sqrt(diag(vcov(e, robust, ver)))
end

StatsBase.vcov{T <: MDEstimator}(e::MomentBasedEstimator{T})  = vcov(e, false, :weighted)
function StatsBase.vcov{T <: MDEstimator}(e::MomentBasedEstimator{T}, robust::Bool, ver::Symbol)
    n, p, m = size(e)
    Ω = mfvcov(e, ver)
    G = jacobian(e, ver)
    V = G'pinv(Ω)*G
    if robust
        #H = inv(hessian(e))
        #V = H'*V*H
    else
        V = pinv(G'pinv(Ω)*G)
    end
    return scale!(n.^2/(n-p), V)
end

function J_test(e::MomentBasedEstimator)
    g = mean(momentfunction(e), 1)
    S = pinv(mfvcov(e))
    j = (nobs(e)*(g*S*g'))[1]
    p = df(e) > 0 ? ccdf(Chisq(df(e)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j, clamp(p, eps(), Inf)
end

df{T}(e::MomentBasedEstimator{T}) = nmom(e) - npar(e)
z_stats{T}(e::MomentBasedEstimator{T}) = coef(e) ./ stderr(e)
p_values{T}(e::MomentBasedEstimator{T}) = 2*ccdf(Normal(), z_stats(e))

function StatsBase.coeftable(e::MomentBasedEstimator)
    cc = coef(e)
    se = stderr(e)
    zz = z_stats(e)
    CoefTable(hcat(cc, se, zz, 2.0*ccdf(Normal(), abs(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              ["x$i" for i = 1:npar(e)],
              4)
end
