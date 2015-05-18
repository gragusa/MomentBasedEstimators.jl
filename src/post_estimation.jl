# --------------------- #
# Post-estimation tools #
# --------------------- #

function optimal_W(mf::Function, theta::Vector, k::RobustVariance)
    h = mf(theta)
    n = size(h, 1)
    S = vcov(h, k) * n
    W = pinv(S)
    W
end

status(me::MomentBasedEstimator) = me.r.status
StatsBase.coef(me::MomentBasedEstimator) = me.r.coef
objval(me::MomentBasedEstimator) = me.r.objval
momentfunction(me::MomentBasedEstimator) = me.e.mf(coef(me))
jacobian(me::GMMEstimator) = me.e.Dmf(coef(me))
mfvcov(me::MomentBasedEstimator, k::RobustVariance) = vcov(momentfunction(me), k)
nobs(me::MomentBasedEstimator) = me.r.nobs
npar(me::MomentBasedEstimator) = me.r.npar
nmom(me::MomentBasedEstimator) = me.r.nmom
df(me::MomentBasedEstimator) = nmom(me) - npar(me)
z_stats(me::MomentBasedEstimator, k::RobustVariance) = coef(me) ./ stderr(me, k)
p_values(me::MomentBasedEstimator, k::RobustVariance) = 2*ccdf(Normal(), z_stats(me, k))
shat(me::GMMEstimator, k::RobustVariance) = mfvcov(me, k)
optimal_W(me::GMMEstimator, k::RobustVariance) = pinv(full(shat(me, k)*nobs(me)))


StatsBase.vcov(me::GMMEstimator, k::RobustVariance) = vcov(me, k, me.e.mgr)
StatsBase.vcov(me::GMMEstimator) = vcov(me, me.e.mgr.k, me.e.mgr)

function StatsBase.vcov(me::GMMEstimator, k::RobustVariance, mgr::TwoStepGMM)
    G = jacobian(me)
    n = nobs(me)
    p = npar(me)
    S = shat(me, k)
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \not Var(\sqrt{N}
    ## A = pinv(G'*pinv(S)*G)
    ## B = G'*pinv(S)**G
    (n.^2/(n-p))*pinv(G'*pinv(S)*G)
end

function StatsBase.vcov(me::GMMEstimator, k::RobustVariance, mgr::OneStepGMM)
    G = jacobian(me)
    n = nobs(me)
    p = npar(me)
    S = shat(me, k)
    W = me.e.W
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \not Var(\sqrt{N}
    A = pinv(G'*W*G)
    B = G'*W*S*W*G
    (n.^2/(n-p))*A*B*A
end

function StatsBase.stderr(me::GMMEstimator)
    sqrt(diag(vcov(me, me.e.mgr.k, me.e.mgr)))
end

function StatsBase.stderr(me::GMMEstimator, mgr::IterationManager)
    sqrt(diag(vcov(me, mgr.k, mgr)))
end

function StatsBase.stderr(me::GMMEstimator, k::RobustVariance)
    sqrt(diag(vcov(me, k, me.e.mgr)))
end

function J_test(me::GMMEstimator, k::RobustVariance=me.e.mgr.k)
    g = mean(momentfunction(me), 1)
    S = pinv(shat(me, k))
    j = (nobs(me)*(g*S*g'))[1]
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j, clamp(p, eps(), Inf)
end
