# --------------------- #
# Post-estimation tools #
# --------------------- #

# Post-estimation methods           #
#-----------------------------------#

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

# jacobian of moment function       #
#-----------------------------------#

function jacobian{T <: GMMEstimator}(e::MomentBasedEstimator{T})
    s(θ) = vec(sum(e.e.mf.s(θ), 1))
    θ = coef(e)
    ForwardDiff.jacobian(s, θ, Chunk{min(10, length(θ))}())
end

function jacobian{T <: MDEstimator}(e::MomentBasedEstimator{T}, ver::Symbol)
    if ver == :weighted
        ws(θ) = e.e.mf.s(θ)'*mdweights(e, :shrunk)
    else
        ws(θ) = vec(sum(e.e.mf.s(θ), 1))
    end
    θ = coef(e)
    ForwardDiff.jacobian(ws, θ, Chunk{min(10, length(θ))}())
end

jacobian{T <: MDEstimator}(e::MomentBasedEstimator{T}) = jacobian(e, :weighted)

# smoothing of the moment function  #
#-----------------------------------#

κ₂{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₂
κ₁{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₁
κ₃{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.κ₃
bw{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern.S

smoothing_kernel{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = e.e.mgr.k
smoothing_kernel{T <: MDEstimator}(e::MomentBasedEstimator{T}) = e.e.mf.kern

iteration_manager{T <: GMMEstimator}(e::MomentBasedEstimator{T}) = e.e.mgr

# covariance of the moment function #
#-----------------------------------#

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

# covariance of the estimator       #
#-----------------------------------#

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

StatsBase.vcov{T <: MDEstimator}(e::MomentBasedEstimator{T})  = vcov(e, false, :weighted)
function StatsBase.vcov{T <: MDEstimator}(e::MomentBasedEstimator{T}, robust::Bool, ver::Symbol)
    n, p, m = size(e)
    Ω = mfvcov(e, ver)
    G = jacobian(e, ver)
    V = G'pinv(Ω)*G/n
    sc = n/(n-p)
    V  = robust ? (H = inv(objhessian(e)); H'*V*H) : pinv(V)
    return scale!(sc, V)
end

# hessian of the MD objective       #
#-----------------------------------#

function objhessian{T <: MDEstimator}(m::MomentBasedEstimator{T})
    @assert status(m) == :Optimal "the status of `::MDEstimator` is not :Optimal"
    n, _, p = size(m)
    ## In case KNITRO has ms_enable on disactivate it
    solver = m.s
    fixsolver!(solver)

    mdp = MinimumDivergenceProblem(Array(Float64, (n,p)), zeros(p), wlb = m.e.wlb,
                                   wub = m.e.wub, solver = solver, div = m.e.div)
    function obj(theta)
        mdp.e.mm.S[:] = m.e.mf.s(theta)
        solve!(mdp)
        mdp.m.inner.obj_val[1]
    end
    H = Calculus.hessian(obj, coef(m))
    k1 = κ₁(m)
    k2 = κ₂(m)
    S  = bw(m)
    k2*H/(S*k1^2)
end

# standard error of the estimator   #
#-----------------------------------#

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

# J test                            #
#-----------------------------------#

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

# coef table                        #
#-----------------------------------#

function StatsBase.coeftable(e::MomentBasedEstimator)
    cc = coef(e)
    se = stderr(e)
    zz = z_stats(e)
    CoefTable(hcat(cc, se, zz, 2.0*ccdf(Normal(), abs(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              ["x$i" for i = 1:npar(e)],
              4)
end

# hack: fix solver                  #
#-----------------------------------#

fixsolver!(s::Ipopt.IpoptSolver) = s

function fixsolver!(s::KnitroSolver)
    opt = Array(Any, length(s.options))
    for j in enumerate(s.options)
        r = j[1]; o = j[2]
        idx = (:outlev in o[1]) | (:ms_enable in o[1])
        opt[r] = (o[1], idx ? o[2] : 0)
    end
    soptions[:] = opt
end

# function Gadfly.plot{T <: MDEstimator}(m::MomentBasedEstimator{T}; xc = nothing, yc = nothing,  na...)
#     n, p, k = size(m)
#     @assert p <= 2 "no plotting for high dimensional MDEstimators"
#     solver = m.s
#     fixsolver!(solver)
#     mdp = MinimumDivergenceProblem(Array(Float64, (n, k)), zeros(k), wlb = m.e.wlb,
#                                    wub = m.e.wub, solver = solver, div = m.e.div)

#     cf = coef(m)
#     se = stderr(m)

#     xc = xc == nothing ? collect(cf[1]-5*se[1]:10*se[1]/20:cf[1]+5*se[1]) : xc
#     yc = yc == nothing && p > 1 ? collect(cf[2]-5*se[2]:10*se[2]/20:cf[2]+5*se[2]) : yc

#     if p == 1
#         objv = Array(Float64, length(xc))
#         for j in enumerate(xc)
#             r = j[1]
#             θ = j[2]
#             mdp.e.mm.S[:] = m.e.mf.s(θ)
#             solve!(mdp)
#             if mdp.m.inner.status == 0
#                 v = mdp.m.inner.obj_val[1]
#             else
#                 v = NaN
#             end
#             k1 = κ₁(m)
#             k2 = κ₂(m)
#             S  = bw(m)
#             objv[r] = k2*v/(S*k1^2)
#         end
#         plot(x = xc, y = objv, Geom.line,
#              Guide.xlabel("θ"), Guide.ylabel("P(θ)"))
#     else
#         objv = Array(Float64, (length(xc), length(yc)))
#         for j in enumerate(yc)
#             i  = j[1]
#             θ₂ = j[2]
#             for h in enumerate(xc)
#                 s  = h[1]
#                 θ₁ = h[2]
#                 mdp.e.mm.S[:] = m.e.mf.s([θ₁; θ₂])
#                 solve!(mdp)
#                 if mdp.m.inner.status == 0
#                     v = mdp.m.inner.obj_val[1]
#                 else
#                     v = NaN
#                 end
#                 k1 = MomentBasedEstimators.κ₁(m)
#                 k2 = MomentBasedEstimators.κ₂(m)
#                 S  = MomentBasedEstimators.bw(m)
#                 objv[s,i] = k2*v/(S*k1^2)
#             end
#         end
#         plot(z = objval, Geom.contour)
#         # fig = figure(figsize=(8,6))
#         # ax = fig[:gca](projection="3d")
#         # xgrid, ygrid = meshgrid(xc, yc)
#         # ax[:plot_surface](xgrid, ygrid, objv,
#         #                   rstride=2, cstride=2,
#         #                   cmap=ColorMap("magma"),
#         #                   alpha=0.7, linewidth=0.25)
#         # ax[:contour](xgrid, ygrid, objv, zdir = "z")
#     end
# end
