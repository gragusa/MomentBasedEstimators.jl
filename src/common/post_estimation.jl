# --------------------- #
# Post-estimation tools #
# --------------------- #

# Post-estimation methods           #
#-----------------------------------#

StatsBase.coef(e::MomentBasedEstimator{T}) where {T} = e.r.coef
StatsBase.coef(e::GenericMomentBasedEstimator) = e.coef

status(e::MomentBasedEstimator{T}) where {T} = e.r.status
objvalstatus(e::MomentBasedEstimator{T}) where {T} = e.r.objval

momentfunction(e::MomentBasedEstimator{T}) where {T} = momentfunction(e, Val{:smoothed})
momentfunction(e::MomentBasedEstimator{T}, ::Type{Val{:smoothed}}) where {T} = e.e.mf.s(coef(e))

momentfunction(e::MomentBasedEstimator{T}, ::Type{Val{:unsmoothed}}) where {T} = e.e.mf.g(coef(e))

momentfunction(e::MomentBasedEstimator{T}, theta) where {T} = e.e.mf.s(theta)
momentfunction(e::GenericMomentBasedEstimator, theta) = e.mf.s(theta)

function shrinkweight(p::Array{T}) where {T}
    mp = minimum(p)
    ϵ  = -min(mp, 0)
    (p + ϵ)./(1 + ϵ)    # n/(1 + e) + ne/(1+e) = n(1+e)/(1+e) = n
end


impliedprob(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = impliedprob(e, Val{:unshrunk})
impliedprob(e::MomentBasedEstimator{T}, ::Type{Val{:unshrunk}}) where {T <: MDEstimator} = e.m.inner.x[1:nobs(e)]

function impliedprob(e::MomentBasedEstimator{T}, ::Type{Val{:shrunk}}) where {T <: MDEstimator}
    shrinkweight(impliedprob(e))::Array{Float64, 1}
end

function impliedprob(e::MomentBasedEstimator{T}, ::Type{Val{:shrunk}}) where {T <: GMMEstimator}
    ones(first(size(e)))
end

function impliedprob(e::MomentBasedEstimator{T}, ::Type{Val{:unshrunk}}) where {T <: GMMEstimator}
    ones(first(size(e)))
end


multiplier(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = -e.m.inner.mult_g[1:end]
multiplier_eta(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = -e.m.inner.mult_g[end]
multiplier_lambda(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = -e.m.inner.mult_g[1:end-1]


# jacobian of moment function       #
#-----------------------------------#

function jacobian(e::MomentBasedEstimator{T}; weighted = true, shrinkweights = false) where {T <: GenericMomentBasedEstimator}
    t = weighted ? Val{:weighted} : Val{:unweighted}
    w = shrinkweights ? Val{:shrunk} : Val{:unshrunk}
    MomentBasedEstimators.jacobian(e, t, w)
end

function jacobian(e::MomentBasedEstimator, t::Type{Val{:weighted}}, w)
    if isa(e.e.mf, MomentBasedEstimators.FADMomFun)
      p = impliedprob(e, w)
      ws(theta) = momentfunction(e, theta)'*p
      theta = coef(e)
      ForwardDiff.jacobian(ws, theta)::Matrix{Float64}
    else
      e.e.mf.Dws(coef(e), impliedprob(e, w))
    end
end

function jacobian(e::MomentBasedEstimator, t::Type{Val{:unweighted}}, w)
    if isa(e.e.mf, MomentBasedEstimators.FADMomFun)
      p = ones(first(size(e)))
      ws(theta) = At_mul_B(momentfunction(e, theta), p)
      theta = coef(e)
      ForwardDiff.jacobian(ws, theta)::Matrix{Float64}
    else
      e.e.mf.Dsn(coef(e))
    end
end



# smoothing of the moment function  #
#-----------------------------------#

κ₂(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = e.e.mf.kern.κ₂
κ₁(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = e.e.mf.kern.κ₁
κ₃(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = e.e.mf.kern.κ₃
bw(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = e.e.mf.kern.S

smoothing_kernel(e::MomentBasedEstimator{T}) where {T <: GMMEstimator} = e.e.mgr.k
smoothing_kernel(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = e.e.mf.kern
iteration_manager(e::MomentBasedEstimator{T}) where {T <: GMMEstimator} = e.e.mgr

# covariance of the moment function #
#-----------------------------------#

function mfvcov(e::MomentBasedEstimator{T}, weighted::Bool = true, shrinkweights::Bool = false) where {T <: MDEstimator}
    t = weighted ? Val{:weighted} : Val{:unweighted}
    w = shrinkweights ? Val{:shrunk} : Val{:unshrunk}
    mfvcov(e, t, w)
end

function mfvcov(e::MomentBasedEstimator{T}, t::Type{Val{:weighted}}, w) where {T <: MDEstimator}
    mf = copy(momentfunction(e))
    p  = sqrt.(impliedprob(e, w))
    broadcast!(*, mf, mf, p)
    S  = bw(e)
    k1 = κ₁(e)
    k2 = κ₂(e)
    Omega = vcov(mf, HC0())
    Omega = scale!(k1^2*S/k2, Omega)
    return Omega
end

function mfvcov(e::MomentBasedEstimator{T}, t::Type{Val{:unweighted}}, w) where {T <: MDEstimator}
    mf = momentfunction(e)
    S  = bw(e)
    k1 = κ₁(e)
    k2 = κ₂(e)
    Omega = vcov(mf, HC0())
    Omega = scale!(k1^2*S/k2, Omega)
    return Omega
end

mfvcov(e::MomentBasedEstimator{T}) where {T <: GMMEstimator} = vcov(momentfunction(e), smoothing_kernel(e))

adjfactor(e::MomentBasedEstimator, k::RobustVariance) = 1.0
adjfactor(e::MomentBasedEstimator, k::HC1) = nobs(e)/(nobs(e)-npar(e))

function mfvcov(e::MomentBasedEstimator{T}, k::RobustVariance) where {T <: GMMEstimator}
    ## FIXME: Add methods to calculate variances that respond to k::RobustVariance
    ## duck type
    V = vcov(momentfunction(e), k)
end

initial_weighting(e::MomentBasedEstimator{T}) where {T <: GMMEstimator} = e.e.W[end]

# covariance of the estimator       #
#-----------------------------------#

StatsBase.vcov(e::MomentBasedEstimator{T}) where {T <: GMMEstimator} = vcov(e, smoothing_kernel(e), iteration_manager(e))

function StatsBase.vcov(e::MomentBasedEstimator{T}, k::RobustVariance, mgr::TwoStepGMM) where {T <: GMMEstimator}
    n, p, m = size(e)
    G = MomentBasedEstimators.jacobian(e)
    S = mfvcov(e, k)
    #(n.^2/(n-p))*pinv(G'*pinv(S)*G)
    n*pinv(G'*pinv(S)*G)*adjfactor(e, k)
end

function StatsBase.vcov(e::MomentBasedEstimator{T}, k::RobustVariance, mgr::OneStepGMM) where {T <: GMMEstimator}
    n, p, m = size(e)
    G = jacobian(e)
    S = mfvcov(e, k)
    W = initial_weighting(e)
    ## Use the general form of the variance covariance matrix
    ## that gives the correct covariance even when S \neq Var(\sqrt{N}g_N(\theta_0))
    A = pinv(G'*W*G)
    B = G'*W*S*W*G
    #(n.^2/(n-p))*A*B*A
    n*A*B*A*adjfactor(e, k)
end

function StatsBase.vcov(e::MomentBasedEstimator{T}, k::RobustVariance) where {T <: GMMEstimator}
    vcov(e, k, iteration_manager(e))
end

function StatsBase.vcov(e::MomentBasedEstimator{T}, mgr::IterationManager) where {T <: GMMEstimator}
    vcov(e, smoothing_kernel(e), mgr)
end

function StatsBase.vcov(e::MomentBasedEstimator{T}; robust::Bool = false, weighted::Bool = true, shrinkweights = false) where {T <: MDEstimator}
    r = robust ? Val{:robust} : Val{:unrobust}
    t = weighted ? Val{:weighted} : Val{:unweighted}
    w = shrinkweights ? Val{:shrunk} : Val{:unshrunk}
    vcov(e, r, t, w)
end

function StatsBase.vcov(e::MomentBasedEstimator{T}, ::Type{Val{:robust}}, t, w) where {T <: MDEstimator}
    n, p, m = size(e)
    G = jacobian(e, t, w)
    S = mfvcov(e, t, w)
    V = G'pinv(S)*G/n
    H = inv(objhessian(e))
    H'*V*H
    ## Do not degree-of-freedom correct
    #sc = n/(n-p)
    #return scale!(sc, V)
    #V
end

function StatsBase.vcov(e::MomentBasedEstimator{T}, ::Type{Val{:unrobust}}, t, w) where {T <: MDEstimator}
    n, p, m = size(e)
    S = mfvcov(e, t, w)
    G = jacobian(e, t, w)
    V = pinv(G'pinv(S)*G/n)
    ## Do not degree-of-freedom correct
    #sc = n/(n-p)
    #return scale!(sc, V)
    V
end


function StatsBase.vcov(e::MomentBasedEstimator{T}, k::RobustVariance) where {T <: MDEstimator}
    n, p, m = size(e)
    mf = momentfunction(e, Val{:unsmoothed})
    Ω = CovarianceMatrices.vcov(mf, k)
    G = jacobian(e, weighted = false, shrinkweights = false)
    V = pinv(G'pinv(Ω)*G/n)
    #sc = n/(n-p)
    scale!(adjfactor(e, k), V)
end


# hessian of the MD objective       #
#-----------------------------------#


ρ(d::KullbackLeibler, x::Real) = exp(x) - one(x)
ρ(d::ReverseKullbackLeibler, x::Real) = -log(1-x)
ρ(d::ChiSquared, x::Real) = x^2/2 + x

ρ₁(e::MomentBasedEstimator{T}) where {T <: MDEstimator} = ρ₁(e.e.div)
ρ₁(d::KullbackLeibler, x::Real) = exp(x)
ρ₁(d::ReverseKullbackLeibler, x::Real) = 1/(1-x)
ρ₁(d::ChiSquared, x::Real) = 1+x

function ρ₁(d::CressieRead, x::Real)
    α = d.α
    x^α/2-α
end

ρ₂(d::KullbackLeibler, x::Real) = exp(x)
ρ₂(d::ReverseKullbackLeibler, x::Real) = 1/((1-x)^2)
ρ₂(d::ChiSquared, x::Real) = one(x)

function ρ₂(d::CressieRead, x::Real)
    α = d.α
    .5*α*x^(α-1)
end


objhessian(md::MomentBasedEstimator{T}) where {T <: MDEstimator} = objhessian(md, Val{:forwarddiff})

function objhessian(md::MomentBasedEstimator{T}, ::Type{Val{:forwarddiff}}) where {T <: MDEstimator}
    Θ = coef(md)
    QQ = (theta) -> Qhessian(md, theta)
    cfg1 = ForwardDiff.HessianConfig(QQ, Θ, ForwardDiff.Chunk{length(Θ)}())
    HAD = ForwardDiff.hessian(QQ, Θ, cfg1)
    k1 = κ₁(md)
    k2 = κ₂(md)
    S  = bw(md)
    scale!(HAD, -k2/(S*k1^2))
end

function objhessian(md::MomentBasedEstimator{T}, ::Type{Val{:finitediff}}) where {T <: MDEstimator}
    Θ = coef(md)
    QQ(theta) = Qhessian(md, theta)
    #cfg1 = ForwardDiff.HessianConfig(QQ, Θ, ForwardDiff.Chunk{length(Θ)}())
    HFD = Calculus.hessian(QQ, coef(md))
    k1 = κ₁(md)
    k2 = κ₂(md)
    S  = bw(md)
    scale!(HFD, -k2/(S*k1^2))
end

function objhessian(m::MomentBasedEstimator{T}, ::Type{Val{:bruteforce}}) where {T <: MDEstimator}
    @assert status(m) == :Optimal "the status of `::MDEstimator` is not :Optimal"
    n, _, p = size(m)
    ## In case KNITRO has ms_enable on disactivate it
    solver = m.s
    fixsolver!(solver)

    mdp = MinimumDivergenceProblem(Array{Float64}(n,p), zeros(p), wlb = m.e.wlb,
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
    scale!(H, k2/(S*k1^2))
end

function Qhessian(e::MomentBasedEstimator{T}, Θ::Array{F,1}) where {T <: MDEstimator,F <: Real}
    # println(Θ)
    d = e.e.div
    γ, iter = .9, 25
    n, p, m = size(e)
    g = MomentBasedEstimators.momentfunction(e, Θ)
    lo = Array{F}(m+1)
    ln = Array{F}(m+1)
    λ = multiplier(e)
    u = g*λ[1:end-1] + λ[end]
    w = MomentBasedEstimators.ρ₁.(d, u)
    w2 = MomentBasedEstimators.ρ₂.(d, u)
    gw = g'w
    sumw = sum(w)
    H = [(w2.*g)'g  gw
         gw' sumw]
    ∇f = [gw; sumw - n]
    ln .= λ .- γ*(H\∇f)
    u .= g*ln[1:end-1] + ln[end]
    # println("Norm:")
    # println(norm(ln-lo, 2))
    while norm(ln-lo, 2) > 1e-08
        # println("Norm 2:")
        # println(norm(ln-lo, 2))
        copy!(lo, ln)
        w .= MomentBasedEstimators.ρ₁.(d, u)
        w2 .= MomentBasedEstimators.ρ₂.(d, u)
        gw .= g'w
        sumw = sum(w)
        H .= [(w2.*g)'g  gw
             gw' sumw]
        ∇f .= [gw; sumw - n]
        ln .= lo .- γ*(H\∇f)
        u .= g*ln[1:end-1] + ln[end]
    end
    sum(ρ.(d, u)-ln[end])
end



# standard error of the estimator   #
#-----------------------------------#

function StatsBase.stderr(e::MomentBasedEstimator{T}) where {T <: GMMEstimator}
    sqrt.(diag(vcov(e, smoothing_kernel(e), iteration_manager(e))))
end

function StatsBase.stderr(e::MomentBasedEstimator{T}, mgr::IterationManager) where {T <: GMMEstimator}
    sqrt.(diag(vcov(e, smoothing_kernel(e), mgr)))
end

function StatsBase.stderr(e::MomentBasedEstimator{T}, k::RobustVariance) where {T}
    sqrt.(diag(vcov(e, k)))
end

function StatsBase.stderr(e::MomentBasedEstimator{T}; robust::Bool = false, weighted::Bool = true, shrinkweights::Bool = true) where {T <: MDEstimator}
    sqrt.(diag(vcov(e, robust = robust, weighted = weighted, shrinkweights = shrinkweights)))
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

df(e::MomentBasedEstimator{T}) where {T} = nmom(e) - npar(e)
z_stats(e::MomentBasedEstimator{T}) where {T} = coef(e) ./ StatsBase.stderr(e)
p_values(e::MomentBasedEstimator{T}) where {T} = 2*ccdf(Normal(), z_stats(e))

# Lagrange multiplier test          #
#-----------------------------------#

## These test have the form
## nλ'Qλ ∼ χ²(m-k_unrestricted) dof

function LM_test(e::MomentBasedEstimator{T}) where {T <: MDEstimator}
    w = impliedprob(e)
    j = sum(w.*gradient(e.e.div, w).^2)
    p = df(e) > 0 ? ccdf(Chisq(df(e)), j) : NaN
    return j, clamp(p, eps(), Inf)
end

function LR_test(e::MomentBasedEstimator{T}) where {T <: MDEstimator}
    j = objval(e)
    p = df(e) > 0 ? ccdf(Chisq(df(e)), j) : NaN
    return j, clamp(p, eps(), Inf)
end


ψ₃(e) = ψ₃(e.e.div)

ψ₃(d::KullbackLeibler) = 1.0
ψ₃(d::ModifiedKullbackLeibler) = 1.0
ψ₃(d::FullyModifiedKullbackLeibler) = 1.0
ψ₃(d::ChiSquared) = 0.0

ψ₃(d::ReverseKullbackLeibler) = 2.0
ψ₃(d::ModifiedReverseKullbackLeibler) = 2.0
ψ₃(d::FullyModifiedReverseKullbackLeibler) = 2.0

ψ₃(d::CressieRead) = d.α-1
ψ₃(d::ModifiedCressieRead) = d.α-1
ψ₃(d::FullyModifiedCressieRead) = d.α-1



function LMe_test(e::MomentBasedEstimator{T}) where {T <: MDEstimator}
    psi3 = ψ₃(e)
    if psi3 != 2.0
        η = multiplier_eta(e)
        j = nobs(e)*η/(1-psi3/2)
        p = df(e) > 0 ? ccdf.(Chisq(df(e)), j) : NaN
    else
        j = NaN
        p = NaN
    end
    return j, clamp(p, eps(), Inf)
end

# coef table                        #
#-----------------------------------#

function StatsBase.coeftable(e::MomentBasedEstimator)
    cc = coef(e)
    se = StatsBase.stderr(e)
    zz = z_stats(e)
    CoefTable(hcat(cc, se, zz, 2.0*ccdf.(Normal(), abs.(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              ["x$i" for i = 1:npar(e)],
              4)
end

# hack: fix solver                  #
#-----------------------------------#

fixsolver!(s::Ipopt.IpoptSolver) = s

function fixsolver!(s::MathProgBase.SolverInterface.AbstractMathProgSolver)
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
