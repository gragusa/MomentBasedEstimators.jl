using ReverseDiff
using Optim
using MomentBasedEstimators


cd(joinpath(Pkg.dir("MomentBasedEstimators"), "test"))
dt = readcsv("iv.csv");

y = dt[:,1];
x = dt[:,2:4];
z = dt[:,5:end];

f(theta) = z.*(y-x*theta);


md = MDEstimator(f, zeros(3))
estimate!(md)

objval(md)

λ = -md.m.inner.mult_g[1:end-1]
η = -md.m.inner.mult_g[end]
Θ = coef(md)

g = f(Θ)
ρ(u) = exp(u) - 1
∇ρ(u) = exp.(u)

u = g*λ + η

-2*sum(ρ.(u)-η)

sum(∇ρ(u)-1)
g'∇ρ(u)



ww = md.m.inner.x[1:MomentBasedEstimators.nobs(md)]



function QQ(Θ, l0)
    g = f(Θ)
    function obj(λ)
        u = g*λ[1:end-1] + λ[end]
        2*sum(ρ.(u)-λ[end])
    end
    Optim.minimum(Optim.optimize(obj, l0, LBFGS()))
end

res = QQ(Θ, ones(8))

inv(-Calculus.hessian(theta -> QQ(theta, ones(8)/3), Θ))


Optim.minimizer(res)











λ
function Q(Θ, η, λ)
    iter = 25
    g = f(Θ)
    n, m = size(g)
    ln = [λ; η]
    γ = .9
    for j in 2:iter
        println(j)
        lo = copy(ln)
        u = g*ln[1:end-1] + ln[end]
        w = ∇ρ.(u)
        gw = g'w
        sumw = sum(w)
        H = [(w.*g)'g  gw
            gw' sumw]
        ∇f = [gw; sumw - n]
        ln = lo .- γ*(H\∇f)
        if norm(ln-lo, 2) < 1e-08
            break
        end
    end
    u = g*ln[1:end-1] + ln[end]
    2*sum(ρ.(u)-ln[end])
end

isposdef(g'g)

Q(Θ, η, λ) <= objval(md)

Q(Θ.+0.01, η, λ)
using ForwardDiff

ForwardDiff.gradient(theta -> Q(theta, η, λ), theta)
dg = Calculus.hessian(theta -> Q(theta, η, λ), theta)

obj(theta) = Q(theta, η, λ)
using ForwardDiff: GradientConfig, Chunk, gradient!
cfg1 = GradientConfig(obj, Θ, ForwardDiff.Chunk{1}())
cfg2 = ForwardDiff.GradientConfig(obj, Θ, Chunk{8}())

@time ForwardDiff.hessian(theta -> Q(theta, η, λ), theta)




@time H = Calculus.hessian(theta -> Q(theta, η, λ), Θ)
inv(-H)

dg


vcov(md)
