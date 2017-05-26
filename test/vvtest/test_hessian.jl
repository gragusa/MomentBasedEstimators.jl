using ForwardDiff
using BenchmarkTools
using Base.Test
cd(joinpath(Pkg.dir("MomentBasedEstimators"), "test"))
dt = readcsv("iv.csv");

y = dt[:,1];
x = dt[:,2:4];
z = dt[:,5:end];

f(theta) = z.*(y-x*theta);


md = MDEstimator(f, zeros(3))
estimate!(md);


@time HAD = objhessian(md)
@time HFD = objhessian(md, Val{:finitediff})
@time HBF = objhessian(md, Val{:bruteforce})
V = vcov(md)
VR = vcov(md, robust=true)

@test norm(inv(HAD)-V, 2)< 1e-3
@test norm(inv(HFD)-V, 2)< 1e-3
@test norm(inv(HBF)-V, 2)< 1e-3

## Low level
QQ(theta) = MomentBasedEstimators.Qhessian(md, theta)

cfg1 = ForwardDiff.HessianConfig(QQ, coef(md), ForwardDiff.Chunk{3}());

## The function Qhessian calculate the hessian of
## -∑γ(Nπ)
@test HAD == -ForwardDiff.hessian(QQ, coef(md), cfg1)
@test HFD == -Calculus.hessian(QQ, coef(md))
