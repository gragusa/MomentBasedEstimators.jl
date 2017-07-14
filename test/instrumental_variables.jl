## Instrumental variables problems

dt = readcsv("iv.csv")

y = dt[:,1];
x = dt[:,2:4];
z = dt[:,5:end];

f(theta) = z.*(y-x*theta);


crossprod(g) = g'g

iv_gmm1s = GMMEstimator(f, [.0, .0, .0], initialW = eye(7), mgr = OneStepGMM())

estimate!(iv_gmm1s)
W = inv(crossprod(f(coef(iv_gmm1s))))

iv_gmm2s = GMMEstimator(f, [.0, .0, .0], initialW = W, mgr = OneStepGMM())
iv_gmm2s_ = GMMEstimator(f, [.0, .0, .0], initialW = eye(7), mgr = TwoStepGMM())
estimate!(iv_gmm2s)
estimate!(iv_gmm2s_)

Dsn(θ)    = -z'x;
Dws(θ, p) = -z'*Diagonal(p)*x;
Dsl(θ, λ) = -x.*(z*λ);
Hwsl(θ, p, λ) = zeros(3,3);

iv_kl = MDEstimator(f, [.0, .0, .0], div = KullbackLeibler(), grad = (Dsn, Dws, Dsl, Hwsl))
iv_el = MDEstimator(f, [.0, .0, .0], div = ReverseKullbackLeibler(), grad = (Dsn, Dws, Dsl, Hwsl))
iv_cu = MDEstimator(f, [.0, .0, .0], div = ChiSquared(), grad = (Dsn, Dws, Dsl, Hwsl));

estimate!(iv_kl)
estimate!(iv_el)
estimate!(iv_cu)



## OV test KullbackLeibler, CUE, ReverseKullbackLeibler
## ----------------------------------------------------
#
# using Base.Test
#
# for est in (iv_kl, iv_cu, iv_el)
#     println(est)
#     LM = LM_test(est)
#     LR = LR_test(est)
#     LMe = LMe_test(est)
#     J   = J_test(est)
#     @test LM[1] ≈ LR[1] atol = 1e-1
#     if isa(est.e.div, ReverseKullbackLeibler)
#         @test isnan(LMe[1])
#     else
#         @test LMe[1] ≈ LR[1] atol = 1e-2
#     end
#     @test J[1] ≈ LR[1] atol = 1e-1
#
#     @test LM[2] ≈ LR[2] atol = 1e-2
#     if isa(est.e.div, ReverseKullbackLeibler)
#         @test isnan(LMe[2])
#     else
#         @test LMe[2] ≈ LR[2] atol = 1e-2
#     end
#
#     @test J[2] ≈ LR[2] atol = 1e-2
# end
#


## Hessian KullbackLeibler, CUE, ReverseKullbackLeibler
## ----------------------------------------------------
# est = iv_el
# for est in (iv_kl, iv_cu, iv_el)
#     @time HAD = objhessian(est)
#     @time HFD = objhessian(est, Val{:finitediff})
#     @time HBF = objhessian(est, Val{:bruteforce})
#     V = vcov(est)
#     VR = vcov(est, robust=true)
#
#     @test norm(inv(HAD)-V, 2)< 1e-2
#     @test norm(inv(HFD)-V, 2)< 1e-2
#     @test norm(inv(HBF)-V, 2)< 1e-2
#
#     ## Low level
#     QQ(theta) = MomentBasedEstimators.Qhessian(est, theta)
#     cfg1 = ForwardDiff.HessianConfig(QQ, coef(est), ForwardDiff.Chunk{3}());
#
#     ## The function Qhessian calculate the hessian of
#     ## -∑γ(Nπ)
#     @test HAD == -ForwardDiff.hessian(QQ, coef(est), cfg1)
#     @test HFD == -Calculus.hessian(QQ, coef(est))
#
# end

# vcov(iv_cu, robust=true)
#
# vcov(iv_cu)
#
#
#
#

## GMM using linear algebra

## Stata Coef
## -.0050209   .0708816   .0593661

## Stata vcov
## symmetric e(V)[3,3]
##             v2          v3          v4
##    .00478911
##   -.00272551   .01026473
##    .00163441  -.00201291   .00983372

##   x1 |  -.0050209   .0692034    -0.07   0.942     -.140657    .1306153
##   x2 |   .0708816    .101315     0.70   0.484    -.1276922    .2694554
##   x3 |   .0593661   .0991651     0.60   0.549     -.134994    .2537261


## Hansen J statistic (overidentification test of all instruments):         1.191
##                                                     Chi-sq(4) P-val =    0.8795





## srand(1)
## using MomentBasedEstimators
## using ModelsGenerators
## y,x,z = ModelsGenerators.sim_iv_d01(n = 10000, CP = 2000)
## xb    = randn(10000,2)
## x     = [x xb]
## z     = [z xb]

## f(theta) = z.*(y-x*theta);

## gmm = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0], initialW = z'z);
## md = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0]);

## MomentBasedEstimators.initialize!(gmm);
## MomentBasedEstimators.initialize!(md);

## estimate!(gmm);
## estimate!(md);


## MomentBasedEstimators.setparLB!(md, coef(md))
## MomentBasedEstimators.setparUB!(md, coef(md))
## estimate!(md);

## MomentBasedEstimators.setparLB!(md, [0.,0,0])
## MomentBasedEstimators.setparUB!(md, [0.,0,0])


## ## Methods

## # 1. Coefficient
## coef(gmm)
## coef(md)

## ## 2. Variances
## vcov(gmm)
## vcov(md)

## ## 3. Objectiv value


## srand(1)
## y,x,z=randiv(100)
## xb = randn(100,2)
## x = [x xb]
## z = [z xb]

## z=sparse(z)

## f(theta) = broadcast(*, z, (y-x*theta))

## g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
## m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


## MomentBasedEstimators.initialize!(g);
## MomentBasedEstimators.initialize!(m);

## @time estimate!(g);
## @time estimate!(m);

## srand(1)
## y,x,z=randiv(100)
## xb = randn(100,2)
## x = [x xb]
## z = [z xb]

## f(theta) = z.*(y-x*theta);

## g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
## m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


## MomentBasedEstimators.initialize!(g)
## MomentBasedEstimators.initialize!(m)

## @time estimate!(g);
## @time estimate!(m);



## Before
## elapsed time: 0.129503039 seconds (69486712 bytes allocated, 40.36% gc time)
