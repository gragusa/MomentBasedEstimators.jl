## Instrumental variables problems

dt = CSV.read("iv.csv"; header = false);

y = convert(Array{Float64}, dt[1]);
x = convert(Array{Float64}, dt[2:4]);
z = convert(Array{Float64}, dt[5:end]);

f(theta) = z.*(y-x*theta);


gmm2s = GMMEstimator(f, [.0, .0, .0])
estimate!(gmm2s)


Dsn(θ)    = -z'x;
Dws(θ, p) = -z'*Diagonal(p)*x;
Dsl(θ, λ) = -x.*(z*λ);
Hwsl(θ, p, λ) = zeros(3,3);

mds = MDEstimator(f, [.0, .0, .0], grad = (Dsn, Dws, Dsl, Hwsl));

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
