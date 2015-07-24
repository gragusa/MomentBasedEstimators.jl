## Instrumental variables problems

dt = readcsv("iv.csv");

y = dt[:,1];
x = dt[:,2];
z = dt[:,3:7];

f(theta) = z.*(y-x*theta);

gmms2 = GMMEstimator(f, [.0])
initialize!(gmm2s)
estimate!(gmm2s)




srand(1)
using MomentBasedEstimators
using ModelsGenerators
y,x,z = ModelsGenerators.sim_iv_d01(n = 10000, CP = 2000)
xb    = randn(10000,2)
x     = [x xb]
z     = [z xb]

f(theta) = z.*(y-x*theta);

gmm = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0]);
md = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0]);

MomentBasedEstimators.initialize!(gmm);
MomentBasedEstimators.initialize!(md);

estimate!(gmm);
estimate!(md);


maximum(vcov(gmm)-vcov(md))<1e-07
maximum(vcov(md, false, :Unweighted) - vcov(md, false, :Weighted))<1e-05



MomentBasedEstimators.setparLB!(md, coef(md))
MomentBasedEstimators.setparUB!(md, coef(md))
estimate!(md);

MomentBasedEstimators.setparLB!(md, [0.,0,0])
MomentBasedEstimators.setparUB!(md, [0.,0,0])


## Methods

# 1. Coefficient
coef(gmm)
coef(md)

## 2. Variances
vcov(gmm)
vcov(md)

## 3. Objectiv value


srand(1)
y,x,z=randiv(100)
xb = randn(100,2)
x = [x xb]
z = [z xb]

z=sparse(z)

f(theta) = broadcast(*, z, (y-x*theta))

g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


MomentBasedEstimators.initialize!(g);
MomentBasedEstimators.initialize!(m);

@time estimate!(g);
@time estimate!(m);

srand(1)
y,x,z=randiv(100)
xb = randn(100,2)
x = [x xb]
z = [z xb]

f(theta) = z.*(y-x*theta);

g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


MomentBasedEstimators.initialize!(g)
MomentBasedEstimators.initialize!(m)

@time estimate!(g);
@time estimate!(m);



## Before
## elapsed time: 0.129503039 seconds (69486712 bytes allocated, 40.36% gc time)
