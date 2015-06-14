## New test

srand(1)
using MomentBasedEstimators
using ModelsGenerators
y,x,z=randiv(1000)
xb = randn(1000,2)
x = [x xb]
z = [z xb]

f(theta) = z.*(y-x*theta);

g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


MomentBasedEstimators.initialize!(g)
MomentBasedEstimators.initialize!(m)

estimate!(g)
estimate!(m)

srand(1)
y,x,z=randiv(1000)
xb = randn(1000,2)
x = [x xb]
z = [z xb]

f(theta) = z.*(y-x*theta);

g = MomentBasedEstimators.GMMEstimator(f, [.1, 0, 0])
m = MomentBasedEstimators.MDEstimator(f, [.1, 0, 0])


MomentBasedEstimators.initialize!(g)
MomentBasedEstimators.initialize!(m)

@time estimate!(g)
@time estimate!(m)


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
