using GMM
using ModelsGenerators
using Ipopt
using NLopt

srand(1)
y, x, z = ModelsGenerators.sim_iv_d01(CP = 100)


g(theta) = z.*(y-x*theta)

W = eye(5);
out = gmm(g, [.1], W)

W = pinv(GMM.mfvcov(out, HC1()))
out = GMM.gmm(g, [.1], W)

coef(out)
sqrt(vcov(out))

sqrt(vcov(out, ParzenKernel(3))

W = eye(5);
out = GMM.gmm(g, [.1], W, solver = NLoptSolver(algorithm = :LD_LBFGS))

W = pinv(GMM.mfvcov(out, HC1()))
out = GMM.gmm(g, [.1], W, solver = NLoptSolver(algorithm = :LD_LBFGS))



