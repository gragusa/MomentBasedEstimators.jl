using GMM
using ModelsGenerators
using CovarianceMatrices
using Ipopt
using NLopt

y, x, z = randiv();
mf(theta) = reshape(sum(z.*(y-x*theta),1), 5, 1);
W = eye(5);

g(theta) = z.*(y-x*theta)

obj, minx, statu = GMM.gmm(g, [.1], W)

W = vcov(g(minx), HC0())

obj_2, minx_2, status = GMM.gmm(g, [.1], W)



obj_3, minx_3, status = GMM.gmm(g, [.1], W,
                                solver = NLoptSolver(algorithm = :LD_LBFGS))


obj_3, minx_3, status = GMM.gmm(g, [.1], [-10], [10], W,
                                solver = NLoptSolver(algorithm = :GD_STOGO,
                                                     maxeval = 10000))


