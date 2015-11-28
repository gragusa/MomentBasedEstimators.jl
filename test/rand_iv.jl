using MomentBasedEstimators
using ModelsGenerators

srand(1)
y, x, z = ModelsGenerators.sim_iv_d01(CP = 100);

g(theta) = z.*(y-x*theta);

gmm_fad = GMMEstimator(g, [.0]);
estimate!(gmm_fad);

gmm_ana = GMMEstimator(g, [.0], grad = θ -> -z'x);
estimate!(gmm_ana);

kl_base = MDEstimator(g, [.0]);
estimate!(kl_base);

## Example instrumental Variables

## Dsn = ∂gn/ ∂θ' (m x k)
## Dws = ∂∑wᵢgᵢ/∂θ' (m x k)
## Dsl = ∂∑gᵢ*λ (k x 1)

Dsn(θ)    = -z'x;
Dws(θ, p) = -z'*scale(p,x);
Dsl(θ, λ) = -x.*(z*λ);
Hwsl(θ, p, λ) = zeros(1,1);

kl_ana_grad = MDEstimator(g, [.0], grad = (Dsn, Dws, Dsl));
estimate!(kl_ana_grad);
@time estimate!(kl_ana_grad);
kl_ana_full = MDEstimator(g, [.0], grad = (Dsn, Dws, Dsl, Hwsl));
estimate!(kl_ana_full);
@time estimate!(kl_ana_full);

begin
    srand(1)
    sim = 100;
    n = 100
    k = 4
    coeff = zeros(sim);
    statu = Array(Symbol, sim);
    stder = Array(Float64, sim);
    x_add = randn(n, k-1)
    Dsn(θ)    = -z'x;
    Dws(θ, p) = -z'*scale(p,x);
    Dsl(θ, λ) = -x.*(z*λ);
    Hwsl(θ, p, λ) = zeros(1,1);

    @time for j = 1:sim
        y, x, z = ModelsGenerators.sim_iv_d01(CP = 100, n = n);
        x = [x x_add];
        z = [z x_add];
        kl_base = MDEstimator(g, zeros(k));
        estimate!(kl_base);
        coeff[j] = coef(kl_base)[1];
        statu[j] = status(kl_base);
        stder[j] = stderr(kl_base)[1];
    end
end

begin
    srand(1)
    sim = 1000;
    n = 1000
    k = 4
    coeff = zeros(sim);
    x_add = randn(n, k-1)
    Dsn(θ)    = -z'x;
    Dws(θ, p) = -z'*scale(p,x);
    Dsl(θ, λ) = -x.*(z*λ);
    Hwsl(θ, p, λ) = zeros(k,k);

    @time for j = 1:sim
        y, x, z = ModelsGenerators.sim_iv_d01(CP = 100, n = n);
        x = [x x_add];
        z = [z x_add];
        kl_base = MDEstimator(g, zeros(k), grad = (Dsn, Dws, Dsl, Hwsl));
        estimate!(kl_base);
        coeff[j] = coef(kl_base)[1];
    end
end


# W = inv(z'z);

# W = eye(5);
# out = gmm(g, [.1], W)


# W = pinv(GMM.mfvcov(out, HC0())*100)
# out = GMM.gmm(g, [.0], W)

# coef(out)
# sqrt(vcov(out))

# sqrt(vcov(out, ParzenKernel(3))

# W = eye(5);
# out = GMM.gmm(g, [.1], W, solver = NLoptSolver(algorithm = :LD_LBFGS))

# W = pinv(GMM.mfvcov(out, HC1()))
# out = GMM.gmm(g, [.1], W, solver = NLoptSolver(algorithm = :LD_LBFGS))
