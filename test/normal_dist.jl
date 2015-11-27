#################################################################
### This test was lifted from the vingette for the R gmm package.
#################################################################
## library(gmm)

## g1 <- function(tet,x) {
##     m1 <- (tet[1]-x)
##     m2 <- (tet[2]^2 - (x - tet[1])^2)
##     m3 <- x^3-tet[1]*(tet[1]^2+3*tet[2]^2)
##     f <- cbind(m1,m2,m3)
##     return(f)
## }

## Dg <- function(tet,x)
## {
## G <- matrix(c(1, 2*(-tet[1]+mean(x)),
##               -3*tet[1]^2-3*tet[2]^2, 0,
##               2*tet[2], -6*tet[1]*tet[2]),
##             nrow=3,ncol=2)
## return(G)
## }

## x <- as.numeric(as.vector(read.csv2('rand_norm.csv',
##                           header=FALSE)[["V1"]]))
## res <-gmm(g1, x, c(mu = 1.0, sig = 1.0), grad = Dg)

using Distributions

srand(42)
x = rand(Normal(4, 2), 1000)

function h(θ)
    m1 = θ[1] - x
    m2 = θ[2]^2 - (x - θ[1]).^2
    m3 = x.^3 - θ[1].*(θ[1]^2 + 3*θ[2]^2)
    return [m1 m2 m3]
end

step_1     = GMMEstimator(h, [1.0, 1.0], initialW = eye(3), mgr = OneStepGMM())
initialize!(step_1);
estimate!(step_1);

ch(θ) = [1 -1]*θ
hlb = [0.]
hub = [0.]

cstep_1 = GMMEstimator(h, [1.0, 1.0], initialW = eye(3), mgr = OneStepGMM(), constraints = Constrained(ch, hlb, hub, 1))


step_2_hac = GMMEstimator(h, coef(step_1), initialW = MomentBasedEstimators.optimal_W(step_1, QuadraticSpectralKernel(0.91469)));

initialize!(step_2_hac);
estimate!(step_2_hac);

step_2_iid = GMMEstimator(h, coef(step_1), initialW = optimal_W(step_1, HC0()));
initialize!(step_2_iid);
estimate!(step_2_iid);

gmm_qs_mgr = GMMEstimator(h, [1.,1.], initialW = eye(3), mgr = TwoStepGMM(QuadraticSpectralKernel(0.91469)));

initialize!(gmm_qs_mgr);
estimate!(gmm_qs_mgr);

gmm_iid_mgr = GMMEstimator(h, [1.,1.], initialW = eye(3), mgr = TwoStepGMM(HC0()));
initialize!(gmm_iid_mgr);
estimate!(gmm_iid_mgr);

### MDE
kl_iid = MDEstimator(h, [1.0, 1.0])
initialize!(kl_iid)
estimate!(kl_iid)

kl_smt = MDEstimator(h, [1.0, 1.0], kernel = MomentBasedEstimators.TruncatedSmoother(2))
initialize!(kl_smt)
estimate!(kl_smt)

el_iid = MDEstimator(h, [1.0, 1.0], div = Divergences.ReverseKullbackLeibler())
cu_iid = MDEstimator(h, [1.0, 1.0], div = Divergences.ChiSquared())
