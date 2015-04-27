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

function g(θ, x)
    m1 = θ[1] - x
    m2 = θ[2]^2 - (x - θ[1]).^2
    m3 = x.^3 - θ[1].*(θ[1]^2 + 3*θ[2]^2)
    return [m1 m2 m3]
end

step_1     = gmm(g, [1.0, 1.0], eye(3), data=x)
step_2_hac = gmm(g, coef(step_1),
                 optimal_W(step_1, QuadraticSpectralKernel(0.91469)), data=x)
step_2_iid = gmm(g, coef(step_1), optimal_W(step_1, HC0()),  data=x)


step_qs_mgr = gmm(g, [1.,1.], eye(3), data = x, mgr = TwoStepGMM(QuadraticSpectralKernel(0.91469)))

step_iid_mgr = gmm(g, [1.,1.], eye(3), data = x, mgr = TwoStepGMM(HC0()))

