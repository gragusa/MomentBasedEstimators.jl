dt = readcsv("iv_large.csv");

y = dt[:, 1];
x = dt[:, 2];
x = reshape(x, size(x,1), 1)
z = dt[:, 3:end];

h(theta) = z.*(y-x*theta);

#=
KNITRO
=#
HAS_KNITRO = true
try
    kl_base = MDEstimator(h, [.0], s = KNITRO.KnitroSolver())
    estimate!(kl_base)
catch
    HAS_KNITRO = false
end

sgmm = KNITRO.KnitroSolver(hessopt=2, outlev=0)
smd = KNITRO.KnitroSolver(outlev=0)

knitro_gmm_base = GMMEstimator(h, [.0], s = sgmm)
estimate!(knitro_gmm_base)

knitro_gmm_one = GMMEstimator(h, [0.], mgr = OneStepGMM(), s=sgmm)
estimate!(knitro_gmm_one)

knitro_kl_base = MDEstimator(h, [.0], s = smd)
estimate!(knitro_kl_base)

knitro_kl_base_truncated = MDEstimator(h, [.0], kernel = TruncatedSmoother(10), s = smd)
estimate!(knitro_kl_base_truncated)

knitro_el_base = MDEstimator(h, [.0], div = Divergences.ReverseKullbackLeibler(), s = smd)
estimate!(knitro_el_base)

knitro_cue_base = MDEstimator(h, [.0], div = Divergences.ChiSquared(), s = smd)
estimate!(knitro_cue_base)

knitro_cue_base_truncated = MDEstimator(h, [.0], div = Divergences.ChiSquared(), kernel = TruncatedSmoother(10), s = smd)
estimate!(knitro_cue_base_truncated)

knitro_el_fm = MDEstimator(h, [.0], div = Divergences.FullyModifiedReverseKullbackLeibler(.8, .8), s = smd)
estimate!(knitro_el_fm)

## Let constraint on weights be relaxed
knitro_cue_uncon = MDEstimator(h, [.0], div = Divergences.ChiSquared(), s = smd)
MomentBasedEstimators.setwtsLB!(knitro_cue_uncon, [-Inf for j=1:10000])
estimate!(knitro_cue_uncon)


## Anlytic gradient
Dsn(θ)    = -z'x;
Dws(θ, p) = -z'*Diagonal(p)*x;
Dsl(θ, λ) = -x.*(z*λ);
Hwsl(θ, p, λ) = zeros(1,1);

knitro_kl_ana_grad = MDEstimator(h, [.0], grad = (Dsn, Dws, Dsl), s = smd);
estimate!(knitro_kl_ana_grad);

knitro_kl_ana_full = MDEstimator(h, [.0], grad = (Dsn, Dws, Dsl, Hwsl), s = smd);
estimate!(knitro_kl_ana_full);
