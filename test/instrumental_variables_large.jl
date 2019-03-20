## Instrumental variables problems

dt = CSV.read("iv_large.csv", header=false);

y = convert(Array{Float64}, dt[1]);
x = convert(Array{Float64}, dt[2:2]);
z = convert(Array{Float64}, dt[3:end]);

h(theta) = z.*(y-x.*theta);

gmm_base = GMMEstimator(h, [.0])
estimate!(gmm_base)

gmm_one = GMMEstimator(h, [0.], mgr = OneStepGMM())
estimate!(gmm_one)

kl_base = MDEstimator(h, [.0])
estimate!(kl_base)

kl_base_truncated = MDEstimator(h, [.0], kernel = TruncatedSmoother(10))
estimate!(kl_base_truncated)

el_base = MDEstimator(h, [.0], div = Divergences.ReverseKullbackLeibler())
estimate!(el_base)

cue_base = MDEstimator(h, [.0], div = Divergences.ChiSquared())
estimate!(cue_base)

cue_base_truncated = MDEstimator(h, [.0], div = Divergences.ChiSquared(), kernel = TruncatedSmoother(10))
estimate!(cue_base_truncated)

el_fm = MDEstimator(h, [.0], div = Divergences.FullyModifiedReverseKullbackLeibler(.8, .8))
estimate!(el_fm)

## Let constraint on weights be relaxed
cue_uncon = MDEstimator(h, [.0], div = Divergences.ChiSquared())
MomentBasedEstimators.setwtsLB!(cue_uncon, [-Inf for j=1:10000])
estimate!(cue_uncon)


## Anlytic gradient
Dsn(θ)    = -z'x;
Dws(θ, p) = -z'*Diagonal(p)*x;
Dsl(θ, λ) = -x.*(z*λ);
Hwsl(θ, p, λ) = zeros(1,1);

kl_ana_grad = MDEstimator(h, [.0], grad = (Dsn, Dws, Dsl));
estimate!(kl_ana_grad);

kl_ana_full = MDEstimator(h, [.0], grad = (Dsn, Dws, Dsl, Hwsl));
estimate!(kl_ana_full);


# HAS_KNITRO = true
# try
#     cue_knitro = deepcopy(cue_base)
#     solver!(cue_knitro, KnitroSolver(hessopt = 6, print_level = 2))
#     estimate!(cue_knitro)
# catch
#     HAS_KNITRO = false
# end

# if HAS_KNITRO
#     @fact coef(cue_knitro) --> roughly(coef(cue_base))
# end
