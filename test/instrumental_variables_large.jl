## Instrumental variables problems

dt = readcsv("iv_large.csv");

y = dt[:, 1];
x = dt[:, 2];
x = reshape(x, size(x,1), 1)
z = dt[:, 3:end];

h(theta) = z.*(y-x*theta);

gmm_base = GMMEstimator(h, [.0])
estimate!(gmm_base)


kl_base = MDEstimator(h, [.0])
estimate!(kl_base)

el_base = MDEstimator(h, [.0], div = Divergences.ReverseKullbackLeibler())
estimate!(el_base)

cue_base = MDEstimator(h, [.0], div = Divergences.ChiSquared())
estimate!(cue_base)

## Let constraint on weights be relaxed
cue_uncon = MDEstimator(h, [.0], div = Divergences.ChiSquared())
MomentBasedEstimators.setwtsLB!(cue_uncon, [-Inf for j=1:10000])
estimate!(cue_uncon)

HAS_KNITRO = true
try
    cue_knitro = deepcopy(cue_base)
    solver!(cue_knitro, KnitroSolver(hessopt = 6, print_level = 2))
    estimate!(cue_knitro)    
catch
    HAS_KNITRO = false
end

if HAS_KNITRO
    @fact coef(cue_knitro)





