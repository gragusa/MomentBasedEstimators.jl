#################################################################
### This test was lifted from the vingette for the R gmm package.
#################################################################
using Distributions

srand(42)
x = rand(Normal(4, 2), 1000)

function h_norm(θ)
    m1 = [θ[1]] .- x
    m2 = [θ[2]].^2 .- (x .- [θ[1]]).^2
    m3 = x.^3 .- [θ[1]].*([θ[1]].^2 .+ 3*[θ[2]].^2)
    return [m1 m2 m3]
end



step_1 = GMMEstimator(h_norm, [1.0, 1.0], initialW = eye(3), mgr = OneStepGMM())
@time estimate!(step_1);

ch(θ) = [1 -1]*θ
hlb = [0.]
hub = [0.]

cstep_1 = GMMEstimator(h_norm, [1.0, 1.0], initialW = eye(3), mgr = OneStepGMM(), constraints = Constrained(ch, hlb, hub, 1))

step_2_hac = GMMEstimator(h_norm, coef(step_1), initialW = MomentBasedEstimators.optimal_W(step_1, QuadraticSpectralKernel(0.91469)));


estimate!(step_2_hac);

step_2_iid = GMMEstimator(h_norm, coef(step_1), initialW = optimal_W(step_1, HC0()));
estimate!(step_2_iid);

gmm_qs_mgr = GMMEstimator(h_norm, [1.,1.], initialW = eye(3), mgr = TwoStepGMM(QuadraticSpectralKernel(0.91469)));
estimate!(gmm_qs_mgr);

gmm_iid_mgr = GMMEstimator(h_norm, [1.,1.], initialW = eye(3), mgr = TwoStepGMM(HC0()));
estimate!(gmm_iid_mgr);

### MDE
kl_iid = MDEstimator(h_norm, [1.0, 1.0])
estimate!(kl_iid)

kl_smt = MDEstimator(h_norm, [1.0, 1.0], kernel = MomentBasedEstimators.TruncatedSmoother(2))
estimate!(kl_smt)

el_iid = MDEstimator(h_norm, [1.0, 1.0], div = Divergences.ReverseKullbackLeibler())
cu_iid = MDEstimator(h_norm, [1.0, 1.0], div = Divergences.ChiSquared())
