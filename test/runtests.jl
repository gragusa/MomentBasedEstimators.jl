using GMM
using FactCheck

facts("Testing basic interface") do
    context("Test from vingette for the R gmm package.") do
        include("normal_dist.jl")
        cft = [3.84376, 2.06728]
        cfe = coef(step_1)
        sdt = [0.06527666675890574,0.04672629429325811]
        sde = stderr(step_1)
        for j = 1:2
            @fact cfe[j] => roughly(cft[j])
            @fact sde[j] => roughly(sdt[j])
        end
        Je, pe = GMM.J_test(step_2_iid)
        Jt, pt = (1.4398836656920428,0.23015816678811782)
        @fact Je => roughly(Jt)
        @fact pe => roughly(pt)
    end

    context("Example 13.5 from Greene (2012) -- verified with Stata") do
        include("gamma_dist.jl")
        cf_stata = [3.358432, .1244622]
        cfe = coef(two_step)
        for j = 1:2
            @fact cfe[j] => roughly(cf_stata[j], atol=1e-3)
        end
        Je, pe = GMM.J_test(two_step)
        Jt, pt = (1.97522, 0.3725)
        @fact Je => roughly(Jt, atol=1e-3)
        @fact pe => roughly(pt, atol=1e-3)
    end
end
