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
        @fact GMM.J_test(step_2_iid) => (1.4398836656920428,0.23015816678811782)
    end
end
