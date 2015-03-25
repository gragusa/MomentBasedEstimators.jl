module GMMtests

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

facts("Test utilities") do
    context("test row_kron") do
        h = ["a" "b"; "c" "d"]
        z = ["1" "2" "3"; "4" "5" "6"]
        want = ["a1" "a2" "a3" "b1" "b2" "b3"; "c4" "c5" "c6" "d4" "d5" "d6"]
        @fact GMM.row_kron(h, z) => want

        # now test on some bigger matrices
        a = randn(400, 3)
        b = randn(400, 5)
        out = GMM.row_kron(a, b)
        @fact size(out) => (400, 15)

        rows_good = true
        for row=1:400
            rows_good = out[row, :] == kron(a[row, :], b[row, :])
        end
        @fact rows_good => true
    end
end


end  # module
