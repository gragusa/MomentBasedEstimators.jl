using MomentBasedEstimators
using FactCheck

facts("Testing basic interface") do
    context("Test from vignette for the R gmm package. Automatic Differentiation") do
        include("normal_dist.jl")
        cft = [3.843, 2.067]
        cfe = coef(step_1)
        sdt = [0.06527666675890574,0.04672629429325811]
        sde = stderr(step_1, TwoStepGMM())
        for j = 1:2
            @fact cfe[j] --> roughly(cft[j], 0.01)
            @fact sde[j] --> roughly(sdt[j], 0.01)
        end
        Je, pe = MomentBasedEstimators.J_test(gmm_iid_mgr)
        ## This is the objective value, which is the J-test
        ## for other softwares
        ov     = 1.4398836656920428
        Jt, pt = (1.4378658264483137,0.23048500853597673)
        @fact Je --> roughly(Jt, atol = 0.01)
        @fact pe --> roughly(pt, atol = 0.01)
        @fact objval(gmm_iid_mgr) --> roughly(ov, atol = 0.1)
    end

    context("Example 13.5 from Greene (2012) -- verified with Stata") do
        include("gamma_dist.jl")
        cf_stata = [3.358432, .1244622]
        cfe = coef(two_step)
        for j = 1:2
            @fact cfe[j] --> roughly(cf_stata[j], atol=1e-3)
        end
        Je, pe = MomentBasedEstimators.J_test(two_step)
        ## The following test is conditional on version.
        ## What changed is the tol for pinv between 0.3 and 0.4
        ##
        if VERSION < v"0.4-"
            Jt, pt = (1.7433378725860427,0.41825292894063326)
        else
            Jt, pt = (3.0875508893757986,0.21357324340764258)
        end
        ## This is the stata J-test
        ov = 1.97522
        @fact Je --> roughly(Jt, atol=1e-3)
        @fact pe --> roughly(pt, atol=1e-3)
        @fact objval(two_step) --> roughly(ov, atol = 0.1)
    end

    context("Instrumental variables -- verified with Stata") do
        include("instrumental_variables.jl")
        cf_stata = [-.0050209, .0708816, .0593661]
        cfe = coef(gmm2s)
        V = vcov(gmm2s)
        for j = 1:2
            @fact cfe[j] --> roughly(cf_stata[j], atol=1e-3)
        end

        V_stata = [ .00478911  -.00272551   .00163441;
                    -.00272551   .01026473  -.00201291;
                    .00163441   -.00201291   .00983372]

        for j = 1:length(V)
            @fact V[j] --> roughly(V_stata[j], atol = 0.01)
        end

        Je, pe = MomentBasedEstimators.J_test(gmm2s)
        ov = 1.191
        @fact objval(gmm2s) --> roughly(ov, atol = 0.02)
    end
    context("Instrumental variables large --- verified by asymptotics") do
        include("instrumental_variables_large.jl")
        @fact status(gmm_base)    --> :Optimal
        @fact status(gmm_one)     --> :Optimal
        @fact status(el_base)     --> :Optimal
        @fact status(kl_base)     --> :Optimal
        @fact status(cue_base)    --> :Optimal
        @fact status(cue_uncon)   --> :Optimal
        @fact status(kl_ana_grad) --> :Optimal
        @fact status(kl_ana_full) --> :Optimal

        @fact coef(el_base) --> roughly(coef(kl_base), 0.01)
        @fact coef(kl_base) --> roughly(coef(kl_ana_full))

        @fact objval(kl_base) --> roughly(objval(kl_ana_full))

        @fact vcov(kl_base)  --> roughly(vcov(kl_ana_full))
        @fact vcov(kl_base)  --> roughly(vcov(kl_ana_full))
        @fact vcov(gmm_base) --> roughly(vcov(el_base), 0.01)

        @fact vcov(el_base, false, :unweighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_base, false, :unweighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_ana_grad, false, :unweighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_ana_full, false, :unweighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(cue_base, false, :unweighted) --> roughly(vcov(gmm_base), 0.001)

        @fact vcov(el_base, false, :weighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_base, false, :weighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_ana_grad, false, :weighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(kl_ana_full, false, :weighted)  --> roughly(vcov(gmm_base), 0.001)
        @fact vcov(cue_base, false, :weighted) --> roughly(vcov(gmm_base), 0.001)

        @fact stderr(gmm_base)' --> sqrt(vcov(gmm_base))

        @fact stderr(kl_base)' --> sqrt(vcov(kl_base))
        @fact stderr(el_base)' --> sqrt(vcov(el_base))

        @fact stderr(kl_base, false, :weighted) --> stderr(kl_base)
        @fact stderr(el_base, false, :weighted) --> stderr(el_base)

        @fact vcov(kl_base_truncated)  --> roughly(vcov(kl_base), 0.01)
        @fact vcov(cue_base_truncated) --> roughly(vcov(cue_base), 0.01)

        @fact J_test(kl_base_truncated)[1] --> roughly(J_test(kl_base)[1], 0.05)
        @fact J_test(cue_base_truncated)[1]  --> roughly(J_test(cue_base)[1], 0.05)
        @fact J_test(kl_base_truncated)[2] --> roughly(J_test(kl_base)[2], 0.02)
        @fact J_test(cue_base_truncated)[2]  --> roughly(J_test(cue_base)[2], 0.02)


        tmp = J_test(gmm_base)
        @fact tmp[1] --> roughly(7.937738532483664, 1e-07)
        @fact tmp[2] --> roughly(0.5404326890480416, 1e-07)

        tmp = J_test(el_base)
        @fact tmp[1] --> roughly(7.825790062416562, 1e-07)
        @fact tmp[2] --> roughly(0.5517934008321658, 1e-07)

        @fact coeftable(gmm_base).mat[1:2]'  --> [coef(gmm_base) stderr(gmm_base)]
        @fact coeftable(el_base).mat[1:2]'  --> [coef(el_base) stderr(el_base)]

    end
    ## context("Instrumental variables large --- verified by asymptotics") do
    ##     Vgmm = vcov(gmm)
    ##     Vmd  = vcov(md)

    ##     for j = 1:length(V)
    ##         @fact Vgmm[j] --> roughly(Vmd[j], atol = 0.00001)
    ##     end

    ##     Vmd_1 = vcov(md, false, :unweighted)
    ##     Vmd_2 = vcov(md, false, :weighted)

    ##     for j = 1:length(V)
    ##         @fact Vmd_1[j] --> roughly(Vmd_2[j], atol = 0.00001)
    ##     end
    ## end
end





## facts("Test utilities") do
##     context("test row_kron") do
##         h = ["a" "b"; "c" "d"]
##         z = ["1" "2" "3"; "4" "5" "6"]
##         want = ["a1" "a2" "a3" "b1" "b2" "b3"; "c4" "c5" "c6" "d4" "d5" "d6"]
##         @fact MomentBasedEstimators.row_kron(h, z) --> want

##         # now test on some bigger matrices
##         a = randn(400, 3)
##         b = randn(400, 5)
##         out = MomentBasedEstimators.row_kron(a, b)
##         @fact size(out) --> (400, 15)

##         rows_good = true
##         for row=1:400
##             rows_good &= out[row, :] == kron(a[row, :], b[row, :])
##         end
##         @fact rows_good --> true
##     end

##     context("test max_args") do
##         foo(x, y, z) = nothing  # standard 3 args
##         bar(x, z=100) = nothing  # standard 2 args with default value
##         baz = (x, y, z)-> nothing  # anonymous 3 args
##         qux(a; b=100) = nothing  # standard 1 with 1 kwarg (kwarg not counted)

##         @fact MomentBasedEstimators.max_args(foo) --> 3
##         @fact MomentBasedEstimators.max_args(bar) --> 2
##         @fact MomentBasedEstimators.max_args(baz) --> 3
##         @fact MomentBasedEstimators.max_args(qux) --> 1
##     end
## end

FactCheck.exitstatus()
