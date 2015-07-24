using MomentBasedEstimators
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
        Je, pe = MomentBasedEstimators.J_test(step_iid_mgr)
        ## This is the objective value, which is the J-test
        ## for other softwares
        ov     = 1.4398836656920428
        Jt, pt = (1.4378658264483137,0.23048500853597673)
        @fact Je => roughly(Jt, atol = 0.01)
        @fact pe => roughly(pt, atol = 0.01)
        @fact objval(step_iid_mgr) => roughly(ov, atol = 0.1)
    end

    context("Example 13.5 from Greene (2012) -- verified with Stata") do
        include("gamma_dist.jl")
        cf_stata = [3.358432, .1244622]
        cfe = coef(two_step)
        for j = 1:2
            @fact cfe[j] => roughly(cf_stata[j], atol=1e-3)
        end
        Je, pe = MomentBasedEstimators.J_test(two_step)
        Jt, pt = (1.7433378725860427,0.41825292894063326)
        ## This is the stata J-test
        ov = 1.97522
        @fact Je => roughly(Jt, atol=1e-3)
        @fact pe => roughly(pt, atol=1e-3)
        @fact objval(two_step) => roughly(ov, atol = 0.1)
    end

    context("Instrumental variables -- verified with Stata") do
        include("instrumental_variables.jl")
        cf_stata = [-.0050209, .0708816, .0593661]
        cfe = coef(gmm2s)
        for j = 1:2
            @fact cfe[j] => roughly(cf_stata[j], atol=1e-3)
        end
        V = vcov(gmm2s)
        V_stata = [ .00478911  -.00272551   .00163441;
                    -.00272551   .01026473  -.00201291;
                    .00163441   -.00201291   .00983372]
        
        for j = 1:length(V)
            @fact V[j] => roughly(V_stata[j], atol = 0.00001)
        end

        @fact maximum(V - cf_stata) => roughly(0.01, atol = 0.01)
        Je, pe = MomentBasedEstimators.J_test(gmm2s)
        ov = 1.191
        @fact objval(gmm2s) => roughly(ov, atol = 0.02)
    end
    context("Instrumental variables large --- verified by asymptotics") do
        Vgmm = vcov(gmm)
        Vmd  = vcov(md)
        
        for j = 1:length(V)
            @fact Vgmm[j] => roughly(Vmd[j], atol = 0.00001)
        end    
    
        Vmd_1 = vcov(md, false, :Unweighted)
        Vmd_2 = vcov(md, false, :Weighted)

        for j = 1:length(V)
            @fact Vmd_1[j] => roughly(Vmd_2[j], atol = 0.00001)
        end               
    end
end





## facts("Test utilities") do
##     context("test row_kron") do
##         h = ["a" "b"; "c" "d"]
##         z = ["1" "2" "3"; "4" "5" "6"]
##         want = ["a1" "a2" "a3" "b1" "b2" "b3"; "c4" "c5" "c6" "d4" "d5" "d6"]
##         @fact MomentBasedEstimators.row_kron(h, z) => want

##         # now test on some bigger matrices
##         a = randn(400, 3)
##         b = randn(400, 5)
##         out = MomentBasedEstimators.row_kron(a, b)
##         @fact size(out) => (400, 15)

##         rows_good = true
##         for row=1:400
##             rows_good &= out[row, :] == kron(a[row, :], b[row, :])
##         end
##         @fact rows_good => true
##     end

##     context("test max_args") do
##         foo(x, y, z) = nothing  # standard 3 args
##         bar(x, z=100) = nothing  # standard 2 args with default value
##         baz = (x, y, z)-> nothing  # anonymous 3 args
##         qux(a; b=100) = nothing  # standard 1 with 1 kwarg (kwarg not counted)

##         @fact MomentBasedEstimators.max_args(foo) => 3
##         @fact MomentBasedEstimators.max_args(bar) => 2
##         @fact MomentBasedEstimators.max_args(baz) => 3
##         @fact MomentBasedEstimators.max_args(qux) => 1
##     end
## end

FactCheck.exitstatus()
