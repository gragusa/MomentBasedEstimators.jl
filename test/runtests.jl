using MomentBasedEstimators
##using FactCheck
using Base.Test

@testset "Test from vignette for the R gmm package........(normal_dist.jl)" begin
    include("normal_dist.jl")
    cft = [3.843, 2.067]
    cfe = coef(step_1)
    sdt = [0.06527666675890574,0.04672629429325811]
    sde = stderr(step_1, TwoStepGMM())
    for j = 1:2
        @test cfe[j]≈cft[j] atol=0.001
        @test sde[j]≈sdt[j] atol=0.001
    end
    Je, pe = MomentBasedEstimators.J_test(gmm_iid_mgr)
    ## This is the objective value, which is the J-test
    ## for other softwares
    ov     = 1.4398836656920428
    Jt, pt = (1.4378658264483137,0.23048500853597673)
    @test Je≈Jt
    @test pe≈pt
    @test objval(gmm_iid_mgr)≈ov
end

@testset "Example 13.5 from Greene (2012)..................(gamma_dist.jl)" begin
    include("gamma_dist.jl")
    cf_stata = [3.358432, .1244]
    cfe = coef(two_step)
    for j = 1:2
        @test cfe[j]≈cf_stata[j] atol=0.001
    end
    Je, pe = MomentBasedEstimators.J_test(two_step)
    Jt, pt = (3.0875508893757986,0.21357324340764258)
    ## This is the stata J-test, which is the minimum of the GMM objective
    ## function
    ov = 1.97522
    @test Je≈Jt atol = 0.001
    @test pe≈pt atol = 0.001
    ## Our criterion is indeed smaller
    @test objval(two_step)<=ov
end

@testset "Instrumental variables...............(instrumental_variables.jl)" begin
    include("instrumental_variables.jl")
        @testset "One step GMM" begin
            cf_stata  = [-.0004677045508287847,	.07832334190607070923,	.07939878851175308228]
            obj_stata = .0092804*nobs(iv_gmm1s)^2
            V_stata   = [.0051886963997978992   -.002925597247777894   .0021495241949285961
                         -.002925597247777894   .0105187022380289937  -.0024974289616277101
                         .0021495241949285961  -.0024974289616277101   .0123292464859444551]
            cf = coef(iv_gmm1s)
            V  = vcov(iv_gmm1s)
            obj= MomentBasedEstimators.objval(iv_gmm1s)

            for j in 1:length(cf)
                @test cf_stata[j]≈cf[j] atol = 1e-08
            end

            for j in  1:length(V)
              @test V[j]≈V_stata[j] atol = 1e-03
            end

            @test obj≈obj_stata atol = 1e-4

        end

        @testset "Two step GMM" begin
            cf_stata = [-.005020876118737405  .0708815796904491147   .059366051523307109]
            V_stata  = [.0051886963997978992   -.002925597247777894   .0021495241949285961
                        -.002925597247777894   .0105187022380289937  -.0024974289616277101
                         .0021495241949285961  -.0024974289616277101   .0123292464859444551]

            obj_stata = .0119120165901957821*nobs(iv_gmm2s)
            J_stata =  1.191201659019578196

            cf = coef(iv_gmm2s)
            V  = vcov(iv_gmm2s)
            obj = MomentBasedEstimators.objval(iv_gmm2s)
            J = J_test(iv_gmm2s)

            for j in 1:3
                @test cf[j]≈cf_stata[j] atol = 1e-08
            end

            for j = 1:length(V)
              @test V[j]≈V_stata[j] atol = 1e-2
            end

            Je, pe = MomentBasedEstimators.J_test(iv_gmm2s)

            @test obj≈obj_stata atol = 1e-7
            @test obj<obj_stata
            @test J_stata≈obj atol = 1e-7
        end

        @testset "Coherence one-step vs two-step" begin
            @test coef(iv_gmm2s)≈coef(iv_gmm2s_)
            @test vcov(iv_gmm2s)≈vcov(iv_gmm2s_) atol = 1e-5
            @test stderr(iv_gmm2s)≈stderr(iv_gmm2s_) atol = 1e-4
            @test objval(iv_gmm2s)≈objval(iv_gmm2s_)
        end
end

@testset "Instrumental variables.........(instrumental_variables_large.jl)" begin
    include("instrumental_variables_large.jl")
    @testset "Ipopt" begin
        @test status(gmm_base) == :Optimal
        @test status(gmm_one) == :Optimal
        @test status(el_base) == :Optimal
        @test status(kl_base) == :Optimal
        @test status(kl_base_truncated) == :Optimal
        @test status(cue_base) == :Optimal
        @test status(cue_uncon) == :Optimal
        @test status(kl_ana_grad) == :Optimal
        @test status(kl_ana_full) == :Optimal

        for k in (gmm_one, el_base, kl_base, kl_base_truncated, cue_base, cue_uncon, kl_ana_grad, kl_ana_full)
            @test coef(gmm_base) ≈ coef(k) atol = 0.01
            V  = vcov(gmm_base)
            Vm = vcov(k)
            vv = sqrt.(diag(Vm))
            ss = stderr(gmm_base)
            ee = stderr(k)
            J  = J_test(gmm_base)[1]/100
            Jm = J_test(k)[1]/100
            @test J≈Jm atol = 1e-2
            for j in 1:length(V)
                @test V[j]≈Vm[j] atol = 1e-4
            end
            for j in 1:length(ss)
                @test ss[j]≈ee[j] atol = 1e-3
                @test ee[j]≈vv[j]
            end
        end
    end
end



HAS_KNITRO = false
isa(Pkg.installed("KNITRO"), VersionNumber) && begin
    pkg = Symbol("KNITRO") ## for example
    eval(:($(Expr(:using, pkg))))
    HAS_KNITRO = true
    h(theta) = reshape(x.-theta, 100, 1)
    x = randn(100)
    try
        klbase = MDEstimator(h, [.0], s = KNITRO.KnitroSolver(outlev=0))
        estimate!(klbase)
    catch
        HAS_KNITRO = false
    end
end

if HAS_KNITRO
@testset "Instrumental variables (KNITRO)(instrumental_variables_large.jl)" begin
    include("instrumental_variables_large_KNITRO.jl")
    @test status(knitro_gmm_base)   == :Optimal
    @test status(knitro_gmm_one)    == :Optimal
    @test status(knitro_el_base)    == :Optimal
    @test status(knitro_kl_base)    == :Optimal
    @test status(knitro_kl_base_truncated)    == :Optimal
    @test status(knitro_cue_base)   == :Optimal
    @test status(knitro_cue_uncon)  == :Optimal
    @test status(knitro_kl_ana_grad)== :Optimal
    @test status(knitro_kl_ana_full)== :Optimal

    for k in (knitro_gmm_one, knitro_el_base, knitro_kl_base, knitro_kl_base_truncated, knitro_cue_base, knitro_cue_uncon, knitro_kl_ana_grad, knitro_kl_ana_full)
        @test coef(gmm_base) ≈ coef(k) atol = 0.01
        V  = vcov(gmm_base)
        Vm = vcov(k)
        vv = sqrt.(diag(Vm))
        ss = stderr(gmm_base)
        ee = stderr(k)
        J  = J_test(gmm_base)[1]/100
        Jm = J_test(k)[1]/100
        @test J≈Jm atol = 1e-2
        for j in 1:length(V)
            @test V[j]≈Vm[j] atol = 1e-4
        end
        for j in 1:length(ss)
            @test ss[j]≈ee[j] atol = 1e-3
            @test ee[j]≈vv[j]
        end
    end
end
end



#
#
#
#
#
# ## facts("Test utilities") do
# ##     context("test row_kron") do
# ##         h = ["a" "b"; "c" "d"]
# ##         z = ["1" "2" "3"; "4" "5" "6"]
# ##         want = ["a1" "a2" "a3" "b1" "b2" "b3"; "c4" "c5" "c6" "d4" "d5" "d6"]
# ##         @fact MomentBasedEstimators.row_kron(h, z) --> want
#
# ##         # now test on some bigger matrices
# ##         a = randn(400, 3)
# ##         b = randn(400, 5)
# ##         out = MomentBasedEstimators.row_kron(a, b)
# ##         @fact size(out) --> (400, 15)
#
# ##         rows_good = true
# ##         for row=1:400
# ##             rows_good &= out[row, :] == kron(a[row, :], b[row, :])
# ##         end
# ##         @fact rows_good --> true
# ##     end
#
# ##     context("test max_args") do
# ##         foo(x, y, z) = nothing  # standard 3 args
# ##         bar(x, z=100) = nothing  # standard 2 args with default value
# ##         baz = (x, y, z)-> nothing  # anonymous 3 args
# ##         qux(a; b=100) = nothing  # standard 1 with 1 kwarg (kwarg not counted)
#
# ##         @fact MomentBasedEstimators.max_args(foo) --> 3
# ##         @fact MomentBasedEstimators.max_args(bar) --> 2
# ##         @fact MomentBasedEstimators.max_args(baz) --> 3
# ##         @fact MomentBasedEstimators.max_args(qux) --> 1
# ##     end
# ## end
#
# FactCheck.exitstatus()
