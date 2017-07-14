using Genoud
using MomentBasedEstimators
using Ipopt
using MathProgBase

cd(joinpath(Pkg.dir("MomentBasedEstimators"), "test"))
dt = readcsv("iv.csv");

y = dt[:,1];
x = dt[:,2:4];
z = dt[:,5:end];

f(theta) = z.*(y-x*theta);
m = MDEstimator(f, [.0, .0, .0], div = KullbackLeibler())
estimate!(m)

function solvedual!(m::MomentBasedEstimator,
                   init_x::Vector{Float64} = m.e.x0;
                   sizepop::Int = 100*length(init_x),
                   solver_inner:: MathProgBase.SolverInterface.AbstractMathProgSolver = IpoptSolver(print_level=0),
                   optimize_best::Bool = true,
                   domain::Genoud.Domain = Genoud.Domain(clamp.([m.e.lb m.e.ub], -1e1, 1e1)),
                   opt::Genoud.Options = Genoud.Options(),
                   operator_o::Genoud.Operators = Genoud.Operators(),
                   optimizer_o::Optim.Options = Optim.Options())

    n,k,p = size(m)
    mdp = MinimumDivergenceProblem(Array{Float64}(n,p), zeros(p),
                                   wlb = m.e.wlb,
                                   wub = m.e.wub,
                                   solver = solver_inner,
                                   div = m.e.div)
    function obj(theta)
        mdp.e.mm.S[:] = m.e.mf.s(theta)
        try
            solve!(mdp)
            mdp.m.inner.obj_val[1]
        catch
            +Inf
        end
    end

    obj(coef(m))
    #out = Genoud.genoud(obj, coef(m), sizepop = 50, domain = Genoud.Domain([-10*ones(3) 10*ones(3)]))
    out = Genoud.genoud(obj, coef(m), sizepop = sizepop, optimize_best = optimize_best,
                 domain = domain, opt = opt, operator_o = operator_o,
                 optimizer_o = optimizer_o, sense = :Min)
    mdp.e.mm.S[:] = m.e.mf.s(out.bestindiv)
    solve!(mdp)
    lambda = -mdp.m.inner.mult_g
    eta = mdp.m.inner.x[end]
    m.m.inner.obj_val = out.bestfitns
    m.m.inner.x[1:k] = out.bestindiv
    m.m.inner.x[k+1:end] = mdp.m.inner.x
    m.m.inner.mult_g = -mdp.m.inner.mult_g
    m.r.status = ifelse(Genoud.f_converged(out), :GenoudOptimal, :GenoudFailure)
    m
end


solvedual!(m)

m.e














m.e.SmoothingKernel
