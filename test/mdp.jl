## Test the Minimum Divergence Problem
## min D
## subject to
## A'p < k
using ProgressMeter
using ModelsGenerators
srand(1)

y, x, z = ModelsGenerators.sim_iv_d01(CP = 50);
y = vec(y)

data = Dict{Symbol, Any}(
  :y => y,
  :x => x,
  :z => z
)


size(x)
G = Array(Float64, 100, 5)

f!(G, theta) = broadcast!(*, G, data[:z], data[:y]-data[:x]*theta)

m = MinimumDivergenceProblem(G, zeros(5))

function loglik(theta::DenseVector, m::MinimumDivergenceProblem, f!::Function)
    ## f! is a mutating function
    f!(m.e.mm.S, theta)
    solve!(m)
    sum(log(m.m.inner.x)) + normlogpdf(0, 10, theta[1])
end

_loglik(theta::DenseVector) = loglik(theta, m, f!)

## MCMC simulation
n = 5000
burnin = 1000
sim = Chains(n, 1, names = ["Θ"])
beta = AMWGVariate([0.0])

p = Progress(n, 1)
for i in 1:n
  amwg!(beta, [.5], _loglik, batchsize = 10, adapt = (i <= burnin))
  sim[i, :, 1] = [beta;]
  next!(p)
end
describe(sim)
p = plot(sim)
draw(p, filename="summaryplot.svg")


#####


    srand(1)

    y, x, z = ModelsGenerators.sim_iv_d01(CP = 50);
    y = vec(y)

    data = Dict{Symbol, Any}(
    :y => y,
    :z => z,
    :x => x
    )


    function f(beta)
        delta = beta[1:5]
        theta = beta[6]
        G = broadcast(*, data[:z], data[:y]-data[:z]*delta.*theta)
        H = broadcast(*, data[:z], data[:x]-data[:z]*delta)
        [G H]
    end

    G = f([ones(5); 0])
    m = MinimumDivergenceProblem(G, zeros(10))


    function loglik(theta::DenseVector, m::MinimumDivergenceProblem, f!::Function)
        ## f! is a mutating function
        m.e.mm.S[:] = f(theta)
        solve!(m)
        if m.m.inner.status == 0
            sum(log(m.m.inner.x)) + norm(theta[1:5])*sqrt((1+theta[6]^2))
        else
            -maxintfloat()
        end
    end

    _loglik_beta(beta::DenseVector) = loglik([delta; beta], m, f!)
    _loglik_delta(delta::DenseVector) = loglik([delta; beta], m, f!)

    V           = vcov(estimate!(MDEstimator(f, [zeros(5); .1])))
    Sigma_delta = cholfact(V[1:5,1:5])
    Sigma_beta = [sqrt(V[6,6])]

    n = 100000
    burnin = 2000
    sim = Chains(n, 6, names = ["γ₁", "γ₂", "γ₃", "γ₄", "γ₅", "Θ"])
    beta  = AMWGVariate([0.01])
    delta = AMMVariate(ones(5)*0.3)
    p = Progress(n, 1)
    for i in 1:n
        amwg!(beta, Sigma_beta, _loglik_beta, batchsize = 50, adapt = (i <= burnin))
        amm!(delta, Sigma_delta, _loglik_delta, adapt = (i <= burnin))
        sim[i, :, 1] = [delta; beta]
        next!(p)
    end
    describe(sim)


sim2 = Chains(80001, 2, names = ["||δ||", "Θ"]);
sim2[:,1,1] = mapslices(u -> norm(u), sim.value[20000:100000,1:5,1], 2)
sim2[:,2,1] = sim.value[20000:100000,6,1]

p = plot(sim)
draw(p, filename="summaryplot.svg")

p = contourplot(sim2, bins = 40);
draw(p, filename="contourplot.svg")



function contourplot(c::AbstractChains; bins::Integer=100, na...)
  nrows, nvars, nchains = size(c.value)
  plots = Plot[]
  offset = 1e4 * eps()
  n = nrows * nchains
  for i in 1:(nvars - 1)
    X = c.value[:, i, :]
    qx = linspace(minimum(X) - offset, maximum(X) + offset, bins + 1)
    mx = map(k -> mean([qx[k], qx[k + 1]]), 1:bins)
    idx = Int[findfirst(k -> qx[k] <= x < qx[k + 1], 1:bins) for x in X]
    for j in (i + 1):nvars
      Y = c.value[:, j, :]
      qy = linspace(minimum(Y) - offset, maximum(Y) + offset, bins + 1)
      my = map(k -> mean([qy[k], qy[k + 1]]), 1:bins)
      idy = Int[findfirst(k -> qy[k] <= y < qy[k + 1], 1:bins) for y in Y]
      density = zeros(bins, bins)
      for k in 1:n
        density[idx[k], idy[k]] += 1.0 / n
      end
      p = plot(x=mx, y=my, z=density, Geom.contour,
               Guide.colorkey("Density"),
               Guide.xlabel(c.names[i], orientation=:horizontal),
               Guide.ylabel(c.names[j], orientation=:vertical))
      push!(plots, p)
    end
  end
  return plots
end
