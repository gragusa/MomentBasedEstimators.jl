# --------------- #
# Display methods #
# --------------- #

function show_extra(me::MomentBasedEstimator{T}) where T<:GMMEstimator
    j, p = J_test(me)
    "\nJ-test: $(round(j; digits = 3)) (P-value: $(round(p, digits = 3)))\n"
end

show_extra(me::MomentBasedEstimator{T}) where T<:MDEstimator = "\n"


function rshow(me::MomentBasedEstimator{T}) where T<:GMMEstimator
    println("GMM")
end

function rshow(me::MomentBasedEstimator{T}) where T<:MDEstimator
    println("MD")
end

function Base.show(io::IO, ::MIME"text/plain", me::T) where T<:MomentBasedEstimator
  s = symbolfy(me)
  println(io, "\n$s [k = $(npar(me)), m = $(nmom(me))]\n")

  if me.status[1] == :Solved
    # Only gives information if the solver
    # converged
    if (status(me) == :Optimal)
      # get coef table and j-test
      ct = coeftable(me)
      # print coefficient table
      println(io, "Coefficients:\n")
      show(io, ct)
      # Then show extra information for this type
      println(io, show_extra(me))
    else
      println("The optimization did not converge to a local solution")
    end
  end
end

function symbolfy(me::T) where T<:MomentBasedEstimator
    s = split(string(T), "MomentBasedEstimator")[2]
    v = split(s, "{")[2]
    Symbol(v)

    # if (v == "GMMEstimator{")
    #     Symbol(v, split(s[4], "{")[1], ", ", split(s[10], ",")[1], ", ", split(s[11], ",")[1], "}")
    # elseif (s[3] == "MDEstimator{")
    #     if (s[4] == "AnaGradMomFun{" || s[4] == "AnaFullMomFun{")
    #       Symbol(s[3], split(s[4], "{")[1], ", ", split(s[7], "},Divergences.")[2], " ", split(s[8], ",")[1], "}")
    #     else
    #       Symbol(s[3], split(s[4], "{")[1], ", ", split(s[9], "},Divergences.")[2], " ", split(s[10], ",")[1], "}")
    #     end
    # end
end
