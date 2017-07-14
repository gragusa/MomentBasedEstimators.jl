# --------------- #
# Display methods #
# --------------- #

function show_extra{T <: GMMEstimator}(me::MomentBasedEstimator{T})
    j, p = J_test(me)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

function show_extra{T <: MDEstimator}(me::MomentBasedEstimator{T})
    j, p = LR_test(me)
    "\nLR-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
    j, p = LM_test(me)
    "\nLM-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
    j, p = LMe_test(me)
    "\nLMe-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
    j, p = LMe_test(me)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

function Base.show{T<:MomentBasedEstimator}(io::IO, ::MIME"text/plain", me::T)
  s = symbolfy(me)
  println(io, "\n$s [k = $(npar(me)), m = $(nmom(me))]\n")

  if me.status[1] == :Solved
    # Only gives information if the solver
    # converged
    if (status(me) == :Optimal) || (status(me) == :GenoudOptimal)
      # get coef table and j-test
      ct = coeftable(me)
      # print coefficient table
      println(io, "Coefficients:\n")
      show(io, ct)
      # Then show extra information for this type
      println(io, show_extra(me))
    else
      println("The optimization did not converge")
    end
  end
end

function symbolfy{T<:MomentBasedEstimator}(me::T)
    s = split(string(T), "MomentBasedEstimators.")
    if (s[3] == "GMMEstimator{")
        Symbol(s[3], split(s[4], "{")[1], ", ", split(s[10], ",")[1], ", ", split(s[11], ",")[1], "}")
    elseif (s[3] == "MDEstimator{")
        if (s[4] == "AnaGradMomFun{" || s[4] == "AnaFullMomFun{")
          Symbol(s[3], split(s[4], "{")[1], ", ", split(s[7], "},Divergences.")[2], " ", split(s[8], ",")[1], "}")
        else
          Symbol(s[3], split(s[4], "{")[1], ", ", split(s[9], "},Divergences.")[2], " ", split(s[10], ",")[1], "}")
        end
    end
end
