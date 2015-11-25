# --------------- #
# Display methods #
# --------------- #

function show_extra{T <: GMMEstimator}(me::MomentBasedEstimator{T})
    j, p = J_test(me, me.e.mgr.k)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

show_extra{T <: MDEstimator}(me::MomentBasedEstimator{T}) = ""


function Base.writemime{T<:MomentBasedEstimator}(io::IO, ::MIME"text/plain", me::T)
    if me.status == :Solved
        s = symbolfy(me)
        println(io, s[1],"{",s[2],"}: $(npar(me)) parameter(s) with $(nmom(me)) moment(s)")
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
    elseif me.status == :Initialized
        s = symbolfy(me)
        println(io, s[1],"{",s[2],"}: $(npar(me)) parameter(s) with $(nmom(me)) moment(s)")
    else
        println("Uninitialized MomentBasedEstimator type")
    end
end

function symbolfy{T<:MomentBasedEstimator}(me::T)
    s = split(string(T), "MomentBasedEstimators.")
    if (s[3] == "GMMEstimator{")
        s = (split(s[3],"{")[1],
             split(s[5], ",")[1],
             split(s[6], ",")[1],
             split(s[7], "}}")[1])
    elseif (s[3] == "MDEstimator{")
        s = (split(s[3],"{")[1],
             split(split(s[4], ",Divergences.")[2],",")[1],
             split(s[5], ",")[1],
             split(s[6], "}}")[1])
    end
    map((x) -> symbol(x), s)
end
