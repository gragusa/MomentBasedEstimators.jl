# --------------- #
# Display methods #
# --------------- #

function show_extra(me::GMMEstimator)
    j, p = J_test(me, me.e.mgr.k)
    "\nJ-test: $(round(j, 3)) (P-value: $(round(p, 3)))\n"
end

# default to nothing
show_extra(me::MomentBasedEstimator) = ""

function Base.writemime{T<:MomentBasedEstimator}(io::IO, ::MIME"text/plain", me::T)
    # get coef table and j-test
    ct = coeftable(me, me.e.mgr.k)

    # show info for our model
    println(io, "$(T): $(npar(me)) parameter(s) with $(nmom(me)) moment(s)")

    # Show extra information for this type
    println(io, show_extra(me))

    # print coefficient table
    println(io, "Coefficients:\n")

    # then show coeftable
    show(io, ct)
end
