# ----------------- #
# Utility functions #
# ----------------- #

"Tells maximum number of arguments for a generic or anonymous function"
function max_args(f::Function)
    if isgeneric(f)
        return methods(f).max_args
    else
        # anonymous function
        # NOTE: This might be quite fragile, but works on 0.3.6 and 0.4-dev
        return length(Base.uncompressed_ast(f.code).args[1])
    end
end

"""
Inplace version of `row_kron`. Sets the `i`th row of `out` equal to
`kron(A[i, :], B[i, :])`
"""
function row_kron!(A::AbstractMatrix, B::AbstractMatrix, out::AbstractMatrix)
    # get input dimensions
    nobsa, na = size(A)
    nobsb, nb = size(B)

    nobsa != nobsb && error("A and B must have same number of rows")

    # fill in each element. To do this we make sure we access each array
    # consistent with its column major memory layout.
    @inbounds for ia=1:na, ib=1:nb, t=1:nobsa
        out[t, nb*(ia-1) + ib] = A[t, ia] * B[t, ib]
    end
    out
end

"""
Compute the row-wise Kronecker product between two matrices `A` and `B`.
The matrices are assumed to be of size `(r, ca)` and `(r, cb)`,
respectively. The output matrix will be `(r, ca*cb)`

The `i`th row of the output matrix will be equal to
`kron(A[i,:], B[i,:])`
"""
:row_kron

function row_kron{S,T}(A::AbstractMatrix{S}, B::AbstractMatrix{T})
    nobsa, na = size(A)
    nobsb, nb = size(B)
    out = Array(promote_type(S, T), nobsa, na*nb)
    row_kron!(A, B, out)
    out
end

# specialized version for sparse matrices
function row_kron{S,T}(A::SparseMatrixCSC{S}, B::SparseMatrixCSC{T})
    nobsa, na = size(A)
    nobsb, nb = size(B)
    out = spzeros(promote_type(S, T), nobsa, na*nb)
    row_kron!(A, B, out)
    out
end

function gettril(x::Array{Float64, 2})
  n, m = size(x)
  a = Array(eltype(x), convert(Int, n.*(n+1)/2))
  k::Int = 1
  @inbounds for i = 1:n
    for j = 1:i
      a[k] = x[i, j]
      k += 1
    end
  end
  return a
end
