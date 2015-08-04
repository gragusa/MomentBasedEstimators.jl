## Dual based jacobian wich allows arguments

function args_dual_fad{T<:Real}(f!::Function, x::Vector{T}, jac_out::Matrix{T}, dual_in, dual_out, args...)
  for i in 1:length(x)
    dual_in[i] = DualNumbers.Dual(x[i], zero(T))
  end
  for i in 1:length(x)
    dual_in[i] = DualNumbers.Dual(x[i], one(T))
    f!(dual_in, dual_out, args...)
    for k in 1:length(dual_out)
      jac_out[k, i] = epsilon(dual_out[k])
    end
    dual_in[i] = DualNumbers.Dual(real(dual_in[i]), zero(T))
  end
end

function args_dual_fad_jacobian!{T<:Real}(f!::Function, ::Type{T}; n::Int=1, m::Int=1)
  dual_in = Array(DualNumbers.Dual{T}, n)
  dual_out = Array(DualNumbers.Dual{T}, m)
  g!(x, jac_out, args...) = args_dual_fad(f!, x, jac_out, dual_in, dual_out, args...)
  return g!
end

function args_dual_fad_jacobian{T<:Real}(f!::Function, ::Type{T}; n::Int=1, m::Int=1)
  dual_in = Array(DualNumbers.Dual{T}, n)
  dual_out = Array(DualNumbers.Dual{T}, m)
  jac_out = Array(T, m, n)
  function g(x, args...)
    args_dual_fad(f!, x, jac_out, dual_in, dual_out, args...)
    jac_out
  end
  return g
end

############################################################
## Typed
############################################################
function args_typed_fad_gradient{T<:Real}(f::Function, ::Type{T})
  g(x::Vector{T}, args...) = grad(f(ForwardDiff.GraDual(x), args...))
  return g
end


function args_typed_fad_hessian{T<:Real}(f::Function, ::Type{T})
  g(x::Vector{T}, args...) = ForwardDiff.hessian(f(ForwardDiff.FADHessian(x), args...))
  return g
end

############################################################
## Finite difference
############################################################

##############################################################################
##
## Utility macro
##
##############################################################################

macro forwardrule(x, e)
    x, e = esc(x), esc(e)
    quote
        $e = sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end

macro centralrule(x, e)
    x, e = esc(x), esc(e)
    quote
        $e = cbrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end

macro hessianrule(x, e)
    x, e = esc(x), esc(e)
    quote
        $e = eps(eltype($x))^(1/4) * max(one(eltype($x)), abs($x))
    end
end

macro complexrule(x, e)
    x, e = esc(x), esc(e)
    quote
        $e = eps($x)
    end
end

##############################################################################
##
## Jacobian derivative of f: R^n -> R^m
##
##############################################################################

function fd_jacobian!{R <: Number, S <: Number, T <: Number}(f::Function,
                                                             x::Vector{R},
                                                             f_x::Vector{S},
                                                             J::Array{T},
                                                             dtype::Symbol = :central,
                                                             args...)
    # What is the dimension of x?
    m, n = size(J)

    # Iterate over each dimension of the gradient separately.
    if dtype == :forward
        for i = 1:n
            @forwardrule x[i] epsilon
            oldx = x[i]
            x[i] = oldx + epsilon
            f_xplusdx = f(x, args...)
            x[i] = oldx
            J[:, i] = (f_xplusdx - f_x) / epsilon
        end
    elseif dtype == :central
        for i = 1:n
            @centralrule x[i] epsilon
            oldx = x[i]
            x[i] = oldx + epsilon
            f_xplusdx = f(x, args...)
            x[i] = oldx - epsilon
            f_xminusdx = f(x, args...)
            x[i] = oldx
            J[:, i] = (f_xplusdx - f_xminusdx) / (epsilon + epsilon)
        end
    else
        error("dtype must :forward or :central")
    end

    return
end
function fd_jacobian{T <: Number}(f::Function,
                                  x::Vector{T},
                                  dtype::Symbol = :central, args...)
    # Establish a baseline for f_x
    f_x = f(x, args...)

    # Allocate space for the Jacobian matrix
    J = zeros(length(f_x), length(x))

    # Compute Jacobian inside allocated matrix
    fd_jacobian!(f, x, f_x, J, dtype, args...)

    # Return Jacobian
    return J
end


##############################################################################
##
## Hessian of f: R^n -> R
##
##############################################################################

function fd_hessian!{S <: Number,
                     T <: Number}(f::Function,
                                  x::Vector{S},
                                  H::Array{T}, args...)
    # What is the dimension of x?
    n = length(x)

    epsilon = NaN
    # TODO: Remove all these copies
    xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
    fx = f(x, args...)
    for i = 1:n
        xi = x[i]
        @hessianrule x[i] epsilon
        xpp[i], xmm[i] = xi + epsilon, xi - epsilon
        H[i, i] = (f(xpp, args...) - 2*fx + f(xmm, args...)) / epsilon^2
        @centralrule x[i] epsiloni
        xp = xi + epsiloni
        xm = xi - epsiloni
        xpp[i], xpm[i], xmp[i], xmm[i] = xp, xp, xm, xm
        for j = i+1:n
            xj = x[j]
            @centralrule x[j] epsilonj
            xp = xj + epsilonj
            xm = xj - epsilonj
            xpp[j], xpm[j], xmp[j], xmm[j] = xp, xm, xp, xm
            H[i, j] = (f(xpp, args...) - f(xpm, args...) - f(xmp, args...) + f(xmm, args...))/(4*epsiloni*epsilonj)
            xpp[j], xpm[j], xmp[j], xmm[j] = xj, xj, xj, xj
        end
        xpp[i], xpm[i], xmp[i], xmm[i] = xi, xi, xi, xi
    end
    Base.LinAlg.copytri!(H,'U')
end

function fd_hessian{T <: Number}(f::Function,
                                 x::Vector{T}, args...)
    # What is the dimension of x?
    n = length(x)

    # Allocate an empty Hessian
    H = Array(Float64, n, n)

    # Mutate the allocated Hessian
    fd_hessian!(f, x, H, args...)

    # Return the Hessian
    return H
end



## Test case:
## Notice that the inplace assignment is on the second argument
## function f!(x, y, z)
##   y[1] = z[1]*(x[1]^2+x[2])
##   y[2] = z[1]*(3*x[1])
##   y[3] = z[1]*(x[1]^2*x[2]^3)
## end

## y = zeros(3);
## f!([2.1, 1.5],y, [1]);

## j = args_dual_fad_jacobian(f!, Float64, n = 2, m = 3)
## j([2.1, 1.5], [0.2])

## function f(x, z)
##   [z[1]*(x[1]^2+x[2]), z[1]*(3*x[1]), z[1]*(x[1]^2*x[2]^3)]
## end


## function h(x)
##     x[1]^3
## end 

## function ah(x, a)
##     x[1]^a
## end 


## Dh = ForwardDiff.typed_fad_jacobian(h, Float64)
## Dh([2.1,2.1])

## Dh = ForwardDiff.typed_fad_hessian(h, Float64)
## Dh([3.,])

## Dh = args_typed_fad_hessian(ah, Float64)
## Dh([3.0], 3.)


## Dh = args_typed_fad_gradient(f, Float64)
## Dh([2.1,2.1], .1)



