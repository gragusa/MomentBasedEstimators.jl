using ModelsGenerators
srand(1)
y,x,z=randiv(1000)
xb = randn(1000,2)
x = [x xb]
z = [z xb]

f(theta) = z.*(y-x*theta);

function f_p(theta)
    n, m = size(z)
    v = Array(Float64, n, m)
    u = (y-x*theta)    
    @inbounds for k = 1:m
        for j = 1:n
            v[j,k] = z[j,k]*u[j]
        end 
    end
    v
end


function f_p!(v, theta)
    n, m = size(z)    
    u = (y-x*theta)    
    @inbounds for k = 1:m
        for j = 1:n
            v[j,k] = z[j,k]*u[j]
        end 
    end
    v
end

n, m = size(z)
v = Array(Float64, n, m)

f([.1, 0, 0])
f_p([.1, 0, 0])
f_p!(v, [.1, 0, 0])
@time f([.1, 0, 0])
@time f_p([.1, 0, 0])
@time f_p!(v, [.1, 0, 0])


p = ones(10000);
l = ones(7,1);

ws() = p'v*l

@time ws()

using Base.LinAlg.BLAS




using FastAnonymous


af = @anon theta->z.*(y-x*theta)

af([.1,0,0])


type dummy
    a::DataType
end

uu = dummy(af)


type dummy_
    a::Function
end 

uuu = dummy_(f)

fdummy() = for j = 1:1000 f([.1,0,0]) end
fdummy_2() = for j = 1:1000 uu.a([.1,0,0]) end
fdummy_() = for j = 1:1000 uuu.a([.1,0,0]) end
@time fdummy()
@time fdummy_2()
@time fdummy_()
