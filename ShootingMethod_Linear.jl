""" Solution of a boundary value problem for a linear ODE
    y'' = -(2/x)y'+(2/x^2)y+ sin(ln(x))/x^2, y(1) = 1, y(2) = 2
    by shooting method.
"""


# Importing Packages
import Pkg
Pkg.import("Plots")
Pkg.import("LaTeXStrings")
Pkg.import("LinearAlgebra")
Pkg.import("DifferentialEquations")


# Loading Packages
using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using Plots

# Interval of integration and Nodes
a = 1          
b = 2          
N = 10;  

# Step size
h = (b-a)/N   


# System of IVPs for shooting method (For further details, see the book Numerical Anaysis
# by Burden and Faires, 9th Edition, Chapter 11, Section 11.1, page 672)

function System_IVP!(du,u,p,t)
    du[1] = u[2]
    du[2] = -2/t*(u[2])+2*u[1]/t^2+sin(log(t))/t^2
    du[3] = u[4]
    du[4] = -2/t*(u[4])+2*u[3]/t^2
end

u0 = [1.0;0.0;0.0;1.0]
tspan = (1.0,2.0)
prob = ODEProblem(System_IVP!,u0,tspan)
sol = solve(prob,saveat=h)

# Renaming the ODEs solution vectors

y1 = sol[1,:]
y2 = sol[2,:]
y3 = sol[3,:]
y4 = sol[4,:]

y = zeros(N+1)

for i in 1:N+1
    y[i]=y1[i]+((b-y1[N+1])/y3[N+1])*y3[i]
end



# Exact Solution 
x = collect(a:h:b)
c2 = 1/70*(8-12*sin(log(2))-4*cos(log(2)))
c1 = 11/10-c2
y_exact = c1*x .+ c2./x.^2 -(3/10).*sin.(log.(x))- (1/10).*cos.(log.(x))


# Infinity norm of y-y_exact
error = norm(y-y_exact,Inf)

plot(sol.t,y,linewidth=2.0,xlabel=L"x",ylabel=L"y(x)",label="Numerical Solution", legend=:right)
scatter!(x,y_exact,linewidth=2.0,xlabel=L"x",ylabel=L"y(x)",label="Exact Solution", legend=:right)



