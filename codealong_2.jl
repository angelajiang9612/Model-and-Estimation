####
function f(a, b)
    y = (a + 8b)^2
    return 7y
end
f(1, 2)
@code_native f(1, 2)

####
a = 3.0
@elapsed for i = 1:1000000
    a += i
end


function timetest()
    a = 3.0
    for i = 1:1000000
        a += i
    end
    a
end

@time a = timetest()



###Optimal Savings
using Parameters, Plots

#struct to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 50, stop= 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
end

mutable struct Results
    val_func::Array{Float64,1} #value function
    pol_func::Array{Float64,1} #policy function
end

#function to solve the model
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk), zeros(prim.nk)
    res = Results(val_func, pol_func)


    error, n = 100, 0
    while error>eps() #loop until convergence
        n+=1

        ######stuff to write here
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.val_func))
        res.val_func = v_next

        #println("Current error: ", error)
        if mod(n, 5000) == 0 || error <eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end
    prim, res
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid = prim #unpack primitive structure
    v_next = zeros(nk) #preallocate next guess of value function

    for i_k = 1:nk #loop over state space
        max_util = -1e10
        k = k_grid[i_k] #value of capital

        for i_kp = 1:nk ###loop over choices of k'
            budget = k^θ + (1-δ)*k #budget ##do this here instead of below so we are not caculating stuff again and again?
            kp = k_grid[i_kp] #value of k'
            c = budget - kp #consumption
            if c>0 #check if postiive
                #more stuff to write here
                val = log(c) + β * res.val_func[i_kp]
                if val>max_util #wow new max!
                    max_util = val
                    res.pol_func[i_k] = kp
                end
            end
        end
        v_next[i_k] = max_util #update value function
    end
    v_next
end





@elapsed prim, res = Solve_model() #solve the model.
Plots.plot(prim.k_grid, res.val_func) #plot value function
Plots.plot(prim.k_grid, res.pol_func) #plot value function


using Profile # to see why the code is slow

Profile.clear() ##make sure nothing is there
@profile prim, res = Solve_model()
Profile.print()
#####snapshots is a time unit


##my bottle neck stuff looks weird
