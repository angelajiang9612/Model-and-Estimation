#=
Migration and the Family testing Julia file

Reads in programs, estimates models, and performs counterfactuals while reporting/exporting results
Author: Modified from GA
=#

#add more cores!

using Distributed
addprocs(3)

@everywhere using Interpolations,Parameters,Distributions,DataFrames,CSV,LinearAlgebra,Statistics,Random,Profile,NLopt,SharedArrays,Optim


@everywhere include("Amf_readin.jl") #just need to be in the same folder as this document
@everywhere include("Amf_setup.jl")
@everywhere include("Amf_funcs_estimate.jl")

@everywhere dir = "/Users/bubbles/Desktop/Research /Migration-family ties /Model and Synthetic Data/Synthetic Data/output"
@everywhere cd(dir)

#read in model utilities, estimation sample, etc.
@everywhere data = Readin_ind(dir)
@everywhere df_20 = Readin_ind_20(dir)
@everywhere locdata = Readin_loc(dir)
@everywhere data_m =Readin_data_m(dir)

##how can we have an objective function that is in terms of more than just the parameters
#objective function to minimize
@everywhere function Objective_function(g::Array{Float64,1})
    obj = 0.0 #preallocation of objective function value
    println("Current Guess")
    println(round.(g, digits = 7)) ##print the guess g

    prim, est, res = Initialize(g) #initialize primitives, estimands, and results vectors
    Backwards(prim, est, res) #compute all value functions
    likelihood = Likelihood(prim, est, res, data_m, locdata)
    obj = -1*likelihood

    #garbage collection
    finalize(res.vxj)
    finalize(res.E_ζ)
    finalize(res.E_ϵ)
    @everywhere GC.gc()

    #report how we're doing
    println("")
    #println("Current Guess")
    #println(round.(g, digits = 4))
    println("Current error: ", round(obj, digits = 4))
    println("")
    obj #return deliverable
end




initial=[0.00002,0.03,0.01,0.01,0.01,0.08,0.02,0.07,0.03,0.06,0.05,10000,10000,0.1,0.5,100]

x1=[0.000025,0.035,0.015,0.015,0.015,0.085,0.025,0.075,0.035,0.065,0.055,11000,11000,0.15,0.55,110]


@elapsed obj=Objective_function(initial)
#some guess

obj2=Objective_function(x1)

obj

result = Optim.optimize(Objective_function, x1, autodiff = :forward, g_tol = 0.000001)


mini = Optim.minimizer(result)
print(mini)
writedlm("mini.txt", mini)







function Test_Value2(
    guess::Array{Float64,1},
    data::Array{Float64,2},
    locdata::Array{Float64,2},
)
    prim, est, res = Initialize(guess) #initialize primitives, estimands, and results vectors
    Backwards(prim, est, res) #compute all value functions
end


@elapsed Test_Value2(initial, data, locdata)


function Test_Value(
    guess::Array{Float64,1},
    data::Array{Float64,2},
    locdata::Array{Float64,2},
)
    prim, est, res = Initialize(guess) #initialize primitives, estimands, and results vectors
    Backwards(prim, est, res) #compute all value functions
    probability = probability_lambda2(prim, res)
    return probability
end
