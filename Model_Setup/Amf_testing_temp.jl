#=
Migration and the Family testing Julia file

Reads in programs, estimates models, and performs counterfactuals while reporting/exporting results
Author: GA
=#

#add more cores!

using Pkg
Pkg.add("Interpolations")
Pkg.add("Parameters")
Pkg.add("Distributions")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("Profile")
Pkg.add("NLopt")



using Interpolations, Parameters, Distributions, DataFrames, CSV, LinearAlgebra, Statistics, Random, Profile, NLopt, SharedArrays

include("mf_setup.jl")
include("mf_valfunc.jl")
include("mf_estimation.jl")
include("mf_simulation.jl")
include("mf_readin.jl")

dir = "/Users/bubbles/Desktop/Research /Migration-family ties /Model and Synthetic Data/Model_Setup"
cd(dir)

#read in model utilities, estimation sample, etc.
package = Readin(dir)
estim_data, simul_data, trans_probs, div_chars = package[1], package[2], package[3], package[4]

#testing function that lets us avoid holding huge matrices in memory (not sure what this means)
function Test_simul(guess::Array{Float64,1}, estim_data::Array{Float64,2}, simul_data::Array{Float64,2}, trans_probs::Array{Float64,2}, div_chars::Array{Float64, 2})
    prim, est, res = Initialize(guess, trans_probs, div_chars) #initialize primitives, estimands, and results vectors
    Backward_Induct(prim, est, res) #compute all value functions
    likelihood, lhoods = Likelihood(prim, est, res, estim_data)
    simul_output = Simulate(prim, est, res, simul_data, lhoods)

    #garbage collection
    finalize(res.emax_1)
    finalize(res.cutoff_1)
    finalize(res.emax_2)
    finalize(res.cutoff_2)
    finalize(res.emax_3)
    finalize(res.cutoff_3)
    GC.gc()

    #likelihood, lhoods
    likelihood, simul_output, lhoods
end

guess_init = [0.11955, 1.35589, -0.13578, -0.38116, 0.0876, 0.74239, -0.39751, 0.00241, 0.55468, .2047, 0.99425, 0.45154, 1.97082, 0.46376, 0.05816, -0.00155, 0.27499, 0.35161, 3.7756, 0.12475, 0.29067, 0.47421, 0.01641, -0.0057, 0.71562, -0.08991, -0.0488, -0.00721, 0.07889, 0.19231]
obj, simul_output, lhoods = Test_simul(guess_init, estim_data, simul_data, trans_probs, div_chars)
