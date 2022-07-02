#=
Migration and the Family testing Julia file

Reads in programs, estimates models, and performs counterfactuals while reporting/exporting results
Author: Modified from GA
=#

#add more cores!

using Interpolations, Parameters, Distributions, DataFrames, CSV, LinearAlgebra, Statistics, Random, Profile, NLopt, SharedArrays

include("Amf_readin.jl") #just need to be in the same folder as this document
include("Amf_setup.jl")
include("Amf_valfunc.jl")



dir = "/Users/bubbles/Desktop/Research /Migration-family ties /Model and Synthetic Data/Synthetic Data/output"
cd(dir)

#read in model utilities, estimation sample, etc.
data = Readin_ind(dir)
df_20 = Readin_ind_20(dir)
locdata = Readin_loc(dir)


function Test_Value(guess::Array{Float64,1}, data::Array{Float64,2}, locdata::Array{Float64,2})
    prim, est, res = Initialize(guess) #initialize primitives, estimands, and results vectors
    Backwards(prim, est, res) #compute all value functions
end


initial=[0.00002,0.03,0.01,0.01,0.01,0.08,0.02,0.07,0.03,0.06,0.05,10000,0.1,0.5,100]


@elapsed Test_Value(initial,data,locdata)

vxj=res.vxj
E_ζ=res.E_ζ
pf_l=res.pf_l


lambda=probability_lambda()

ϵ_cutoffs=cutoffs()

l_j, l_o, ic=solve_forward()

v_dat,η_dat,ι_dat,μ_dat= misc()

l_o

data_sim=DataFrames.DataFrame(hcat(df_20,vec(l_o'),vec(l_j'),vec(ic'),v_dat,vec(η_dat'),ι_dat,μ_dat),:auto)   ##can also just add stuff at the end

CSV.write("generated.csv", data_sim)













##junk##

#how to use res?
##

function probability_lambda() ##function returns probability, the inputs are vxj and v_bar
    prim=Primitives()
    res=Results(vxj,E_ζ)
    @unpack n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n,γ =prim #dimensions
    @unpack lambda,vxj,E_ζ = res

    #lambda=zeros(n_n,n_ad,n_l,n_ξ,n_v,n_h,n_T,n_l)
    for i=1:n_n, a= 1:n_ad,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j= 1:n_j,
        res.lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-res.E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
end

probability_lambda()



##how to use the result from res?


    #testing function that lets us avoid holding huge matrices in memory (not sure what this means)
function Test_Value(guess::Array{Float64,1}, data::Array{Float64,2}, locdata::Array{Float64,2})
    prim, est, res = Initialize(guess) #initialize primitives, estimands, and results vectors
    res=Backwards(prim, est, res) #compute all value functions
    probability_lambda(prim,res)
    cutoffs(prim,est,res)
end



function probability(prim::Primitives,res::Results)
    prim=Primitives()
    res=Results()
    probability_lambda(prim,res)
end

@elapsed probability(initial,data,locdata)




res.lambda

function probability_lambda(prim::Primitives,res::Results) ##function returns probability, the inputs are vxj and v_bar
    @unpack n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n,γ =prim #dimensions
    @unpack lambda,vxj,E_ζ = res
    for i=1:n_n, a= 1:n_ad,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j= 1:n_j,
        res.lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
end

@elapsed Test_Value(initial,data,locdata) ##

@elapsed probability(initial,data,locdata) ##




res.ϵ_cutoffs

vxj=res.vxj
pf_l=res.pf_l

@elapsed probability_lambda(prim::Primitives,res::Results)















data_sim=DataFrames.DataFrame(hcat(data_20,vec(l_o'),vec(l_j'),vec(ic'),v_dat,vec(η_dat'),ι_dat,μ_dat),:auto)   ##can also just add stuff at the end

CSV.write("generated.csv", data_sim)
