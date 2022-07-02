###########Code to initialize model primitives and estimands and compute value functions and other useful things

#structure to house model primitives
@with_kw struct Primitives

    ##wage parameters estimated out of model
    β0::Int64=10000
    β1::Float64=0.5
    β2::Float64=-0.01
    β3::Int64=30000
    β4::Int64=0
    #discount variable
    β::Float64=0.95
    γ::Float64=Base.MathConstants.eulergamma
    #state grids

    v_grid::Array{Int64,1} = [-10000, 10000] ##
    η_grid::Array{Int64,1} =[-10000,10000]
    l_grid::Array{Int64,1} = [1,2,3] #location
    ξ_grid::Array{Float64,1} = [-0.1, 0.1]
    h_grid::Array{Int64,1} =[1,2,3]
    T_grid::Array{Int64,1} =[0,1]

    a_grid::Array{Int64,1} = collect(25:1:64) #age grid
    ad_grid::Array{Int64,1} = collect(25:1:44) #age grid for data

    #grid dimensions

    n_η::Int64 = length(η_grid) #number of fixed effects
    n_l::Int64 = length(l_grid)
    n_ξ::Int64 = length(ξ_grid)
    n_v::Int64 = length(v_grid)
    n_h::Int64 = length(h_grid)
    n_T::Int64 = length(T_grid)
    n_j::Int64 = length(l_grid)
    n_a::Int64 = length(a_grid)
    n_ad::Int64 = length(ad_grid)

    n_n::Int64=1000

    #div_chars::Array{Float64, 2} = zeros(9, 5)

    ##transition matrix
    markov_h::Array{Float64,2} =[0.9 0.09 0.01; 0.05 0.90 0.05; 0 0 1 ]
    markov_T::Array{Float64,2} =[0.9 0.1; 0.1 0.9]
end

#structure containing model estimands
@with_kw mutable struct Estimands
    guess::Vector{Float64} = zeros(16)

    α0::Float64 =0.00002
    α1::Float64 =0.03
    αp::Float64 =0.01
    γ0::Float64 =0.01
    γ1::Float64 =0.01
    θ1::Float64 =0.08
    θ2::Float64 =0.02
    θ3::Float64 =0.07
    θ4::Float64 =0.03
    θ5::Float64 =0.06
    cost::Float64 =0.05
    v_d::Float64 =10000 #the cutoff points
    η_d::Float64 =10000 #also normalised
    ξ_d::Float64 =0.1
    σ_ϵ::Float64 =0.5 #the preference shock
    σ_ι::Float64 =100 #the shock in wages, bigger than the shock in kappa preference
end

mutable struct Results
    vxj::Array{Float64,8}
    E_ζ::Array{Float64,7}
    E_ϵ::Array{Float64,7}
    pf_v::Array{Float64,7}
    pf_l::Array{Float64,7}
    lambda::Array{Float64,8}
    ϵ_cutoffs::Array{Float64,2}
end



#Initialize model primitives
function Initialize(guess::Array{Float64,1})
    prim = Primitives() #initialize model primitives
    est = Estimands(guess=guess) #could have different guesses, then guess is replaced by this one, changing here probably will not require restart julia

    #unpack primitives
    @unpack n_n,n_a,n_ad,n_l,n_ξ,n_v,n_h,n_T,n_l = prim

    vxj::Array{Float64, 8}=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T,n_l).-1000 ##this has n_l next in it too. In reality the i_h stuff doesn't affect v, do can probably drop them later
    E_ζ::Array{Float64, 7}=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T) #this is the previous v_bar
    E_ϵ::Array{Float64, 7}=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T)
    pf_v::Array{Float64, 7}=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T).-1000 ##value function in terms of all the state variables
    pf_l::Array{Float64, 7}=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T) ##location decision
    lambda::Array{Float64,8}=zeros(n_n,n_ad,n_l,n_ξ,n_v,n_h,n_T,n_l)
    ϵ_cutoffs::Array{Float64,2}=zeros(n_n,n_ad)

    res = Results(vxj,E_ζ,E_ϵ,pf_v,pf_l,lambda,ϵ_cutoffs) #initialize value funciton vectors
    prim, est, res #return deliverables
end

##why do we need both the results and the initialize?
