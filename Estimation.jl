using DataFrames
using CSV
using Statistics
using DataStructures
using Distributions
using DelimitedFiles
using Random
using Pkg
using Optim

##the initial parameters

cd("/Users/bubbles/Desktop/Research /Migration-family ties /Model/Synthetic Data/Data")
loc_data=CSV.File("locdata.csv")
dfloc = DataFrames.DataFrame(loc_data)
locdata=Matrix(dfloc)

data=CSV.File("mydata.csv") #the original data with more periods
df = DataFrames.DataFrame(data) ##must define Dataframe.DataFrame or does not work
data=Matrix(df)


data_m=CSV.File("regenerated.csv") #the simulated data
df_m = DataFrames.DataFrame(data_m) ##must define Dataframe.DataFrame or does not work
data_m=Matrix(df_m)

i_m=Integer.(data_m[:,1])
t_m=Integer.(data_m[:,2])
age_m=Integer.(data_m[:,3])
g_m=Integer.(data_m[:,4])
school_m=Integer.(data_m[:,5])
sib_m=Integer.(data_m[:,6])
lp_m=Integer.(data_m[:,7])
loc_in_m=Integer.(data_m[:,8])
low_m=Integer.(data_m[:,9])
ph_m=Integer.(data_m[:,10])
G_m=data_m[:,11]
T_m=Integer.(data_m[:,12])
loc_m=Integer.(data_m[:,13])
j_m=Integer.(data_m[:,14])
ic_m=Integer.(data_m[:,15])
v_m=data_m[:,16]
eta_m=data_m[:,17]
iota_m=data_m[:,18]
mu_m=data_m[:,19]
ndistinct_m=Integer.(data_m[:,20])
k_m=Integer.(data_m[:,21])
w_m=data_m[:,22]


id_d=Integer.(data[:,1])
t_d=Integer.(data[:,2])
age_d=Integer.(data[:,3])
g_d=Integer.(data[:,4])
school_d=Integer.(data[:,5])
sib_d=Integer.(data[:,6])
lp_d=Integer.(data[:,7])
loc_in_d=Integer.(data[:,8])
low_d=Integer.(data[:,9])
ph_d=Integer.(data[:,10])
G_d=data[:,11]
T_d=Integer.(data[:,12])



##parameters estimated out of model

β0=10000
β1=0.5
β2=-0.01
β3=30000
β4=0

locations=Integer.(locdata[:,1])
μ=locdata[:,2] #mean wages
c=locdata[:,3] #costs
N=locdata[:,5] #population
hc=locdata[:,4] #care cost


##Parameters that do not change
n_n=1000
n_a=40
n_data=20

n_η=2
n_l=3
n_ξ=2
n_v=2
n_h=3
n_T=2
β=0.95

γ=Base.MathConstants.eulergamma #euler's constant


##

α0_in=0.00002
α1_in=0.03
αp_in=0.01
γ0_in=0.01
γ1_in=0.01
θ1_in=0.08
θ2_in=0.02
θ3_in=0.07
θ4_in=0.03
θ5_in=0.06

σ_ϵ_in=0.5 #the preference shock
v_d_in=10000 #the cutoff points
ξ_d_in=0.1
η_d_in=10000 #also normalised
σ_ι_in=100 #the shock in wages, bigger than the shock in kappa preference
cost_in=0.05


#grids


markov=[0.9 0.09 0.01; 0.05 0.90 0.05; 0 0 1 ] #healthy, sick and dead transition matrix
h_grid=[1,2,3]
T_grid=[0,1]
v_grid =[-v_d_in v_d_in]
ξ_grid =[-ξ_d_in ξ_d_in]
η_grid =[-η_d_in η_d_in]



##the shocks that can be drawn outside, does not depend on the parameter estimates

Random.seed!(1234);

v=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #draw from binary distribution
v_index=copy(v) ##preparing to create an index of shocks
v_index[v_index.==1].=2 #this is the grid conversion for the shock
v_index[v_index.==0].=1

η=convert(Array{Float64,1},rand(Binomial(1,0.5), n_n))

ξ=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l))
ξ_index=copy(ξ) #need to use copy not equal
ξ_index[ξ_index.==1].=2 #this is the grid conversion for the shock
ξ_index[ξ_index.==0].=1
ζ= reshape(rand(Gumbel(0, 1), n_n*n_l*n_a),n_n,n_l,n_a) #this type one distribution thing should stay the same, j is ag n_l is location


function Backwards(α0,α1,αp,γ0,γ1,θ1,θ2,θ3,θ4,θ5,σ_ϵ,v_d,ξ_d,η_d,σ_ι,cost,v_grid,ξ_grid,η_grid,η,ι,ϵ,ζ)
    vxj=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T,n_l).-1000 ##this has n_l next in it too. In reality the i_h stuff doesn't affect v, do can probably drop them later
    E_ζ=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T) #this is the previous v_bar
    E_ϵ=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T)
    pf_v=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T).-1000 ##value function in terms of all the state variables
    pf_l=zeros(n_n,n_a,n_l,n_ξ,n_v,n_h,n_T) ##location decision
    for i=1:n_n
        println("this is", i, "person")
        #use the person's actual information, these variables do not change over time
        g=g_d[(i-1)*n_a+1]
        sib=sib_d[(i-1)*n_a+1]
        lp=lp_d[(i-1)*n_a+1]
        low=low_d[(i-1)*n_a+1]

        for a = n_a:-1:1 #this is number of periods
            ##these variables change over time(age)
            age = age_d[(i-1)*n_a+a]
            G=G_d[(i-1)*n_a+a] #the wage relationship estimated outside, doesn't depend on location
            Δ=γ0 + γ1*age #right now the moving cost is not location specific
            if a == n_a #no location decision in this period, so no location related stuff this period
                for i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v, i_h = 1:n_h, i_T = 1:n_T #for any current state variables
                    wages= μ[i_l] + v_grid[i_v]+ G + η[i] + ι[i,i_l,a]
                    kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T_grid[i_T] - cost
                    u_l0= α0*wages/c[i_l] + α1*N[i_l] + αp*(i_l==lp) + (h_grid[i_h]==2)*kappa + ξ_grid[i_ξ]
                    if i_l != lp || (i_l == lp && i_h!=2)
                        E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 #########got to here
                    else
                        E_ϵ[i,a,i_l,i_ξ,i_v,i_h, i_T]=u_l0 + σ_ϵ*(cdf.(Normal(), kappa/σ_ϵ)*kappa/σ_ϵ + pdf.(Normal(), kappa/σ_ϵ))
                    end
                end
            elseif a < n_a #in other periods do the migration decision first (drom the back)
                for i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v, i_h = 1:n_h, i_T = 1:n_T
                    for i_j=1:n_l #next
                        if i_l == i_j #no location match shocks
                             #remember to use next period's things, a+1 and next period's shocks (in this case same as this period)
                            e_next=markov[i_h,1]*markov[i_T,1]*E_ϵ[i,a+1,i_j,i_ξ,i_v,1,1] + markov[i_h,1]*markov[i_T,2]*E_ϵ[i,a+1,i_j,i_ξ,i_v,1,2] + markov[i_h,2]*markov[i_T,1]*E_ϵ[i,a+1,i_j,i_ξ,i_v,2,1]
                            + markov[i_h,2]*markov[i_T,2]*E_ϵ[i,a+1,i_j,i_ξ,i_v,2,2] +  markov[i_h,3]*markov[i_T,1]*E_ϵ[i,a+1,i_j,i_ξ,i_v,3,1]
                            + markov[i_h,3]*markov[i_T,2]*E_ϵ[i,a+1,i_j,i_ξ,i_v,3,2] #this takes into consideration those expected values may all be the same in the case where i_j is not equal to lp

                        elseif i_j != i_l
                            e_next=0
                            E_ϵ_next=zeros(n_v,n_ξ,n_h,n_T) # a place to hold all the possibilities
                            for i_ξ_j = 1:n_ξ, i_v_j = 1:n_v, i_h_next = 1:n_h, i_T_next = 1:n_T #summing over all x'
                                E_ϵ_next[i_ξ_j,i_v_j,i_h_next,i_T_next] = E_ϵ[i,a+1,i_j,i_ξ_j,i_v_j,i_h_next,i_T_next] #pick out the expectation to use, again use all next period's shocks
                                e_next = e_next+ E_ϵ_next[i_ξ_j,i_v_j,i_h_next, i_T_next]*markov[i_h,i_h_next]*markov[i_T,i_T_next]*(1/n_v)*(1/n_ξ)
                            end
                        end
                        vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]= - Δ*(i_l!=i_j) + β*e_next
                        V_temp =vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]+ ζ[i,i_j,a]
                        if V_temp > pf_v[i,a,i_l,i_ξ,i_v,i_h,i_T]
                            pf_v[i,a,i_l,i_ξ,i_v,i_h,i_T]= V_temp
                            pf_l[i,a,i_l,i_ξ,i_v,i_h,i_T]= i_j #making the choice
                        end
                    end
                    constant=maximum(vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,:]) #defined this way so exp term don't blow up
                    E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T] = γ + constant + log(sum(exp.(vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,:].-constant)))
                    #now for the the new E_ϵ_next
                    wages= μ[i_l] + v_grid[i_v]+ G + η[i] + ι[i,i_l,a]
                    kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T_grid[i_T]- cost
                    u_l0= α0*wages/c[i_l] + α1*N[i_l] + αp*(i_l==lp) + h_grid[i_h]*kappa + ξ_grid[i_ξ]
                    if i_l != lp || (i_l == lp && i_h!=2)
                        E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 + E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T] #use this period's E_ζ state variable to compute E_ϵ
                    elseif i_l == lp && i_h==2
                        E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 + E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T]  + σ_ϵ*(cdf.(Normal(), kappa/σ_ϵ)*kappa/σ_ϵ + pdf.(Normal(), kappa/σ_ϵ))
                    end
                end
            end
        end
    end
    return vxj, E_ζ, E_ϵ, pf_v, pf_l
end



function probability_lambda(vxj,E_ζ) ##function returns probability, the inputs are vxj and E_ζ
    lambda=zeros(n_n,n_data,n_l,n_ξ,n_v,n_h,n_T,n_l)
    for i=1:n_n, a= 1:n_data,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j=1:n_l
        lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
    return lambda
end


all_perm(xs, n) = vec(map(collect, Iterators.product(ntuple(_ -> xs, n)...)))



function llikelihood(x)
    α0=x[1]
    α1=x[2]
    αp=x[3]
    γ0=x[4]
    γ1=x[5]
    θ1=x[6]
    θ2=x[7]
    θ3=x[8]
    θ4=x[9]
    θ5=x[10]

    σ_ϵ=x[11]
    v_d=x[12]
    ξ_d=x[13]
    η_d=x[14]
    σ_ι=x[15]
    cost=x[16]

    Random.seed!(1234);

    v[v.==1].=v_d #replace with the cutoffs
    v[v.==0].=-v_d

    ξ[ξ.==1].=ξ_d #replace with the cutoffs
    ξ[ξ.==0].=-ξ_d

    η[η.==1].=η_d #replace with the cutoffs
    η[η.==0].=-η_d
    ι= reshape(rand(Normal(0,σ_ι), n_n*n_l*n_a),n_n,n_l,n_a) #the wage shocks, this has a age dimension
    ϵ= reshape(rand(Normal(0,σ_ι), n_n*n_a),n_n,n_a) #preference shock probably no need to differ by location

    v_grid =[-v_d v_d]
    ξ_grid =[-ξ_d ξ_d]
    η_grid =[-η_d η_d]

    vxj, E_ζ, E_ϵ, pf_v, pf_l= Backwards(α0,α1,αp,γ0,γ1,θ1,θ2,θ3,θ4,θ5,σ_ϵ,v_d,ξ_d,η_d,σ_ι,cost,v_grid,ξ_grid,η_grid,η,ι,ϵ,ζ)
    lambda=probability_lambda(vxj,E_ζ)

    log_L=0
    log_Li=zeros(n_n)

    for i=1:n_n
        N_total=Integer(ndistinct_m[(i-1)*n_data+1])
        g=Integer(g_m[(i-1)*n_data+1])
        school=Integer(school_m[(i-1)*n_data+1])
        sib=Integer(sib_m[(i-1)*n_data+1])
        lp=Integer(lp_m[(i-1)*n_data+1])
        low=Integer(low_m[(i-1)*n_data+1])

        ω_eta=zeros(1) ##a place for omega eta
        ω_v=zeros(N_total) ##indice for v
        ω_ξ=zeros(N_total) ##indice for xi
        IT = n_η*(n_ξ^N_total)*(n_v^N_total) #the total number of different permutations
        indice=all_perm([1, 2], 2*N_total +1) #generate all possible distinct permutations with number of digits 2*N +1 and digits 1 or 2

        Liω=ones(IT)
        Li_sum=0
        for i_it = 1:IT #for each iteration
            index=indice[i_it] #take out the relevant indice
            ω_eta=index[1]
            ω_v=index[2:N_total+1]
            ω_xi=index[N_total+2:2*N_total+1]
            for a=1:n_data #now for things that change through time, no next period info in last period
                ph=Integer(ph_m[(i-1)*n_data+a])
                G=G_m[(i-1)*n_data+a]
                T=Integer(T_m[(i-1)*n_data+a]) +1
                loc=Integer(loc_m[(i-1)*n_data+a])
                j=Integer(j_m[(i-1)*n_data+a])
                κ=Integer(k_m[(i-1)*n_data+a])
                w=w_m[(i-1)*n_data+a]
                ic=Integer(ic_m[(i-1)*n_data+a])

                kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T - cost

                ωv=ω_v[κ] ##this period's shock indices, for each person, k should be less than or equal to N total, this is 1 or 2
                ωxi=ω_xi[κ]
                ψ=(1/σ_ι)pdf(Normal(0,σ_ι), (w-μ[loc]-G- v_grid[ωv]-η_grid[ω_eta])/σ_ι) #the problem is psi is always zero
                #println("this is", ψ, "psi")
                if a == n_data
                    λ=1 #last period the care decision cannot be observed
                else
                    λ=lambda[i,a,loc,ωxi,ωv,ph,T,j]
                    println("this is", λ, "lambda")
                end

                if loc != lp || (loc == lp && ph!=2) #no care decision, probability is 1
                    ρ=1
                  # println("this is", ρ, "rho")
                elseif loc == lp && ph==2
                    if ic==1
                        ρ=cdf.(Normal(), kappa/σ_ϵ)
                        #println("this is", ρ, "rho")
                    else
                        ρ=(1-cdf.(Normal(), kappa/σ_ϵ))
                        #println("this is", ρ, "rho")
                    end
                end
                Liω[i_it]=Liω[i_it]*ψ*λ*ρ #this is the likelihood of observation for one set of history for one iteration of shocks
            end
        end
        Li_sum=sum(Liω)
        #println("this is", Li_sum, "for" ,i, "person")
        log_Li[i]=log(Li_sum*(1/(n_η*(n_ξ^N_total)*(n_v^N_total)))) #the Li_sum here is for all omega #this is before the end
    end
    log_L=sum(log_Li)
    return -log_L
end


x1=x0.+0.2 ##start from not the input

res0=llikelihood(x0)
res1=llikelihood(x1)


x0=[α0_in,α1_in,αp_in,γ0_in,γ1_in,θ1_in,θ2_in,θ3_in,θ4_in,θ5_in,σ_ϵ_in,v_d_in,ξ_d_in,η_d_in,σ_ι_in,cost_in]


x1=[0.00001,0.02,0.015,0.02,0.05,0.06,0.01,0.05,0.02,0.05,0.3,8000,0.2,11000,90,0.02]

#@elapsed testres=llikelihood(x0)
#@show testres

result=optimize(llikelihood, x1, g_tol =10)

minimum=Optim.minimizer(result)
print(minimum)
writedlm("mini.txt",minimum)
#writedlm("res0.txt",res0)
#writedlm("res1.txt",res1)
