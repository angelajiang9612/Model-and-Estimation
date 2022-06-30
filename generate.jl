using DataFrames
using CSV
using Statistics
using DataStructures
using Distributions
using DelimitedFiles
using Random

##############################

#this script solves the model using exogenous variables from the data, generates the endogenous variables, stores export the generated data in a dataframe for estimation

#backwards() solves the model , solve_forward() solves the model, the rest stores and exports data.

#this is done for one set of parameters and one realisation of the shocks.

#############################

##data

cd("/Users/bubbles/Desktop/Research /Migration-family ties /Model/Synthetic Data")

data=CSV.File("mydata.csv")
df = DataFrames.DataFrame(data) ##must define Dataframe.DataFrame or does not work
data=Matrix(df)

loc_data=CSV.File("locdata.csv")
dfloc = DataFrames.DataFrame(loc_data)
locdata=Matrix(dfloc)

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
μ=locdata[:,2] #mean wages normalise
c=locdata[:,3] #costs
N=locdata[:,5] #population
hc=locdata[:,4] #care cost


#grids

markov=[0.9 0.09 0.01; 0.05 0.90 0.05; 0 0 1 ] #healthy, sick and dead transition matrix
v_grid =[0.5 -0.5]
ξ_grid =[0.5 -0.5]
h_grid=[1,2,3]
T_grid=[0,1]

##numbers that do not change
n_n=1000
n_a=40 ##this is the total periods in the model
n_data=20 #this is the total periods in the data
n_η=2
n_l=3
n_ξ=2
n_v=2
n_h=3
n_T=2
n_j=3

β=0.95

γ=Base.MathConstants.eulergamma #euler's constant


##parameters to be estimated


α0=0.00002
α1=0.03
αp=0.01
γ0=0.01
γ1=0.01
θ1=0.08
θ2=0.02
θ3=0.07
θ4=0.03
θ5=0.06
cost=0.05



v_d=10000 #the cutoff points
η_d=10000 #also normalised
ξ_d=0.1
σ_ϵ=0.5 #the preference shock
σ_ι=100 #the shock in wages, bigger than the shock in kappa preference



##draw a matrix of shocks for the initial backwards solve

Random.seed!(1234);

##some of these shocks probably only need n_data number
v=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #draw from binary distribution, this replaced by v_d is the actual shocks
v_index=copy(v) ##preparing to create an index of shocks #this is the index of the actual shocks

v[v.==1].=v_d #replace with the cutoffs
v[v.==0].=-v_d
v
v_index[v_index.==1].=2 #this is the grid conversion for the shock
v_index[v_index.==0].=1


ξ=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #the actual shocks
ξ_index=copy(ξ) #need to use copy not equal #the indexes

ξ[ξ.==1].=ξ_d #replace with the cutoffs
ξ[ξ.==0].=-ξ_d

ξ_index[ξ_index.==1].=2 #this is the grid conversion for the shock
ξ_index[ξ_index.==0].=1


η=convert(Array{Float64,1},rand(Binomial(1,0.5), n_n)) #the actual shocks
η[η.==1].=η_d #replace with the cutoffs ##why no index?
η[η.==0].=-η_d

ι= reshape(rand(Normal(0,σ_ι), n_n*n_l*n_a),n_n,n_l,n_a) #the wage shocks, this has a age dimension
ϵ= reshape(rand(Normal(0,σ_ϵ), n_n*n_a),n_n,n_a) #preference shock probably no need to differ by location, probably only need to have n_data periods
ζ= reshape(rand(Gumbel(0, 1), n_n*n_l*n_a),n_n,n_l,n_a) #this type one distribution thing should stay the same, j is ag n_l is location



#2880 states (including age) for each person
function Backwards()
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


@elapsed vxj, E_ζ, E_ϵ, pf_v, pf_l = Backwards()


##to calculate everything else just need df_20, the data df


df_20=filter(row -> row.t <1989, df)
data_20=Matrix(df_20)

id_d20=Integer.(data_20[:,1])
t_d20=Integer.(data_20[:,2])
age_d20=Integer.(data_20[:,3])
g_d20=Integer.(data_20[:,4])
school_d20=Integer.(data_20[:,5])
sib_d20=Integer.(data_20[:,6])
lp_d20=Integer.(data_20[:,7])
loc_in_d20=Integer.(data_20[:,8])
low_d20=Integer.(data_20[:,9])
ph_d20=Integer.(data_20[:,10])
G_d20=data_20[:,11]
T_d20=Integer.(data_20[:,12])


function probability_lambda() ##function returns probability, the inputs are vxj and v_bar
    lambda=zeros(n_n,n_data,n_l,n_ξ,n_v,n_h,n_T,n_l)
    for i=1:n_n, a= 1:n_data,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j= 1:n_j,
        lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
    return lambda
end

@elapsed lambda=probability_lambda()

minimum(BigFloat.(lambda))

function cutoffs()
    ϵ_cutoffs=zeros(n_n,n_data) #the care cytoff only depends on
    for i=1:n_n
        #use the person's actual information, these variables do not change over time
        g=g_d20[(i-1)*n_data+1]
        sib=sib_d20[(i-1)*n_data+1]
        lp=lp_d20[(i-1)*n_data+1]
        low=low_d20[(i-1)*n_data+1]

         for a = 1:n_data, i_T=1:n_T #this is number of periods
             T=T_d20[(i-1)*n_data+a]
             kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T - cost
             ϵ_cutoffs[i,a]=-kappa
         end
     end
     return ϵ_cutoffs
end ##this just computes the cutoffs

@elapsed ϵ_cutoffs =  cutoffs()


##use the initial value and the data's health and T values to solve forward



function solve_forward()
    l_o=zeros(n_n,n_data) #to store results, only solve forward up to when data is available
    l_j=zeros(n_n,n_data) #to store choices
    ic=zeros(n_n,n_data) #to store current locations
    for i=1:n_n
        lp =lp_d20[(i-1)*n_data+1]
        for a=1:n_data #going forward
            ph=ph_d20[(i-1)*n_data+a]
            i_T=T_d20[(i-1)*n_data+a] +1 #add 1 for index
            ϵ_star=ϵ_cutoffs[i,a]
            ϵ_now=ϵ[i,a]
            if a==1
                i_l=loc_in_d20[(i-1)*n_data+1]
                l_o[i,a]=i_l
            elseif 1<a<=n_a
                i_l=Integer(l_j[i,a-1]) #location is last period's choice
                l_o[i,a]=i_l
            end
            i_v=Integer(v_index[i,i_l]) #take the index of the shocks of the initial location
            i_ξ=Integer(ξ_index[i,i_l])
            l_j[i,a]=Integer(pf_l[i,a,i_l,i_ξ,i_v,ph,i_T]) #chosen location as a function of current locations
            if l_o[i,a]==lp && ph==2 && ϵ_now >=ϵ_star #if at parent location and parent has need and the shock is larger than cutoff
                ic[i,a]=1
            else
                ic[i,a]=0
            end
        end
    end
    return l_j, l_o, ic
end

l_j, l_o, ic=solve_forward()


####################

##Now put everything together

#####################

function v_data(v,lo) ##this use the shocks generated and the location to get location match shock in each period
    v_data=zeros(n_n*n_data)
    for i=1:n_n, a = 1:n_data
        location=Integer(l_o[i,a]) ##figure out the location in that period
        v_data[(i-1)*n_data+a]=v[i,location] #figure out the shock corresponding shock in that period by choosing from the v the relevant entry
    end
    return v_data
end

v_dat=v_data(v,l_o)

function η_data(η)
    η_data=zeros(n_n, n_data)
    for i=1:n_n
        η_data[i,:].=η[i]
    end
    return η_data
end

η_dat=η_data(η) #i does not depend on time

function ι_data(ι,lo)
    ι_data=zeros(n_n*n_data)
    for i=1:n_n, a = 1:n_data
        location=Integer(l_o[i,a])
        ι_data[(i-1)*n_data+a]=ι[i,location,a]
    end
    return ι_data
end

ι_dat=ι_data(ι,l_o)



function mu(μ,lo) #mean waves for each period depends on location
    mu_data=zeros(n_n*n_data)
    for i=1:n_n, a = 1:n_data
        location=Integer(l_o[i,a]) ##figure out the location in that period
        mu_data[(i-1)*n_data+a]=μ[location] #figure out the shock corresponding shock in that period by choosing from the v the relevant entry
    end
    return mu_data
end

μ_dat=mu(μ,l_o)


data_sim=DataFrames.DataFrame(hcat(data_20,vec(l_o'),vec(l_j'),vec(ic'),v_dat,vec(η_dat'),ι_dat,μ_dat),:auto)   ##can also just add stuff at the end

CSV.write("generated.csv", data_sim)





##play


u=α0.*μ./c + α1.*N

kappa= θ1*1 + θ2 + θ3 + θ4*hc[3] - cost

kappa= θ4*hc[2] - cost

moving=γ0 + γ1*44



##play
