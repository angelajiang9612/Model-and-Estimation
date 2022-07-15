#this is now a bit messy can probably get it to just compute vxj
#2880 states (including age) for each person
function Backwards(prim::Primitives, est::Estimands, res::Results) #instead of doing this can just do prim=Primitives(), but this is only possible after initialize has already been completed, otherwise will est and res are not really defined.
    @unpack β0, β1, β2, β3, β4, β, γ=prim #alternatively can just use prim.β0
    @unpack v_grid, η_grid, l_grid, ξ_grid, h_grid, T_grid, a_grid = prim #grids
    @unpack n_η, n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n =prim #dimensions
    @unpack markov_h, markov_T =prim
    @unpack α0,α1,αp,γ0,γ1,θ1,θ2,θ3,θ4,θ5,cost,v_d,η_d, ξ_d, σ_ϵ,σ_ι = est
    @unpack vxj,E_ζ, E_ϵ,pf_v,pf_l = res


    age_d=Integer.(data[:,3]) #check what happens is used type declaration
    g_d=Integer.(data[:,4])
    school_d=Integer.(data[:,5])
    sib_d=Integer.(data[:,6])
    lp_d=Integer.(data[:,7])
    low_d=Integer.(data[:,9])
    G_d=data[:,11]

    μ=locdata[:,2] #mean wages
    c=locdata[:,3] #costs
    N=locdata[:,5] #population
    hc=locdata[:,4] #care cost

    Random.seed!(1234);
    η=convert(Array{Float64,1},rand(Binomial(1,0.5), n_n)) #the actual shocks, a different realisation of these shocks would give different value functions
    η[η.==1].=η_d #replace with the cutoffs ##why no index?
    η[η.==0].=-η_d
    ζ= reshape(rand(Gumbel(0, 1), n_n*n_l*n_a),n_n,n_l,n_a) #this type one distribution thing should stay the same, j is ag n_l is location

    @sync @distributed for i=1:n_n
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
                    wages= μ[i_l] + v_grid[i_v]+ G + η[i]
                    kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T_grid[i_T] - cost
                    u_l0= α0*wages/c[i_l] + α1*N[i_l] + αp*(i_l==lp) + (h_grid[i_h]==2)*kappa + ξ_grid[i_ξ]
                    if i_l != lp || (i_l == lp && i_h!=2)
                        res.E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 #########got to here
                    else
                        res.E_ϵ[i,a,i_l,i_ξ,i_v,i_h, i_T]=u_l0 + σ_ϵ*(cdf.(Normal(), kappa/σ_ϵ)*kappa/σ_ϵ + pdf.(Normal(), kappa/σ_ϵ))
                    end
                end
            elseif a < n_a #in other periods do the migration decision first (drom the back)
                for i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v, i_h = 1:n_h, i_T = 1:n_T
                        for i_j=1:n_l #next
                        if i_l == i_j #no location match shocks
                             #remember to use next period's things, a+1 and next period's shocks (in this case same as this period)
                            e_next=markov_h[i_h,1]*markov_T[i_T,1]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,1,1] + markov_h[i_h,1]*markov_T[i_T,2]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,1,2] + markov_h[i_h,2]*markov_T[i_T,1]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,2,1]
                            + markov_h[i_h,2]*markov_T[i_T,2]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,2,2] +  markov_h[i_h,3]*markov_T[i_T,1]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,3,1]
                            + markov_h[i_h,3]*markov_T[i_T,2]*res.E_ϵ[i,a+1,i_j,i_ξ,i_v,3,2] #this takes into consideration those expected values may all be the same in the case where i_j is not equal to lp

                        elseif i_j != i_l
                            e_next=0
                            E_ϵ_next=zeros(n_v,n_ξ,n_h,n_T) # a place to hold all the possibilities
                            for i_ξ_j = 1:n_ξ, i_v_j = 1:n_v, i_h_next = 1:n_h, i_T_next = 1:n_T #summing over all x'
                                E_ϵ_next[i_ξ_j,i_v_j,i_h_next,i_T_next] = res.E_ϵ[i,a+1,i_j,i_ξ_j,i_v_j,i_h_next,i_T_next] #pick out the expectation to use, again use all next period's shocks
                                e_next = e_next+ E_ϵ_next[i_ξ_j,i_v_j,i_h_next, i_T_next]*markov_h[i_h,i_h_next]*markov_T[i_T,i_T_next]*(1/n_v)*(1/n_ξ)
                            end
                        end
                        res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]= - Δ*(i_l!=i_j) + β*e_next
                        V_temp =res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]+ ζ[i,i_j,a]
                        if V_temp > res.pf_v[i,a,i_l,i_ξ,i_v,i_h,i_T]
                            res.pf_v[i,a,i_l,i_ξ,i_v,i_h,i_T]= V_temp
                            res.pf_l[i,a,i_l,i_ξ,i_v,i_h,i_T]= i_j #making the choice
                        end
                    end
                    constant=maximum(vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,:]) #defined this way so exp term don't blow up
                    res.E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T] = γ + constant + log(sum(exp.(res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,:].-constant)))
                    #now for the the new E_ϵ_next
                    wages= μ[i_l] + v_grid[i_v]+ G + η[i]
                    kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T_grid[i_T]- cost
                    u_l0= α0*wages/c[i_l] + α1*N[i_l] + αp*(i_l==lp) + h_grid[i_h]*kappa + ξ_grid[i_ξ]
                    if i_l != lp || (i_l == lp && i_h!=2)
                        res.E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 + res.E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T] #use this period's E_ζ state variable to compute E_ϵ
                    elseif i_l == lp && i_h==2
                        res.E_ϵ[i,a,i_l,i_ξ,i_v,i_h,i_T]=u_l0 + res.E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T]  + σ_ϵ*(cdf.(Normal(), kappa/σ_ϵ)*kappa/σ_ϵ + pdf.(Normal(), kappa/σ_ϵ))
                    end
                end
            end
        end
    end
    return res
end


function probability_lambda() ##function returns probability, the inputs are vxj and v_bar
    prim=Primitives()
    @unpack n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n,γ =prim #dimensions
    lambda=zeros(n_n,n_ad,n_l,n_ξ,n_v,n_h,n_T,n_l)
    for i=1:n_n, a= 1:n_ad,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j= 1:n_j,
        lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
    lambda
end


function cutoffs()
    prim=Primitives()
    est=Estimands()
    @unpack n_ad, n_n,n_T =prim
    @unpack θ1,θ2,θ3,θ4,θ5,cost =est

    ϵ_cutoffs=zeros(n_n,n_ad)

    hc=locdata[:,4] #care cost

    id_d20=Integer.(df_20[:,1])
    t_d20=Integer.(df_20[:,2])
    age_d20=Integer.(df_20[:,3])
    g_d20=Integer.(df_20[:,4])
    school_d20=Integer.(df_20[:,5])
    sib_d20=Integer.(df_20[:,6])
    lp_d20=Integer.(df_20[:,7])
    loc_in_d20=Integer.(df_20[:,8])
    low_d20=Integer.(df_20[:,9])
    ph_d20=Integer.(df_20[:,10])
    G_d20=df_20[:,11]
    T_d20=Integer.(df_20[:,12])

    for i=1:n_n
        #use the person's actual information, these variables do not change over time
        g=g_d20[(i-1)*n_ad+1]
        sib=sib_d20[(i-1)*n_ad+1]
        lp=lp_d20[(i-1)*n_ad+1]
        low=low_d20[(i-1)*n_ad+1]

         for a = 1:n_ad, i_T=1:n_T #this is number of periods
             T=T_d20[(i-1)*n_ad+a]
             kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T - cost
             ϵ_cutoffs[i,a]=-kappa
         end
     end
     ϵ_cutoffs
end ##this just computes the cutoffs


function solve_forward()
    prim=Primitives()
    est=Estimands()
    @unpack n_ad, n_n,n_l =prim
    @unpack v_grid, ξ_grid = prim #grids
    @unpack σ_ϵ =est

    Random.seed!(1234);
    ϵ= reshape(rand(Normal(0,σ_ϵ), n_n*n_ad),n_n,n_ad)

    v=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #draw from binary distribution, this replaced by v_d is the actual shocks
    v_index=copy(v) ##preparing to create an index of shocks #this is the index of the actual shocks
    v_index[v_index.==1].=2 #this is the grid conversion for the shock
    v_index[v_index.==0].=1


    ξ=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #the actual shocks
    ξ_index=copy(ξ) #need to use copy not equal #the indexes
    ξ_index[ξ_index.==1].=2 #this is the grid conversion for the shock
    ξ_index[ξ_index.==0].=1


    l_o=zeros(n_n,n_ad) #to store results, only solve forward up to when data is available
    l_j=zeros(n_n,n_ad) #to store choices
    ic=zeros(n_n,n_ad) #to store current location

    lp_d20=Integer.(df_20[:,7])
    ph_d20=Integer.(df_20[:,10])
    T_d20=Integer.(df_20[:,12])
    loc_in_d20=Integer.(df_20[:,8])

    for i=1:n_n
        lp =lp_d20[(i-1)*n_ad+1]
        for a=1:n_ad #going forward
            ph=ph_d20[(i-1)*n_ad+a]
            i_T=T_d20[(i-1)*n_ad+a] +1 #add 1 for index
            ϵ_star=ϵ_cutoffs[i,a]
            ϵ_now=ϵ[i,a]
            if a==1
                i_l=loc_in_d20[(i-1)*n_ad+1]
                l_o[i,a]=i_l
            elseif 1<a<=n_ad
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


function misc() ##drawing other things useful for data
    prim=Primitives()
    est=Estimands()

    @unpack n_ad, n_n, n_l,n_a =prim
    @unpack η_d,v_d,σ_ι = est

    Random.seed!(1234);
    η=convert(Array{Float64,1},rand(Binomial(1,0.5), n_n))
    η[η.==1].=η_d
    η[η.==0].=-η_d

    v=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #draw from binary distribution
    v[v.==1].=v_d #replace with the cutoffs ##why no index?
    v[v.==0].=-v_d

    ι= reshape(rand(Normal(0,σ_ι), n_n*n_l*n_ad),n_n,n_l,n_ad)

    v_data=zeros(n_n*n_ad)
    η_data=zeros(n_n,n_ad)
    ι_data=zeros(n_n*n_ad)
    mu_data=zeros(n_n*n_ad)

    μ=locdata[:,2] #mean wages

    for i=1:n_n, a = 1:n_ad
        location=Integer(l_o[i,a]) ##figure out the location in that period
        mu_data[(i-1)*n_ad+a]=μ[location]
        ι_data[(i-1)*n_ad+a]=ι[i,location,a]
        v_data[(i-1)*n_ad+a]=v[i,location]
        η_data[i,:].=η[i]
    end
    return v_data,vec(η_data'),ι_data,mu_data
end


#=


function probability_lambda(prim::Primitives,res::Results) ##function returns probability, the inputs are vxj and v_bar
    @unpack n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n,γ =prim #dimensions
    @unpack lambda,vxj,E_ζ = res
    for i=1:n_n, a= 1:n_ad,i_l = 1:n_l, i_ξ = 1:n_ξ, i_v = 1:n_v,i_h = 1:n_h,i_T= 1:n_T, i_j= 1:n_j,
        res.lambda[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]=exp(γ+ res.vxj[i,a,i_l,i_ξ,i_v,i_h,i_T,i_j]-E_ζ[i,a,i_l,i_ξ,i_v,i_h,i_T])
    end
end


function cutoffs(prim::Primitives,est::Estimands,res::Results)
    @unpack n_ad, n_n,n_T =prim
    @unpack θ1,θ2,θ3,θ4,θ5,cost =est
    @unpack ϵ_cutoffs = res

    hc=locdata[:,4] #care cost

    id_d20=Integer.(df_20[:,1])
    t_d20=Integer.(df_20[:,2])
    age_d20=Integer.(df_20[:,3])
    g_d20=Integer.(df_20[:,4])
    school_d20=Integer.(df_20[:,5])
    sib_d20=Integer.(df_20[:,6])
    lp_d20=Integer.(df_20[:,7])
    loc_in_d20=Integer.(df_20[:,8])
    low_d20=Integer.(df_20[:,9])
    ph_d20=Integer.(df_20[:,10])
    G_d20=df_20[:,11]
    T_d20=Integer.(df_20[:,12])

    for i=1:n_n
        #use the person's actual information, these variables do not change over time
        g=g_d20[(i-1)*n_ad+1]
        sib=sib_d20[(i-1)*n_ad+1]
        lp=lp_d20[(i-1)*n_ad+1]
        low=low_d20[(i-1)*n_ad+1]

         for a = 1:n_ad, i_T=1:n_T #this is number of periods
             T=T_d20[(i-1)*n_ad+a]
             kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T - cost
             res.ϵ_cutoffs[i,a]=-kappa
         end
     end
end ##this just computes the cutoffs


function solve_forward(prim::Primitives,res::Results)
    @unpack n_ad, n_n =prim
    @unpack v_grid, ξ_grid = prim #grids
    @unpack ϵ_cutoffs,pf_l= res

    Random.seed!(1234);
    ϵ= reshape(rand(Normal(0,σ_ι), n_n*n_ad),n_n,n_ad)

    l_o=zeros(n_n,n_ad) #to store results, only solve forward up to when data is available
    l_j=zeros(n_n,n_ad) #to store choices
    ic=zeros(n_n,n_ad) #to store current location

    lp_d20=Integer.(df_20[:,7])
    ph_d20=Integer.(df_20[:,10])
    T_d20=Integer.(df_20[:,12])
    loc_in_d20=Integer.(df_20[:,8])

    for i=1:n_n
        lp =lp_d20[(i-1)*n_ad+1]
        for a=1:n_ad #going forward
            ph=ph_d20[(i-1)*n_ad+a]
            i_T=T_d20[(i-1)*n_ad+a] +1 #add 1 for index
            ϵ_star=ϵ_cutoffs[i,a]
            ϵ_now=ϵ[i,a]
            if a==1
                i_l=loc_in_d20[(i-1)*n_ad+1]
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

function misc() ##drawing other things useful for data
    @unpack n_ad, n_n, n_l,n_a =prim
    @unpack η_d,v_d,σ_ι = est

    Random.seed!(1234);
    η=convert(Array{Float64,1},rand(Binomial(1,0.5), n_n))
    η[η.==1].=η_d
    η[η.==0].=-η_d

    v=convert(Array{Float64,2}, reshape(rand(Binomial(1,0.5), n_n*n_l),n_n,n_l)) #draw from binary distribution
    v[v.==1].=v_d #replace with the cutoffs ##why no index?
    v[v.==0].=-v_d

    ι= reshape(rand(Normal(0,σ_ι), n_n*n_l*n_ad),n_n,n_l,n_ad)

    v_data=zeros(n_n*n_ad)
    η_data=zeros(n_n*n_ad)
    ι_data=zeros(n_n*n_ad)
    mu_data=zeros(n_n*n_ad)

    μ=locdata[:,2] #mean wages

    for i=1:n_n, a = 1:n_ad
        location=Integer(l_o[i,a]) ##figure out the location in that period
        mu_data[(i-1)*n_ad+a]=μ[location]
        ι_data[(i-1)*n_ad+a]=ι[i,location,a]
        v_data[(i-1)*n_ad+a]=v[i,location]
        η_data[i,:].=η[i]
    end
    return v_dat,η_dat,ι_dat,μ_dat
end
=#
