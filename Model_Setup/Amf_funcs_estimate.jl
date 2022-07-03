#this is now a bit messy can probably get it to just compute vxj
#2880 states (including age) for each person
function Backwards(prim::Primitives, est::Estimands, res::Results)
    @unpack β0, β1, β2, β3, β4, β, γ=prim
    @unpack v_grid, η_grid, l_grid, ξ_grid, h_grid, T_grid, a_grid = prim #grids
    @unpack n_η, n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n =prim #dimensions
    @unpack markov_h, markov_T =prim
    @unpack α0,α1,αp,γ0,γ1,θ1,θ2,θ3,θ4,θ5,cost,v_d,η_d, ξ_d, σ_ϵ,σ_ι = est
    @unpack vxj,E_ζ, E_ϵ = res


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
        #println("this is", i, "person")
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


###############
all_perm(xs, n) = vec(map(collect, Iterators.product(ntuple(_ -> xs, n)...)))

function Likelihood(prim::Primitives, est::Estimands, res::Results, data_m::Array{Float64,2},locdata::Array{Float64,2})
    @unpack n_η, n_l, n_ξ, n_v, n_h, n_T, n_j, n_a, n_ad, n_n =prim #dimensions
    @unpack γ =prim #dimensions
    @unpack α0,α1,αp,γ0,γ1,θ1,θ2,θ3,θ4,θ5,cost,v_d,η_d, ξ_d, σ_ϵ,σ_ι = est
    @unpack vxj,E_ζ, E_ϵ = res

    v_grid =[-v_d v_d]
    ξ_grid =[-ξ_d ξ_d]
    η_grid =[-η_d η_d]
    μ=locdata[:,2]
    hc=locdata[:,4] #care cost

    g_m=Integer.(data_m[:,4])
    sib_m=Integer.(data_m[:,6])
    lp_m=Integer.(data_m[:,7])
    low_m=Integer.(data_m[:,9])
    ph_m=Integer.(data_m[:,10])
    G_m=data_m[:,11]
    T_m=Integer.(data_m[:,12])
    loc_m=Integer.(data_m[:,13])
    j_m=Integer.(data_m[:,14])
    ic_m=Integer.(data_m[:,15])
    ndistinct_m=Integer.(data_m[:,20])
    k_m=Integer.(data_m[:,21])
    w_m=data_m[:,22]

    log_L=0
    log_Li=zeros(n_n)

    for i=1:n_n
        N_total=Integer(ndistinct_m[(i-1)*n_ad+1])
        g=Integer(g_m[(i-1)*n_ad+1])
        sib=Integer(sib_m[(i-1)*n_ad+1])
        lp=Integer(lp_m[(i-1)*n_ad+1])
        low=Integer(low_m[(i-1)*n_ad+1])

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

            for a=1:n_ad #now for things that change through time, no next period info in last period
                ph=Integer(ph_m[(i-1)*n_ad+a])
                G=G_m[(i-1)*n_ad+a]
                T=Integer(T_m[(i-1)*n_ad+a]) +1
                loc=Integer(loc_m[(i-1)*n_ad+a])
                j=Integer(j_m[(i-1)*n_ad+a])
                κ=Integer(k_m[(i-1)*n_ad+a])
                w=w_m[(i-1)*n_ad+a]
                ic=Integer(ic_m[(i-1)*n_ad+a])

                kappa= θ1*g + θ2*(1-sib) + θ3*low + θ4*hc[lp] - θ5*T - cost

                ωv=ω_v[κ] ##this period's shock indices, for each person, k should be less than or equal to N total, this is 1 or 2
                ωxi=ω_xi[κ]
                ψ=(1/σ_ι)pdf(Normal(0,σ_ι), (w-μ[loc]-G- v_grid[ωv]-η_grid[ω_eta])/σ_ι) #the problem is psi is always zero
                    #println("this is", ψ, "psi"
                if a == n_ad
                    λ=1 #last period the care decision cannot be observed
                else
                    λ=exp(γ+ vxj[i,a,loc,ωxi,ωv,ph,T,j]-E_ζ[i,a,loc,ωxi,ωv,ph,T]) #compute the probability inside
                end

                if loc != lp || (loc == lp && ph!=2) #no care decision, probability is 1
                    ρ=1  # println("this is", ρ, "rho")
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
    return log_L
end
