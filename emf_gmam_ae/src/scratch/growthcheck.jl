using Distributions, Statistics, StatsBase

function est(;S,T, σ2f = 2, σ2ε=1, β = 0.1)
    @assert T≥ S
    f=rand(Normal(0,σ2f^0.5),T)
    err = rand(Normal(0,σ2ε^0.5), S)
    
    
    fs = view(f, 1:S)
    r = β*fs + err
    βest = cov(fs,r)/var(fs)
    


    return (;βest, σ=std(r), varfβest= var(f*βest),  varε=var(r-fs*βest),σβ=(var(f*βest) + var(r-fs*βest))^0.5 )
end

function accuracy(N;S,T,σ2f,σ2ε,β)
    rawests = [est(;S,T,σ2f,σ2ε,β) for n ∈ 1:N]
    σ = (rawest->rawest.σ).(rawests)
    σβ = (rawest->rawest.σβ).(rawests)
    βest = (rawest->rawest.βest).(rawests)
    varfβest = (rawest->rawest.varfβest).(rawests)
    varε = (rawest->rawest.varε).(rawests)

    analyticalσ=(2*(β^2*σ2f+σ2ε)^2/S)^0.5
    vβ = σ2ε/(σ2f*S)
    #analyticalσβ=((2*vβ^2+4*vβ*β^2)*σ2f^2+2*σ2ε^2/S)^0.5   
    μβ2=vβ+β^2
    σβ2=2*vβ^2+4*vβ*β^2
    #g = est(β2)*est(σ2f)
    ∇g = [σ2f,μβ2]
    Σβ2σ2f = [σβ2 0.0; 0.0 2*σ2f^2/T; ]
    analyticalσβ = (∇g'Σβ2σ2f*∇g+2*σ2ε^2/S)^0.5


    @info "var(βest)=$(var(βest)); σ2ε/(σ2f*S)=$(σ2ε/(σ2f*S))"  
    @info "var(varfβest)=$(var(varfβest)); ((2*(σ2ε/(σ2f*S))^2*(1+2β^2*(σ2ε/(σ2f*S))^-1))*σ2f^2=$(((2*(σ2ε/(σ2f*S))^2*(1+2β^2*(σ2ε/(σ2f*S))^-1))*σ2f^2))"  
    @info "var(varε)=$(var(varε)); 2*σ2ε^2/S=$(2*σ2ε^2/S)"  
    @info "var(r)=$(β^2*σ2f+σ2ε)"


    @info "Results with standard estimation: mean vol $(mean(σ)); mean var $(mean(σ.^2)); std dev var $(std(σ .^2)), analytical est = $analyticalσ"
    @info "Results using factor history: mean vol $(mean(σβ)); mean var $(mean(σβ.^2)); std dev var $(std(σβ .^2)), analytical est = $analyticalσβ"

    return nothing
end

@time accuracy(10^5; S=12, T=1000, σ2f = 1, σ2ε=10, β = 0)

  

