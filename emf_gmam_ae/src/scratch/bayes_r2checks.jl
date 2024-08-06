
#Test t distribution
function testt(;iter, μ=-1.0, τ=1/2.3, ν=8.0)
    direct = rand(TDist(ν), iter)/τ^(0.5) .+ μ
    hierarch = rand(Gamma(ν/2,2/ν),iter) .|> ψi->rand(Normal(μ,(τ*ψi)^-0.5))
  
    @info "direct t stats (μ=$μ, τ=$τ, ν=$ν):"
    println("mean=",mean(direct))
    println("variance=",var(direct))
    println("kurtosis=",kurtosis(direct))
    println("median =", median(direct))
  
    @info "hierarch t stats:"
    println("mean=",mean(hierarch))
    println("variance=",var(hierarch))
    println("kurtosis=",kurtosis(hierarch))
    println("median=", median(hierarch))
  
    @info "caculated t stats"
    println("mean=", μ)
    println("var=",1/τ*(ν/(ν-2)))
    println("median=", μ)
  
  end
  
  testt(;iter=10^7)